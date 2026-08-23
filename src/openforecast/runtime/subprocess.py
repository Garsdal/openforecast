"""A provider in another process and another environment, behaving like a local one.

```python
provider = SubprocessProvider([python, "-m", "openforecast_nixtla"])

provider.descriptors()                        # over a handshake
engine = Engine(providers=ProviderRegistry([provider]))
```

:class:`SubprocessProvider` implements the same three calls the in-process
provider does, so the engine cannot tell the difference — which is the whole
point of Step 9. What it adds is the transport: a long-lived child process,
JSON Lines for control, Arrow bundles for bulk data, and a set of failures that
only exist once a boundary does.

```text
handshake         who are you, and what do you provide
fit               here is a view bundle and a directory; train
forecast          here is an origin; write the answer here
```

The process is started lazily and kept alive. Starting a forecasting
environment costs seconds — imports, CUDA contexts, model registries — and a
provider that was restarted per call would pay that for every leaf of every
ensemble. What ends it is :meth:`SubprocessProvider.close`, or the end of the
program.

Five failure modes are treated as first-class rather than as surprises, because
each of them otherwise looks like a working provider:

```text
the process dies          reported with its exit code and its last log lines
it writes to stdout       stdout is protocol; anything else corrupts the stream
it answers nonsense       a line that is not a response is not an answer
it speaks another version bundles are laid out by that number, so it is refused
it never answers          a request has a deadline, and the process is killed
```

Anything the child writes to stderr is a log: it is drained continuously (a full
pipe would otherwise deadlock the child mid-fit), forwarded to the
``openforecast.provider.<name>`` logger, and the tail of it is quoted in the
error when something goes wrong — a provider's own message about the missing
CUDA driver is usually the answer.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import subprocess
import tempfile
import threading
from collections import deque
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import IO, Any, NoReturn

import pyarrow as pa
from pydantic import BaseModel, ValidationError

from openforecast.errors import (
    DataError,
    InvalidModelParameters,
    ProviderError,
    UnknownModelError,
)
from openforecast.models.descriptor import ModelDescriptor
from openforecast.models.ref import ModelRef
from openforecast.protocol import PROTOCOL_VERSION
from openforecast.protocol.messages import (
    ErrorCode,
    ErrorPayload,
    FitRequest,
    ForecastRequest,
    HandshakeRequest,
    HandshakeResult,
    Response,
    ViewRef,
    parse_response,
)
from openforecast.views.bundle import read_answer, write_view
from openforecast.views.forecast import ForecastView
from openforecast.views.planner import FitView

__all__ = ["DEFAULT_TIMEOUT", "SubprocessProvider"]

#: How long one request may take before the process is considered hung. Long,
#: because the request it is most likely to be measuring is a neural fit.
DEFAULT_TIMEOUT = 1800.0

#: How long a closing process is given to exit before it is killed.
SHUTDOWN_GRACE = 5.0

#: How long a failing process is given to exit, so that its exit code and its
#: last log lines can be reported rather than guessed at.
REAP_GRACE = 1.0

#: How many lines of the child's stderr are kept for error messages.
LOG_TAIL = 40

ANSWER_FILENAME = "answer.arrow"

_LOGGER = logging.getLogger("openforecast.provider")

# An error the provider raised is re-raised as the error the same failure would
# have been in-process, so that a caller's handling does not depend on where the
# model happened to run.
_CODE_ERRORS = {
    ErrorCode.UNKNOWN_MODEL: UnknownModelError,
    ErrorCode.INVALID_MODEL_PARAMETERS: InvalidModelParameters,
    ErrorCode.INVALID_VIEW: DataError,
}


class SubprocessProvider:
    """A provider executed by a child process speaking the wire protocol."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        name: str | None = None,
        version: str | None = None,
        descriptors: Sequence[ModelDescriptor] | None = None,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        if not command:
            raise ProviderError("a subprocess provider needs a command to run")
        self._command = tuple(command)
        self._cwd = None if cwd is None else Path(cwd)
        self._env = None if env is None else dict(env)
        self._timeout = timeout
        # Recorded at install time, so that listing models and registering them
        # starts no process. A handshake still verifies them before anything is
        # executed — see :meth:`_start`.
        self._name = name
        self._version = version
        self._descriptors = None if descriptors is None else tuple(descriptors)
        self._process: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._log: deque[str] = deque(maxlen=LOG_TAIL)
        self._pumps: tuple[threading.Thread, ...] = ()

    # -- what the engine asks of any provider ------------------------------

    @property
    def name(self) -> str:
        if self._name is None:
            self._start()
        assert self._name is not None  # noqa: S101 - established by the handshake
        return self._name

    @property
    def version(self) -> str:
        if self._version is None:
            self._start()
        assert self._version is not None  # noqa: S101 - established by the handshake
        return self._version

    @property
    def command(self) -> tuple[str, ...]:
        return self._command

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def descriptors(self) -> tuple[ModelDescriptor, ...]:
        """Every model the provider advertises, from its handshake."""
        if self._descriptors is None:
            self._start()
        assert self._descriptors is not None  # noqa: S101 - established by the handshake
        return self._descriptors

    def fit(
        self,
        *,
        model: ModelRef | str,
        params: Mapping[str, Any],
        view: FitView,
        seed: int | None,
        into: Path,
    ) -> None:
        """Send ``view`` as a bundle and wait for the fit to finish."""
        with _workspace(self.name) as workspace:
            bundle = write_view(view, workspace / "view")
            self._exchange(
                FitRequest(
                    model=str(model),
                    params=dict(params),
                    seed=seed,
                    view=ViewRef(kind=view.kind, path=str(bundle)),
                    into=str(into),
                )
            )

    def forecast(
        self,
        *,
        model: ModelRef | str,
        params: Mapping[str, Any],
        view: ForecastView,
        output: Mapping[str, Any],
        state: Path,
    ) -> pa.Table:
        """Send one origin and read back the Arrow forecast the provider wrote."""
        with _workspace(self.name) as workspace:
            bundle = write_view(view, workspace / "view")
            answer = workspace / ANSWER_FILENAME
            self._exchange(
                ForecastRequest(
                    model=str(model),
                    params=dict(params),
                    view=ViewRef(kind=view.kind, path=str(bundle)),
                    output=dict(output),
                    state=str(state),
                    answer=str(answer),
                )
            )
            if not answer.is_file():
                raise ProviderError(
                    f"{self.name} reported a successful forecast and wrote nothing to "
                    f"{answer.name}{self._tail()}"
                )
            return read_answer(answer)

    # -- lifecycle ----------------------------------------------------------

    def _start(self) -> None:
        """Spawn the process and handshake with it, once."""
        if self.is_running:
            return
        try:
            self._process = subprocess.Popen(  # noqa: S603 - the command is the caller's
                self._command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=None if self._cwd is None else str(self._cwd),
                env=None if self._env is None else {**os.environ, **self._env},
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except OSError as error:
            raise ProviderError(
                f"the provider command {list(self._command)} could not be started: {error}"
            ) from error
        self._lines = queue.Queue()
        self._log = deque(maxlen=LOG_TAIL)
        # ``None`` on the line queue is end of output: a child that exited owes
        # an answer it can no longer give, and the reader must not wait for it.
        self._pumps = (
            _drain(self._process.stdout, self._lines.put, lambda: self._lines.put(None)),
            _drain(self._process.stderr, self._record),
        )
        self._handshake()

    def _handshake(self) -> None:
        """Establish who answered, and check it is who was expected.

        A mismatch stops the process on the way out. Whatever is running is not
        the provider that was asked for, and leaving it alive would leave the
        next call talking to it anyway.
        """
        result = HandshakeResult.model_validate(self._exchange(HandshakeRequest()).payload)
        advertised = tuple(ModelDescriptor.model_validate(payload) for payload in result.models)
        if self._name is not None and result.provider != self._name:
            self._kill()
            raise ProviderError(
                f"the command {list(self._command)} was installed as provider "
                f"{self._name!r} and identifies itself as {result.provider!r}; reinstall it"
            )
        if self._descriptors is not None and advertised != self._descriptors:
            self._kill()
            raise ProviderError(
                f"{result.provider} advertises different models than were recorded when it "
                f"was installed; reinstall it so that what is discoverable is what runs"
            )
        self._name = result.provider
        self._version = result.provider_version
        self._descriptors = advertised

    def close(self) -> None:
        """Close the child's input and wait for it to exit."""
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        if process.stdin is not None:
            with suppress(OSError, ValueError):  # already gone, or already closed
                process.stdin.close()
        try:
            process.wait(timeout=SHUTDOWN_GRACE)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def __enter__(self) -> SubprocessProvider:
        self._start()
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"SubprocessProvider(name={self._name!r}, command={list(self._command)}, "
            f"running={self.is_running})"
        )

    # -- the wire -----------------------------------------------------------

    def _exchange(self, request: BaseModel) -> Response:
        """Send one request and return the successful response to it."""
        if not isinstance(request, HandshakeRequest):
            self._start()
        response = self._transact(request)
        if response.protocol_version != PROTOCOL_VERSION:
            self._fail(
                f"speaks protocol version {response.protocol_version} and this build speaks "
                f"{PROTOCOL_VERSION}; install a provider built for this version"
            )
        if response.error is not None:
            raise self._error(response.error)
        return response

    def _transact(self, request: BaseModel) -> Response:
        process = self._process
        if process is None or process.stdin is None:  # pragma: no cover - _start guarantees it
            raise ProviderError(f"the provider {self._label()} is not running")
        try:
            process.stdin.write(request.model_dump_json() + "\n")
            process.stdin.flush()
        except (BrokenPipeError, ValueError):
            self._fail("exited before it could be asked anything", ended=True)
        return self._read()

    def _read(self) -> Response:
        """The next line of stdout, as a response, or an explanation of why not."""
        try:
            line = self._lines.get(timeout=self._timeout)
        except queue.Empty:
            self._kill()
            raise ProviderError(
                f"the provider {self._label()} did not answer within {self._timeout:g}s and "
                f"was stopped{self._tail()}"
            ) from None
        if line is None:
            self._fail("exited without answering", ended=True)
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            self._fail(
                f"wrote something that is not a protocol message to stdout: {line.strip()!r}; "
                f"stdout carries protocol only, and logs belong on stderr"
            )
        try:
            return parse_response(payload)
        except ValidationError as error:
            self._fail(f"answered with something that is not a response: {error}")

    def _error(self, payload: ErrorPayload) -> Exception:
        """The provider's envelope as the exception it would have been in-process.

        The details cross unchanged, and the provider's own code joins them: the
        envelope this raises reports the OpenForecast code a caller branches on,
        so the finer-grained thing the provider actually said stays available
        without being the thing recovery depends on.
        """
        cls = _CODE_ERRORS.get(payload.code, ProviderError)
        return cls(
            f"{self._label()} failed [{payload.code}]: {payload.message}",
            provider=self._label(),
            provider_code=str(payload.code),
            **payload.details,
        )

    def _fail(self, what: str, *, ended: bool = False) -> NoReturn:
        """Stop the process and raise; the transport is not trusted after this.

        ``ended`` says the child's output has already closed, which is the one
        case worth waiting on: a process that just died has usually not been
        reaped yet, and the lines explaining why are still in flight. Reporting
        no exit code and no log for a provider that crashed would drop the most
        useful thing there is to say.
        """
        status = self._reap() if ended else ""
        self._kill()
        raise ProviderError(f"the provider {self._label()} {what}{status}{self._tail()}")

    def _reap(self) -> str:
        process = self._process
        if process is None:  # pragma: no cover - only reachable after _kill
            return ""
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=REAP_GRACE)
        for pump in self._pumps:
            pump.join(timeout=REAP_GRACE)
        code = process.poll()
        return "" if code is None else f" (exit code {code})"

    def _kill(self) -> None:
        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()

    def _record(self, line: str) -> None:
        text = line.rstrip("\n")
        self._log.append(text)
        _LOGGER.getChild(self._name or "unknown").debug("%s", text)

    def _tail(self) -> str:
        if not self._log:
            return ""
        return "\n--- provider log ---\n" + "\n".join(self._log)

    def _label(self) -> str:
        return self._name if self._name is not None else str(list(self._command))


def _drain(
    stream: IO[str] | None,
    sink: Callable[[str], None],
    on_close: Callable[[], None] | None = None,
) -> threading.Thread:
    """Read ``stream`` on a thread until it closes, then call ``on_close``.

    Continuously, and on a thread, for two reasons: a pipe nobody reads fills up
    and blocks the child mid-fit, and a request has to be able to time out while
    the child is still producing output.
    """

    def pump() -> None:
        if stream is not None:  # pragma: no branch - the pipes are always requested
            try:
                for line in stream:
                    sink(line)
            finally:
                if on_close is not None:
                    on_close()

    thread = threading.Thread(target=pump, daemon=True, name="openforecast-provider")
    thread.start()
    return thread


@contextmanager
def _workspace(name: str) -> Generator[Path]:
    """A directory for one request's bundles, removed when the request is done."""
    path = Path(tempfile.mkdtemp(prefix=f"openforecast-{name}-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
