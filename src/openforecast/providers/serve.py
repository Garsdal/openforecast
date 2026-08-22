"""The provider side of the wire: read a request, do the work, answer.

An integration's ``__main__`` is expected to be two lines:

```python
from openforecast.providers import serve
from openforecast_nixtla.provider import NixtlaProvider

raise SystemExit(serve(NixtlaProvider()))
```

Everything the transport requires happens here rather than in each integration:
requests are validated, view bundles are read back into real execution views,
answers are written as Arrow, and failures become the standard error envelope
instead of a traceback on a stream that is supposed to hold protocol.

Two properties are worth stating outright.

**stdout is protocol only.** Forecasting libraries print — progress bars,
convergence notices, deprecation warnings — and a single stray line would make
the next response unparseable. So the harness replaces ``sys.stdout`` with
``sys.stderr`` for the duration of every provider call and writes responses to
the real stream it captured at startup. A provider does not have to be careful;
it has to be correct.

**A failed request does not end the process.** An exception becomes an error
response and the loop continues, because the failure the caller cares about is
"this fit did not work", and a provider that dies also loses the environment it
took a minute to start. What does end the loop is end of input, which is how the
engine says it is finished.
"""

from __future__ import annotations

import json
import sys
import traceback
from collections.abc import Iterable
from contextlib import redirect_stdout
from pathlib import Path
from typing import IO, Any

from pydantic import ValidationError

from openforecast.errors import (
    DataError,
    ModelError,
    OpenForecastError,
    RecipeError,
    SchemaError,
)
from openforecast.protocol import PROTOCOL_VERSION
from openforecast.protocol.messages import (
    ErrorCode,
    FitRequest,
    FitResult,
    ForecastRequest,
    ForecastResult,
    HandshakeRequest,
    HandshakeResult,
    Request,
    Response,
    ViewRef,
    parse_request,
)
from openforecast.providers.client import ProviderClient
from openforecast.views import ViewKind
from openforecast.views.bundle import read_fit_view, read_forecast_view, write_answer

__all__ = ["ProviderServer", "serve"]


def serve(
    provider: ProviderClient,
    *,
    stdin: IO[str] | None = None,
    stdout: IO[str] | None = None,
    stderr: IO[str] | None = None,
) -> int:
    """Answer requests on ``stdin`` until it closes. Returns a process exit code."""
    server = ProviderServer(provider, stdout=stdout, stderr=stderr)
    return server.run(stdin if stdin is not None else sys.stdin)


class ProviderServer:
    """One provider, answering the wire protocol on a pair of streams."""

    def __init__(
        self,
        provider: ProviderClient,
        *,
        stdout: IO[str] | None = None,
        stderr: IO[str] | None = None,
    ) -> None:
        self._provider = provider
        # Captured now, so that redirecting ``sys.stdout`` during a provider
        # call cannot take the protocol stream with it.
        self._out = stdout if stdout is not None else sys.stdout
        self._log = stderr if stderr is not None else sys.stderr

    def run(self, stdin: Iterable[str]) -> int:
        """Read one request per line until the stream ends."""
        for line in stdin:
            if not line.strip():
                continue
            self._respond(self.handle(line))
        return 0

    def handle(self, line: str) -> Response:
        """The response to one request line, whatever it turns out to be."""
        try:
            request = parse_request(json.loads(line))
        except (ValidationError, ValueError) as error:
            return Response.failed(
                ErrorCode.MALFORMED_REQUEST,
                f"the request could not be parsed: {error}",
            )
        if request.protocol_version != PROTOCOL_VERSION:
            return Response.failed(
                ErrorCode.PROTOCOL_MISMATCH,
                f"this provider speaks protocol version {PROTOCOL_VERSION} and the "
                f"request declares {request.protocol_version}",
                {"expected": PROTOCOL_VERSION, "received": request.protocol_version},
            )
        return self._dispatch(request)

    def _dispatch(self, request: Request) -> Response:
        try:
            if isinstance(request, HandshakeRequest):
                return Response.of(self._handshake())
            if isinstance(request, FitRequest):
                return Response.of(self._fit(request))
            return Response.of(self._forecast(request))
        except ModelError as error:
            return _failure(ErrorCode.UNKNOWN_MODEL, error)
        except RecipeError as error:
            # A recipe error out here is the model rejecting its own parameters:
            # everything OpenForecast owns was checked before the request left.
            return _failure(ErrorCode.INVALID_MODEL_PARAMETERS, error)
        except (DataError, SchemaError) as error:
            return _failure(ErrorCode.INVALID_VIEW, error)
        except OpenForecastError as error:
            return _failure(ErrorCode.EXECUTION_FAILED, error)
        except Exception as error:  # noqa: BLE001 - a provider must answer, not die
            return _failure(ErrorCode.INTERNAL_ERROR, error, {"traceback": traceback.format_exc()})

    # -- the three operations ----------------------------------------------

    def _handshake(self) -> HandshakeResult:
        with self._quiet():
            descriptors = self._provider.descriptors()
            name = self._provider.name
            version = self._provider.version
        return HandshakeResult(
            provider=name,
            provider_version=version,
            models=tuple(descriptor.model_dump(mode="json") for descriptor in descriptors),
        )

    def _fit(self, request: FitRequest) -> FitResult:
        view = read_fit_view(_bundle(request.view, expected=None))
        if view.kind is not request.view.kind:
            raise DataError(
                f"the request announces a {request.view.kind} view and the bundle at "
                f"{request.view.path} holds a {view.kind} one"
            )
        with self._quiet():
            self._provider.fit(
                model=request.model,
                params=request.params,
                view=view,
                seed=request.seed,
                into=Path(request.into),
            )
        return FitResult(model=request.model)

    def _forecast(self, request: ForecastRequest) -> ForecastResult:
        view = read_forecast_view(_bundle(request.view, expected=ViewKind.FORECAST))
        with self._quiet():
            answer = self._provider.forecast(
                model=request.model,
                params=request.params,
                view=view,
                output=request.output,
                state=Path(request.state),
            )
        write_answer(answer, request.answer)
        return ForecastResult(answer=request.answer)

    # -- streams ------------------------------------------------------------

    def _quiet(self) -> redirect_stdout[IO[str]]:
        """Everything a provider prints goes to the log stream, never to stdout."""
        return redirect_stdout(self._log)

    def _respond(self, response: Response) -> None:
        self._out.write(response.model_dump_json() + "\n")
        self._out.flush()


def _bundle(ref: ViewRef, expected: ViewKind | None) -> Path:
    if expected is not None and ref.kind is not expected:
        raise DataError(f"a {expected} view was expected and the request names a {ref.kind} one")
    path = Path(ref.path)
    if not path.is_dir():
        raise DataError(f"no view bundle at {path}")
    return path


def _failure(code: ErrorCode, error: Exception, details: dict[str, Any] | None = None) -> Response:
    return Response.failed(
        code, f"{type(error).__name__}: {error}", {"error": type(error).__name__, **(details or {})}
    )
