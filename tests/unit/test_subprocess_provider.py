"""The transport: a provider in another process, and everything that can go wrong.

Every test here runs a real child process. The happy path is the least
interesting part — what matters is that each way a provider can misbehave
produces an error that says what happened, rather than a forecast that looks
fine. A provider that dies mid-fit, prints a progress bar onto the protocol
stream, answers a different protocol version or simply never answers are all
things that will happen, and none of them may be mistaken for a result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from openforecast import DataError, ProviderError, RecipeError, UnknownModelError
from openforecast.protocol import PROTOCOL_VERSION, ErrorCode, Operation
from openforecast.runtime.subprocess import SubprocessProvider
from openforecast.views import ForecastView
from tests import wire

HANDSHAKE: dict[str, Any] = {
    "provider": "fake",
    "provider_version": "9.9.9",
    "models": [],
}


def ok(result: object, *, version: int = PROTOCOL_VERSION) -> str:
    return json.dumps({"protocol_version": version, "status": "ok", "result": result})


def failed(code: str, message: str = "no") -> str:
    return json.dumps(
        {
            "protocol_version": PROTOCOL_VERSION,
            "status": "error",
            "error": {"code": code, "message": message},
        }
    )


def responder(tmp_path: Path, body: str) -> list[str]:
    """A provider whose whole behavior is the ``respond`` function in ``body``."""
    return wire.script(tmp_path, body, wire.ANSWERS_ANYTHING)


# -- handshake --------------------------------------------------------------


def test_a_handshake_is_how_a_provider_says_what_it_is(tmp_path: Path) -> None:
    command = responder(tmp_path, f"def respond(request): return {ok(HANDSHAKE)!r}\n")

    with SubprocessProvider(command) as provider:
        assert provider.name == "fake"
        assert provider.version == "9.9.9"
        assert provider.descriptors() == ()
        assert provider.is_running


def test_the_handshake_happens_once_and_is_the_first_thing_sent(tmp_path: Path) -> None:
    """A second question must not re-introduce itself."""
    command = responder(
        tmp_path,
        f"""
        seen = []

        def respond(request):
            seen.append(request["operation"])
            assert seen == ["handshake"], seen
            return {ok(HANDSHAKE)!r}
        """,
    )

    with SubprocessProvider(command) as provider:
        assert provider.name == "fake"
        assert provider.descriptors() == ()
        assert provider.version == "9.9.9"


def test_a_provider_that_is_not_who_it_was_installed_as_is_refused(tmp_path: Path) -> None:
    command = responder(tmp_path, f"def respond(request): return {ok(HANDSHAKE)!r}\n")
    provider = SubprocessProvider(command, name="nixtla")

    with pytest.raises(ProviderError, match=r"installed as provider 'nixtla'"):
        provider.descriptors()

    # Whatever is running is not the provider that was asked for, so it is
    # stopped rather than left for the next call to talk to.
    assert not provider.is_running


def test_a_recorded_provider_advertises_nothing_until_it_is_used(tmp_path: Path) -> None:
    """Discovery is a recorded handshake, so listing models starts no process."""
    provider = SubprocessProvider(
        responder(tmp_path, "def respond(request): raise SystemExit(1)\n"),
        name="fake",
        version="9.9.9",
        descriptors=(),
    )

    assert provider.name == "fake"
    assert provider.version == "9.9.9"
    assert provider.descriptors() == ()
    assert not provider.is_running


def test_a_command_that_cannot_be_started_says_so() -> None:
    with pytest.raises(ProviderError, match=r"could not be started"):
        SubprocessProvider(["./definitely-not-a-provider"]).descriptors()


def test_a_provider_needs_a_command_to_run() -> None:
    with pytest.raises(ProviderError, match=r"needs a command"):
        SubprocessProvider([])


# -- failure modes ----------------------------------------------------------


def test_a_provider_that_dies_is_reported_with_its_exit_code_and_its_log(
    tmp_path: Path,
) -> None:
    command = wire.script(
        tmp_path,
        """
        import sys

        print("CUDA driver not found", file=sys.stderr)
        sys.stderr.flush()
        raise SystemExit(3)
        """,
    )

    with pytest.raises(ProviderError) as failure:
        SubprocessProvider(command).descriptors()

    assert "exited without answering" in str(failure.value)
    assert "exit code 3" in str(failure.value)
    assert "CUDA driver not found" in str(failure.value)


def test_something_that_is_not_protocol_on_stdout_is_a_violation(tmp_path: Path) -> None:
    """stdout is protocol only. A progress bar there is not noise to skip."""
    command = wire.script(
        tmp_path,
        """
        import sys

        for line in sys.stdin:
            sys.stdout.write("Epoch 1/500 [====>       ]\\n")
            sys.stdout.flush()
        """,
    )

    with pytest.raises(ProviderError, match=r"not a protocol message to stdout"):
        SubprocessProvider(command).descriptors()


def test_a_response_that_is_not_a_response_is_not_an_answer(tmp_path: Path) -> None:
    command = responder(tmp_path, 'def respond(request): return {"status": "maybe"}\n')

    with pytest.raises(ProviderError, match=r"not a response"):
        SubprocessProvider(command).descriptors()


def test_a_provider_speaking_another_protocol_version_is_refused(tmp_path: Path) -> None:
    command = responder(tmp_path, f"def respond(request): return {ok(HANDSHAKE, version=99)!r}\n")

    with pytest.raises(ProviderError, match=r"speaks protocol version 99"):
        SubprocessProvider(command).descriptors()


def test_a_provider_that_never_answers_is_stopped(tmp_path: Path) -> None:
    command = responder(tmp_path, "def respond(request): return None\n")
    provider = SubprocessProvider(command, timeout=0.5)

    with pytest.raises(ProviderError, match=r"did not answer within 0.5s"):
        provider.descriptors()
    assert not provider.is_running


def test_a_torrent_of_logs_does_not_stop_a_provider_from_answering(tmp_path: Path) -> None:
    """A pipe nobody drains fills up, and the child blocks with the answer unsent."""
    command = responder(
        tmp_path,
        f"""
        import sys

        def respond(request):
            for index in range(20000):
                print(f"line {{index}} of a very chatty library", file=sys.stderr)
            return {ok(HANDSHAKE)!r}
        """,
    )

    with SubprocessProvider(command, timeout=30) as provider:
        assert provider.name == "fake"


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (ErrorCode.UNKNOWN_MODEL, UnknownModelError),
        (ErrorCode.INVALID_MODEL_PARAMETERS, RecipeError),
        (ErrorCode.INVALID_VIEW, DataError),
        (ErrorCode.EXECUTION_FAILED, ProviderError),
        (ErrorCode.INTERNAL_ERROR, ProviderError),
    ],
)
def test_an_error_envelope_raises_what_the_same_failure_would_raise_locally(
    tmp_path: Path, code: ErrorCode, expected: type[Exception]
) -> None:
    command = responder(
        tmp_path,
        f"""
        def respond(request):
            if request["operation"] == {Operation.HANDSHAKE.value!r}:
                return {ok(HANDSHAKE)!r}
            return {failed(code.value, "it did not work")!r}
        """,
    )

    with SubprocessProvider(command) as provider, pytest.raises(expected, match=r"did not work"):
        provider.forecast(
            model="fake/thing",
            params={},
            view=_forecast_view(),
            output={},
            state=tmp_path,
        )


def test_a_forecast_that_wrote_no_answer_is_not_an_empty_forecast(tmp_path: Path) -> None:
    command = responder(
        tmp_path,
        f"""
        def respond(request):
            if request["operation"] == {Operation.HANDSHAKE.value!r}:
                return {ok(HANDSHAKE)!r}
            return {ok({"answer": "/nowhere/answer.arrow"})!r}
        """,
    )

    with SubprocessProvider(command) as provider, pytest.raises(ProviderError, match=r"wrote noth"):
        provider.forecast(
            model="fake/thing",
            params={},
            view=_forecast_view(),
            output={},
            state=tmp_path,
        )


def test_closing_a_provider_twice_is_not_an_error(tmp_path: Path) -> None:
    command = responder(tmp_path, f"def respond(request): return {ok(HANDSHAKE)!r}\n")
    provider = SubprocessProvider(command)
    provider.descriptors()

    provider.close()
    provider.close()

    assert not provider.is_running
    assert "fake" in repr(provider)


def _forecast_view() -> ForecastView:
    """One origin, the smallest one that is still a valid view."""
    from tests.unit.test_view_bundle import forecast_view

    return forecast_view()
