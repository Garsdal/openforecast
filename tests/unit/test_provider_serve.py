"""The serving harness: the provider side of the wire, without a process.

The transport is tested against real subprocesses elsewhere. Here the streams
are in memory, so what is under test is the harness's own contract: a request
becomes a provider call, a failure becomes an error envelope rather than a
traceback, and stdout carries protocol and nothing else however loud the
provider is.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import openforecast as of
from openforecast.models import ModelDescriptor
from openforecast.protocol import PROTOCOL_VERSION, ErrorCode, Operation, Status, parse_response
from openforecast.providers import BUILTIN_PROVIDER, ProviderServer, serve
from openforecast.tasks.forecast import OutputSpec
from openforecast.views import ViewKind, ViewPlanner, ViewRequest, read_answer, write_view

MODEL = "builtin/seasonal-naive"
PARAMS = {"season_length": 2}

planner = ViewPlanner()


def frame(periods: int = 8) -> of.TimeSeriesFrame:
    rows: list[dict[str, Any]] = [
        {
            "zone": "DE",
            "timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=step),
            "load": float(step),
        }
        for step in range(periods)
    ]
    return of.TimeSeriesFrame.from_pandas(
        history=pd.DataFrame(rows),
        time="timestamp",
        frequency="1h",
        instance_keys=["zone"],
        targets=["load"],
    )


def request(operation: Operation, **fields: object) -> str:
    return json.dumps(
        {"protocol_version": PROTOCOL_VERSION, "operation": operation.value, **fields}
    )


def answer(server: ProviderServer, line: str) -> Any:
    return parse_response(json.loads(server.handle(line).model_dump_json()))


@pytest.fixture
def server() -> ProviderServer:
    return ProviderServer(BUILTIN_PROVIDER, stdout=io.StringIO(), stderr=io.StringIO())


# -- the three operations ---------------------------------------------------


def test_a_handshake_reports_the_provider_and_its_models(server: ProviderServer) -> None:
    response = answer(server, request(Operation.HANDSHAKE))

    assert response.is_ok
    assert response.payload["provider"] == "builtin"
    assert response.payload["provider_version"] == BUILTIN_PROVIDER.version
    advertised = [ModelDescriptor.model_validate(model) for model in response.payload["models"]]
    assert advertised == list(BUILTIN_PROVIDER.descriptors())
    assert [str(descriptor.ref) for descriptor in advertised] == [MODEL]


def test_a_fit_and_a_forecast_cross_the_boundary_as_bundles(
    server: ProviderServer, tmp_path: Path
) -> None:
    """The whole point: bulk data travels as Arrow, control as one line of JSON."""
    view = planner.fit_view(frame(), ViewRequest(kind=ViewKind.SERIES))
    bundle = write_view(view, tmp_path / "fit-view")
    state = tmp_path / "state"
    state.mkdir()

    fitted = answer(
        server,
        request(
            Operation.FIT,
            model=MODEL,
            params=PARAMS,
            view={"kind": "series", "path": str(bundle)},
            into=str(state),
        ),
    )

    assert fitted.is_ok
    assert list(state.iterdir()), "the provider persisted nothing"

    context = of.ForecastContext(origin_time="2026-01-01T07:00:00", frame=frame())
    inference = write_view(
        planner.forecast_view(context, ViewRequest(kind=ViewKind.FORECAST, horizon=2)),
        tmp_path / "forecast-view",
    )
    forecast = answer(
        server,
        request(
            Operation.FORECAST,
            model=MODEL,
            params=PARAMS,
            view={"kind": "forecast", "path": str(inference)},
            output=OutputSpec.point().model_dump(mode="json"),
            state=str(state),
            answer=str(tmp_path / "answer.arrow"),
        ),
    )

    assert forecast.is_ok
    table = read_answer(forecast.payload["answer"])
    assert table.num_rows == 2
    assert set(table.column("target").to_pylist()) == {"load"}


# -- failures ---------------------------------------------------------------


def test_a_line_that_is_not_a_request_is_answered_rather_than_crashed_on(
    server: ProviderServer,
) -> None:
    response = answer(server, '{"operation": "levitate"}')

    assert response.status is Status.ERROR
    assert response.error is not None
    assert response.error.code is ErrorCode.MALFORMED_REQUEST


def test_a_request_from_another_protocol_version_is_refused(server: ProviderServer) -> None:
    response = answer(
        server, json.dumps({"protocol_version": 99, "operation": Operation.HANDSHAKE.value})
    )

    assert response.error is not None
    assert response.error.code is ErrorCode.PROTOCOL_MISMATCH
    assert response.error.details == {"expected": PROTOCOL_VERSION, "received": 99}


def test_a_model_the_provider_does_not_advertise_is_an_unknown_model(
    server: ProviderServer, tmp_path: Path
) -> None:
    bundle = write_view(
        planner.fit_view(frame(), ViewRequest(kind=ViewKind.SERIES)), tmp_path / "view"
    )

    response = answer(
        server,
        request(
            Operation.FIT,
            model="builtin/nhits",
            view={"kind": "series", "path": str(bundle)},
            into=str(tmp_path),
        ),
    )

    assert response.error is not None
    assert response.error.code is ErrorCode.UNKNOWN_MODEL


def test_a_bundle_that_is_not_where_the_request_says_is_an_invalid_view(
    server: ProviderServer, tmp_path: Path
) -> None:
    response = answer(
        server,
        request(
            Operation.FIT,
            model=MODEL,
            view={"kind": "series", "path": str(tmp_path / "nothing")},
            into=str(tmp_path),
        ),
    )

    assert response.error is not None
    assert response.error.code is ErrorCode.INVALID_VIEW


def test_a_bundle_that_is_not_the_view_it_was_announced_as_is_refused(
    server: ProviderServer, tmp_path: Path
) -> None:
    """The kind travels beside the path so that the two can be checked against."""
    bundle = write_view(
        planner.fit_view(frame(), ViewRequest(kind=ViewKind.SERIES)), tmp_path / "view"
    )

    response = answer(
        server,
        request(
            Operation.FIT,
            model=MODEL,
            view={"kind": "sequences", "path": str(bundle)},
            into=str(tmp_path),
        ),
    )

    assert response.error is not None
    assert response.error.code is ErrorCode.INVALID_VIEW
    assert "announces a sequences view" in response.error.message


def test_a_forecast_asked_of_a_training_bundle_is_refused(
    server: ProviderServer, tmp_path: Path
) -> None:
    bundle = write_view(
        planner.fit_view(frame(), ViewRequest(kind=ViewKind.SERIES)), tmp_path / "view"
    )

    response = answer(
        server,
        request(
            Operation.FORECAST,
            model=MODEL,
            view={"kind": "series", "path": str(bundle)},
            state=str(tmp_path),
            answer=str(tmp_path / "answer.arrow"),
        ),
    )

    assert response.error is not None
    assert response.error.code is ErrorCode.INVALID_VIEW


def test_a_provider_that_raises_something_unexpected_still_answers() -> None:
    """A provider must answer, not die: the environment took a minute to start."""

    class Exploding:
        name = "boom"
        version = "0.0.1"

        def descriptors(self) -> tuple[Any, ...]:
            raise RuntimeError("the registry is on fire")

    server = ProviderServer(Exploding(), stdout=io.StringIO(), stderr=io.StringIO())  # type: ignore[arg-type]

    response = answer(server, request(Operation.HANDSHAKE))

    assert response.error is not None
    assert response.error.code is ErrorCode.INTERNAL_ERROR
    assert "the registry is on fire" in response.error.message
    assert "traceback" in response.error.details


# -- streams ----------------------------------------------------------------


def test_a_chatty_provider_cannot_corrupt_the_protocol_stream() -> None:
    """stdout is protocol only, and a provider does not have to be careful."""

    class Chatty:
        name = "chatty"
        version = "0.0.1"

        def descriptors(self) -> tuple[Any, ...]:
            print("Epoch 1/500 [====>    ]")  # noqa: T201 - the thing under test
            return ()

    out, log = io.StringIO(), io.StringIO()
    server = ProviderServer(Chatty(), stdout=out, stderr=log)  # type: ignore[arg-type]

    exit_code = server.run(iter([request(Operation.HANDSHAKE) + "\n", "\n"]))

    assert exit_code == 0
    assert "Epoch" in log.getvalue()
    assert [json.loads(line)["status"] for line in out.getvalue().splitlines()] == ["ok"]


def test_serving_ends_when_its_input_does() -> None:
    """The two lines an integration's ``__main__`` is: streams in, exit code out."""
    out = io.StringIO()

    exit_code = serve(
        BUILTIN_PROVIDER,
        stdin=io.StringIO(request(Operation.HANDSHAKE) + "\n"),
        stdout=out,
        stderr=io.StringIO(),
    )

    assert exit_code == 0
    assert json.loads(out.getvalue())["result"]["provider"] == "builtin"


def test_a_failed_request_does_not_end_the_conversation(server: ProviderServer) -> None:
    out = io.StringIO()
    server = ProviderServer(BUILTIN_PROVIDER, stdout=out, stderr=io.StringIO())

    server.run(iter(['{"nonsense": true}\n', request(Operation.HANDSHAKE) + "\n"]))

    statuses = [json.loads(line)["status"] for line in out.getvalue().splitlines()]
    assert statuses == ["error", "ok"]
