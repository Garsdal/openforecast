"""Step 16's "done when", over a real socket.

> ``OpenForecast(LocalTransport())`` and ``OpenForecast(HttpTransport(...))``
> provide the same user-facing forecasting semantics.

So the tests below are written as comparisons rather than as assertions about
plumbing, the same way Step 9's subprocess suite is: two clients, one local and
one talking to a service over HTTP, are handed the same data and the same calls,
and what comes back has to be equal. A difference in a status code or a header
is invisible to that; a difference in what a forecast *is* is not.

The service is a real ``uvicorn`` in a thread, reached over loopback. Nothing is
patched, and the two clients own separate artifact stores — so an artifact
fitted remotely is resolved remotely, by reference, which is the one thing that
genuinely changes shape when the model lives somewhere else.
"""

from __future__ import annotations

import socket
import threading
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import uvicorn

import openforecast as of
from openforecast.errors import DataError, ModelRequiresFit, UnknownModelError
from openforecast.server.app import create_app
from openforecast.server.transport import HttpTransport, LocalTransport

MODEL = "builtin/seasonal-naive"
PARAMS = {"season_length": 4}
START = datetime(2026, 1, 1)
HOUR = timedelta(hours=1)


def at(step: int) -> datetime:
    return START + HOUR * step


def frame(periods: int = 12) -> of.TimeSeriesFrame:
    """A panel whose value at hour ``t`` is ``t`` plus an offset per zone."""
    rows: list[dict[str, Any]] = [
        {"zone": zone, "timestamp": at(step), "load": float(step + offset * 100)}
        for offset, zone in enumerate(("DE", "FR"))
        for step in range(periods)
    ]
    return of.TimeSeriesFrame.from_pandas(
        history=pd.DataFrame(rows),
        time="timestamp",
        frequency="1h",
        instance_keys=["zone"],
        targets=["load"],
    )


def dataset() -> of.ForecastDataset:
    """Real vintages: every origin published its own view of every event time."""
    rows: list[dict[str, Any]] = [
        {
            "zone": "DE",
            "ref_time": at(origin),
            "target_time": at(event),
            "load": float(event),
            "wind_fc": float(origin * 100 + event),
        }
        for origin in range(12)
        for event in range(12)
    ]
    return of.ForecastDataset.from_pandas(
        pd.DataFrame(rows),
        origin_time="ref_time",
        event_time="target_time",
        targets=["load"],
        instance_keys=["zone"],
        known_features=["wind_fc"],
        event_frequency="1h",
        origin_frequency="1h",
    )


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port: int = probe.getsockname()[1]
    return port


@pytest.fixture(scope="module")
def service(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """A real OpenForecast service, on loopback, for the length of this module."""
    root = tmp_path_factory.mktemp("service")
    port = free_port()
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(LocalTransport(store=root / "openforecast")),
            host="127.0.0.1",
            port=port,
            log_level="warning",
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        _await_ready(server)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def _await_ready(server: uvicorn.Server, attempts: int = 200) -> None:
    """Wait for the socket to be listening, rather than for a fixed sleep."""
    for _ in range(attempts):
        if server.started:
            return
        threading.Event().wait(0.05)
    raise AssertionError("the forecasting service did not start")


@pytest.fixture
def remote(service: str) -> of.OpenForecast:
    return of.OpenForecast(transport=HttpTransport(service, timeout=60.0))


@pytest.fixture
def local(tmp_path: Path) -> of.OpenForecast:
    return of.OpenForecast(store=tmp_path / "openforecast")


# -- the same semantics -----------------------------------------------------


def test_both_transports_discover_the_same_models(
    local: of.OpenForecast, remote: of.OpenForecast
) -> None:
    assert local.models.list() == remote.models.list()
    assert local.models.get(MODEL) == remote.models.get(MODEL)


def test_a_descriptor_crosses_without_losing_what_it_declares(
    local: of.OpenForecast, remote: of.OpenForecast
) -> None:
    """What ``of.models.get`` is for: planning against a model without running it."""
    here = local.models.get(MODEL)
    there = remote.models.get(MODEL)

    assert there.lifecycle == here.lifecycle
    assert there.training.view == here.training.view
    assert there.capabilities == here.capabilities
    assert there.parameters_schema == here.parameters_schema


def test_the_same_fit_and_forecast_produce_the_same_numbers(
    local: of.OpenForecast, remote: of.OpenForecast
) -> None:
    """The claim of Step 16, as a comparison rather than as an assertion."""
    data = frame()
    here = local.forecast(local.fit(MODEL, data, params=PARAMS, name="parity"), data, horizon=4)
    there = remote.forecast(remote.fit(MODEL, data, params=PARAMS, name="parity"), data, horizon=4)

    assert there.table.equals(here.table)
    assert there.origin_time == here.origin_time
    assert there.targets == here.targets
    assert there.instance_keys == here.instance_keys
    assert there.to_wide().equals(here.to_wide())


def test_point_in_time_data_crosses_as_point_in_time_data(
    local: of.OpenForecast, remote: of.OpenForecast
) -> None:
    """Real vintages, fitted at one origin and forecast from another one.

    The dataset is the thing most likely to be quietly flattened on the way
    across: it is two frames with axes that have to agree, and an origin that
    must not move. Comparing the forecasts is what says it did not.
    """
    data = dataset()
    context = data.at_origin(at(8))
    plan = of.FitPlan(origins=of.AtOrigin(at(8)))

    here = local.forecast(
        local.fit(MODEL, data, params=PARAMS, plan=plan, name="pit"), context, horizon=3
    )
    there = remote.forecast(
        remote.fit(MODEL, data, params=PARAMS, plan=plan, name="pit"), context, horizon=3
    )

    assert there.table.equals(here.table)
    assert there.origin_time == at(8)


def test_a_pipeline_crosses_as_a_recipe_rather_than_as_a_reference(
    local: of.OpenForecast, remote: of.OpenForecast
) -> None:
    """Fitted transform statistics are the service's, and are inverted there."""
    recipe = of.Pipeline(
        steps=(
            of.StandardScaler(columns=of.ColumnSet.TARGETS),
            of.Model(MODEL, params=PARAMS),
        )
    )
    data = frame()

    here = local.forecast(local.fit(recipe, data, name="scaled"), data, horizon=4)
    there = remote.forecast(remote.fit(recipe, data, name="scaled"), data, horizon=4)

    assert there.table.equals(here.table)


def test_the_artifact_a_remote_fit_published_is_described_by_reference(
    remote: of.OpenForecast,
) -> None:
    """A fitted model is a resource with an identity; the reference is the handle."""
    handle = remote.fit(MODEL, frame(), params=PARAMS, name="remote-only")

    assert str(handle.ref).startswith("local/remote-only@")
    assert remote.artifact("local/remote-only").ref == handle.ref
    assert remote.artifact(handle.ref).manifest.provider == "builtin"


def test_a_remote_alias_follows_the_latest_fit(remote: of.OpenForecast) -> None:
    first = remote.fit(MODEL, frame(), params=PARAMS, name="rolling")
    second = remote.fit(MODEL, frame(periods=16), params=PARAMS, name="rolling")

    assert first.ref != second.ref
    assert remote.artifact("local/rolling").ref == second.ref


def test_a_backtest_runs_over_a_transport_without_knowing_it(
    local: of.OpenForecast, remote: of.OpenForecast
) -> None:
    """Step 17 over Step 16: backtesting is a loop over fit and forecast.

    So it inherits the property this module exists to assert. The models are
    fitted and forecast on the service and scored here, and the measurements have
    to be the ones a local backtest of the same data produces — everything that
    differs between the two runs is a timing or an artifact reference.
    """
    data = frame()
    validation = of.RollingOrigin(horizon=2, windows=2)

    here = of.backtest(
        models=[MODEL], data=data, validation=validation, metrics=[of.MAE()], client=local
    )
    there = of.backtest(
        models=[MODEL], data=data, validation=validation, metrics=[of.MAE()], client=remote
    )

    scored = ["model", "fold", "origin", "metric", "value", "pairs", "origin_fidelity", "provider"]
    assert there.table.select(scored).equals(here.table.select(scored))
    assert there.best("mae") == here.best("mae") == MODEL


# -- the same failures ------------------------------------------------------


def test_an_unknown_model_is_the_error_it_would_have_been_here(
    remote: of.OpenForecast,
) -> None:
    with pytest.raises(UnknownModelError, match="nowhere/nothing"):
        remote.models.get("nowhere/nothing")


def test_forecasting_an_unfitted_reference_still_refuses(remote: of.OpenForecast) -> None:
    """The failure crosses as the failure, not as a status code."""
    with pytest.raises(ModelRequiresFit, match="has to be fitted"):
        remote.forecast(MODEL, frame(), horizon=4)


def test_data_the_model_was_not_fitted_on_is_refused_remotely(
    remote: of.OpenForecast,
) -> None:
    handle = remote.fit(MODEL, frame(), params=PARAMS, name="mismatch")
    other = of.TimeSeriesFrame.from_pandas(
        history=pd.DataFrame(
            [{"zone": "DE", "timestamp": at(step), "price": float(step)} for step in range(12)]
        ),
        time="timestamp",
        frequency="1h",
        instance_keys=["zone"],
        targets=["price"],
    )

    with pytest.raises(DataError, match="was fitted to forecast"):
        remote.forecast(handle, other, horizon=4)


def test_an_artifact_that_was_never_fitted_is_reported_as_missing(
    remote: of.OpenForecast,
) -> None:
    with pytest.raises(UnknownModelError):
        remote.artifact("local/never-fitted")
