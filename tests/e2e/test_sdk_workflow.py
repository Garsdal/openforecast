"""Step 24's "done when": the entire normal workflow, in one file.

> The entire normal workflow can be taught on one README page.

Which is only true if the page is true, so this is that page executed. One
client, built the way the README builds one, and then every operation on it:
discovery, a fit, a forecast off what the fit returned, and a backtest that
compares two candidates — with no object constructed that the short form does
not need.

The reference provider is the only model involved, deliberately: what is under
test is the shape of the API, and a workflow that needed a forecasting library
installed to be taught would not be the normal one.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import openforecast as of

MODEL = "builtin/seasonal-naive"
START = datetime(2026, 1, 1)
HOUR = timedelta(hours=1)


def at(step: int) -> datetime:
    return START + HOUR * step


@pytest.fixture
def client(tmp_path: Path) -> of.OpenForecast:
    return of.OpenForecast(store=tmp_path / "openforecast")


@pytest.fixture
def data() -> of.TimeSeriesFrame:
    rows: list[dict[str, Any]] = [
        {"zone": "DE", "timestamp": at(step), "load": float(step)} for step in range(48)
    ]
    return of.TimeSeriesFrame.from_pandas(
        history=pd.DataFrame(rows),
        time="timestamp",
        frequency="1h",
        instance_keys=["zone"],
        targets=["load"],
    )


def test_the_whole_workflow_is_one_client_and_four_calls(
    client: of.OpenForecast, data: of.TimeSeriesFrame
) -> None:
    """Discovery, fit, forecast, backtest. Nothing else has to be named."""
    assert MODEL in {str(descriptor.ref) for descriptor in client.models.list()}
    assert client.models.get(MODEL).provider == "builtin"

    model = client.fit(MODEL, data=data, name="de-load")
    assert str(model.ref).startswith("local/de-load@")

    forecast = client.forecast(model, data=data, horizon=24)
    assert forecast.horizon == 24
    assert forecast.point().num_rows == 24

    result = client.backtest(
        [MODEL, of.Candidate(of.Model(MODEL, params={"season_length": 2}), name="sn-2")],
        data=data,
        validation=of.RollingOrigin(horizon=6, windows=2),
        metrics=[of.MAE()],
    )
    assert result.models == (MODEL, "sn-2")
    assert result.best("mae") in result.models


def test_a_method_and_its_function_are_the_same_call(
    client: of.OpenForecast, data: of.TimeSeriesFrame
) -> None:
    """``client.backtest(...)`` is ``of.backtest(..., client=client)``, exactly.

    Signature parity is asserted in ``tests/unit/test_sdk_surface.py``; what is
    checked here is that the two produce the same numbers, so the method is a
    spelling of the function rather than a second implementation of it.
    """
    arguments: dict[str, Any] = {
        "data": data,
        "validation": of.RollingOrigin(horizon=6, windows=2),
        "metrics": [of.MAE(), of.Bias()],
    }
    through_the_method = client.backtest([MODEL], **arguments)
    through_the_function = of.backtest([MODEL], client=client, **arguments)

    assert through_the_method.metrics.column("value").to_pylist() == (
        through_the_function.metrics.column("value").to_pylist()
    )

    assert client.eligible_models(data, horizon=6, models=[MODEL]) == of.eligible_models(
        data, horizon=6, models=[MODEL], client=client
    )


def test_the_short_form_needs_no_request_objects(
    client: of.OpenForecast, data: of.TimeSeriesFrame
) -> None:
    """A reference, some data, a horizon — and the explicit forms still fit.

    Both of these are the same fit said two ways, which is what "more explicit
    forms remain available" has to mean if the short one is not a special case.
    """
    short = client.fit(MODEL, data=data, horizon=24, params={"season_length": 24})
    explicit = client.fit(
        of.Model(MODEL, params={"season_length": 24}),
        data=data,
        plan=of.FitPlan(origins=of.LatestOrigin()),
        horizon=24,
    )

    assert client.forecast(short, data=data, horizon=24).point().to_pylist() == (
        client.forecast(explicit, data=data, horizon=24).point().to_pylist()
    )


def test_a_forecast_off_a_reference_reads_the_same_as_off_a_handle(
    client: of.OpenForecast, data: of.TimeSeriesFrame
) -> None:
    """The handle prints as the reference, and both are what a forecast takes."""
    model = client.fit(MODEL, data=data, name="de-load")

    by_handle = client.forecast(model, data=data, horizon=12)
    by_alias = client.forecast("local/de-load", data=data, horizon=12)
    by_revision = client.forecast(str(model.ref), data=data, horizon=12)

    assert by_handle.point().to_pylist() == by_alias.point().to_pylist()
    assert by_handle.point().to_pylist() == by_revision.point().to_pylist()
    assert client.artifact(str(model.ref)).ref == model.ref
