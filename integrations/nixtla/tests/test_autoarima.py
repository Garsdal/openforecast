"""``nixtla/autoarima`` through the public API, on both semantic sources.

Step 11's claim is that StatsForecast works through the isolated provider
without understanding point-in-time semantics — so what is asserted here is the
behavior a caller sees:

```text
event time     fit a frame, forecast the steps after it
point in time  fit one selected vintage, and be refused every vintage at once
lifecycle      a model reference that was never fitted is not silently fitted
```

The targets are straight lines, which an ARIMA with a drift term continues
exactly, so the numbers below are arithmetic rather than a recorded blob.
"""

from __future__ import annotations

from pathlib import Path

import golden
import pytest
from golden import AUTOARIMA, at

import openforecast as of
from openforecast.errors import DataError, ModelRequiresFit, OriginScopeError, RecipeError
from openforecast.protocol import ForecastColumn

HORIZON = 3


def test_a_straight_line_is_continued_from_the_end_of_the_history(tmp_path: Path) -> None:
    frame = golden.event_time_frame(periods=24)
    client = golden.client(tmp_path)

    handle = client.fit(AUTOARIMA, frame, name="de-load")
    forecast = client.forecast(handle, frame, horizon=HORIZON)

    assert forecast.origin_time == at(23)
    assert forecast.event_times == (at(24), at(25), at(26))
    assert golden.values(forecast) == pytest.approx([240.0, 250.0, 260.0], abs=1e-6)
    assert handle.manifest.provider == "nixtla"
    assert handle.manifest.provider_version == golden.PROVIDER.version


def test_a_panel_is_one_model_per_series_labeled_with_its_instance(tmp_path: Path) -> None:
    """A local model fitted on three series answers as three series."""
    frame = golden.event_time_frame(instances=3, periods=24)
    client = golden.client(tmp_path)

    forecast = client.forecast(client.fit(AUTOARIMA, frame, name="zones"), frame, horizon=HORIZON)
    table = forecast.table

    assert forecast.instance_keys == ("zone",)
    assert table.num_rows == 3 * HORIZON
    zones: list[str] = table.column("zone").to_pylist()
    values: list[float] = table.column(ForecastColumn.VALUE.value).to_pylist()
    by_zone = {
        zone: [value for value, name in zip(values, zones, strict=True) if name == zone]
        for zone in ("DE", "FR", "NL")
    }
    for index, zone in enumerate(("DE", "FR", "NL")):
        expected = [golden.target_value(index, step) for step in (24, 25, 26)]
        assert by_zone[zone] == pytest.approx(expected, abs=1e-6)


def test_known_features_are_handed_to_the_model_as_exogenous_regressors(
    tmp_path: Path,
) -> None:
    """A known feature is a value the model may condition a future step on."""
    frame = golden.event_time_frame(periods=24, future_periods=HORIZON, known=True)
    client = golden.client(tmp_path)

    forecast = client.forecast(
        client.fit(AUTOARIMA, frame, name="with-exog"), frame, horizon=HORIZON
    )

    assert forecast.event_times == (at(24), at(25), at(26))
    assert golden.values(forecast) == pytest.approx([240.0, 250.0, 260.0], rel=1e-3)


def test_a_forecast_dataset_at_one_origin_is_an_ordinary_series(tmp_path: Path) -> None:
    """The point-in-time API of Step 11: one vintage becomes one SeriesView."""
    dataset = golden.point_in_time_dataset(origins=6, first_origin=2)
    origin = at(7)
    client = golden.client(tmp_path)

    handle = client.fit(
        AUTOARIMA, dataset, plan=of.FitPlan(origins=of.AtOrigin(origin)), name="vintage"
    )
    forecast = client.forecast(handle, dataset.at_origin(origin), horizon=HORIZON)

    assert handle.training.origin_fidelity == "observed"
    assert forecast.origin_time == origin
    assert forecast.event_times == (at(8), at(9), at(10))
    assert golden.values(forecast) == pytest.approx([80.0, 90.0, 100.0], rel=1e-3)


def test_learning_across_every_vintage_at_once_is_refused(tmp_path: Path) -> None:
    """AutoARIMA does not learn jointly across historical forecast origins."""
    dataset = golden.point_in_time_dataset()
    client = golden.client(tmp_path)

    with pytest.raises(OriginScopeError, match="one forecast origin"):
        client.fit(AUTOARIMA, dataset, plan=of.FitPlan(origins=of.AllOrigins()), name="all")


def test_a_model_reference_that_was_never_fitted_is_not_fitted_here(tmp_path: Path) -> None:
    """``of.forecast(model="nixtla/autoarima", ...)`` is a lifecycle error."""
    client = golden.client(tmp_path)

    with pytest.raises(ModelRequiresFit, match="nixtla/autoarima"):
        client.forecast(AUTOARIMA, golden.event_time_frame(periods=24), horizon=HORIZON)


def test_forecasting_from_an_origin_the_model_was_not_fitted_at_is_refused(
    tmp_path: Path,
) -> None:
    """A local model continues the series it saw; it does not re-read a new one.

    ``predict`` extrapolates from the last observation of the fit, so answering
    at a different origin would produce the right numbers for the wrong event
    times. The alternative to refusing is fitting again, which is a fit.
    """
    client = golden.client(tmp_path)
    handle = client.fit(AUTOARIMA, golden.event_time_frame(periods=24), name="de-load")

    with pytest.raises(DataError, match="fitted on"):
        client.forecast(handle, golden.event_time_frame(periods=20), horizon=HORIZON)


def test_a_parameter_the_model_does_not_have_is_refused_by_name(tmp_path: Path) -> None:
    client = golden.client(tmp_path)
    frame = golden.event_time_frame(periods=24)

    with pytest.raises(RecipeError, match=r"no parameter \['nonsense'\]"):
        client.fit(AUTOARIMA, frame, params={"nonsense": 1}, name="broken")

    with pytest.raises(RecipeError, match="season_length of at least 1"):
        client.fit(AUTOARIMA, frame, params={"season_length": 0}, name="broken")

    with pytest.raises(RecipeError, match="season_length as integer"):
        client.fit(AUTOARIMA, frame, params={"season_length": True}, name="broken")

    assert not list((tmp_path / "models").glob("*")), "a refused fit left an artifact"


def test_the_parameters_are_compiled_into_the_native_model(tmp_path: Path) -> None:
    """Restricting the order search changes the forecast, which is how we know.

    Left free, AutoARIMA picks a drift term and continues the line exactly.
    Restricted to a stationary model with no AR or MA terms, it cannot, and the
    difference is the parameters arriving where they were meant to.
    """
    frame = golden.event_time_frame(periods=48)
    client = golden.client(tmp_path)

    free = client.forecast(client.fit(AUTOARIMA, frame, name="free"), frame, horizon=HORIZON)
    restricted = client.forecast(
        client.fit(
            AUTOARIMA,
            frame,
            params={"season_length": 24, "stationary": True, "max_p": 0, "max_q": 0},
            name="restricted",
        ),
        frame,
        horizon=HORIZON,
    )

    assert golden.values(free) == pytest.approx([480.0, 490.0, 500.0], abs=1e-6)
    assert golden.values(restricted) != pytest.approx(golden.values(free), abs=1.0)
