"""The transforms OpenForecast executes itself, on the views a provider sees.

The property that matters is not that scaling is arithmetically right — it is
that the *same* statistics are used at fit and at inference, and that what comes
back out is on the caller's scale. A scaler that recomputed its mean from the
forecast context would be wrong in a way no output makes visible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import openforecast as of
from openforecast.errors import RecipeError, UnsupportedPlanError
from openforecast.protocol import ForecastColumn, forecast_columns
from openforecast.runtime.transforms import (
    apply_to_forecast_view,
    fit_transforms,
    invert_forecast,
    read_state,
    write_state,
)
from openforecast.views import SeriesView, ViewKind, ViewPlanner, ViewRequest
from tests import artifacts, factories, providers

PLANNER = ViewPlanner()
SCALER = of.StandardScaler(columns=of.ColumnSet.TARGETS)


def scaled(view: SeriesView, *transforms: of.StandardScaler) -> tuple[SeriesView, list[Any]]:
    result, _ = fit_transforms(view, transforms)
    assert isinstance(result, SeriesView)
    return result, result.temporal.column("load").to_pylist()


def test_scaling_centers_every_instance_on_its_own_history() -> None:
    """A 40 GW zone and a 2 GW zone become learnable by one model."""
    view, values = scaled(artifacts.series_view(artifacts.frame()), SCALER)

    assert view.temporal.column("load").to_pylist() == values
    assert sum(values[:8]) == pytest.approx(0.0)
    assert values[:8] == pytest.approx(values[8:])


def test_scaling_across_the_panel_keeps_the_levels_apart() -> None:
    """``per_instance=False`` is for when the difference between series is signal."""
    _, values = scaled(
        artifacts.series_view(artifacts.frame()),
        of.StandardScaler(columns=of.ColumnSet.TARGETS, per_instance=False),
    )

    assert values[0] < 0 < values[-1]
    assert values[:8] != pytest.approx(values[8:])


def test_a_constant_series_scales_to_zero_rather_than_to_nan() -> None:
    history = factories.history(instances=("DE",), instance_key="zone", periods=6)
    history["load"] = 5.0
    frame = of.TimeSeriesFrame.from_pandas(
        history=history,
        time="timestamp",
        frequency="1h",
        instance_keys=["zone"],
        targets=["load"],
    )

    _, values = scaled(artifacts.series_view(frame), SCALER)

    assert values == [0.0] * 6


def test_a_missing_value_stays_missing_and_does_not_move_the_mean() -> None:
    history = factories.history(instances=("DE",), instance_key="zone", periods=6)
    history.loc[history.index[0], "load"] = factories.NAN
    frame = of.TimeSeriesFrame.from_pandas(
        history=history,
        time="timestamp",
        frequency="1h",
        instance_keys=["zone"],
        targets=["load"],
    )

    _, values = scaled(artifacts.series_view(frame), SCALER)

    assert values[0] is None
    assert sum(value for value in values[1:] if value is not None) == pytest.approx(0.0)


def test_features_can_be_scaled_too() -> None:
    view, state = fit_transforms(
        artifacts.series_view(artifacts.frame()),
        (of.StandardScaler(columns=of.ColumnSet.FEATURES),),
    )
    assert isinstance(view, SeriesView)

    assert view.temporal.column("temp_fc").to_pylist()[:8] == pytest.approx(
        view.temporal.column("temp_fc").to_pylist()[8:]
    )
    assert state.steps[0].columns == ("temp_fc",)


def test_a_transform_naming_nothing_in_the_view_says_so() -> None:
    with pytest.raises(RecipeError, match="does not hold"):
        fit_transforms(
            artifacts.series_view(artifacts.frame()),
            (of.StandardScaler(columns=("nope",)),),
        )


def test_only_the_scaler_executes_today() -> None:
    """A pipeline that silently skipped a step would look like one that ran it."""
    with pytest.raises(UnsupportedPlanError, match="not executable yet"):
        fit_transforms(
            artifacts.series_view(artifacts.frame()),
            (of.MissingIndicator(columns=of.ColumnSet.FEATURES),),
        )


def test_a_tabular_view_has_no_executable_transform_yet() -> None:
    with pytest.raises(UnsupportedPlanError, match="tabular view"):
        fit_transforms(artifacts.tabular_view(), (SCALER,))


def test_inference_is_scaled_by_the_statistics_that_were_fitted(tmp_path: Path) -> None:
    """Not by the context's own, which would leak whatever it happens to hold."""
    frame = artifacts.frame()
    _, state = fit_transforms(artifacts.series_view(frame), (SCALER,))
    write_state(tmp_path / "transforms.json", state)
    reloaded = read_state(tmp_path / "transforms.json")
    context = of.ForecastContext(origin_time=artifacts.at(7), frame=frame)
    view = PLANNER.forecast_view(context, ViewRequest(kind=ViewKind.FORECAST, horizon=2))

    applied = apply_to_forecast_view(view, reloaded)

    assert reloaded == state
    assert applied.history.column("load").to_pylist()[:8] == pytest.approx(
        applied.history.column("load").to_pylist()[8:]
    )


def test_a_forecast_comes_back_on_the_scale_the_data_was_on() -> None:
    """Inverting is the other half; without it the model answers in deviations."""
    frame = artifacts.frame()
    _, state = fit_transforms(artifacts.series_view(frame), (SCALER,))
    context = of.ForecastContext(origin_time=artifacts.at(7), frame=frame)
    view = PLANNER.forecast_view(context, ViewRequest(kind=ViewKind.FORECAST, horizon=2))
    answer = providers.flat_answer(view, value=0.0)

    restored = invert_forecast(answer, ("zone",), state)

    values = restored.column(ForecastColumn.VALUE.value).to_pylist()
    # A scaled zero is the instance's own mean: DE averages 3.5, FR 1003.5.
    assert values == pytest.approx([3.5, 3.5, 1003.5, 1003.5])
    assert restored.column_names == list(forecast_columns(("zone",)))


def test_nothing_fitted_means_nothing_applied(tmp_path: Path) -> None:
    state = read_state(tmp_path / "transforms.json")

    assert state.steps == ()
