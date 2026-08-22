"""The built-in reference model, against the views it actually consumes.

Everything here goes through ``SeriesView`` and ``ForecastView`` rather than
through the engine: this is the provider's own contract — what it persists, what
it answers, and what it refuses — and the engine is tested separately against a
provider that does nothing.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

import openforecast as of
from openforecast.errors import DataError, ProviderError, RecipeError
from openforecast.protocol import ForecastColumn
from openforecast.providers.builtin import seasonal_naive
from openforecast.views import ForecastView, SeriesView, ViewKind, ViewPlanner, ViewRequest
from tests import artifacts, factories

PLANNER = ViewPlanner()
POINT: dict[str, Any] = of.OutputSpec.point().model_dump(mode="json")


def series_view(**overrides: Any) -> SeriesView:
    return artifacts.series_view(artifacts.frame(**overrides))


def forecast_view(
    horizon: int, frame: of.TimeSeriesFrame | None = None, origin: datetime | None = None
) -> ForecastView:
    """The inference view at the end of the frame's history."""
    frame = artifacts.frame() if frame is None else frame
    context = of.ForecastContext(
        origin_time=artifacts.at(7) if origin is None else origin, frame=frame
    )
    return PLANNER.forecast_view(context, ViewRequest(kind=ViewKind.FORECAST, horizon=horizon))


def fitted(tmp_path: Path, view: SeriesView | None = None, **params: Any) -> Path:
    seasonal_naive.fit(series_view() if view is None else view, params, tmp_path)
    return tmp_path


def values(view: ForecastView, state: Path) -> list[float | None]:
    answer = seasonal_naive.forecast(view, POINT, state)
    return answer.column(ForecastColumn.VALUE.value).to_pylist()


# -- the descriptor ---------------------------------------------------------


def test_the_descriptor_declares_a_local_series_model() -> None:
    """Its training unit is one complete series, which is what makes it local."""
    descriptor = seasonal_naive.descriptor("builtin")

    assert str(descriptor.ref) == "builtin/seasonal-naive"
    assert descriptor.training.view is ViewKind.SERIES
    assert not descriptor.training.horizon_bound_at_fit
    assert descriptor.lifecycle.requires_fit


def test_it_declares_that_it_consumes_missing_values_natively() -> None:
    """A season-ago gap yields a gap; nothing is filled in on the way through."""
    capabilities = seasonal_naive.descriptor("builtin").capabilities

    assert capabilities.tolerates_missing_values
    assert capabilities.outputs.point
    assert not capabilities.outputs.quantiles


# -- fitting ----------------------------------------------------------------


def test_fitting_persists_one_season_per_series(tmp_path: Path) -> None:
    state = fitted(tmp_path, season_length=3)

    persisted = (state / seasonal_naive.STATE_FILENAME).read_text(encoding="utf-8")
    assert '"season_length": 3' in persisted
    assert '"DE"' in persisted and '"FR"' in persisted


def test_fitting_refuses_a_season_longer_than_the_series(tmp_path: Path) -> None:
    """Cycling a shorter history would invent a season nobody observed."""
    with pytest.raises(DataError, match="needs 99 observations"):
        fitted(tmp_path, season_length=99)


def test_fitting_refuses_a_view_it_does_not_train_on(tmp_path: Path) -> None:
    with pytest.raises(ProviderError, match="one complete time series"):
        seasonal_naive.fit(artifacts.sequence_view(), {"season_length": 2}, tmp_path)


@pytest.mark.parametrize("season_length", [0, -1, 2.5, "24", True])
def test_the_season_length_is_a_positive_integer(tmp_path: Path, season_length: object) -> None:
    with pytest.raises(RecipeError, match="season_length"):
        fitted(tmp_path, season_length=season_length)


def test_an_unknown_parameter_is_refused_rather_than_ignored(tmp_path: Path) -> None:
    with pytest.raises(RecipeError, match="takes no parameter"):
        fitted(tmp_path, seasonality=24)


# -- forecasting ------------------------------------------------------------


def test_the_forecast_repeats_the_last_season(tmp_path: Path) -> None:
    """``season_length=3`` on a rising series repeats the last three values."""
    state = fitted(tmp_path, season_length=3)

    forecast = values(forecast_view(horizon=4), state)
    # The panel is DE 0..7 and FR 1000..1007; the view lists DE first.
    assert forecast[:4] == [5.0, 6.0, 7.0, 5.0]
    assert forecast[4:] == [1005.0, 1006.0, 1007.0, 1005.0]


def test_a_season_of_one_repeats_the_last_observation(tmp_path: Path) -> None:
    state = fitted(tmp_path, season_length=1)

    assert values(forecast_view(horizon=3), state) == [7.0, 7.0, 7.0, 1007.0, 1007.0, 1007.0]


def test_a_missing_observation_makes_a_missing_forecast(tmp_path: Path) -> None:
    """The model knows what it remembers, and it remembers a gap as a gap."""
    history = factories.history(instances=("DE",), instance_key="zone", periods=8)
    history.loc[history.index[-1], "load"] = factories.NAN
    frame = of.TimeSeriesFrame.from_pandas(
        history=history,
        time="timestamp",
        frequency="1h",
        instance_keys=["zone"],
        targets=["load"],
    )
    state = fitted(tmp_path, artifacts.series_view(frame), season_length=1)

    assert values(forecast_view(horizon=2, frame=frame), state) == [None, None]


def test_an_instance_it_never_saw_has_no_model_to_forecast_with(tmp_path: Path) -> None:
    """A local model is fitted per series, so an unseen series has no parameters."""
    state = fitted(tmp_path, artifacts.series_view(artifacts.frame(instances=("DE",))))

    with pytest.raises(DataError, match="no model for instance"):
        values(forecast_view(horizon=2), state)


def test_it_refuses_to_answer_a_question_it_cannot_answer(tmp_path: Path) -> None:
    state = fitted(tmp_path, season_length=2)

    with pytest.raises(ProviderError, match="point forecasts"):
        seasonal_naive.forecast(forecast_view(horizon=2), {"kind": "quantiles"}, state)


def test_forecasting_without_fitted_state_says_so(tmp_path: Path) -> None:
    with pytest.raises(ProviderError, match="no fitted state"):
        values(forecast_view(horizon=2), tmp_path)


def test_forecasting_from_state_that_is_not_its_own_says_so(tmp_path: Path) -> None:
    (tmp_path / seasonal_naive.STATE_FILENAME).write_text('{"model": "other"}', encoding="utf-8")

    with pytest.raises(ProviderError, match="does not hold the fitted state"):
        values(forecast_view(horizon=2), tmp_path)


def test_forecasting_off_the_grid_of_the_fitted_series_is_refused(tmp_path: Path) -> None:
    """A forecast view built from a different series would answer the wrong phase."""
    state = fitted(tmp_path, season_length=2)
    start = artifacts.START + timedelta(days=7, minutes=30)
    later = artifacts.frame(start=start)

    with pytest.raises(DataError, match="not a 1h step after"):
        values(forecast_view(horizon=2, frame=later, origin=start + timedelta(hours=7)), state)


def test_the_answer_is_the_canonical_long_forecast(tmp_path: Path) -> None:
    """Instance keys first, under the caller's own names, then the six columns."""
    state = fitted(tmp_path, season_length=2)

    answer = seasonal_naive.forecast(forecast_view(horizon=2), POINT, state)

    assert answer.column_names == [
        "zone",
        "event_time",
        "target",
        "kind",
        "quantile",
        "sample",
        "value",
    ]
    assert set(answer.column("kind").to_pylist()) == {"point"}
    assert answer.column("quantile").null_count == answer.num_rows
    assert isinstance(answer.column("event_time").to_pylist()[0], datetime)
