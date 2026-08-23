"""The validation strategies of Step 17: which origins, and what is reachable at one.

Folds are built without a client, a provider or an artifact store, which is the
property worth having: what a model is allowed to see at a historical origin is
decided by the semantic layer, before anything is fitted.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest
from pydantic import TypeAdapter

import openforecast as of
from openforecast.errors import DataError
from openforecast.evaluation.validation import Validation, outcomes
from tests.factories import HOUR, START, history, point_in_time


def at(step: int) -> datetime:
    return START + HOUR * step


def frame(periods: int = 12, **kwargs: Any) -> of.TimeSeriesFrame:
    return of.TimeSeriesFrame.from_pandas(
        history=history(instances=("DE", "FR"), instance_key="zone", periods=periods, **kwargs),
        time="timestamp",
        frequency="1h",
        instance_keys=["zone"],
        targets=["load"],
        known_features=list(kwargs.get("known", ())),
    )


def dataset(origins: int = 8) -> of.ForecastDataset:
    return of.ForecastDataset.from_pandas(
        point_in_time(origins=origins, horizon=4),
        origin_time="ref_time",
        event_time="target_time",
        targets=["price"],
        event_frequency="1h",
        origin_frequency="1h",
        known_features=["wind_fc"],
    )


def values(table: Any, column: str) -> list[Any]:
    found: list[Any] = table.column(column).to_pylist()
    return found


# -- rolling origins over event-time data -----------------------------------


def test_the_last_window_ends_at_the_last_event_time() -> None:
    """Nothing of the recent history is left unevaluated by accident."""
    folds = of.RollingOrigin(horizon=3, windows=2).folds(frame(periods=12))

    assert [fold.origin for fold in folds] == [at(5), at(8)]
    assert max(values(folds[-1].truth, "event_time")) == at(11)


def test_the_default_stride_makes_the_windows_consecutive() -> None:
    folds = of.RollingOrigin(horizon=4, windows=3).folds(frame(periods=16))

    assert [fold.origin for fold in folds] == [at(3), at(7), at(11)]


def test_a_shorter_stride_evaluates_more_often_over_the_same_history() -> None:
    folds = of.RollingOrigin(horizon=4, windows=3, stride=1).folds(frame(periods=16))

    assert [fold.origin for fold in folds] == [at(9), at(10), at(11)]


def test_a_fold_is_trained_on_the_history_up_to_its_own_origin() -> None:
    fold = of.RollingOrigin(horizon=3, windows=1).folds(frame(periods=12))[0]

    assert isinstance(fold.train, of.TimeSeriesFrame)
    assert max(values(fold.train.history, "timestamp")) == fold.origin
    assert fold.context is fold.train


def test_the_truth_of_a_fold_is_the_horizon_after_its_origin() -> None:
    fold = of.RollingOrigin(horizon=3, windows=1).folds(frame(periods=12))[0]

    assert sorted(set(values(fold.truth, "event_time"))) == [at(9), at(10), at(11)]
    assert set(values(fold.truth, "target")) == {"load"}
    assert fold.truth.num_rows == 6  # three steps, two instances


def test_more_history_than_there_is_is_an_error_naming_what_to_shorten() -> None:
    with pytest.raises(DataError, match="before this history begins"):
        of.RollingOrigin(horizon=4, windows=5).folds(frame(periods=12))


def test_a_rolling_origin_refuses_real_vintages() -> None:
    """It would have to choose which vintage each simulated origin means."""
    with pytest.raises(DataError, match="ForecastOriginValidation"):
        of.RollingOrigin(horizon=3, windows=2).folds(dataset())


# -- the origins a point-in-time dataset actually holds ---------------------


def test_every_vintage_is_an_evaluation_origin_by_default() -> None:
    folds = of.ForecastOriginValidation(horizon=2).folds(dataset(origins=5))

    assert [fold.origin for fold in folds] == [at(step) for step in range(5)]


def test_the_origins_are_selected_with_the_same_vocabulary_a_fit_plan_uses() -> None:
    folds = of.ForecastOriginValidation(
        horizon=2, origins=of.OriginsBetween(at(2), at(6), stride=2)
    ).folds(dataset(origins=8))

    assert [fold.origin for fold in folds] == [at(2), at(4), at(6)]


def test_a_fold_forecasts_from_the_vintage_that_existed_at_its_origin() -> None:
    """The Step 17 guarantee: features come from that exact historical origin."""
    fold = of.ForecastOriginValidation(horizon=2, origins=of.AtOrigin(at(3))).folds(dataset())[0]

    assert isinstance(fold.context, of.ForecastContext)
    assert fold.context.origin_time == at(3)
    # wind_fc names the origin that issued it: 100 * origin + event. Where this
    # vintage said nothing — event times before its own origin — the value is
    # null rather than borrowed from a vintage that did.
    known = values(fold.context.frame.history, "wind_fc") + values(
        fold.context.frame.future, "wind_fc"
    )
    assert [value for value in known if value is not None]
    assert all(300 <= value < 400 for value in known if value is not None)


def test_later_vintages_are_absent_from_what_a_fold_trains_on() -> None:
    """Not merely unused. There is nothing for a bug downstream to reach for."""
    fold = of.ForecastOriginValidation(horizon=2, origins=of.AtOrigin(at(3))).folds(dataset())[0]

    assert isinstance(fold.train, of.ForecastDataset)
    assert max(fold.train.origins) == at(3)
    assert max(values(fold.train.truth.history, "target_time")) == at(3)


def test_the_truth_of_a_fold_comes_from_the_truth_frame() -> None:
    """Which reaches past the origin, unlike anything the fold trains on."""
    fold = of.ForecastOriginValidation(horizon=2, origins=of.AtOrigin(at(3))).folds(dataset())[0]

    assert sorted(set(values(fold.truth, "event_time"))) == [at(4), at(5)]


def test_a_forecast_origin_validation_refuses_event_time_data() -> None:
    """It would have to invent origins that were never issued."""
    with pytest.raises(DataError, match="RollingOrigin"):
        of.ForecastOriginValidation(horizon=2).folds(frame())


@pytest.mark.parametrize(
    "validation",
    [of.RollingOrigin(horizon=3, windows=2), of.ForecastOriginValidation(horizon=3)],
)
def test_anything_that_is_not_a_semantic_source_is_refused(validation: Any) -> None:
    with pytest.raises(DataError):
        validation.folds(object())


# -- what the outcomes table is ---------------------------------------------


def test_an_unpublished_outcome_is_a_null_rather_than_a_dropped_row() -> None:
    """How much of a fold was actually scored has to stay visible."""
    data = frame(periods=8)
    truth = outcomes(data, after=at(20), horizon=3)

    assert truth.num_rows == 0
    assert truth.column_names == ["zone", "event_time", "target", "value"]


def test_the_outcomes_of_several_targets_are_long_rather_than_wide() -> None:
    data = of.TimeSeriesFrame.from_pandas(
        history=history(instances=("DE",), periods=8, targets=("load", "wind")),
        time="timestamp",
        frequency="1h",
        targets=["load", "wind"],
    )

    truth = outcomes(data, after=at(4), horizon=2)

    assert set(values(truth, "target")) == {"load", "wind"}
    assert truth.num_rows == 4


# -- the vocabulary ---------------------------------------------------------


@pytest.mark.parametrize(
    "validation",
    [
        of.RollingOrigin(horizon=24, windows=5),
        of.RollingOrigin(horizon=24, windows=5, stride=6),
        of.ForecastOriginValidation(horizon=72, origins=of.AllOrigins(stride=24)),
    ],
)
def test_a_validation_round_trips_as_the_type_it_was_written_from(validation: object) -> None:
    adapter = TypeAdapter[object](Validation)

    assert adapter.validate_python(adapter.dump_python(validation, mode="json")) == validation


def test_a_stride_defaults_to_the_horizon_rather_than_to_one() -> None:
    assert of.RollingOrigin(horizon=24, windows=2).step == 24
    assert of.RollingOrigin(horizon=24, windows=2, stride=6).step == 6
