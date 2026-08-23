"""The metrics of Step 17: the arithmetic, the edges, and the round trip."""

from __future__ import annotations

import math

import pytest
from pydantic import TypeAdapter

import openforecast as of
from openforecast.errors import DataError
from openforecast.evaluation.metrics import Metric

ACTUAL = [10.0, 20.0, 30.0]
PREDICTED = [12.0, 18.0, 33.0]


def test_mean_absolute_error() -> None:
    assert of.MAE().compute(ACTUAL, PREDICTED) == pytest.approx(7 / 3)


def test_root_mean_squared_error() -> None:
    assert of.RMSE().compute(ACTUAL, PREDICTED) == pytest.approx(math.sqrt(17 / 3))


def test_mean_absolute_percentage_error_is_a_percentage() -> None:
    expected = 100 * (2 / 10 + 2 / 20 + 3 / 30) / 3
    assert of.MAPE().compute(ACTUAL, PREDICTED) == pytest.approx(expected)


def test_bias_is_signed_and_says_which_way() -> None:
    """Positive means the model forecast too high, which is what makes it readable."""
    assert of.Bias().compute(ACTUAL, PREDICTED) == pytest.approx(1.0)
    assert of.Bias().compute(PREDICTED, ACTUAL) == pytest.approx(-1.0)


def test_a_perfect_forecast_scores_zero_on_all_of_them() -> None:
    for metric in (of.MAE(), of.RMSE(), of.MAPE(), of.Bias()):
        assert metric.compute(ACTUAL, ACTUAL) == pytest.approx(0.0)


# -- the edges --------------------------------------------------------------


def test_a_percentage_error_of_a_zero_outcome_is_refused_not_skipped() -> None:
    """Skipping it would score a different subset of the horizon for one model."""
    with pytest.raises(DataError, match="undefined for 1 outcomes"):
        of.MAPE().compute([0.0, 20.0], [1.0, 18.0])


def test_a_metric_over_nothing_is_not_a_zero_score() -> None:
    for metric in (of.MAE(), of.RMSE(), of.MAPE(), of.Bias()):
        with pytest.raises(DataError, match="nothing to score"):
            metric.compute([], [])


def test_ranking_a_bias_ranks_its_magnitude() -> None:
    """A model biased by -3 and one biased by +3 are equally biased."""
    assert of.Bias().rank(-3.0) == of.Bias().rank(3.0) == 3.0
    assert of.MAE().rank(3.0) == 3.0


def test_every_metric_is_minimized_after_ranking() -> None:
    """What lets one leaderboard sort them all without a per-metric branch."""
    for metric in (of.MAE(), of.RMSE(), of.MAPE(), of.Bias()):
        assert metric.rank(metric.compute(ACTUAL, ACTUAL)) <= metric.rank(
            metric.compute(ACTUAL, PREDICTED)
        )


# -- the vocabulary ---------------------------------------------------------


def test_a_metric_names_itself_the_way_a_result_table_spells_it() -> None:
    assert [metric.name for metric in (of.MAE(), of.RMSE(), of.MAPE(), of.Bias())] == [
        "mae",
        "rmse",
        "mape",
        "bias",
    ]


@pytest.mark.parametrize("metric", [of.MAE(), of.RMSE(), of.MAPE(), of.Bias()])
def test_a_metric_round_trips_as_the_type_it_was_written_from(metric: object) -> None:
    """The reason each one is a type: a written-down metric reads back as itself."""
    adapter = TypeAdapter[object](Metric)

    assert adapter.validate_python(adapter.dump_python(metric, mode="json")) == metric


def test_a_metric_is_frozen_and_refuses_fields_it_does_not_have() -> None:
    with pytest.raises(Exception, match="frozen|Instance is frozen"):
        of.MAE().metric = "rmse"  # pyright: ignore[reportAttributeAccessIssue]
