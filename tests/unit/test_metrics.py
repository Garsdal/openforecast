"""The metrics: the arithmetic, the edges, and the round trip.

The four point metrics of Step 17 and the three of Step 20 that score a
predictive distribution. Built from ``Prediction`` objects rather than from a
backtest, so that the numbers are chosen to exercise the rules — a pinball loss
charged asymmetrically, an interval that misses, and a distribution a metric
cannot read.
"""

from __future__ import annotations

import math

import pytest
from pydantic import TypeAdapter, ValidationError

import openforecast as of
from openforecast.errors import DataError, RecipeError
from openforecast.evaluation.metrics import Measurement, Metric
from openforecast.evaluation.predictions import PredictedValue, Prediction, predictions_of
from openforecast.protocol import quantile_of_samples

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


# -- the probabilistic three ------------------------------------------------


def quantile_prediction(actual: float, **levels: float) -> Prediction:
    """One outcome and the quantiles a model answered for it."""
    return Prediction(
        actual=actual, quantiles={float(level): value for level, value in levels.items()}
    )


def sample_prediction(actual: float, *draws: float) -> Prediction:
    return Prediction(actual=actual, samples=draws)


def test_a_pinball_loss_charges_being_under_the_outcome_by_its_level() -> None:
    """A 0.9 quantile is meant to be exceeded one time in ten, and is charged for it."""
    under = of.PinballLoss(0.9).measure([quantile_prediction(100.0, **{"0.9": 90.0})])
    over = of.PinballLoss(0.9).measure([quantile_prediction(80.0, **{"0.9": 90.0})])

    assert under.value == pytest.approx(10.0 * 0.9)
    assert over.value == pytest.approx(10.0 * 0.1)
    assert under.pairs == over.pairs == 1


def test_a_pinball_loss_is_minimized_by_the_quantile_it_names() -> None:
    """Nine outcomes below and one above: the 0.9 level is the cheapest answer."""
    outcomes = [float(value) for value in range(1, 11)]

    def loss(forecast: float) -> float:
        measured = of.PinballLoss(0.9).measure(
            [quantile_prediction(outcome, **{"0.9": forecast}) for outcome in outcomes]
        )
        assert measured.value is not None
        return measured.value

    assert loss(9.0) <= loss(5.0)
    assert loss(9.0) <= loss(10.0)


def test_coverage_counts_the_outcomes_inside_the_interval() -> None:
    inside = quantile_prediction(80.0, **{"0.1": 70.0, "0.9": 95.0})
    outside = quantile_prediction(120.0, **{"0.1": 70.0, "0.9": 95.0})

    measured = of.Coverage().measure([inside, inside, inside, outside])

    assert measured.value == pytest.approx(0.75)
    assert measured.pairs == 4


def test_coverage_is_of_the_central_interval_its_level_names() -> None:
    assert of.Coverage().bounds == (pytest.approx(0.1), pytest.approx(0.9))
    assert of.Coverage(0.5).bounds == (pytest.approx(0.25), pytest.approx(0.75))
    assert of.IntervalWidth(0.5).bounds == (pytest.approx(0.25), pytest.approx(0.75))


def test_a_coverage_is_ranked_by_its_distance_from_its_nominal_level() -> None:
    """An 80% interval covering 99% of outcomes is wrong in the other direction."""
    coverage = of.Coverage()

    assert coverage.rank(0.78) == pytest.approx(coverage.rank(0.82))
    assert coverage.rank(0.8) < coverage.rank(0.99)


def test_an_interval_width_is_the_mean_width_of_the_interval() -> None:
    measured = of.IntervalWidth().measure(
        [
            quantile_prediction(80.0, **{"0.1": 70.0, "0.9": 95.0}),
            quantile_prediction(80.0, **{"0.1": 60.0, "0.9": 95.0}),
        ]
    )

    assert measured.value == pytest.approx((25.0 + 35.0) / 2)


def test_a_probabilistic_metric_reads_a_sample_forecast_the_same_way() -> None:
    """The claim of Step 20: the same metric over either form of the same answer."""
    draws = sample_prediction(80.0, 70.0, 80.0, 90.0, 100.0)
    equivalent = quantile_prediction(80.0, **{"0.1": 73.0, "0.9": 97.0})

    assert of.IntervalWidth().measure([draws]).value == pytest.approx(
        of.IntervalWidth().measure([equivalent]).value
    )
    assert of.Coverage().measure([draws]).value == of.Coverage().measure([equivalent]).value


def test_a_point_metric_scores_the_median_of_a_distribution() -> None:
    """The median is a level the model stated, or a reading of the draws it gave."""
    assert of.MAE().measure([quantile_prediction(80.0, **{"0.5": 78.0})]).value == pytest.approx(
        2.0
    )
    assert of.MAE().measure([sample_prediction(80.0, 70.0, 80.0, 90.0, 100.0)]).value == (
        pytest.approx(5.0)
    )


def test_a_metric_over_a_prediction_it_cannot_read_measures_nothing() -> None:
    """Not a zero score: a metric that scored nothing is a null beside a zero count."""
    no_median = quantile_prediction(80.0, **{"0.1": 70.0, "0.9": 95.0})

    assert of.MAE().measure([no_median]) == Measurement(value=None, pairs=0)
    assert of.PinballLoss(0.5).measure([no_median]) == Measurement(value=None, pairs=0)
    assert of.Coverage(0.5).measure([no_median]) == Measurement(value=None, pairs=0)


def test_a_level_between_two_the_model_stated_is_not_one_it_stated() -> None:
    with pytest.raises(DataError, match=r"no quantile 0.25"):
        quantile_prediction(80.0, **{"0.1": 70.0, "0.9": 95.0}).quantile(0.25)
    with pytest.raises(DataError, match="no point estimate"):
        _ = quantile_prediction(80.0, **{"0.1": 70.0, "0.9": 95.0}).point_estimate


def test_a_point_forecast_is_not_a_distribution_with_a_narrow_interval() -> None:
    point = Prediction(actual=80.0, point=78.0)

    assert of.MAE().measure([point]).value == pytest.approx(2.0)
    assert of.Coverage().measure([point]) == Measurement(value=None, pairs=0)


# -- what a metric can be asked to score ------------------------------------


def test_a_probabilistic_metric_refuses_a_point_forecast_before_anything_runs() -> None:
    reason = of.Coverage().requirement(of.OutputSpec.point())

    assert reason is not None and "point forecast is not one" in reason
    assert of.PinballLoss(0.9).requirement(of.OutputSpec.point()) is not None


def test_a_metric_says_which_levels_it_needs() -> None:
    asked = of.OutputSpec.quantiles([0.1, 0.9])

    assert of.PinballLoss(0.9).requirement(asked) is None
    assert of.Coverage().requirement(asked) is None
    assert of.IntervalWidth().requirement(asked) is None

    missing = of.PinballLoss(0.5).requirement(asked)
    assert missing is not None and "0.5" in missing
    narrower = of.Coverage(0.5).requirement(asked)
    assert narrower is not None and "0.25" in narrower


def test_a_point_metric_needs_a_median_to_read_out_of_a_quantile_forecast() -> None:
    reason = of.MAE().requirement(of.OutputSpec.quantiles([0.1, 0.9]))

    assert reason is not None and "0.5 is not among the levels" in reason
    assert of.MAE().requirement(of.OutputSpec.quantiles([0.1, 0.5, 0.9])) is None
    assert of.MAE().requirement(of.OutputSpec.point()) is None


def test_sample_draws_satisfy_every_metric() -> None:
    """Reading a quantile off the draws is a projection of what the model gave."""
    samples = of.OutputSpec.samples(100)

    for metric in (of.MAE(), of.PinballLoss(0.99), of.Coverage(), of.IntervalWidth()):
        assert metric.requirement(samples) is None


# -- the vocabulary of the parameterized three ------------------------------


def test_a_parameterized_metric_carries_its_parameter_in_its_name() -> None:
    """Two pinball losses in one backtest are two rows, so they cannot share a name."""
    assert of.PinballLoss(0.9).name == "pinball[0.9]"
    assert of.Coverage().name == "coverage[0.8]"
    assert of.IntervalWidth(0.5).name == "interval_width[0.5]"


def test_a_level_can_be_given_positionally_or_by_keyword_but_not_twice() -> None:
    assert of.PinballLoss(0.9) == of.PinballLoss(level=0.9)
    assert of.Coverage(0.5) == of.Coverage(level=0.5)
    with pytest.raises(RecipeError, match="both positionally and by keyword"):
        of.PinballLoss(0.9, level=0.9)


def test_a_probabilistic_metric_round_trips_as_the_type_it_was_written_from() -> None:
    adapter = TypeAdapter[object](Metric)

    for metric in (of.PinballLoss(0.9), of.Coverage(0.5), of.IntervalWidth()):
        assert adapter.validate_python(adapter.dump_python(metric, mode="json")) == metric


def test_a_level_outside_the_distribution_is_refused() -> None:
    for level in (0.0, 1.0, 1.5):
        with pytest.raises(ValidationError):
            of.PinballLoss(level)
        with pytest.raises(ValidationError):
            of.Coverage(level)


# -- gathering the rows about one outcome -----------------------------------


def rows(*parts: tuple[str, float | None, int | None, float]) -> list[PredictedValue]:
    return [
        PredictedValue(
            outcome=("DE", 1), kind=kind, level=level, draw=draw, predicted=value, actual=80.0
        )
        for kind, level, draw, value in parts
    ]


def test_the_rows_about_one_outcome_become_one_prediction() -> None:
    gathered = predictions_of(rows(("quantile", 0.1, None, 65.0), ("quantile", 0.9, None, 95.0)))

    assert len(gathered) == 1
    assert gathered[0].quantiles == {0.1: 65.0, 0.9: 95.0}
    assert gathered[0].actual == 80.0


def test_draws_are_gathered_in_draw_order_whatever_order_they_arrive_in() -> None:
    gathered = predictions_of(
        rows(("sample", None, 2, 90.0), ("sample", None, 0, 70.0), ("sample", None, 1, 80.0))
    )

    assert gathered[0].samples == (70.0, 80.0, 90.0)


def test_a_row_with_no_outcome_is_left_out_rather_than_scored_as_zero() -> None:
    unpublished = PredictedValue(
        outcome=("DE", 1), kind="point", level=None, draw=None, predicted=78.0, actual=None
    )
    unanswered = PredictedValue(
        outcome=("DE", 2), kind="point", level=None, draw=None, predicted=float("nan"), actual=80.0
    )

    assert predictions_of([unpublished, unanswered]) == ()


def test_a_row_describing_no_part_of_a_forecast_is_refused() -> None:
    """A value filed under the wrong reading is worse than a missing one."""
    with pytest.raises(DataError, match="describes no part of a forecast"):
        predictions_of(rows(("quantile", None, None, 65.0)))


def test_the_quantile_of_draws_is_defined_once_for_everything_that_reads_one() -> None:
    """The engine's reduction and a metric's reading are one estimator."""
    draws = [70.0, 80.0, 90.0, 100.0]

    assert quantile_of_samples(draws, 0.5) == pytest.approx(85.0)
    assert quantile_of_samples(draws, 0.9) == pytest.approx(97.0)
    assert sample_prediction(80.0, *draws).quantile(0.9) == pytest.approx(97.0)
    with pytest.raises(DataError, match="of no draws is not a number"):
        quantile_of_samples([], 0.5)
