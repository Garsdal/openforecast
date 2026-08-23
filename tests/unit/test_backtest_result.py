"""``BacktestResult``: the two tables, and the projections people read them as.

Built here from rows rather than from a backtest, so that the ranking rules are
tested on numbers chosen to exercise them — including a bias whose best value is
zero rather than lowest, and a prediction whose outcome was never published.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pyarrow as pa
import pytest

import openforecast as of
from openforecast.errors import DataError, RecipeError
from openforecast.evaluation.result import (
    BACKTEST_COLUMNS,
    PREDICTION_COLUMNS,
    BacktestResult,
)

ORIGIN = datetime(2026, 1, 1, 12)
HOUR = timedelta(hours=1)

#: ``(model, zone, horizon_step, prediction, actual)`` — one prediction row.
PredictionRow = tuple[str, str, int, float, float | None]

#: The predictions every metric in :func:`result` is nominally computed from.
#: Every model predicts 10 at both steps; what happened differs by step, so a
#: ``horizon_step`` grouping has something to say.
PREDICTIONS: tuple[PredictionRow, ...] = (
    ("a", "DE", 1, 10.0, 11.0),
    ("a", "DE", 2, 10.0, 14.0),
    ("b", "DE", 1, 10.0, 12.0),
    ("b", "DE", 2, 10.0, 18.0),
)


def predictions(*rows: PredictionRow) -> pa.Table:
    columns: dict[str, pa.Array[Any]] = {
        "model": pa.array([model for model, _, _, _, _ in rows], type=pa.string()),
        "fold": pa.array([0] * len(rows), type=pa.int64()),
        "zone": pa.array([zone for _, zone, _, _, _ in rows], type=pa.string()),
        "origin_time": pa.array([ORIGIN] * len(rows), type=pa.timestamp("us")),
        "event_time": pa.array(
            [ORIGIN + HOUR * step for _, _, step, _, _ in rows], type=pa.timestamp("us")
        ),
        "horizon_step": pa.array([step for _, _, step, _, _ in rows], type=pa.int64()),
        "target": pa.array(["price"] * len(rows), type=pa.string()),
        "prediction": pa.array([value for _, _, _, value, _ in rows], type=pa.float64()),
        "actual": pa.array([outcome for _, _, _, _, outcome in rows], type=pa.float64()),
    }
    return pa.table(columns)


def result(*rows: tuple[str, int, str, float], predicted: pa.Table | None = None) -> BacktestResult:
    """A result over ``(model, fold, metric, value)``, with plausible everything else."""
    columns: dict[str, pa.Array[Any]] = {
        "model": pa.array([model for model, _, _, _ in rows], type=pa.string()),
        "fold": pa.array([fold for _, fold, _, _ in rows], type=pa.int64()),
        "origin": pa.array([ORIGIN] * len(rows), type=pa.timestamp("us")),
        "metric": pa.array([metric for _, _, metric, _ in rows], type=pa.string()),
        "value": pa.array([value for _, _, _, value in rows], type=pa.float64()),
        "pairs": pa.array([24] * len(rows), type=pa.int64()),
        "fit_seconds": pa.array([1.0] * len(rows), type=pa.float64()),
        "forecast_seconds": pa.array([0.5] * len(rows), type=pa.float64()),
        "origin_fidelity": pa.array(["observed"] * len(rows), type=pa.string()),
        "provider": pa.array(["builtin"] * len(rows), type=pa.string()),
        "artifact": pa.array(["local/x@01K"] * len(rows), type=pa.string()),
    }
    return BacktestResult(
        pa.table(columns),
        predictions(*PREDICTIONS) if predicted is None else predicted,
        scored_by=[of.MAE(), of.Bias()],
    )


def values(table: pa.Table, column: str) -> list[Any]:
    found: list[Any] = table.column(column).to_pylist()
    return found


# -- the tables -------------------------------------------------------------


def test_the_metric_columns_are_canonical_and_in_order() -> None:
    assert result(("a", 0, "mae", 1.0)).metrics.column_names == list(BACKTEST_COLUMNS)


def test_the_instance_keys_sit_between_the_fold_and_the_origin() -> None:
    """Under the caller's own names, which is why they are not in the enum."""
    measured = result(("a", 0, "mae", 1.0))

    assert measured.instance_keys == ("zone",)
    assert measured.predictions.column_names == [
        "model",
        "fold",
        "zone",
        *PREDICTION_COLUMNS[2:],
    ]


def test_a_table_that_is_not_a_backtest_result_is_refused() -> None:
    with pytest.raises(DataError, match="missing the metric columns"):
        BacktestResult(
            pa.table({"model": pa.array(["a"])}),
            predictions(*PREDICTIONS),
            scored_by=[of.MAE()],
        )


def test_predictions_that_are_not_predictions_are_refused_too() -> None:
    with pytest.raises(DataError, match="missing the prediction columns"):
        result(("a", 0, "mae", 1.0), predicted=pa.table({"model": pa.array(["a"])}))


def test_it_reports_what_it_measured() -> None:
    measured = result(("a", 0, "mae", 1.0), ("b", 0, "mae", 2.0), ("a", 0, "bias", -1.0))

    assert measured.models == ("a", "b")
    assert measured.metric_names == ("mae", "bias")
    assert measured.origins == (ORIGIN,)
    assert measured.metrics.num_rows == 3


# -- the leaderboard --------------------------------------------------------


def test_a_leaderboard_averages_the_folds_and_ranks_them_best_first() -> None:
    board = result(
        ("a", 0, "mae", 4.0),
        ("a", 1, "mae", 2.0),
        ("b", 0, "mae", 1.0),
        ("b", 1, "mae", 1.0),
    ).leaderboard("mae")

    assert values(board, "model") == ["b", "a"]
    assert values(board, "value") == [1.0, 3.0]
    assert values(board, "folds") == [2, 2]
    assert values(board, "fit_seconds") == [1.0, 1.0]


def test_a_bias_is_ranked_by_how_biased_it_is_not_by_its_sign() -> None:
    """A model biased by -1 beats one biased by +5, and the sign is still reported."""
    board = result(("a", 0, "bias", 5.0), ("b", 0, "bias", -1.0)).leaderboard("bias")

    assert values(board, "model") == ["b", "a"]
    assert values(board, "value") == [-1.0, 5.0]


def test_the_best_model_is_the_one_that_ranked_first() -> None:
    measured = result(("a", 0, "mae", 4.0), ("b", 0, "mae", 1.0))

    assert measured.best("mae") == "b"
    assert measured.best(of.MAE()) == "b"


def test_a_ranking_over_several_metrics_has_to_say_which_one() -> None:
    """Ranking by whichever came first would answer a question nobody asked."""
    measured = result(("a", 0, "mae", 1.0), ("a", 0, "bias", 1.0))

    with pytest.raises(RecipeError, match="has to say which one"):
        measured.leaderboard()


def test_one_metric_needs_no_naming() -> None:
    assert result(("a", 0, "mae", 1.0)).leaderboard().num_rows == 1


def test_a_metric_that_was_not_measured_is_refused() -> None:
    with pytest.raises(RecipeError, match="did not measure"):
        result(("a", 0, "mae", 1.0)).leaderboard("rmse")


def test_a_model_that_was_only_scored_on_some_folds_says_how_many() -> None:
    board = result(("a", 0, "mae", 1.0), ("a", 1, "mae", 3.0), ("b", 0, "mae", 2.0)).leaderboard(
        "mae"
    )

    counted = dict(zip(values(board, "model"), values(board, "folds"), strict=True))

    assert counted == {"a": 2, "b": 1}


# -- grouped metrics --------------------------------------------------------


def test_a_metric_is_regrouped_from_the_predictions_rather_than_re_run() -> None:
    """The question a table of means cannot answer: does it degrade with horizon?"""
    grouped = result(("a", 0, "mae", 1.0)).metrics_by("horizon_step")

    errors = {
        (model, step): value
        for model, step, metric, value in zip(
            values(grouped, "model"),
            values(grouped, "horizon_step"),
            values(grouped, "metric"),
            values(grouped, "value"),
            strict=True,
        )
        if metric == "mae"
    }

    assert errors == {("a", 1): 1.0, ("a", 2): 4.0, ("b", 1): 2.0, ("b", 2): 8.0}
    assert set(values(grouped, "pairs")) == {1}


def test_the_model_is_always_a_group_key_even_unasked() -> None:
    """A number pooled over the candidates would compare nothing."""
    grouped = result(("a", 0, "mae", 1.0)).metrics_by(["model", "zone"])

    assert grouped.column_names == ["model", "zone", "metric", "value", "pairs"]
    assert values(grouped, "model") == ["a", "a", "b", "b"]


def test_every_metric_the_backtest_measured_is_regrouped() -> None:
    grouped = result(("a", 0, "mae", 1.0)).metrics_by("target")

    assert values(grouped, "metric") == ["mae", "bias", "mae", "bias"]
    # Predicting 10 where 11 and 14 happened is an error of 2.5 and, since both
    # forecasts were low, a bias of -2.5. And 12 and 18 for the other model.
    assert values(grouped, "value") == [2.5, -2.5, 5.0, -5.0]


def test_a_group_key_that_is_not_a_prediction_column_names_the_ones_that_are() -> None:
    with pytest.raises(RecipeError, match="not columns of the prediction table"):
        result(("a", 0, "mae", 1.0)).metrics_by("origin_fidelity")


def test_an_unpublished_outcome_is_not_scored_as_a_zero_error() -> None:
    """The same rule the metric rows were computed under, applied again."""
    measured = result(
        ("a", 0, "mae", 1.0),
        predicted=predictions(("a", "DE", 1, 10.0, 11.0), ("a", "DE", 2, 10.0, None)),
    )

    grouped = measured.metrics_by("horizon_step")
    scored = {
        (step, metric): (value, pairs)
        for step, metric, value, pairs in zip(
            values(grouped, "horizon_step"),
            values(grouped, "metric"),
            values(grouped, "value"),
            values(grouped, "pairs"),
            strict=True,
        )
    }

    assert scored[1, "mae"] == (1.0, 1)
    # A group with nothing to score says so rather than reporting a zero.
    assert scored[2, "mae"] == (None, 0)


# -- reading it -------------------------------------------------------------


def test_it_prints_the_ranking_when_there_is_one_metric() -> None:
    printed = str(result(("a", 0, "mae", 4.0), ("b", 0, "mae", 1.0)))

    assert printed == "BacktestResult(mae: b 1, a 4)"


def test_it_prints_its_shape_when_there_are_several() -> None:
    printed = str(result(("a", 0, "mae", 4.0), ("a", 0, "bias", 1.0)))

    assert printed == repr(result(("a", 0, "mae", 4.0), ("a", 0, "bias", 1.0)))
    assert "metrics=['mae', 'bias']" in printed
    assert "predictions=4" in printed


def test_the_long_table_converts_to_pandas_without_pandas_being_a_dependency() -> None:
    frame = result(("a", 0, "mae", 1.0)).to_pandas()

    assert list(frame.columns) == list(BACKTEST_COLUMNS)
