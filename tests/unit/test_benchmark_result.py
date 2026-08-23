"""``BenchmarkResult``: the table, and the projections people read it as.

Built here from rows rather than from a benchmark, so that the ranking rules are
tested on numbers chosen to exercise them — including a bias whose best value is
zero rather than lowest.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pyarrow as pa
import pytest

import openforecast as of
from openforecast.errors import DataError, RecipeError
from openforecast.evaluation.result import BENCHMARK_COLUMNS, BenchmarkResult

ORIGIN = datetime(2026, 1, 1, 12)


def result(*rows: tuple[str, int, str, float]) -> BenchmarkResult:
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
    return BenchmarkResult(pa.table(columns), metrics=[of.MAE(), of.Bias()])


def values(table: pa.Table, column: str) -> list[Any]:
    found: list[Any] = table.column(column).to_pylist()
    return found


# -- the table --------------------------------------------------------------


def test_the_columns_are_canonical_and_in_order() -> None:
    assert result(("a", 0, "mae", 1.0)).table.column_names == list(BENCHMARK_COLUMNS)


def test_a_table_that_is_not_a_benchmark_result_is_refused() -> None:
    with pytest.raises(DataError, match="missing the columns"):
        BenchmarkResult(pa.table({"model": pa.array(["a"])}), metrics=[of.MAE()])


def test_it_reports_what_it_measured() -> None:
    measured = result(("a", 0, "mae", 1.0), ("b", 0, "mae", 2.0), ("a", 0, "bias", -1.0))

    assert measured.models == ("a", "b")
    assert measured.metrics == ("mae", "bias")
    assert measured.origins == (ORIGIN,)
    assert measured.num_rows == 3


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


# -- reading it -------------------------------------------------------------


def test_it_prints_the_ranking_when_there_is_one_metric() -> None:
    printed = str(result(("a", 0, "mae", 4.0), ("b", 0, "mae", 1.0)))

    assert printed == "BenchmarkResult(mae: b 1, a 4)"


def test_it_prints_its_shape_when_there_are_several() -> None:
    printed = str(result(("a", 0, "mae", 4.0), ("a", 0, "bias", 1.0)))

    assert printed == repr(result(("a", 0, "mae", 4.0), ("a", 0, "bias", 1.0)))
    assert "metrics=['mae', 'bias']" in printed


def test_the_long_table_converts_to_pandas_without_pandas_being_a_dependency() -> None:
    frame = result(("a", 0, "mae", 1.0)).to_pandas()

    assert list(frame.columns) == list(BENCHMARK_COLUMNS)
