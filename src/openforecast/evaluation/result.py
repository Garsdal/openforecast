"""``BacktestResult``: what a backtest measured, in one long Arrow table.

```text
model                fold origin  metric value pairs fit_seconds forecast_seconds ...

builtin/seasonal-naive  0  06:00   mae    3.5   48    0.004       0.002
builtin/seasonal-naive  0  06:00   bias  -1.2   48    0.004       0.002
nixtla/autoarima        0  06:00   mae    2.1   48    1.930       0.041
```

One row per model, fold and metric — long for the same reason a forecast is:
the columns do not change with what was asked for, so one reader reads every
backtest. A wide table with a column per metric would change shape with the
argument list.

Three columns are not measurements and are there anyway, because a number
without them is not comparable:

```text
origin_fidelity  simulated windows, or real vintages
provider         who executed it, `openforecast` for a recipe it executes itself
artifact         the pinned revision that produced these numbers
```

``origin_fidelity`` is the one that changes conclusions. A model scored against
simulated historical availability was told the past was cleaner than it was, and
no metric in the table recovers that; carrying it per row is what makes
backtesting the two against each other a thing a caller can do rather than a
caveat they have to remember. ``artifact`` is what makes a result reproducible
rather than reported: the winner of a backtest is a reference you can forecast
with, not a name you have to fit again.

:meth:`BacktestResult.leaderboard` is a projection of the table, in the same
sense ``Forecast.to_wide`` is: it averages the folds, and averaging is a choice
about how to read measurements rather than a fact about them.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

import pyarrow as pa

from openforecast.data._arrow import build_table, column_values
from openforecast.errors import DataError, RecipeError
from openforecast.evaluation.metrics import Metric

__all__ = ["BACKTEST_COLUMNS", "BacktestColumn", "BacktestResult"]


class BacktestColumn(StrEnum):
    """The columns of a backtest result, whatever was backtested."""

    #: The candidate, under the label it was backtested as.
    MODEL = "model"
    FOLD = "fold"
    #: The historical origin the fold was evaluated at.
    ORIGIN = "origin"
    METRIC = "metric"
    VALUE = "value"
    #: How many (event time, target) outcomes the value was computed over. A
    #: fold whose truth is partly unpublished is scored on what exists, and this
    #: is how much that was.
    PAIRS = "pairs"
    FIT_SECONDS = "fit_seconds"
    FORECAST_SECONDS = "forecast_seconds"
    #: ``simulated`` or ``observed``, read off the artifact rather than assumed.
    ORIGIN_FIDELITY = "origin_fidelity"
    PROVIDER = "provider"
    #: The pinned revision these numbers came from: ``local/...@01K...``.
    ARTIFACT = "artifact"


#: The canonical column order.
BACKTEST_COLUMNS = tuple(column.value for column in BacktestColumn)

#: What a leaderboard holds: the model, the metric, the mean over folds, and how
#: many folds that was, with the mean cost of getting there.
LEADERBOARD_COLUMNS = (
    BacktestColumn.MODEL.value,
    BacktestColumn.METRIC.value,
    BacktestColumn.VALUE.value,
    "folds",
    BacktestColumn.FIT_SECONDS.value,
    BacktestColumn.FORECAST_SECONDS.value,
)


class BacktestResult:
    """The measurements of one backtest, and the projections people read them as."""

    def __init__(self, table: pa.Table, metrics: Sequence[Metric]) -> None:
        missing = [name for name in BACKTEST_COLUMNS if name not in table.column_names]
        if missing:
            raise DataError(f"a backtest result is missing the columns {missing}")
        self._table = table.select(list(BACKTEST_COLUMNS))
        self._metrics = {metric.name: metric for metric in metrics}

    # -- accessors ---------------------------------------------------------

    @property
    def table(self) -> pa.Table:
        """The long result, in canonical column order."""
        return self._table

    @property
    def models(self) -> tuple[str, ...]:
        """The candidates, in the order they were backtested."""
        return self._distinct(BacktestColumn.MODEL)

    @property
    def metrics(self) -> tuple[str, ...]:
        return self._distinct(BacktestColumn.METRIC)

    @property
    def origins(self) -> tuple[datetime, ...]:
        """The evaluation origins, in ascending order."""
        values: list[datetime] = column_values(self._table, BacktestColumn.ORIGIN.value)
        return tuple(sorted(set(values)))

    @property
    def num_rows(self) -> int:
        return self._table.num_rows

    # -- projections -------------------------------------------------------

    def leaderboard(self, metric: str | Metric | None = None) -> pa.Table:
        """The models ranked by one metric, averaged over the folds.

        Best first, where "best" is the metric's own idea of it: lowest for an
        error, closest to zero for a bias. A metric is required when the
        backtest measured several, because ranking by whichever came first
        would answer a question nobody asked.
        """
        name = self._one_metric(metric)
        rows = self._rows_for(name)
        ranker = self._metrics[name]
        ordered = sorted(rows, key=lambda entry: ranker.rank(entry.value))
        columns: dict[str, tuple[list[Any], pa.DataType]] = {
            BacktestColumn.MODEL.value: ([entry.model for entry in ordered], pa.string()),
            BacktestColumn.METRIC.value: ([name] * len(ordered), pa.string()),
            BacktestColumn.VALUE.value: ([entry.value for entry in ordered], pa.float64()),
            "folds": ([entry.folds for entry in ordered], pa.int64()),
            BacktestColumn.FIT_SECONDS.value: (
                [entry.fit_seconds for entry in ordered],
                pa.float64(),
            ),
            BacktestColumn.FORECAST_SECONDS.value: (
                [entry.forecast_seconds for entry in ordered],
                pa.float64(),
            ),
        }
        return build_table(columns)

    def best(self, metric: str | Metric | None = None) -> str:
        """The label of the model that ranked first — the winner, as a string.

        A label rather than an artifact, because a backtest fits one artifact
        per fold and there is no single one of them that "won"; which revision
        to keep is a question about the folds, and :attr:`table` holds them.
        """
        return str(column_values(self.leaderboard(metric), BacktestColumn.MODEL.value)[0])

    def to_pandas(self) -> Any:
        """The long result as a pandas ``DataFrame``.

        pandas is not a dependency of OpenForecast — this converts through
        Arrow, which is where the data already is.
        """
        return self._table.to_pandas()  # pyright: ignore[reportUnknownMemberType]

    # -- internals ---------------------------------------------------------

    def _distinct(self, column: BacktestColumn) -> tuple[str, ...]:
        values: list[str] = column_values(self._table, column.value)
        return tuple(dict.fromkeys(values))

    def _one_metric(self, metric: str | Metric | None) -> str:
        available = self.metrics
        if metric is None:
            if len(available) != 1:
                raise RecipeError(
                    f"this backtest measured {list(available)}, so a ranking has to say "
                    f"which one: leaderboard(metric='{available[0]}')"
                )
            return available[0]
        name = metric if isinstance(metric, str) else metric.name
        if name not in available:
            raise RecipeError(
                f"this backtest did not measure {name!r}; it measured {list(available)}"
            )
        return name

    def _rows_for(self, metric: str) -> list[_Summary]:
        """One summary per model, over the folds that metric was measured on."""
        grouped: dict[str, list[tuple[float, float, float]]] = {}
        for model, name, value, fit, predict in zip(
            column_values(self._table, BacktestColumn.MODEL.value),
            column_values(self._table, BacktestColumn.METRIC.value),
            column_values(self._table, BacktestColumn.VALUE.value),
            column_values(self._table, BacktestColumn.FIT_SECONDS.value),
            column_values(self._table, BacktestColumn.FORECAST_SECONDS.value),
            strict=True,
        ):
            if name == metric:
                grouped.setdefault(str(model), []).append((value, fit, predict))
        return [
            _Summary(
                model=model,
                value=_mean(value for value, _, _ in measured),
                folds=len(measured),
                fit_seconds=_mean(fit for _, fit, _ in measured),
                forecast_seconds=_mean(predict for _, _, predict in measured),
            )
            for model, measured in grouped.items()
        ]

    # -- dunder ------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BacktestResult(models={len(self.models)}, folds={len(self.origins)}, "
            f"metrics={list(self.metrics)}, rows={self._table.num_rows})"
        )

    def __str__(self) -> str:
        """The ranking when one metric was measured, and the shape otherwise."""
        if len(self.metrics) != 1:
            return repr(self)
        board = self.leaderboard()
        ranked = ", ".join(
            f"{model} {value:g}"
            for model, value in zip(
                column_values(board, BacktestColumn.MODEL.value),
                column_values(board, BacktestColumn.VALUE.value),
                strict=True,
            )
        )
        return f"BacktestResult({self.metrics[0]}: {ranked})"


@dataclass(frozen=True)
class _Summary:
    """One model's measurements of one metric, averaged over its folds."""

    model: str
    value: float
    folds: int
    fit_seconds: float
    forecast_seconds: float


def _mean(values: Iterable[float]) -> float:
    listed = list(values)
    return sum(listed) / len(listed)
