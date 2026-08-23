"""``BacktestResult``: what a backtest predicted, and what those predictions scored.

Two Arrow tables, and one is derived from the other. The metrics:

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

And the predictions every one of those numbers was computed from:

```text
model             fold zone origin event_time step target kind     quantile sample prediction actual

seasonal-naive     0   DE   06:00  07:00      1    price  point    null     null   78.0       80.1
seasonal-naive     0   DE   06:00  08:00      2    price  point    null     null   78.0       74.6
nixtla/autoarima   0   DE   06:00  07:00      1    price  quantile 0.9      null   84.2       80.1
```

(``origin`` and ``step`` are ``origin_time`` and ``horizon_step``, abbreviated to
fit the page.)

Long over the distribution as well as over the folds, for the same reason a
forecast is: a quantile forecast contributes one row per level and a sample
forecast one per draw, so the columns do not change with what was asked for and
one reader reads every backtest. The rows about one outcome are gathered back
into one prediction before a metric sees them, which is what lets a pinball loss
and a mean absolute error be computed from this one table.

Retained rather than dropped, because the metric rows are derivable from these
and not the reverse: *does it degrade after horizon 48?* is the most common
question asked after a backtest, and it is unanswerable from a table of means.
:meth:`BacktestResult.metrics_by` answers it here rather than by re-running
anything.

:meth:`BacktestResult.leaderboard` and :meth:`BacktestResult.metrics_by` are
both projections, in the same sense ``Forecast.to_wide`` is: they average and
group measurements, and how to read a set of measurements is a choice about them
rather than a fact about them.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

import pyarrow as pa

from openforecast.data._arrow import build_table, column_type, column_values
from openforecast.errors import DataError, RecipeError
from openforecast.evaluation.metrics import Metric
from openforecast.evaluation.predictions import PredictedValue, Prediction, predictions_of

__all__ = [
    "BACKTEST_COLUMNS",
    "PREDICTION_COLUMNS",
    "BacktestColumn",
    "BacktestResult",
    "PredictionColumn",
]


class BacktestColumn(StrEnum):
    """The columns of a backtest's metric table, whatever was backtested."""

    #: The candidate, under the label it was backtested as.
    MODEL = "model"
    FOLD = "fold"
    #: The historical origin the fold was evaluated at.
    ORIGIN = "origin"
    METRIC = "metric"
    VALUE = "value"
    #: How many (event time, target) outcomes *this metric* was computed over. A
    #: fold whose truth is partly unpublished is scored on what exists, and this
    #: is how much that was. Per metric rather than per fold, because two metrics
    #: of one fold do not always have what they need on the same outcomes.
    PAIRS = "pairs"
    #: How long the fit took, or null for a frozen revision that skipped it.
    FIT_SECONDS = "fit_seconds"
    FORECAST_SECONDS = "forecast_seconds"
    #: ``simulated`` or ``observed``, read off the artifact rather than assumed.
    ORIGIN_FIDELITY = "origin_fidelity"
    PROVIDER = "provider"
    #: The pinned revision these numbers came from: ``local/...@01K...``.
    ARTIFACT = "artifact"


class PredictionColumn(StrEnum):
    """The columns of a backtest's prediction table, besides the instance keys.

    The instance keys sit between ``fold`` and ``origin_time`` under the
    caller's own names, for the same reason a forecast carries them: a
    prediction has to say which instance it is about.
    """

    MODEL = "model"
    FOLD = "fold"
    #: The historical origin the forecast was made at.
    ORIGIN_TIME = "origin_time"
    #: The moment being forecast.
    EVENT_TIME = "event_time"
    #: How far ahead of the origin that moment is, counting from 1.
    HORIZON_STEP = "horizon_step"
    TARGET = "target"
    #: ``point``, ``quantile`` or ``sample`` — which part of the predictive
    #: distribution this row holds, spelled as a forecast spells it.
    KIND = "kind"
    #: The level, for a quantile row; null otherwise.
    QUANTILE = "quantile"
    #: The draw index, for a sample row; null otherwise.
    SAMPLE = "sample"
    PREDICTION = "prediction"
    #: What happened. Null where the truth published no outcome, which is also
    #: what makes such a row unscorable.
    ACTUAL = "actual"


#: The canonical column order of the metric table.
BACKTEST_COLUMNS = tuple(column.value for column in BacktestColumn)

#: The columns of the prediction table that are not instance keys. The keys are
#: everything else, and sit after ``fold``.
PREDICTION_COLUMNS = tuple(column.value for column in PredictionColumn)

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
    """The predictions of one backtest, the metrics over them, and the readings.

    The predictions are where the memory goes: one row per model, fold,
    instance, event time and target — origins × horizon × instances × targets
    per model. A year of hourly origins over a wide panel is a large table, and
    it is retained by default anyway, because the metrics are derivable from it
    and not the reverse.
    """

    def __init__(
        self, metrics: pa.Table, predictions: pa.Table, *, scored_by: Sequence[Metric]
    ) -> None:
        missing = [name for name in BACKTEST_COLUMNS if name not in metrics.column_names]
        if missing:
            raise DataError(f"a backtest result is missing the metric columns {missing}")
        absent = [name for name in PREDICTION_COLUMNS if name not in predictions.column_names]
        if absent:
            raise DataError(f"a backtest result is missing the prediction columns {absent}")
        self._metrics = metrics.select(list(BACKTEST_COLUMNS))
        self._instance_keys = tuple(
            name for name in predictions.column_names if name not in PREDICTION_COLUMNS
        )
        self._predictions = predictions.select(list(self._prediction_columns()))
        self._scored_by = {metric.name: metric for metric in scored_by}

    # -- accessors ---------------------------------------------------------

    @property
    def metrics(self) -> pa.Table:
        """The long metric table, in canonical column order."""
        return self._metrics

    @property
    def predictions(self) -> pa.Table:
        """Every point prediction the metrics were computed from."""
        return self._predictions

    @property
    def models(self) -> tuple[str, ...]:
        """The candidates, in the order they were backtested."""
        return self._distinct(BacktestColumn.MODEL)

    @property
    def metric_names(self) -> tuple[str, ...]:
        """What was measured, spelled as the ``metric`` column spells it."""
        return self._distinct(BacktestColumn.METRIC)

    @property
    def instance_keys(self) -> tuple[str, ...]:
        """The caller's own instance key columns, as ``predictions`` carries them."""
        return self._instance_keys

    @property
    def origins(self) -> tuple[datetime, ...]:
        """The evaluation origins, in ascending order."""
        values: list[datetime] = column_values(self._metrics, BacktestColumn.ORIGIN.value)
        return tuple(sorted(set(values)))

    # -- projections -------------------------------------------------------

    def leaderboard(self, metric: str | Metric | None = None) -> pa.Table:
        """The models ranked by one metric, averaged over the folds.

        Best first, where "best" is the metric's own idea of it: lowest for an
        error, closest to zero for a bias, closest to its nominal level for a
        coverage. A metric is required when the backtest measured several,
        because ranking by whichever came first would answer a question nobody
        asked.

        A model this metric could score on no fold at all is last, with a null
        value: it was measured and came back with nothing, which is a different
        result from a bad score and from not having been entered.
        """
        name = self._one_metric(metric)
        rows = self._rows_for(name)
        ranker = self._scored_by[name]
        ordered = sorted(rows, key=lambda entry: _ranked(ranker, entry.value))
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
        to keep is a question about the folds, and :attr:`metrics` holds them.
        """
        return str(column_values(self.leaderboard(metric), BacktestColumn.MODEL.value)[0])

    def metrics_by(self, keys: str | Sequence[str]) -> pa.Table:
        """Every metric again, grouped by columns of :attr:`predictions`.

        ```python
        result.metrics_by("horizon_step")
        result.metrics_by(["horizon_step", "zone"])
        ```

        Computed from the predictions rather than by re-running anything, which
        is why the group keys are prediction columns and an unknown one is an
        error naming the ones that exist. ``model`` is always a key, because a
        number pooled over the candidates compares nothing.

        The folds are pooled inside each group, so this is a different reading
        of the same measurements rather than the fold table sliced: a
        ``horizon_step`` group holds every origin's step *k*.
        """
        requested = (keys,) if isinstance(keys, str) else tuple(keys)
        unknown = [name for name in requested if name not in self._predictions.column_names]
        if unknown:
            raise RecipeError(
                f"{unknown} are not columns of the prediction table, which holds "
                f"{list(self._predictions.column_names)}"
            )
        return self._grouped(
            (
                PredictionColumn.MODEL.value,
                *(name for name in requested if name != PredictionColumn.MODEL.value),
            )
        )

    def to_pandas(self) -> Any:
        """The long metric table as a pandas ``DataFrame``.

        pandas is not a dependency of OpenForecast — this converts through
        Arrow, which is where the data already is.
        """
        return self._metrics.to_pandas()  # pyright: ignore[reportUnknownMemberType]

    # -- internals ---------------------------------------------------------

    def _prediction_columns(self) -> tuple[str, ...]:
        """``model, fold, instance keys..., origin_time ... actual``."""
        head = (PredictionColumn.MODEL.value, PredictionColumn.FOLD.value)
        return (*head, *self._instance_keys, *PREDICTION_COLUMNS[len(head) :])

    def _distinct(self, column: BacktestColumn) -> tuple[str, ...]:
        values: list[str] = column_values(self._metrics, column.value)
        return tuple(dict.fromkeys(values))

    def _one_metric(self, metric: str | Metric | None) -> str:
        available = self.metric_names
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
        """One summary per model, over the folds that metric was measured on.

        A fold this metric scored nothing of carries a null value, and averaging
        it in as a zero would report a score for a fold that produced none. So
        the mean is over the folds that have a value, and ``folds`` says how many
        that was — the same rule ``fit_seconds`` follows for a frozen revision.
        """
        grouped: dict[str, list[tuple[float | None, float | None, float]]] = {}
        for model, name, value, fit, predict in zip(
            column_values(self._metrics, BacktestColumn.MODEL.value),
            column_values(self._metrics, BacktestColumn.METRIC.value),
            column_values(self._metrics, BacktestColumn.VALUE.value),
            column_values(self._metrics, BacktestColumn.FIT_SECONDS.value),
            column_values(self._metrics, BacktestColumn.FORECAST_SECONDS.value),
            strict=True,
        ):
            if name == metric:
                grouped.setdefault(str(model), []).append((value, fit, predict))
        return [
            _Summary(
                model=model,
                value=_mean_or_none(value for value, _, _ in measured),
                folds=sum(1 for value, _, _ in measured if value is not None),
                fit_seconds=_mean_or_none(fit for _, fit, _ in measured),
                forecast_seconds=_mean(predict for _, _, predict in measured),
            )
            for model, measured in grouped.items()
        ]

    def _grouped(self, grouping: Sequence[str]) -> pa.Table:
        """Every metric, computed over the predictions of each group.

        The rows of one group are gathered back into one prediction per outcome
        before anything is scored, exactly as the per-fold metric rows were: a
        quantile forecast holds several rows about one event time, and a metric
        of a distribution needs them together. Which is also why grouping by
        ``quantile`` or ``kind`` is allowed but rarely useful — it splits a
        distribution across groups, and each group then holds whatever part of it
        landed there.
        """
        gathered = self._by_group(grouping)
        rows = [(group, metric) for group in gathered for metric in self._scored_by.values()]
        measured = [metric.measure(gathered[group]) for group, metric in rows]
        built: dict[str, tuple[list[Any], pa.DataType]] = {
            name: ([group[position] for group, _ in rows], column_type(self._predictions, name))
            for position, name in enumerate(grouping)
        }
        built[BacktestColumn.METRIC.value] = ([metric.name for _, metric in rows], pa.string())
        built[BacktestColumn.VALUE.value] = (
            [measurement.value for measurement in measured],
            pa.float64(),
        )
        built[BacktestColumn.PAIRS.value] = (
            [measurement.pairs for measurement in measured],
            pa.int64(),
        )
        return build_table(built)

    def _by_group(self, grouping: Sequence[str]) -> dict[tuple[Any, ...], tuple[Prediction, ...]]:
        """The predictions of each group, one per outcome, in group order."""
        columns = [column_values(self._predictions, name) for name in grouping]
        identity = [column_values(self._predictions, name) for name in self._outcome_columns()]
        kinds = column_values(self._predictions, PredictionColumn.KIND.value)
        levels = column_values(self._predictions, PredictionColumn.QUANTILE.value)
        draws = column_values(self._predictions, PredictionColumn.SAMPLE.value)
        predicted = column_values(self._predictions, PredictionColumn.PREDICTION.value)
        outcomes = column_values(self._predictions, PredictionColumn.ACTUAL.value)

        rows: dict[tuple[Any, ...], list[PredictedValue]] = {}
        for position, group in enumerate(zip(*columns, strict=True)):
            rows.setdefault(group, []).append(
                PredictedValue(
                    outcome=tuple(values[position] for values in identity),
                    kind=str(kinds[position]),
                    level=levels[position],
                    draw=draws[position],
                    predicted=predicted[position],
                    actual=outcomes[position],
                )
            )
        return {group: predictions_of(values) for group, values in rows.items()}

    def _outcome_columns(self) -> tuple[str, ...]:
        """What identifies one outcome, whatever the grouping is.

        Everything but the columns describing *which part* of the distribution a
        row holds, and the value it holds.
        """
        describes_the_row = {
            PredictionColumn.KIND.value,
            PredictionColumn.QUANTILE.value,
            PredictionColumn.SAMPLE.value,
            PredictionColumn.PREDICTION.value,
            PredictionColumn.ACTUAL.value,
        }
        return tuple(
            name for name in self._predictions.column_names if name not in describes_the_row
        )

    # -- dunder ------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BacktestResult(models={len(self.models)}, folds={len(self.origins)}, "
            f"metrics={list(self.metric_names)}, rows={self._metrics.num_rows}, "
            f"predictions={self._predictions.num_rows})"
        )

    def __str__(self) -> str:
        """The ranking when one metric was measured, and the shape otherwise."""
        if len(self.metric_names) != 1:
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
        return f"BacktestResult({self.metric_names[0]}: {ranked})"


@dataclass(frozen=True)
class _Summary:
    """One model's measurements of one metric, averaged over its folds."""

    model: str
    #: Null where no fold of this model could be scored by this metric.
    value: float | None
    folds: int
    #: Null for a frozen revision, which was evaluated rather than fitted.
    fit_seconds: float | None
    forecast_seconds: float


def _ranked(metric: Metric, value: float | None) -> float:
    """A value as the number a leaderboard sorts by.

    A model with no value at all sorts last rather than first: nothing measured
    is not a perfect score, and a ranking that put it above a model that was
    actually scored would name it the winner.
    """
    return math.inf if value is None else metric.rank(value)


def _mean(values: Iterable[float]) -> float:
    listed = list(values)
    return sum(listed) / len(listed)


def _mean_or_none(values: Iterable[float | None]) -> float | None:
    """The mean of the values that exist, or null when none of them do.

    A frozen revision records no ``fit_seconds`` at all, so its mean is not
    zero — zero would say the fit was instant rather than that there was none.
    """
    listed = [value for value in values if value is not None]
    return _mean(listed) if listed else None
