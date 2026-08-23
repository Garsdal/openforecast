"""What a model said about one outcome, whatever form it said it in.

A metric scores an *outcome*: one instance, one event time, one target, and the
number that turned out to be true. A probabilistic forecast holds several rows
about that outcome — one per quantile level, or one per sample path — so the rows
have to be gathered back into one predictive distribution before anything can be
computed over it:

```text
DE 12:00 price quantile 0.1 65          Prediction(actual=80,
DE 12:00 price quantile 0.5 79    ->               quantiles={0.1: 65, 0.5: 79, 0.9: 96})
DE 12:00 price quantile 0.9 96
```

That is the whole of this module, and it is shared rather than done twice: the
per-fold metric rows of a backtest and the regroupings
``BacktestResult.metrics_by`` computes are the same arithmetic over the same
predictions, and a second implementation of the gathering would eventually
disagree with the first about which rows belong to one outcome.

Two readings are defined here rather than in each metric, because they are
statements about a distribution rather than about a score:

```text
prediction.point_estimate    the point, or the median of a distribution
prediction.quantile(level)   a level held, or read off the draws
```

Both follow the rule of Step 20. Reducing a distribution to one of its readings
is a projection of what the model said; manufacturing a distribution around a
point forecast is an invention, and there is no method here that does it.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field

from openforecast.data._arrow import is_missing
from openforecast.errors import DataError
from openforecast.protocol.quantiles import quantile_of_samples
from openforecast.tasks.forecast import OutputKind

__all__ = ["MEDIAN", "PredictedValue", "Prediction", "predictions_of"]

#: The quantile a point estimate is read off a predictive distribution at. The
#: median rather than the mean, because it is a level a forecast can actually
#: hold — a mean is not a quantile, and the draws that would give one are not
#: there once a model has answered in quantiles.
MEDIAN = 0.5


@dataclass(frozen=True)
class Prediction:
    """One outcome, and everything one model said about it.

    Exactly one of the three forms is populated, because a forecast answers one
    request: ``point`` for a point forecast, ``quantiles`` for a quantile one,
    ``samples`` for draws.
    """

    #: What happened. A prediction is only built where the truth published one,
    #: since an outcome that does not exist cannot be scored against.
    actual: float
    point: float | None = None
    #: ``level -> value``, for the levels the forecast holds.
    quantiles: Mapping[float, float] = field(default_factory=lambda: {})
    #: The draws, in draw order.
    samples: tuple[float, ...] = ()

    @property
    def has_point_estimate(self) -> bool:
        return self.point is not None or MEDIAN in self.quantiles or bool(self.samples)

    @property
    def point_estimate(self) -> float:
        """The one number a point metric scores.

        A point forecast is that number. A distribution's is its median, which is
        a reading of what the model said rather than an assumption about it — and
        it is refused rather than interpolated when the median is not a level the
        forecast holds: a 0.5 derived from a 0.1 and a 0.9 is a different number
        from the one the model would have produced.
        """
        if self.point is not None:
            return self.point
        if MEDIAN in self.quantiles:
            return self.quantiles[MEDIAN]
        if self.samples:
            return quantile_of_samples(self.samples, MEDIAN)
        raise DataError(
            f"this forecast holds no point estimate: it holds the quantiles "
            f"{sorted(self.quantiles)}, and a point metric scores the median. Ask for "
            f"{MEDIAN} among the levels, or score it with of.PinballLoss(level)"
        )

    def holds(self, level: float) -> bool:
        return level in self.quantiles or bool(self.samples)

    def quantile(self, level: float) -> float:
        """One quantile of this predictive distribution.

        Read off the draws when the forecast holds draws, which is the one
        conversion Step 20 allows: the draws *are* the distribution. Never
        interpolated between two levels the model did state, and never derived
        from a point forecast.
        """
        if level in self.quantiles:
            return self.quantiles[level]
        if self.samples:
            return quantile_of_samples(self.samples, level)
        raise DataError(
            f"this forecast holds no quantile {level}; it holds {sorted(self.quantiles) or 'none'}"
        )


@dataclass(frozen=True)
class PredictedValue:
    """One row of a long forecast, as a metric's input rather than as a table.

    ``outcome`` is whatever identifies the thing being forecast — the instance,
    the origin, the event time and the target, in whatever form the caller
    already has them. It is only ever compared for equality, so that the rows
    about one outcome can be found without this module deciding what an outcome
    is keyed by.
    """

    outcome: Hashable
    kind: str
    level: float | None
    draw: int | None
    predicted: float | None
    actual: float | None

    @property
    def is_scorable(self) -> bool:
        """Whether this row can contribute to a metric at all.

        Null and NaN are the two spellings of "no value here", and either on
        either side means this row says nothing a metric can score. An
        unanswered event time is not an error of zero, and an outcome the truth
        never published is not one either.
        """
        return not (
            self.predicted is None
            or self.actual is None
            or is_missing(self.predicted)
            or is_missing(self.actual)
        )


def predictions_of(rows: Iterable[PredictedValue]) -> tuple[Prediction, ...]:
    """The rows, gathered into one :class:`Prediction` per outcome.

    In the order the outcomes first appear, so that a metric computed over the
    result is computed over the folds in the order they were run. Unscorable rows
    are left out rather than counted as zero errors: what a metric was able to
    score is reported per metric as ``pairs``, which is the honest answer to a
    fold whose truth is only partly published.
    """
    gathered: dict[Hashable, _Gathering] = {}
    for row in rows:
        if not row.is_scorable:
            continue
        assert row.actual is not None and row.predicted is not None  # what is_scorable checks
        gathering = gathered.setdefault(row.outcome, _Gathering(actual=float(row.actual)))
        gathering.add(row, float(row.predicted))
    return tuple(gathering.prediction() for gathering in gathered.values())


@dataclass
class _Gathering:
    """The rows about one outcome, as they arrive."""

    actual: float
    point: float | None = None
    quantiles: dict[float, float] = field(default_factory=lambda: {})
    samples: dict[int, float] = field(default_factory=lambda: {})

    def add(self, row: PredictedValue, value: float) -> None:
        """One row, filed under what it says it is.

        A row whose ``kind`` says quantile and whose level is null describes no
        part of any distribution, and is refused rather than counted as the point
        forecast — a value filed under the wrong reading is worse than a missing
        one, because a metric would then score it.
        """
        if row.kind == OutputKind.POINT.row_kind:
            self.point = value
        elif row.kind == OutputKind.QUANTILES.row_kind and row.level is not None:
            self.quantiles[row.level] = value
        elif row.kind == OutputKind.SAMPLES.row_kind and row.draw is not None:
            self.samples[row.draw] = value
        else:
            raise DataError(
                f"a prediction of kind {row.kind!r} carrying quantile {row.level} and sample "
                f"{row.draw} describes no part of a forecast"
            )

    def prediction(self) -> Prediction:
        return Prediction(
            actual=self.actual,
            point=self.point,
            quantiles=dict(sorted(self.quantiles.items())),
            samples=tuple(value for _, value in sorted(self.samples.items())),
        )
