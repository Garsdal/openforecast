"""What a forecast is scored by.

```python
of.MAE()    # mean absolute error
of.RMSE()   # root mean squared error
of.MAPE()   # mean absolute percentage error, in percent
of.Bias()   # mean signed error: positive means over-forecasting
```

and, since Step 20, the three that score a predictive distribution rather than a
number:

```python
of.PinballLoss(0.9)   # the loss the 0.9 quantile is the optimal answer to
of.Coverage()         # how often the outcome fell inside the interval
of.IntervalWidth()    # how wide that interval was
```

Each metric is its own type in a discriminated union rather than a string, for
the same reason the origin selections are: a metric has to survive being written
down — in a result table, in a request, in a report — and a name that a reader
has to look up in a table of accepted spellings is a name that will eventually be
misspelled. ``{"metric": "mae"}`` reads back as ``MAE()``. The probabilistic
three are also the argument for it: a pinball loss is a loss *at a level*, and a
string cannot say which one, so their names carry the parameter —
``pinball[0.9]``, ``coverage[0.8]`` — and the leaderboard still ranks by a
string.

## What a metric is given

One :class:`~openforecast.evaluation.predictions.Prediction` per outcome, holding
whatever the model said about it: a point, the quantile levels it answered, or
its sample draws. That is what lets one metric list score any provider — a
pinball loss reads the 0.9 quantile of a quantile forecast and of a sample
forecast alike, and neither reading invents anything the model did not say.

A metric scores the predictions it has what it needs for, and
:class:`Measurement` reports how many that was. A backtest fold whose truth is
half unpublished, or a quantile forecast asked for a coverage it holds no
interval for, is a number over fewer outcomes — and the count is in the result
beside the value rather than left to be inferred from it.

## The edges that are decisions rather than accidents

```text
MAPE           undefined where the outcome is zero, and refused rather than skipped
Bias           best at zero rather than lowest, so ranking it ranks its magnitude
Coverage       best at its nominal level, so ranking it ranks the distance to it
IntervalWidth  lower is better only beside a coverage: alone, zero width wins
```
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from openforecast.errors import DataError, RecipeError
from openforecast.evaluation.predictions import MEDIAN, Prediction
from openforecast.tasks.forecast import OutputKind, OutputSpec

__all__ = [
    "MAE",
    "MAPE",
    "RMSE",
    "Bias",
    "Coverage",
    "IntervalWidth",
    "Measurement",
    "Metric",
    "MetricKind",
    "PinballLoss",
]


#: Decimal places a derived quantile level is rounded to before it is looked up
#: among the levels a forecast holds. Ten is far past any level anybody names and
#: far short of where binary floating point starts disagreeing with itself.
_LEVEL_PRECISION = 10


class MetricKind(StrEnum):
    """The discriminator of the metrics, and their wire spelling."""

    MAE = "mae"
    RMSE = "rmse"
    MAPE = "mape"
    BIAS = "bias"
    PINBALL = "pinball"
    COVERAGE = "coverage"
    INTERVAL_WIDTH = "interval_width"


@dataclass(frozen=True)
class Measurement:
    """One metric over a set of predictions, and how many it could score.

    ``value`` is null exactly when ``pairs`` is zero. A metric over nothing is
    not a zero score — a fold that could not be scored at all and a fold scored
    at zero error are opposite results — so the absence is reported rather than
    filled in.
    """

    value: float | None
    pairs: int


class _Metric(BaseModel, ABC):
    """What the metrics share: immutability, a name, and how to be ranked.

    The ``metric`` tag is declared by each subclass rather than here, because it
    is the discriminator: every subclass narrows it to exactly one value.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    @property
    def name(self) -> str:
        """The spelling that appears in a backtest's ``metric`` column."""
        return str(self.kind)

    @property
    @abstractmethod
    def kind(self) -> MetricKind:
        raise NotImplementedError

    def measure(self, predictions: Sequence[Prediction]) -> Measurement:
        """This metric over the predictions it has what it needs to score."""
        usable = [prediction for prediction in predictions if self.can_score(prediction)]
        if not usable:
            return Measurement(value=None, pairs=0)
        return Measurement(value=self._over(usable), pairs=len(usable))

    @abstractmethod
    def can_score(self, prediction: Prediction) -> bool:
        """Whether this metric can be computed from what the model said here."""
        raise NotImplementedError

    @abstractmethod
    def _over(self, predictions: Sequence[Prediction]) -> float:
        """This metric over predictions :meth:`can_score` accepted."""
        raise NotImplementedError

    @abstractmethod
    def requirement(self, output: OutputSpec) -> str | None:
        """Why this metric cannot score that kind of forecast, or ``None``.

        Answered from the request rather than from the answer, so that a backtest
        asking for a coverage of a point forecast fails before it fits anything
        — the same reason the engine checks an output request against a model's
        declared capabilities before starting a provider.
        """
        raise NotImplementedError

    def rank(self, value: float) -> float:
        """This metric's value as a lower-is-better number.

        Every metric is minimized *after* this transform, which is what lets one
        leaderboard sort them all without a per-metric branch. It is the identity
        for the error metrics, the magnitude for a signed one, and the distance
        from the nominal level for a coverage.
        """
        return value


# -- point metrics ----------------------------------------------------------


class _PointMetric(_Metric, ABC):
    """A metric of one number against one outcome.

    Its arithmetic is stated as :meth:`compute` over paired sequences, which is
    what the metric *is* — the gathering of a distribution down to its median
    happens before it and is not part of the definition of a mean absolute error.
    """

    @abstractmethod
    def compute(self, actual: Sequence[float], predicted: Sequence[float]) -> float:
        """Score one set of paired outcomes and forecasts."""
        raise NotImplementedError

    def can_score(self, prediction: Prediction) -> bool:
        return prediction.has_point_estimate

    def _over(self, predictions: Sequence[Prediction]) -> float:
        return self.compute(
            [prediction.actual for prediction in predictions],
            [prediction.point_estimate for prediction in predictions],
        )

    def requirement(self, output: OutputSpec) -> str | None:
        """A point metric needs a point, or a median it can read as one."""
        if output.kind is not OutputKind.QUANTILES or MEDIAN in output.levels:
            return None
        return (
            f"{self.name} scores one number against each outcome, and a quantile forecast of "
            f"{list(output.levels)} holds none: the median of a distribution is its point "
            f"estimate, and {MEDIAN} is not among the levels asked for. Add it, or score this "
            f"forecast with of.PinballLoss(level)"
        )

    def _errors(
        self, actual: Sequence[float], predicted: Sequence[float]
    ) -> list[tuple[float, float]]:
        """The pairs, guarded: a metric over nothing is not a zero score."""
        if not actual:
            raise DataError(
                f"{self.name} has nothing to score: no forecast event time had an outcome "
                f"to compare against"
            )
        return list(zip(actual, predicted, strict=True))


class MAE(_PointMetric):
    """Mean absolute error. The default choice, and the hardest to misread."""

    metric: Literal[MetricKind.MAE] = MetricKind.MAE

    @property
    def kind(self) -> MetricKind:
        return MetricKind.MAE

    def compute(self, actual: Sequence[float], predicted: Sequence[float]) -> float:
        pairs = self._errors(actual, predicted)
        return sum(abs(outcome - forecast) for outcome, forecast in pairs) / len(pairs)


class RMSE(_PointMetric):
    """Root mean squared error — the same units as the target, weighted to the tails."""

    metric: Literal[MetricKind.RMSE] = MetricKind.RMSE

    @property
    def kind(self) -> MetricKind:
        return MetricKind.RMSE

    def compute(self, actual: Sequence[float], predicted: Sequence[float]) -> float:
        pairs = self._errors(actual, predicted)
        return math.sqrt(sum((outcome - forecast) ** 2 for outcome, forecast in pairs) / len(pairs))


class MAPE(_PointMetric):
    """Mean absolute percentage error, in percent.

    Refused rather than approximated where an outcome is zero: the percentage
    error of a zero outcome is not a large number, it is not a number. Skipping
    those rows would silently score a different subset of the horizon for one
    model than for another, which is exactly the comparison a backtest exists to
    make sound.
    """

    metric: Literal[MetricKind.MAPE] = MetricKind.MAPE

    @property
    def kind(self) -> MetricKind:
        return MetricKind.MAPE

    def compute(self, actual: Sequence[float], predicted: Sequence[float]) -> float:
        pairs = self._errors(actual, predicted)
        zeros = sum(1 for outcome, _ in pairs if outcome == 0.0)
        if zeros:
            raise DataError(
                f"mape is undefined for {zeros} outcomes that are zero; score this data with "
                f"of.MAE() or of.RMSE(), which are defined everywhere"
            )
        return (
            100.0
            * sum(abs((outcome - forecast) / outcome) for outcome, forecast in pairs)
            / len(pairs)
        )


class Bias(_PointMetric):
    """Mean signed error: positive means the model forecast too high.

    Not an accuracy metric and not ranked as one. Zero is the best value a bias
    can have, so :meth:`rank` compares magnitudes — a model biased by -3 and one
    biased by +3 are equally biased, and a leaderboard that put one above the
    other would be reporting the sign as a virtue.
    """

    metric: Literal[MetricKind.BIAS] = MetricKind.BIAS

    @property
    def kind(self) -> MetricKind:
        return MetricKind.BIAS

    def compute(self, actual: Sequence[float], predicted: Sequence[float]) -> float:
        pairs = self._errors(actual, predicted)
        return sum(forecast - outcome for outcome, forecast in pairs) / len(pairs)

    def rank(self, value: float) -> float:
        return abs(value)


# -- probabilistic metrics --------------------------------------------------


class PinballLoss(_Metric):
    """The loss a quantile forecast is the optimal answer to.

    ```python
    of.PinballLoss(0.9)
    ```

    ```text
    outcome above the forecast    (outcome - forecast) * level
    outcome below the forecast    (forecast - outcome) * (1 - level)
    ```

    Asymmetric on purpose, and that asymmetry is the whole content of the metric:
    a 0.9 quantile is meant to be exceeded one time in ten, so being under the
    outcome is charged nine times what being over it is. Minimized in expectation
    by exactly the quantile it names, which is what makes it the metric that says
    whether a *quantile* was any good rather than whether a number was close.

    One level per metric, because one loss is one level. Scoring three levels is
    three metrics, and their sum is not a thing this returns: which levels to
    weight and how is a decision about the result, and
    :attr:`~openforecast.evaluation.result.BacktestResult.metrics` holds them all
    separately so it can be made there.
    """

    metric: Literal[MetricKind.PINBALL] = MetricKind.PINBALL
    #: The quantile level scored, strictly between 0 and 1.
    level: float = Field(gt=0.0, lt=1.0)

    def __init__(self, level: float | None = None, /, **data: Any) -> None:
        """``PinballLoss(0.9)`` as well as ``PinballLoss(level=0.9)``."""
        super().__init__(**_with_level(level, data))

    @property
    def kind(self) -> MetricKind:
        return MetricKind.PINBALL

    @property
    def name(self) -> str:
        """``pinball[0.9]`` — the level is part of what was measured."""
        return f"{self.kind}[{self.level:g}]"

    def can_score(self, prediction: Prediction) -> bool:
        return prediction.holds(self.level)

    def _over(self, predictions: Sequence[Prediction]) -> float:
        return sum(self._loss(prediction) for prediction in predictions) / len(predictions)

    def _loss(self, prediction: Prediction) -> float:
        forecast = prediction.quantile(self.level)
        error = prediction.actual - forecast
        return error * self.level if error >= 0 else -error * (1.0 - self.level)

    def requirement(self, output: OutputSpec) -> str | None:
        return _needs_levels(self.name, (self.level,), output)


class _IntervalMetric(_Metric, ABC):
    """A metric of the central interval of a predictive distribution.

    Parameterized by the nominal coverage of that interval rather than by its two
    bounds, because an interval a coverage can be compared against has to be
    symmetric around the median: an 80% interval is the 0.1 and the 0.9, and
    naming the bounds separately would allow a "90% interval" spanning 0.1 to 0.9
    and a coverage that looks wrong for a reason nothing in the result explains.
    """

    #: The nominal coverage of the interval — 0.8 is the 0.1 to 0.9 interval.
    level: float = Field(default=0.8, gt=0.0, lt=1.0)

    @property
    def name(self) -> str:
        """``coverage[0.8]`` — the interval is part of what was measured."""
        return f"{self.kind}[{self.level:g}]"

    @property
    def bounds(self) -> tuple[float, float]:
        """The two quantile levels of the central interval.

        Rounded, because these levels are *looked up* against the levels a caller
        asked for: ``(1 - 0.8) / 2`` is 0.09999999999999998 in binary floating
        point, and a coverage that could not find the 0.1 the forecast plainly
        holds would be arithmetic leaking into a lookup.
        """
        tail = round((1.0 - self.level) / 2.0, _LEVEL_PRECISION)
        return tail, round(1.0 - tail, _LEVEL_PRECISION)

    def can_score(self, prediction: Prediction) -> bool:
        lower, upper = self.bounds
        return prediction.holds(lower) and prediction.holds(upper)

    def _interval(self, prediction: Prediction) -> tuple[float, float]:
        lower, upper = self.bounds
        return prediction.quantile(lower), prediction.quantile(upper)

    def requirement(self, output: OutputSpec) -> str | None:
        return _needs_levels(self.name, self.bounds, output)


class Coverage(_IntervalMetric):
    """How often the outcome fell inside the interval, as a fraction.

    ```python
    of.Coverage()        # of the 0.1 to 0.9 interval
    of.Coverage(0.5)     # of the 0.25 to 0.75 interval
    ```

    The calibration question, and the one a point metric cannot ask: an 80%
    interval that contains the outcome 55% of the time is overconfident, and one
    that contains it 99% of the time is useless in the other direction. So the
    best value is the nominal level rather than the highest, and :meth:`rank`
    compares the distance to it — a leaderboard that ranked coverage upwards
    would be ranking width.
    """

    metric: Literal[MetricKind.COVERAGE] = MetricKind.COVERAGE

    def __init__(self, level: float | None = None, /, **data: Any) -> None:
        """``Coverage(0.5)`` as well as ``Coverage(level=0.5)``."""
        super().__init__(**_with_level(level, data))

    @property
    def kind(self) -> MetricKind:
        return MetricKind.COVERAGE

    def _over(self, predictions: Sequence[Prediction]) -> float:
        inside = sum(1 for prediction in predictions if self._contains(prediction))
        return inside / len(predictions)

    def _contains(self, prediction: Prediction) -> bool:
        lower, upper = self._interval(prediction)
        return lower <= prediction.actual <= upper

    def rank(self, value: float) -> float:
        """The distance from the nominal coverage: 0.78 and 0.82 rank alike at 0.8."""
        return abs(value - self.level)


class IntervalWidth(_IntervalMetric):
    """How wide the interval was, in the units of the target.

    ```python
    of.IntervalWidth()   # mean width of the 0.1 to 0.9 interval
    ```

    Sharpness, and only half a question on its own: the narrowest interval any
    model can produce is a degenerate one, which scores perfectly here and fails
    a :class:`Coverage` completely. Ranked lower-is-better anyway, because that
    is what it means among models whose coverage is comparable — reading it
    without one beside it is the mistake, and no ranking rule prevents that.
    """

    metric: Literal[MetricKind.INTERVAL_WIDTH] = MetricKind.INTERVAL_WIDTH

    def __init__(self, level: float | None = None, /, **data: Any) -> None:
        """``IntervalWidth(0.5)`` as well as ``IntervalWidth(level=0.5)``."""
        super().__init__(**_with_level(level, data))

    @property
    def kind(self) -> MetricKind:
        return MetricKind.INTERVAL_WIDTH

    def _over(self, predictions: Sequence[Prediction]) -> float:
        widths = [upper - lower for lower, upper in map(self._interval, predictions)]
        return sum(widths) / len(widths)


def _with_level(level: float | None, data: dict[str, Any]) -> dict[str, Any]:
    """A positionally given level, as the field it names.

    Written once and called from each metric that takes one rather than
    inherited: pydantic regenerates ``__init__`` for every model class, so an
    inherited positional argument is one a type checker cannot see.
    """
    if level is None:
        return data
    if "level" in data:
        raise RecipeError("level was given both positionally and by keyword")
    return {**data, "level": level}


def _needs_levels(name: str, levels: Sequence[float], output: OutputSpec) -> str | None:
    """Why ``name`` cannot score ``output``, when it needs these quantile levels.

    Sample draws satisfy any level, since reading a quantile out of the draws is
    a projection of the distribution the model gave. A point forecast satisfies
    none, and that is the rule of Step 20 stated where it is enforced: a
    deterministic model's distribution would have to be invented, so a
    probabilistic metric over a point forecast is refused rather than
    approximated.
    """
    wanted = [f"{level:g}" for level in levels]
    if output.kind is OutputKind.SAMPLES:
        return None
    if output.kind is OutputKind.POINT:
        return (
            f"{name} scores a predictive distribution and a point forecast is not one; ask for "
            f"of.OutputSpec.quantiles([{', '.join(wanted)}]), or score points with of.MAE()"
        )
    missing = [f"{level:g}" for level in levels if level not in output.levels]
    if missing:
        return (
            f"{name} needs the quantiles [{', '.join(wanted)}] and this forecast was asked for "
            f"{list(output.levels)}; add [{', '.join(missing)}] to the levels. A quantile "
            f"between two the model stated is not one it stated"
        )
    return None


#: Any of the seven. Annotated with the discriminator so that a serialized metric
#: deserializes back into the same type it was written from.
Metric = Annotated[
    MAE | RMSE | MAPE | Bias | PinballLoss | Coverage | IntervalWidth,
    Field(discriminator="metric"),
]
