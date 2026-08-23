"""What a forecast is scored by.

```python
of.MAE()    # mean absolute error
of.RMSE()   # root mean squared error
of.MAPE()   # mean absolute percentage error, in percent
of.Bias()   # mean signed error: positive means over-forecasting
```

Each metric is its own type in a discriminated union rather than a string, for
the same reason the origin selections are: a metric has to survive being written
down — in a result table, in a request, in a report — and a name that a reader
has to look up in a table of accepted spellings is a name that will eventually be
misspelled. ``{"metric": "mae"}`` reads back as ``MAE()``.

They score *point* forecasts, which is what :func:`~openforecast.evaluation.backtest.backtest`
asks models for. Scoring a predictive distribution needs the probabilistic
output protocol to be normalized across providers first — a quantile means the
same thing everywhere before a pinball loss over it means anything — so those
metrics arrive with it rather than being guessed at here.

Two of the four have an edge that has to be a decision rather than an accident:

```text
MAPE   undefined where the outcome is zero, and refused rather than skipped
Bias   best at zero rather than lowest, so ranking it ranks its magnitude
```
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from openforecast.errors import DataError

__all__ = ["MAE", "MAPE", "RMSE", "Bias", "Metric", "MetricKind"]


class MetricKind(StrEnum):
    """The discriminator of the metrics, and their wire spelling."""

    MAE = "mae"
    RMSE = "rmse"
    MAPE = "mape"
    BIAS = "bias"


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

    @abstractmethod
    def compute(self, actual: Sequence[float], predicted: Sequence[float]) -> float:
        """Score one set of paired outcomes and forecasts."""
        raise NotImplementedError

    def rank(self, value: float) -> float:
        """This metric's value as a lower-is-better number.

        Every metric is minimized *after* this transform, which is what lets one
        leaderboard sort them all without a per-metric branch. It is the identity
        for the error metrics and the magnitude for a signed one.
        """
        return value

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


class MAE(_Metric):
    """Mean absolute error. The default choice, and the hardest to misread."""

    metric: Literal[MetricKind.MAE] = MetricKind.MAE

    @property
    def kind(self) -> MetricKind:
        return MetricKind.MAE

    def compute(self, actual: Sequence[float], predicted: Sequence[float]) -> float:
        pairs = self._errors(actual, predicted)
        return sum(abs(outcome - forecast) for outcome, forecast in pairs) / len(pairs)


class RMSE(_Metric):
    """Root mean squared error — the same units as the target, weighted to the tails."""

    metric: Literal[MetricKind.RMSE] = MetricKind.RMSE

    @property
    def kind(self) -> MetricKind:
        return MetricKind.RMSE

    def compute(self, actual: Sequence[float], predicted: Sequence[float]) -> float:
        pairs = self._errors(actual, predicted)
        return math.sqrt(sum((outcome - forecast) ** 2 for outcome, forecast in pairs) / len(pairs))


class MAPE(_Metric):
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


class Bias(_Metric):
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


#: Any of the four. Annotated with the discriminator so that a serialized metric
#: deserializes back into the same type it was written from.
Metric = Annotated[MAE | RMSE | MAPE | Bias, Field(discriminator="metric")]
