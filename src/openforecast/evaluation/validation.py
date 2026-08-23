"""Which historical origins a backtest evaluates at, and what "correct" means there.

```python
of.RollingOrigin(horizon=24, windows=5)

of.ForecastOriginValidation(
    origins=of.OriginsBetween(start, end, stride=24),
    horizon=72,
)
```

The two are not two ways of writing one thing. They differ in what a historical
origin *is*, which is the same distinction the semantic data layer is built on:

```text
RollingOrigin              a TimeSeriesFrame, cut back to the shape the past had
ForecastOriginValidation   a ForecastDataset, at the vintages that really existed
```

So each names the source it can honestly evaluate, and refuses the other rather
than quietly doing something adjacent. A rolling origin over real vintages would
have to pick which vintage each origin means; a forecast-origin validation over
event-time data would have to invent origins that were never issued.

## The guarantee

For every origin a fold is built at:

```text
features come from that exact historical origin
truth comes from the truth TimeSeriesFrame
later vintages are inaccessible, not merely unused
```

Inaccessible rather than unused is the whole point, and it is a property of the
objects rather than of the loop below. A fold does not hold the dataset and an
origin — it holds
:meth:`~openforecast.data.forecast_dataset.ForecastDataset.up_to`'s answer, which
is a *different dataset* that does not contain the later vintages at all. There
is nothing for a model, a provider or a bug in this module to reach for.

The event-time counterpart is the same arrangement with the weaker guarantee it
can actually make: ``TimeSeriesFrame.up_to`` truncates the history, so no
outcome after the origin is reachable, but the feature values are still today's.
That difference survives into every backtest row as ``origin_fidelity``, which
is what makes "simulated availability versus true point-in-time availability" a
comparison a caller can run rather than a caveat they have to remember.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field

from openforecast.data._arrow import InstanceKey, build_table, column_type, column_values, key_rows
from openforecast.data.forecast_context import ForecastContext
from openforecast.data.forecast_dataset import ForecastDataset
from openforecast.data.frame import TimeSeriesFrame
from openforecast.errors import DataError
from openforecast.protocol.vocabulary import ForecastColumn
from openforecast.tasks.origins import AllOrigins, OriginSelection

__all__ = [
    "Fold",
    "ForecastOriginValidation",
    "RollingOrigin",
    "Validation",
    "ValidationMode",
]

#: What a fold's training data can be. Both are semantic source datasets, which
#: is what keeps a fold something ``of.fit`` accepts rather than a private shape.
TrainingSource = TimeSeriesFrame | ForecastDataset

#: What a fold is forecast from. A frame stands for its own last event time; a
#: context is one origin and says so.
OriginSource = TimeSeriesFrame | ForecastContext


class ValidationMode(StrEnum):
    """The discriminator of the validation strategies, and their wire spelling."""

    ROLLING = "rolling"
    ORIGINS = "origins"


@dataclass(frozen=True)
class Fold:
    """One evaluation origin: what to fit on, what to forecast from, what happened.

    ``truth`` is in exactly the shape :meth:`~openforecast.runtime.Forecast.point`
    returns — instance keys, ``event_time``, ``target``, ``value`` — because the
    two are about to be compared row by row, and a comparison between two shapes
    is a place for an alignment bug to live.
    """

    index: int
    origin: datetime
    #: The data as it stood at the origin. A *different* dataset, not the same
    #: one with a cut-off remembered alongside it.
    train: TrainingSource
    #: The single origin the forecast is made at.
    context: OriginSource
    truth: pa.Table

    def __repr__(self) -> str:
        return (
            f"Fold(index={self.index}, origin={self.origin.isoformat()}, "
            f"outcomes={self.truth.num_rows})"
        )


class _Validation(BaseModel, ABC):
    """What the strategies share: immutability, a horizon, and how to fold data."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    horizon: int = Field(ge=1)

    @abstractmethod
    def folds(self, data: object) -> tuple[Fold, ...]:
        """The evaluation origins ``data`` supports, in ascending order."""
        raise NotImplementedError


class RollingOrigin(_Validation):
    """``windows`` origins stepping back from the end of an event-time history.

    The last fold's forecast window ends at the last event time the data holds,
    and each earlier fold steps back by ``stride`` — the horizon by default,
    which makes the windows consecutive and non-overlapping. A shorter stride
    evaluates more often over the same history and is honest about the folds
    then sharing outcomes.

    Only a ``TimeSeriesFrame``. Point-in-time data has origins of its own and
    :class:`ForecastOriginValidation` is how they are used; choosing one vintage
    per rolling origin here would be this module inventing the very thing the
    data already records.
    """

    mode: Literal[ValidationMode.ROLLING] = ValidationMode.ROLLING
    windows: int = Field(ge=1)
    #: How far apart consecutive origins are. ``None`` means the horizon.
    stride: int | None = Field(default=None, ge=1)

    @property
    def step(self) -> int:
        return self.horizon if self.stride is None else self.stride

    def folds(self, data: object) -> tuple[Fold, ...]:
        if isinstance(data, ForecastDataset):
            raise DataError(
                "a rolling origin cuts historical origins out of one event-time history, and "
                "this data holds real forecast vintages; evaluate them with "
                "of.ForecastOriginValidation(origins=..., horizon=...), or backtest "
                "dataset.truth to compare against simulated availability"
            )
        if not isinstance(data, TimeSeriesFrame):
            raise DataError(f"a rolling origin folds a TimeSeriesFrame, got {type(data).__name__}")
        schema = data.schema
        times: list[datetime] = column_values(data.history, schema.time)
        end = max(times)
        origins = [
            schema.frequency.shift(end, -(self.horizon + index * self.step))
            for index in reversed(range(self.windows))
        ]
        earliest = min(times)
        if origins[0] < earliest:
            raise DataError(
                f"{self.windows} windows of {self.horizon} steps at a stride of {self.step} "
                f"reach back to {origins[0].isoformat()}, before this history begins at "
                f"{earliest.isoformat()}; ask for fewer windows, a shorter horizon or a "
                f"shorter stride"
            )
        return tuple(self._fold(data, index, origin) for index, origin in enumerate(origins))

    def _fold(self, data: TimeSeriesFrame, index: int, origin: datetime) -> Fold:
        """One fold. The truncated frame is both the training data and the origin.

        A ``TimeSeriesFrame`` whose history ends at the origin *is* that origin,
        which is what ``of.forecast`` reads it as, so there is nothing to keep in
        step between the two halves of a fold.
        """
        past = data.up_to(origin)
        return Fold(
            index=index,
            origin=origin,
            train=past,
            context=past,
            truth=outcomes(data, after=origin, horizon=self.horizon),
        )


class ForecastOriginValidation(_Validation):
    """The origins a point-in-time dataset actually holds, as evaluation origins.

    This is the strategy the semantic model exists for. At each selected vintage
    the model is fitted on
    :meth:`~openforecast.data.forecast_dataset.ForecastDataset.up_to` that
    origin, forecast from
    :meth:`~openforecast.data.forecast_dataset.ForecastDataset.at_origin` it, and
    scored against the truth frame — so what it is given is what was on the wire
    that day, revisions included, and the artifact records
    ``origin_fidelity: observed`` to say so.

    Only a ``ForecastDataset``: an event-time frame has no vintages to select,
    and :class:`RollingOrigin` is how its origins are simulated instead.
    """

    mode: Literal[ValidationMode.ORIGINS] = ValidationMode.ORIGINS
    #: Which vintages to evaluate at. The same four selections a fit plan uses,
    #: because "which origins" is one question with one vocabulary.
    origins: OriginSelection = AllOrigins()

    def folds(self, data: object) -> tuple[Fold, ...]:
        if isinstance(data, TimeSeriesFrame):
            raise DataError(
                "this validation evaluates at the forecast origins a ForecastDataset holds, "
                "and an event-time frame holds none; simulate them with "
                "of.RollingOrigin(horizon=..., windows=...)"
            )
        if not isinstance(data, ForecastDataset):
            raise DataError(
                f"a forecast-origin validation folds a ForecastDataset, got {type(data).__name__}"
            )
        return tuple(
            Fold(
                index=index,
                origin=origin,
                train=data.up_to(origin),
                context=data.at_origin(origin),
                truth=outcomes(data.truth, after=origin, horizon=self.horizon),
            )
            for index, origin in enumerate(self.origins.select(data.origins))
        )


#: Either strategy. Annotated with the discriminator so that a serialized
#: validation deserializes back into the same type it was written from.
Validation = Annotated[RollingOrigin | ForecastOriginValidation, Field(discriminator="mode")]


def outcomes(truth: TimeSeriesFrame, *, after: datetime, horizon: int) -> pa.Table:
    """What happened over the ``horizon`` steps following ``after``.

    Long rather than wide, in the shape a point forecast comes back in, so that
    scoring is a lookup rather than a reshape. Missing outcomes are kept as
    nulls: an event time whose value was never published is not an event time
    with a value of nothing, and dropping it here would hide how much of a fold
    was actually scored.
    """
    schema = truth.schema
    end = schema.frequency.shift(after, horizon)
    keys = key_rows(truth.history, schema.instance_keys)
    times: list[datetime] = column_values(truth.history, schema.time)
    values = {target: column_values(truth.history, target) for target in schema.targets}

    rows = [
        (key, moment, index)
        for index, (key, moment) in enumerate(zip(keys, times, strict=True))
        if after < moment <= end
    ]
    columns: dict[str, tuple[list[Any], pa.DataType]] = {}
    for position, name in enumerate(schema.instance_keys):
        columns[name] = (
            [key[position] for key, _, _ in rows for _ in schema.targets],
            column_type(truth.history, name),
        )
    columns[ForecastColumn.EVENT_TIME.value] = (
        [moment for _, moment, _ in rows for _ in schema.targets],
        column_type(truth.history, schema.time),
    )
    columns[ForecastColumn.TARGET.value] = (
        [target for _ in rows for target in schema.targets],
        pa.string(),
    )
    columns[ForecastColumn.VALUE.value] = (
        [values[target][index] for _, _, index in rows for target in schema.targets],
        pa.float64(),
    )
    return build_table(columns)


def truth_lookup(
    truth: pa.Table, instance_keys: Sequence[str]
) -> dict[tuple[InstanceKey, datetime, str], float | None]:
    """``(instance, event time, target) -> outcome``, for scoring against."""
    keys = key_rows(truth, instance_keys)
    times: list[datetime] = column_values(truth, ForecastColumn.EVENT_TIME.value)
    targets: list[str] = column_values(truth, ForecastColumn.TARGET.value)
    values: list[float | None] = column_values(truth, ForecastColumn.VALUE.value)
    return dict(zip(zip(keys, times, targets, strict=True), values, strict=True))
