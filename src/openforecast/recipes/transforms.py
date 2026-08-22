"""Transforms a recipe can put in front of a model.

Two of them exist because of point-in-time data:

```python
of.LeadTimeFeature(name="lead_hours", unit="hour")
of.OriginCalendarFeatures(hour=True, weekday=True)
```

Lead time is derived from the two time axes rather than stored, so asking for it
as a feature is asking OpenForecast to materialize it — a provider never sees an
``origin_time`` column to compute it from.

The other three are how missing values get handled *explicitly*:

```python
of.Pipeline(steps=[
    of.MissingIndicator(columns="features"),
    of.Impute(columns="features", method="median"),
    of.Model("nixtla/nhits"),
])
```

Nothing in OpenForecast imputes on its own. A model that cannot consume missing
values declares ``MissingValueSupport.REQUIRES_TRANSFORM``, and the caller writes
the transform down — where it is recorded in the artifact manifest and visible
to whoever reads the forecast later. In point-in-time data a missing value is
information: it says a feature had not been published at that origin.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import model_validator

from openforecast.data.frequency import FrequencyUnit
from openforecast.errors import RecipeError
from openforecast.recipes.base import ColumnTransform, RecipeKind, RecipeNode

__all__ = [
    "Impute",
    "ImputeMethod",
    "LeadTimeFeature",
    "MissingIndicator",
    "OriginCalendarFeatures",
    "StandardScaler",
    "Transform",
]


class StandardScaler(ColumnTransform):
    """Center and scale columns, undoing it on the way back out.

    ``per_instance`` scales every series by its own history, which is what makes
    a panel of a 40 GW zone and a 2 GW zone learnable by one global model. Turn
    it off when the levels are comparable and the differences between them are
    the signal.
    """

    kind: Literal[RecipeKind.STANDARD_SCALER] = RecipeKind.STANDARD_SCALER
    per_instance: bool = True


class MissingIndicator(ColumnTransform):
    """Add a boolean column recording where a value was missing.

    Put it *before* an imputation, never after: an indicator computed on imputed
    data is constant, and the fact that a feature was not published at an origin
    is exactly what would have been lost.
    """

    kind: Literal[RecipeKind.MISSING_INDICATOR] = RecipeKind.MISSING_INDICATOR
    suffix: str = "_is_missing"

    @model_validator(mode="after")
    def _check_suffix(self) -> Self:
        if not self.suffix.strip():
            raise RecipeError(
                "a missing indicator needs a non-empty suffix; without one the new "
                "column would overwrite the column it describes"
            )
        return self


class ImputeMethod(StrEnum):
    MEAN = "mean"
    MEDIAN = "median"
    ZERO = "zero"


class Impute(ColumnTransform):
    """Fill missing values with a stated statistic.

    ``method`` has no default on purpose. Which fill is right depends on what
    the column means, and a default would be OpenForecast quietly choosing for
    every dataset that forgot to say.

    The statistic is computed on the fitted data and recorded in the artifact,
    so inference fills the same way training did rather than leaking the context
    it happens to be given.
    """

    kind: Literal[RecipeKind.IMPUTE] = RecipeKind.IMPUTE
    method: ImputeMethod


class LeadTimeFeature(RecipeNode):
    """``event_time - origin_time``, as a feature the model can condition on.

    Genuinely useful on point-in-time data: a wind forecast issued six hours
    ahead is not the same quality of information as one issued sixty, and a
    model given the lead time can learn that. Derived rather than stored, which
    is why asking for it is a recipe step instead of a column in the source data.
    """

    kind: Literal[RecipeKind.LEAD_TIME_FEATURE] = RecipeKind.LEAD_TIME_FEATURE
    name: str = "lead_time"
    unit: FrequencyUnit = FrequencyUnit.HOUR

    @model_validator(mode="after")
    def _check_name(self) -> Self:
        if not self.name.strip():
            raise RecipeError("a lead-time feature needs a column name")
        return self


class OriginCalendarFeatures(RecipeNode):
    """Calendar features of the *origin*, not of the event time.

    The distinction matters on real vintages: a forecast issued at 06:00 for
    tomorrow noon is built from a different information set than one issued at
    18:00 for the same noon, and when a model is learning across origins that is
    a systematic effect rather than noise.
    """

    kind: Literal[RecipeKind.ORIGIN_CALENDAR_FEATURES] = RecipeKind.ORIGIN_CALENDAR_FEATURES
    hour: bool = False
    weekday: bool = False
    month: bool = False
    #: Prefixed so the new columns cannot collide with the caller's own.
    prefix: str = "origin"

    @model_validator(mode="after")
    def _check_something_is_requested(self) -> Self:
        if not (self.hour or self.weekday or self.month):
            raise RecipeError(
                "origin calendar features must request at least one of hour, weekday or month"
            )
        if not self.prefix.strip():
            raise RecipeError("origin calendar features need a non-empty column prefix")
        return self

    @property
    def columns(self) -> tuple[str, ...]:
        """The column names this step adds."""
        requested = (("hour", self.hour), ("weekday", self.weekday), ("month", self.month))
        return tuple(f"{self.prefix}_{name}" for name, wanted in requested if wanted)


#: Every transform. Not a discriminated union of its own: a pipeline step can
#: also be an estimator, so the union that admits both lives with the estimators.
Transform = StandardScaler | MissingIndicator | Impute | LeadTimeFeature | OriginCalendarFeatures
