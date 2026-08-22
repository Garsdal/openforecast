"""What to predict, and in what form.

```python
of.ForecastTask(horizon=24)

of.OutputSpec.point()
of.OutputSpec.quantiles([0.1, 0.5, 0.9])
of.OutputSpec.samples(100)
```

The task says how far ahead; the output spec says what kind of answer. They are
separate because they are checked against different things — the horizon against
the data and the artifact, the output kind against the model's declared
capabilities — and because asking for quantiles should not require restating the
horizon.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from openforecast.errors import RecipeError
from openforecast.models.capabilities import OutputCapabilities

__all__ = ["ForecastTask", "OutputKind", "OutputSpec"]


class ForecastTask(BaseModel):
    """How far ahead to forecast, from whatever origin is being asked about.

    In V1 a horizon is a count of steps of the data's frequency, not a duration:
    24 on hourly data is a day, and on daily data it is more than three weeks.
    The frequency is declared on the data, so restating it here would let the two
    disagree about what "24" meant.

    The origin is deliberately absent. At fit time the origins come from the
    plan, and at inference time the context *is* one origin — a task that also
    named one could contradict the data it was handed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    horizon: int = Field(ge=1)

    def __init__(self, horizon: int | None = None, /, **data: Any) -> None:
        """``ForecastTask(24)`` as well as ``ForecastTask(horizon=24)``."""
        if horizon is not None:
            if "horizon" in data:
                raise RecipeError("horizon was given both positionally and by keyword")
            data["horizon"] = horizon
        super().__init__(**data)


class OutputKind(StrEnum):
    #: One number per target and event time.
    POINT = "point"
    #: Named quantiles of the predictive distribution.
    QUANTILES = "quantiles"
    #: Draws from the predictive distribution, for the caller to reduce.
    SAMPLES = "samples"


class OutputSpec(BaseModel):
    """What kind of forecast to produce.

    The fields are ``levels`` and ``draws`` rather than ``quantiles`` and
    ``samples`` because the constructors own those two names — and the
    constructors are the interface:

    ```python
    of.OutputSpec.quantiles([0.1, 0.5, 0.9])   # spec.levels
    of.OutputSpec.samples(100)                 # spec.draws
    ```
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: OutputKind = OutputKind.POINT
    #: The quantile levels asked for, strictly between 0 and 1, ascending.
    levels: tuple[float, ...] = ()
    #: How many sample paths to draw.
    draws: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _check_kind_matches_fields(self) -> Self:
        if self.kind is not OutputKind.QUANTILES and self.levels:
            raise RecipeError(f"a {self.kind} forecast does not take quantile levels")
        if self.kind is not OutputKind.SAMPLES and self.draws is not None:
            raise RecipeError(f"a {self.kind} forecast does not take a draw count")
        if self.kind is OutputKind.QUANTILES and not self.levels:
            raise RecipeError("a quantile forecast must name the levels it wants")
        if self.kind is OutputKind.SAMPLES and self.draws is None:
            raise RecipeError("a sample forecast must say how many draws to take")
        outside = [level for level in self.levels if not 0.0 < level < 1.0]
        if outside:
            raise RecipeError(
                f"quantile levels lie strictly between 0 and 1: {sorted(outside)}; "
                f"0 and 1 name the bounds of the distribution, not quantiles of it"
            )
        if len(set(self.levels)) != len(self.levels):
            raise RecipeError(f"duplicate quantile levels: {sorted(self.levels)}")
        if list(self.levels) != sorted(self.levels):
            raise RecipeError(
                f"quantile levels must be ascending: {list(self.levels)}; "
                f"the forecast columns come back in this order"
            )
        return self

    @classmethod
    def point(cls) -> OutputSpec:
        """One number per target and event time."""
        return cls(kind=OutputKind.POINT)

    @classmethod
    def quantiles(cls, levels: Sequence[float]) -> OutputSpec:
        """``OutputSpec.quantiles([0.1, 0.5, 0.9])``."""
        return cls(kind=OutputKind.QUANTILES, levels=tuple(levels))

    @classmethod
    def samples(cls, draws: int) -> OutputSpec:
        """``OutputSpec.samples(100)``."""
        return cls(kind=OutputKind.SAMPLES, draws=draws)

    @property
    def is_probabilistic(self) -> bool:
        return self.kind is not OutputKind.POINT

    def is_supported_by(self, outputs: OutputCapabilities) -> bool:
        """Whether a model declaring ``outputs`` can answer this request.

        A quantile request is not silently downgraded to a point forecast, nor
        derived from samples the caller did not ask for: those are different
        answers, and picking one would hide that the model cannot give this one.
        """
        if self.kind is OutputKind.QUANTILES:
            return outputs.quantiles
        if self.kind is OutputKind.SAMPLES:
            return outputs.samples
        return outputs.point
