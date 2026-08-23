"""``FitPlan``: everything about a fit that is not the model and not the data.

```python
plan = of.FitPlan(
    origins=of.AllOrigins(),
    window=of.WindowPlan(context=168),
    seed=42,
    resources=of.Resources(accelerator="auto"),
)
```

The window is the reason this exists. A context length is an OpenForecast
concept — how much history one training sample conditions on — and every
library spells it differently:

```text
WindowPlan(context=168)  ->  Nixtla input_size=168
                         ->  Darts input_chunk_length=168
```

Those are compilation targets, so the caller states it once, here, and never in
provider parameters. It keeps the word *window* because it sizes the window of a
sample; the materialized result is a ``SequenceView``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from openforecast.errors import RecipeError, UnsupportedPlanError
from openforecast.tasks.origins import AllOrigins, OriginSelection

__all__ = [
    "Accelerator",
    "FitPlan",
    "Resources",
    "SearchPlan",
    "SearchStrategy",
    "WindowPlan",
]


class WindowPlan(BaseModel):
    """How much history one training sample conditions on.

    Steps of the data's frequency, not a duration: 168 on hourly data is a week,
    and the same number on daily data is half a year. The frequency lives on the
    data, so saying it again here would let the two disagree.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    context: int = Field(ge=1)


class Accelerator(StrEnum):
    #: Let the provider pick whatever it finds. The only defensible default:
    #: OpenForecast cannot know what hardware a provider's environment sees.
    AUTO = "auto"
    CPU = "cpu"
    GPU = "gpu"


class Resources(BaseModel):
    """What hardware a fit may use.

    Deliberately thin. A provider knows how to talk to its own accelerators; the
    point of naming them here is that the *request* is OpenForecast's, so two
    providers do not need two different spellings of "use the GPU".
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    accelerator: Accelerator = Accelerator.AUTO
    #: How many devices to use. ``None`` means the provider decides.
    devices: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _check_devices(self) -> Self:
        if self.accelerator is Accelerator.CPU and self.devices is not None:
            raise RecipeError(
                "a cpu fit has no devices to allocate; drop devices, or ask for an "
                "accelerator that has them"
            )
        return self


class SearchStrategy(StrEnum):
    GRID = "grid"
    RANDOM = "random"


class SearchPlan(BaseModel):
    """Hyperparameter search — reserved, not yet executable.

    The shape is fixed now so that the plan that reaches a provider does not
    have to change when search lands. Attaching one to a ``FitPlan`` raises
    :class:`~openforecast.errors.UnsupportedPlanError` today, because the
    alternative — accepting it and fitting once with the first candidate — would
    look exactly like a search that ran.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: SearchStrategy
    #: Parameter name -> the candidate values to try.
    space: dict[str, list[Any]]
    #: How many candidates to draw. Required for a random search, meaningless
    #: for a grid, which is exhaustive by definition.
    trials: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _check_strategy(self) -> Self:
        if not self.space:
            raise RecipeError("a search plan must name at least one parameter to search over")
        empty = sorted(name for name, values in self.space.items() if not values)
        if empty:
            raise RecipeError(f"search parameters with no candidate values: {empty}")
        if self.strategy is SearchStrategy.RANDOM and self.trials is None:
            raise RecipeError("a random search must say how many trials to draw")
        if self.strategy is SearchStrategy.GRID and self.trials is not None:
            raise RecipeError("a grid search is exhaustive, so it does not take a trial count")
        return self


class FitPlan(BaseModel):
    """How to fit: which origins, how much context, how reproducibly, on what.

    Not *what* to fit — that is the recipe — and not *what to predict*, which is
    the forecast task. Keeping the three apart is what lets the same recipe be
    fitted at one origin and at every origin without being rewritten.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Every origin by default. On event-time data that means every origin the
    #: history can simulate; on vintages, every vintage that exists.
    origins: OriginSelection = AllOrigins()
    #: Required by models that learn from context -> horizon sequences, unused
    #: by the others. Which of those a model is, is its contract's business.
    window: WindowPlan | None = None
    #: Seeds whatever the provider seeds. Recorded in the artifact manifest, so
    #: an unseeded fit is honest about being unreproducible rather than pretending.
    seed: int | None = Field(default=None, ge=0)
    resources: Resources = Resources()
    #: Reserved. See :class:`SearchPlan`.
    search: SearchPlan | None = None

    @model_validator(mode="after")
    def _check_support(self) -> Self:
        if self.search is not None:
            raise UnsupportedPlanError(
                "hyperparameter search is not implemented yet; the field is reserved so "
                "that the protocol does not change when it lands. Fit the configurations "
                "you want to compare and use of.backtest to rank them."
            )
        return self

    @property
    def context(self) -> int | None:
        """The context length this plan sizes samples with, if it sizes any."""
        return None if self.window is None else self.window.context
