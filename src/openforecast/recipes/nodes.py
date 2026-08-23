"""What to fit: leaf models, pipelines, ensembles and reductions.

```python
of.Model("nixtla/nhits", params={"max_steps": 500})

of.Pipeline(steps=[
    of.StandardScaler(columns="targets"),
    of.Model("nixtla/nhits"),
])

of.Ensemble(
    models=[of.Model("nixtla/nhits"), of.Model("nixtla/autoarima")],
    combine=of.Mean(),
)

of.Reduction(estimator="lightgbm/regressor", strategy="direct", lags=[1, 24, 168])
```

A recipe is a tree, and every node is one of these four. Pipelines and ensembles
compose recipes rather than models, so an ensemble of pipelines needs no new
vocabulary. The four form a discriminated union on ``kind``, which is what lets a
recipe round-trip through JSON — into an artifact manifest in Step 7, over
provider RPC in Step 9, over HTTP in Step 16 — without a reader having to guess
what it is holding.

Nothing here mentions a provider. ``of.Model("nixtla/nhits")`` names a model, and
which subprocess ends up executing it is the registry's business.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from openforecast.errors import RecipeError
from openforecast.models.ref import ModelRef
from openforecast.recipes.base import RecipeKind, RecipeNode
from openforecast.recipes.transforms import (
    Impute,
    LeadTimeFeature,
    MissingIndicator,
    OriginCalendarFeatures,
    StandardScaler,
    Transform,
)

__all__ = [
    "Combiner",
    "CombinerKind",
    "Ensemble",
    "Mean",
    "Model",
    "Pipeline",
    "PipelineStep",
    "Recipe",
    "Reduction",
    "ReductionStrategy",
    "WeightedMean",
    "declared_transforms",
    "estimator_refs",
    "parse_recipe",
]

#: Provider parameters that name something OpenForecast owns. Passing one would
#: put the same number in two places, free to disagree, with the provider's
#: spelling winning silently.
_OWNED_PARAMETERS: dict[str, str] = {
    "input_size": "of.WindowPlan(context=...)",
    "input_chunk_length": "of.WindowPlan(context=...)",
    "context_length": "of.WindowPlan(context=...)",
    "window_length": "of.WindowPlan(context=...)",
    "output_chunk_length": "the horizon of of.ForecastTask",
    "h": "the horizon of of.ForecastTask",
    "horizon": "the horizon of of.ForecastTask",
    "fh": "the horizon of of.ForecastTask",
    "freq": "the frequency declared on the data",
    "frequency": "the frequency declared on the data",
    "random_state": "of.FitPlan(seed=...)",
    "random_seed": "of.FitPlan(seed=...)",
    "seed": "of.FitPlan(seed=...)",
    "hist_exog_list": "the observed features declared on the data",
    "futr_exog_list": "the known features declared on the data",
    "stat_exog_list": "the static features declared on the data",
    "past_covariates": "the observed features declared on the data",
    "future_covariates": "the known features declared on the data",
}


class Model(RecipeNode):
    """One model, by reference, with the parameters only its provider understands.

    ```python
    of.Model("nixtla/nhits", params={"max_steps": 500})
    ```

    ``params`` is opaque to OpenForecast and reaches the provider unchanged —
    which is exactly why it may not carry anything OpenForecast owns. A context
    length passed as ``input_size`` would be a second copy of
    ``WindowPlan(context=...)``, invisible to the planner that has to materialize
    the samples, so it is rejected with the field to use instead.
    """

    kind: Literal[RecipeKind.MODEL] = RecipeKind.MODEL
    #: Accepts a plain ``"nixtla/nhits"``; stored parsed.
    ref: ModelRef
    params: dict[str, Any] = {}

    def __init__(self, ref: ModelRef | str | None = None, /, **data: Any) -> None:
        """``of.Model("nixtla/nhits")`` as well as ``Model(ref=...)``."""
        if ref is not None:
            if "ref" in data:
                raise RecipeError("ref was given both positionally and by keyword")
            data["ref"] = ref
        super().__init__(**data)

    @model_validator(mode="after")
    def _check_params(self) -> Self:
        owned = sorted(set(self.params) & set(_OWNED_PARAMETERS))
        if owned:
            replacements = sorted({_OWNED_PARAMETERS[name] for name in owned})
            raise RecipeError(
                f"{owned} name concepts OpenForecast owns, so they cannot be passed as "
                f"provider parameters of {self.ref}; use {replacements} instead"
            )
        try:
            json.dumps(self.params)
        except (TypeError, ValueError) as error:
            raise RecipeError(
                f"the parameters of {self.ref} must survive serialization: a recipe is "
                f"recorded in the artifact manifest and sent to a provider as JSON ({error})"
            ) from error
        return self

    def estimator_refs(self) -> tuple[ModelRef, ...]:
        return (self.ref,)


class CombinerKind(StrEnum):
    MEAN = "mean"
    WEIGHTED_MEAN = "weighted_mean"


class Mean(BaseModel):
    """Average the members equally."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    combine: Literal[CombinerKind.MEAN] = CombinerKind.MEAN


class WeightedMean(BaseModel):
    """Average the members by given weights.

    ```python
    of.WeightedMean(weights=[0.7, 0.3])
    ```

    The weights are relative: they are normalized by their sum, so ``[7, 3]``
    means the same thing as ``[0.7, 0.3]``. Every weight must be positive — a
    zero weight is a member that is fitted and then ignored, which is better
    said by leaving it out.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    combine: Literal[CombinerKind.WEIGHTED_MEAN] = CombinerKind.WEIGHTED_MEAN
    weights: tuple[float, ...]

    @model_validator(mode="after")
    def _check_weights(self) -> Self:
        if not self.weights:
            raise RecipeError("a weighted mean needs one weight per ensemble member")
        if any(weight <= 0 for weight in self.weights):
            raise RecipeError(
                f"ensemble weights must be positive: {list(self.weights)}; a zero-weighted "
                f"member is fitted and then ignored, so leave it out instead"
            )
        return self

    @property
    def normalized(self) -> tuple[float, ...]:
        """The weights as fractions summing to one."""
        total = sum(self.weights)
        return tuple(weight / total for weight in self.weights)


#: How an ensemble combines its members.
Combiner = Annotated[Mean | WeightedMean, Field(discriminator="combine")]

#: The steps that are not estimators, for telling the two apart in a pipeline.
_TRANSFORMS = (StandardScaler, MissingIndicator, Impute, LeadTimeFeature, OriginCalendarFeatures)


class ReductionStrategy(StrEnum):
    """How a horizon is turned into supervised regression problems."""

    #: One model, applied repeatedly, feeding its own predictions back in.
    RECURSIVE = "recursive"
    #: One model per horizon step, each predicting that step directly.
    DIRECT = "direct"
    #: One model predicting every horizon step at once.
    MULTIOUTPUT = "multioutput"


class Reduction(RecipeNode):
    """Forecasting reduced to tabular regression.

    ```python
    of.Reduction(estimator="lightgbm/regressor", strategy="direct", lags=[1, 24, 168])
    ```

    This is the recipe behind the point-in-time LightGBM setups that motivated
    the whole design: the ``ViewPlanner`` materializes a ``TabularView`` holding
    one row per instance, origin and horizon step, and the estimator never learns
    that some of those rows came from different vintages of the same event time.

    The protocol is defined here; execution arrives in Step 14. A recipe that
    cannot be executed yet is still worth being able to write down and store,
    which is why constructing one is not an error.
    """

    kind: Literal[RecipeKind.REDUCTION] = RecipeKind.REDUCTION
    #: The tabular regressor to reduce onto, e.g. ``"lightgbm/regressor"``.
    estimator: ModelRef
    strategy: ReductionStrategy
    #: Which past values become features, in steps of the data's frequency.
    lags: tuple[int, ...]

    @model_validator(mode="after")
    def _check_lags(self) -> Self:
        if not self.lags:
            raise RecipeError(
                "a reduction needs at least one lag; without one there is nothing for "
                "the regressor to condition on"
            )
        if any(lag < 1 for lag in self.lags):
            raise RecipeError(
                f"lags count steps into the past and are therefore positive: {list(self.lags)}; "
                f"a lag of 0 is the value being predicted and a negative one is the future"
            )
        if len(set(self.lags)) != len(self.lags):
            raise RecipeError(f"duplicate lags: {sorted(self.lags)}")
        if list(self.lags) != sorted(self.lags):
            raise RecipeError(
                f"lags must be ascending: {list(self.lags)}; two spellings of one lag set "
                f"would be two recipes that mean the same thing"
            )
        return self

    def estimator_refs(self) -> tuple[ModelRef, ...]:
        return (self.estimator,)


class Ensemble(RecipeNode):
    """Several recipes, combined into one forecast.

    Members are recipes rather than models, so an ensemble of pipelines — or of
    ensembles — needs no extra vocabulary. Whether a given combination can
    actually be fitted on the data at hand is a question about the members'
    training contracts, and the engine answers it in Step 8; a member's contract
    is not knowable here, where nothing has been resolved yet.
    """

    kind: Literal[RecipeKind.ENSEMBLE] = RecipeKind.ENSEMBLE
    models: tuple[Recipe, ...]
    combine: Combiner = Mean()

    @model_validator(mode="after")
    def _check_members(self) -> Self:
        if len(self.models) < 2:
            raise RecipeError(
                f"an ensemble combines at least two members, got {len(self.models)}; "
                f"one member is that member"
            )
        if isinstance(self.combine, WeightedMean) and len(self.combine.weights) != len(self.models):
            raise RecipeError(
                f"a weighted mean needs one weight per member: {len(self.models)} members, "
                f"{len(self.combine.weights)} weights"
            )
        return self

    def estimator_refs(self) -> tuple[ModelRef, ...]:
        return tuple(ref for member in self.models for ref in estimator_refs(member))


class Pipeline(RecipeNode):
    """Transforms, then exactly one estimator, in order.

    ```python
    of.Pipeline(steps=[
        of.MissingIndicator(columns="features"),
        of.Impute(columns="features", method="median"),
        of.Model("nixtla/nhits"),
    ])
    ```

    The estimator has to come last, and there has to be exactly one: a pipeline
    is a recipe for a forecast, and steps after the thing that forecasts have
    nothing to act on. Transform order is otherwise the caller's, with one
    exception — an imputation before a missing indicator over the same columns
    would make the indicator constant, silently discarding the very fact it was
    added to record.

    Pipelines do not nest. Flattening one would change nothing about what it
    means, so a nested one is a recipe written two ways.
    """

    kind: Literal[RecipeKind.PIPELINE] = RecipeKind.PIPELINE
    steps: tuple[PipelineStep, ...]

    @model_validator(mode="after")
    def _check_steps(self) -> Self:
        estimators = [step for step in self.steps if not isinstance(step, _TRANSFORMS)]
        if not estimators:
            raise RecipeError(
                "a pipeline must end in something that forecasts; add a model, an "
                "ensemble or a reduction as its last step"
            )
        if len(estimators) > 1:
            raise RecipeError(
                f"a pipeline holds one estimator, got {len(estimators)}; combine several "
                f"with of.Ensemble and put the ensemble in the pipeline"
            )
        if isinstance(self.steps[-1], _TRANSFORMS):
            raise RecipeError(
                "a pipeline's estimator must be its last step; a transform after it has "
                "nothing left to transform"
            )
        _check_indicator_order(self.steps)
        return self

    @property
    def estimator(self) -> Recipe:
        """The step that actually forecasts — always the last one."""
        last = self.steps[-1]
        if isinstance(last, _TRANSFORMS):  # pragma: no cover - the validator forbids it
            raise RecipeError("a pipeline's estimator must be its last step")
        return last

    @property
    def transforms(self) -> tuple[Transform, ...]:
        return tuple(step for step in self.steps if isinstance(step, _TRANSFORMS))

    def estimator_refs(self) -> tuple[ModelRef, ...]:
        return estimator_refs(self.estimator)


#: A whole recipe: what ``of.fit(model=...)`` accepts.
Recipe = Annotated[Model | Pipeline | Ensemble | Reduction, Field(discriminator="kind")]

#: A step of a pipeline: any transform, or the one estimator it ends with.
PipelineStep = Annotated[
    StandardScaler
    | MissingIndicator
    | Impute
    | LeadTimeFeature
    | OriginCalendarFeatures
    | Model
    | Ensemble
    | Reduction,
    Field(discriminator="kind"),
]

# The recipe union is recursive — an ensemble holds recipes, a pipeline holds an
# estimator — so the two composites are completed once the aliases above exist.
Ensemble.model_rebuild()
Pipeline.model_rebuild()

_RECIPE_ADAPTER: TypeAdapter[Recipe] = TypeAdapter(Recipe)


def parse_recipe(value: object) -> Recipe:
    """Read a recipe back from what ``recipe.model_dump()`` produced.

    The entry point for every place a recipe crosses a boundary: an artifact
    manifest, a provider request, an HTTP body. Nothing about it is
    provider-specific, which is the point — a recipe written by one client is
    readable by any other.
    """
    return _RECIPE_ADAPTER.validate_python(value)


def estimator_refs(recipe: Recipe) -> tuple[ModelRef, ...]:
    """Every model reference the recipe names, in order, duplicates kept.

    What the engine resolves before it can plan: each reference has a descriptor,
    and the descriptors are what say whether the recipe can be fitted on the data
    it was handed.
    """
    return recipe.estimator_refs()


def declared_transforms(recipe: Recipe) -> tuple[Transform, ...]:
    """Every transform anywhere in the recipe, in the order it would be applied.

    What the artifact manifest records so that reading it answers "was anything
    done to this data before the model saw it" without walking the tree again —
    an imputation in particular, since a manifest that cannot say a missing value
    was filled describes a model that is not the one that was fitted.
    """
    if isinstance(recipe, Pipeline):
        return (*recipe.transforms, *declared_transforms(recipe.estimator))
    if isinstance(recipe, Ensemble):
        return tuple(
            transform for member in recipe.models for transform in declared_transforms(member)
        )
    return ()


def _check_indicator_order(steps: Sequence[PipelineStep]) -> None:
    imputed: list[Impute] = []
    for step in steps:
        if isinstance(step, Impute):
            imputed.append(step)
        elif isinstance(step, MissingIndicator):
            culprits = [transform for transform in imputed if transform.may_overlap(step)]
            if culprits:
                raise RecipeError(
                    "a missing indicator after an imputation of the same columns would be "
                    "constant, discarding the missingness it was added to record; put the "
                    "indicator first"
                )
