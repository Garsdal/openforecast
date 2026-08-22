"""The model-construction intermediate representation.

A recipe says what to fit, in OpenForecast's own vocabulary:

```python
of.Model("nixtla/nhits", params={"max_steps": 500})

of.Pipeline(steps=[
    of.MissingIndicator(columns="features"),
    of.Impute(columns="features", method="median"),
    of.Model("nixtla/nhits"),
])

of.Ensemble(
    models=[of.Model("nixtla/nhits"), of.Model("nixtla/autoarima")],
    combine=of.WeightedMean(weights=[0.7, 0.3]),
)

of.Reduction(estimator="lightgbm/regressor", strategy="direct", lags=[1, 24, 168])
```

Two rules shape all of it.

**OpenForecast owns what OpenForecast can own.** A context length, a horizon, a
seed and the feature roles are recipe and plan fields, never provider
parameters — a `params={"input_size": 168}` would be a second copy of
``WindowPlan(context=168)``, invisible to the planner that has to cut the
samples. Passing one is an error that names the field to use instead.

**Nothing is imputed silently.** Missing values in point-in-time data are
information: a feature that had not been published yet. A model that cannot
consume them declares so, and the caller writes ``MissingIndicator`` and
``Impute`` down as steps — recorded in the artifact, visible to whoever reads
the forecast.

Recipes are a serializable AST, discriminated on ``kind``, and
:func:`parse_recipe` reads one back. The same JSON is what reaches an artifact
manifest in Step 7, a provider subprocess in Step 9 and an HTTP body in Step 16.
Execution of these nodes arrives with the engine in Step 8; being able to write
one down, store it and read it back is what Step 6 delivers.
"""

from openforecast.recipes.base import ColumnSelector, ColumnSet, ColumnTransform, RecipeKind
from openforecast.recipes.nodes import (
    Combiner,
    CombinerKind,
    Ensemble,
    Mean,
    Model,
    Pipeline,
    PipelineStep,
    Recipe,
    Reduction,
    ReductionStrategy,
    WeightedMean,
    estimator_refs,
    parse_recipe,
)
from openforecast.recipes.transforms import (
    Impute,
    ImputeMethod,
    LeadTimeFeature,
    MissingIndicator,
    OriginCalendarFeatures,
    StandardScaler,
    Transform,
)

__all__ = [
    "ColumnSelector",
    "ColumnSet",
    "ColumnTransform",
    "Combiner",
    "CombinerKind",
    "Ensemble",
    "Impute",
    "ImputeMethod",
    "LeadTimeFeature",
    "Mean",
    "MissingIndicator",
    "Model",
    "OriginCalendarFeatures",
    "Pipeline",
    "PipelineStep",
    "Recipe",
    "RecipeKind",
    "Reduction",
    "ReductionStrategy",
    "StandardScaler",
    "Transform",
    "WeightedMean",
    "estimator_refs",
    "parse_recipe",
]
