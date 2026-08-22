"""OpenForecast: the unified interface for forecasting.

The public surface grows only as the implementation does, so that no stub API
outlives the design it was guessing at. Today it is the semantic data layer —
``TimeSeriesFrame`` for ordinary event-time data, ``PointInTimeFrame`` and
``ForecastDataset`` for real forecast vintages, ``ForecastContext`` for one
inference origin, and the vocabulary needed to describe them — ``of.models``,
where a model reference resolves to a descriptor, and the recipes and plans that
say what to fit and how:

```python
of.fit(                                     # Step 8
    model=of.Pipeline(steps=[
        of.StandardScaler(columns="targets"),
        of.Model("nixtla/nhits", params={"max_steps": 500}),
    ]),
    data=dataset,
    horizon=24,
    plan=of.FitPlan(
        origins=of.AllOrigins(),
        window=of.WindowPlan(context=168),
    ),
)
```

Everything in that call except ``of.fit`` itself exists today, and every part of
it is provider-independent: a context length is stated once and compiled into
whatever the provider calls it, and the origin selection means the same thing on
ordinary event-time data as on real forecast vintages.

The execution views of Step 4 are deliberately not re-exported here: they are a
provider-facing boundary, imported from :mod:`openforecast.views`, not something
a user of the library needs to name.
"""

from openforecast import models
from openforecast.data import (
    FeatureAvailability,
    FeatureKind,
    FeatureSpec,
    ForecastContext,
    ForecastDataset,
    Frequency,
    FrequencyUnit,
    PointInTimeFrame,
    PointInTimeSchema,
    TimeSeriesFrame,
    TimeSeriesSchema,
)
from openforecast.errors import (
    DataError,
    DuplicateModelError,
    FrequencyError,
    InconsistentTruthError,
    ModelError,
    ModelRefError,
    OpenForecastError,
    OriginScopeError,
    RecipeError,
    SchemaError,
    UnknownModelError,
    UnsupportedPlanError,
)
from openforecast.recipes import (
    ColumnSet,
    Ensemble,
    Impute,
    ImputeMethod,
    LeadTimeFeature,
    Mean,
    MissingIndicator,
    Model,
    OriginCalendarFeatures,
    Pipeline,
    Recipe,
    Reduction,
    ReductionStrategy,
    StandardScaler,
    WeightedMean,
    parse_recipe,
)
from openforecast.tasks import (
    Accelerator,
    AllOrigins,
    AtOrigin,
    FitPlan,
    ForecastTask,
    LatestOrigin,
    OriginsBetween,
    OriginSelection,
    OutputKind,
    OutputSpec,
    Resources,
    WindowPlan,
)

__all__ = [
    "Accelerator",
    "AllOrigins",
    "AtOrigin",
    "ColumnSet",
    "DataError",
    "DuplicateModelError",
    "Ensemble",
    "FeatureAvailability",
    "FeatureKind",
    "FeatureSpec",
    "FitPlan",
    "ForecastContext",
    "ForecastDataset",
    "ForecastTask",
    "Frequency",
    "FrequencyError",
    "FrequencyUnit",
    "Impute",
    "ImputeMethod",
    "InconsistentTruthError",
    "LatestOrigin",
    "LeadTimeFeature",
    "Mean",
    "MissingIndicator",
    "Model",
    "ModelError",
    "ModelRefError",
    "OpenForecastError",
    "OriginCalendarFeatures",
    "OriginScopeError",
    "OriginSelection",
    "OriginsBetween",
    "OutputKind",
    "OutputSpec",
    "Pipeline",
    "PointInTimeFrame",
    "PointInTimeSchema",
    "Recipe",
    "RecipeError",
    "Reduction",
    "ReductionStrategy",
    "Resources",
    "SchemaError",
    "StandardScaler",
    "TimeSeriesFrame",
    "TimeSeriesSchema",
    "UnknownModelError",
    "UnsupportedPlanError",
    "WeightedMean",
    "WindowPlan",
    "__version__",
    "models",
    "parse_recipe",
]

__version__ = "0.1.0"
