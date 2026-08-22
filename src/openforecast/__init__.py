"""OpenForecast: the unified interface for forecasting.

The public surface grows only as the implementation does, so that no stub API
outlives the design it was guessing at. Today it is the semantic data layer —
``TimeSeriesFrame`` for ordinary event-time data, ``PointInTimeFrame`` and
``ForecastDataset`` for real forecast vintages, ``ForecastContext`` for one
inference origin, and the vocabulary needed to describe them — ``of.models``,
where a model reference resolves to a descriptor, the recipes and plans that say
what to fit and how, and ``of.fit`` and ``of.forecast``, which do it:

```python
model = of.fit(
    model=of.Pipeline(steps=[
        of.StandardScaler(columns="targets"),
        of.Model("builtin/seasonal-naive", params={"season_length": 24}),
    ]),
    data=dataset,
    plan=of.FitPlan(origins=of.LatestOrigin()),
    name="de-price",
)

forecast = of.forecast(model=model, data=context, horizon=24)
```

Every part of that is provider-independent: a context length is stated once and
compiled into whatever the provider calls it, and the origin selection means the
same thing on ordinary event-time data as on real forecast vintages.

What the fit returns is a reference to an immutable artifact —
``local/de-price@01K...`` — which is the string a forecast then takes. The
machinery behind it is not user vocabulary: manifests, staging directories and
aliases live in :mod:`openforecast.artifacts`, and only the errors a caller can
act on are exported here.

The execution views of Step 4 are deliberately not re-exported here either: they
are a provider-facing boundary, imported from :mod:`openforecast.views`, not
something a user of the library needs to name.
"""

from openforecast import models
from openforecast.client import OpenForecast, fit, forecast
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
    ArtifactError,
    DataError,
    DuplicateModelError,
    FrequencyError,
    IncompatibleForecastTask,
    InconsistentTruthError,
    ModelError,
    ModelRefError,
    ModelRequiresFit,
    OpenForecastError,
    OriginScopeError,
    ProviderError,
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
from openforecast.runtime import Forecast
from openforecast.runtime.providers import install_default_providers as _install_providers
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
    "ArtifactError",
    "AtOrigin",
    "ColumnSet",
    "DataError",
    "DuplicateModelError",
    "Ensemble",
    "FeatureAvailability",
    "FeatureKind",
    "FeatureSpec",
    "FitPlan",
    "Forecast",
    "ForecastContext",
    "ForecastDataset",
    "ForecastTask",
    "Frequency",
    "FrequencyError",
    "FrequencyUnit",
    "Impute",
    "ImputeMethod",
    "IncompatibleForecastTask",
    "InconsistentTruthError",
    "LatestOrigin",
    "LeadTimeFeature",
    "Mean",
    "MissingIndicator",
    "Model",
    "ModelError",
    "ModelRefError",
    "ModelRequiresFit",
    "OpenForecast",
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
    "ProviderError",
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
    "fit",
    "forecast",
    "models",
    "parse_recipe",
]

__version__ = "0.1.0"

# The models this build ships with become discoverable when the package is
# imported, exactly as an external provider's will when it answers a handshake.
# Nothing here touches the filesystem: an artifact store is named only when
# something is actually fitted.
_install_providers()
