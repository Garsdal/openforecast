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

Since Step 16, where that happens is a client's transport rather than a fact
about the library:

```python
client = of.OpenForecast(transport=of.LocalTransport())
client = of.OpenForecast(transport=of.HttpTransport("http://localhost:8321"))
```

Both answer ``client.models.list()``, ``client.models.get(...)``,
``client.fit(...)`` and ``client.forecast(...)`` with the same objects, because
HTTP is a projection of what forecasting means here rather than a second
architecture. The request and response models it projects to live in
:mod:`openforecast.server`; the service that answers them is
``openforecast serve``, behind the ``openforecast[server]`` extra.

Since Step 17 the same vocabulary also compares models rather than only running
them:

```python
result = of.backtest(
    models=["builtin/seasonal-naive", "nixtla/autoarima", "nixtla/nhits"],
    data=dataset,
    validation=of.ForecastOriginValidation(origins=of.AllOrigins(stride=24), horizon=72),
    metrics=[of.MAE(), of.Bias()],
)

result.leaderboard("mae")
```

Which is a loop over ``of.fit`` and ``of.forecast`` and nothing else — no
provider knows it is being backtested, and for a ``ForecastDataset`` each origin
is evaluated on the vintage that actually existed. That is what
:mod:`openforecast.evaluation` is, along with ``of.eligible_models``, which
answers which models this data could fit at all.

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
from openforecast.evaluation import (
    MAE,
    MAPE,
    RMSE,
    BacktestResult,
    Bias,
    Candidate,
    Eligibility,
    ForecastOriginValidation,
    Metric,
    RollingOrigin,
    Validation,
    backtest,
    eligible_models,
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
from openforecast.server import HttpTransport, LocalTransport, Transport
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
    "BacktestResult",
    "Bias",
    "Candidate",
    "ColumnSet",
    "DataError",
    "DuplicateModelError",
    "Eligibility",
    "Ensemble",
    "FeatureAvailability",
    "FeatureKind",
    "FeatureSpec",
    "FitPlan",
    "Forecast",
    "ForecastContext",
    "ForecastDataset",
    "ForecastOriginValidation",
    "ForecastTask",
    "Frequency",
    "FrequencyError",
    "FrequencyUnit",
    "HttpTransport",
    "Impute",
    "ImputeMethod",
    "IncompatibleForecastTask",
    "InconsistentTruthError",
    "LatestOrigin",
    "LeadTimeFeature",
    "LocalTransport",
    "MAE",
    "MAPE",
    "Mean",
    "Metric",
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
    "RMSE",
    "Recipe",
    "RecipeError",
    "Reduction",
    "ReductionStrategy",
    "Resources",
    "RollingOrigin",
    "SchemaError",
    "StandardScaler",
    "TimeSeriesFrame",
    "TimeSeriesSchema",
    "Transport",
    "UnknownModelError",
    "UnsupportedPlanError",
    "Validation",
    "WeightedMean",
    "WindowPlan",
    "__version__",
    "backtest",
    "eligible_models",
    "fit",
    "forecast",
    "models",
    "parse_recipe",
]

__version__ = "0.1.0"

# Every model this build can execute becomes discoverable when the package is
# imported: the ones it ships with, and the ones each installed provider
# environment recorded when it answered its handshake. That reads the provider
# cache directory and nothing else — no provider process is started, and an
# artifact store is named only when something is actually fitted.
_install_providers()
