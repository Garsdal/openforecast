"""OpenForecast: the unified interface for forecasting.

The public surface grows only as the implementation does, so that no stub API
outlives the design it was guessing at. Today it is the semantic data layer —
``TimeSeriesFrame`` for ordinary event-time data, ``PointInTimeFrame`` and
``ForecastDataset`` for real forecast vintages, ``ForecastContext`` for one
inference origin, and the vocabulary needed to describe them — plus
``of.models``, where a model reference resolves to a descriptor.

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
    SchemaError,
    UnknownModelError,
)

__all__ = [
    "DataError",
    "DuplicateModelError",
    "FeatureAvailability",
    "FeatureKind",
    "FeatureSpec",
    "ForecastContext",
    "ForecastDataset",
    "Frequency",
    "FrequencyError",
    "FrequencyUnit",
    "InconsistentTruthError",
    "ModelError",
    "ModelRefError",
    "OpenForecastError",
    "OriginScopeError",
    "PointInTimeFrame",
    "PointInTimeSchema",
    "SchemaError",
    "TimeSeriesFrame",
    "TimeSeriesSchema",
    "UnknownModelError",
    "__version__",
    "models",
]

__version__ = "0.1.0"
