"""OpenForecast: the unified interface for forecasting.

The public surface grows only as the implementation does, so that no stub API
outlives the design it was guessing at. Today it is the semantic data layer:
``TimeSeriesFrame`` for ordinary event-time data, ``PointInTimeFrame`` and
``ForecastDataset`` for real forecast vintages, ``ForecastContext`` for one
inference origin, and the vocabulary needed to describe them.
"""

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
    FrequencyError,
    InconsistentTruthError,
    OpenForecastError,
    SchemaError,
)

__all__ = [
    "DataError",
    "FeatureAvailability",
    "FeatureKind",
    "FeatureSpec",
    "ForecastContext",
    "ForecastDataset",
    "Frequency",
    "FrequencyError",
    "FrequencyUnit",
    "InconsistentTruthError",
    "OpenForecastError",
    "PointInTimeFrame",
    "PointInTimeSchema",
    "SchemaError",
    "TimeSeriesFrame",
    "TimeSeriesSchema",
    "__version__",
]

__version__ = "0.1.0"
