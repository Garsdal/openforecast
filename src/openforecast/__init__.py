"""OpenForecast: the unified interface for forecasting.

The public surface grows only as the implementation does, so that no stub API
outlives the design it was guessing at. Today it is the event-time semantic
model: ``TimeSeriesFrame`` and the vocabulary needed to describe one.
"""

from openforecast.data import (
    FeatureAvailability,
    FeatureKind,
    FeatureSpec,
    Frequency,
    FrequencyUnit,
    TimeSeriesFrame,
    TimeSeriesSchema,
)
from openforecast.errors import DataError, FrequencyError, OpenForecastError, SchemaError

__all__ = [
    "DataError",
    "FeatureAvailability",
    "FeatureKind",
    "FeatureSpec",
    "Frequency",
    "FrequencyError",
    "FrequencyUnit",
    "OpenForecastError",
    "SchemaError",
    "TimeSeriesFrame",
    "TimeSeriesSchema",
    "__version__",
]

__version__ = "0.1.0"
