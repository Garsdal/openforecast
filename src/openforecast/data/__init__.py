"""Semantic source datasets.

Event-time primitives (``Frequency``, ``TimeSeriesSchema``, ``TimeSeriesFrame``)
live here; the point-in-time primitives (``PointInTimeFrame``,
``ForecastDataset``, ``ForecastContext``) arrive in Step 3 alongside them rather
than as options on these types.

Nothing here may import :mod:`openforecast.views` — views are materialized
*from* semantic datasets, never the other way around.
"""

from openforecast.data.features import FeatureAvailability, FeatureKind, FeatureSpec
from openforecast.data.frame import TimeSeriesFrame
from openforecast.data.frequency import Frequency, FrequencyUnit
from openforecast.data.schema import TimeSeriesSchema

__all__ = [
    "FeatureAvailability",
    "FeatureKind",
    "FeatureSpec",
    "Frequency",
    "FrequencyUnit",
    "TimeSeriesFrame",
    "TimeSeriesSchema",
]
