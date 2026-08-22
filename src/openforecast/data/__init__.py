"""Semantic source datasets.

Two representations, deliberately distinct rather than one with optional fields:

```text
TimeSeriesFrame     instance x event_time x variable
PointInTimeFrame    instance x origin_time x event_time x variable
```

``ForecastDataset`` pairs a ``PointInTimeFrame`` (what was knowable) with a
``TimeSeriesFrame`` (what happened), and ``ForecastContext`` is one origin of
either — the shape inference always has.

Nothing here may import :mod:`openforecast.views` — views are materialized
*from* semantic datasets, never the other way around.
"""

from openforecast.data.features import FeatureAvailability, FeatureKind, FeatureSpec
from openforecast.data.forecast_context import ForecastContext
from openforecast.data.forecast_dataset import ForecastDataset
from openforecast.data.frame import TimeSeriesFrame
from openforecast.data.frequency import Frequency, FrequencyUnit
from openforecast.data.point_in_time import PointInTimeFrame, PointInTimeSchema
from openforecast.data.schema import TimeSeriesSchema

__all__ = [
    "FeatureAvailability",
    "FeatureKind",
    "FeatureSpec",
    "ForecastContext",
    "ForecastDataset",
    "Frequency",
    "FrequencyUnit",
    "PointInTimeFrame",
    "PointInTimeSchema",
    "TimeSeriesFrame",
    "TimeSeriesSchema",
]
