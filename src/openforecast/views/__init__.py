"""Provider-neutral execution views.

``SeriesView``, ``SequenceView``, ``TabularView`` and ``ForecastView`` are the
only data representations that cross the provider boundary. The ``ViewPlanner``
materializes them from semantic source datasets so that no provider ever has to
branch on whether the data originated as event-time or point-in-time.

The three fit views are named after the training unit they hold, not after a
model family:

```text
SeriesView     one complete time series           ARIMA, ETS, Theta
SequenceView   many context -> horizon sequences  NHiTS, TFT, PatchTST
TabularView    individual supervised target rows  LightGBM, XGBoost, CatBoost
```

``ForecastView`` is the inference counterpart of all three: one origin, one
horizon.

This module is the whole import surface a provider needs. ``FeatureSpec``,
``Frequency``, the origin selections and the forecast columns a provider answers
with are re-exported here so that an integration never reaches into
:mod:`openforecast.data`, where the semantic source datasets live, nor into
:mod:`openforecast.tasks`, where the user-facing plans do.
"""

from openforecast.data.features import FeatureAvailability, FeatureKind, FeatureSpec
from openforecast.data.frequency import Frequency, FrequencyUnit
from openforecast.protocol.vocabulary import ForecastColumn, forecast_columns
from openforecast.tasks.origins import (
    AllOrigins,
    AtOrigin,
    LatestOrigin,
    OriginMode,
    OriginsBetween,
    OriginSelection,
)
from openforecast.views.base import (
    CONTEXT_END,
    CONTEXT_START,
    EVENT_TIME,
    FORECAST_END,
    FORECAST_START,
    HORIZON_STEP,
    ORIGIN_TIME,
    ROW_ID,
    SAMPLE_ID,
    SERIES_ID,
    ViewKind,
)
from openforecast.views.bundle import (
    read_answer,
    read_fit_view,
    read_forecast_view,
    read_view,
    write_answer,
    write_view,
)
from openforecast.views.forecast import ForecastView, ForecastViewMetadata
from openforecast.views.planner import FitView, ViewPlanner, ViewRequest
from openforecast.views.provenance import (
    MATERIALIZER_VERSION,
    OriginFidelity,
    SourceKind,
    ViewProvenance,
)
from openforecast.views.sequences import SequenceView, SequenceViewSchema
from openforecast.views.series import SeriesView, SeriesViewSchema
from openforecast.views.tabular import TabularView, TabularViewSchema

__all__ = [
    "AllOrigins",
    "AtOrigin",
    "CONTEXT_END",
    "CONTEXT_START",
    "EVENT_TIME",
    "FORECAST_END",
    "FORECAST_START",
    "FeatureAvailability",
    "FeatureKind",
    "FeatureSpec",
    "FitView",
    "ForecastColumn",
    "ForecastView",
    "ForecastViewMetadata",
    "Frequency",
    "FrequencyUnit",
    "HORIZON_STEP",
    "LatestOrigin",
    "MATERIALIZER_VERSION",
    "ORIGIN_TIME",
    "OriginFidelity",
    "OriginMode",
    "OriginSelection",
    "OriginsBetween",
    "ROW_ID",
    "SAMPLE_ID",
    "SERIES_ID",
    "SequenceView",
    "SequenceViewSchema",
    "SeriesView",
    "SeriesViewSchema",
    "SourceKind",
    "TabularView",
    "TabularViewSchema",
    "ViewKind",
    "ViewPlanner",
    "ViewProvenance",
    "ViewRequest",
    "forecast_columns",
    "read_answer",
    "read_fit_view",
    "read_forecast_view",
    "read_view",
    "write_answer",
    "write_view",
]
