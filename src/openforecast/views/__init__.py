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

This module is the whole import surface a provider needs. ``FeatureSpec`` and
``Frequency`` are re-exported here so that an integration never reaches into
:mod:`openforecast.data`, where the semantic source datasets live.
"""

from openforecast.data.features import FeatureAvailability, FeatureKind, FeatureSpec
from openforecast.data.frequency import Frequency, FrequencyUnit
from openforecast.views.base import ViewKind
from openforecast.views.forecast import ForecastView, ForecastViewMetadata
from openforecast.views.planner import (
    FitView,
    OriginMode,
    OriginSelection,
    ViewPlanner,
    ViewRequest,
)
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
    "FeatureAvailability",
    "FeatureKind",
    "FeatureSpec",
    "FitView",
    "ForecastView",
    "ForecastViewMetadata",
    "Frequency",
    "FrequencyUnit",
    "MATERIALIZER_VERSION",
    "OriginFidelity",
    "OriginMode",
    "OriginSelection",
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
]
