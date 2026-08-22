"""Fit and forecast task descriptions.

What to fit over, and what to predict:

```python
plan = of.FitPlan(
    origins=of.AllOrigins(),
    window=of.WindowPlan(context=168),
    seed=42,
)

task = of.ForecastTask(horizon=24)
output = of.OutputSpec.quantiles([0.1, 0.5, 0.9])
```

Three things are deliberately separate here. The *recipe* says what model to
build, the *plan* says how to fit it, and the *task* says what to predict. That
is what lets one recipe be fitted at a single origin and at every origin, or
asked for a different horizon, without being rewritten — and it is why nothing
in this package knows which provider will execute it.

Everything is OpenForecast-native. A context length is stated once, as
``WindowPlan(context=168)``, and compiled into whatever the provider calls it
(``input_size``, ``input_chunk_length``); a horizon is a count of steps of the
data's frequency. The caller never says either thing twice.
"""

from openforecast.tasks.forecast import ForecastTask, OutputKind, OutputSpec
from openforecast.tasks.origins import (
    AllOrigins,
    AtOrigin,
    LatestOrigin,
    OriginMode,
    OriginsBetween,
    OriginSelection,
)
from openforecast.tasks.plan import (
    Accelerator,
    FitPlan,
    Resources,
    SearchPlan,
    SearchStrategy,
    WindowPlan,
)

__all__ = [
    "Accelerator",
    "AllOrigins",
    "AtOrigin",
    "FitPlan",
    "ForecastTask",
    "LatestOrigin",
    "OriginMode",
    "OriginSelection",
    "OriginsBetween",
    "OutputKind",
    "OutputSpec",
    "Resources",
    "SearchPlan",
    "SearchStrategy",
    "WindowPlan",
]
