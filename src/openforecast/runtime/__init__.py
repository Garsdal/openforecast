"""The execution engine and the providers it executes models with.

```python
engine = Engine(store=..., providers=default_providers())

handle = engine.fit("builtin/seasonal-naive", data=train, params={"season_length": 24})
forecast = engine.forecast(handle, data=context, horizon=48)
```

Three things live here. :class:`Engine` is the sequence a fit and a forecast go
through — resolve, materialize, check, execute, publish — and it is deliberately
provider-blind: nothing in it can name Nixtla or Darts, because a descriptor
already says which view to build and the provider registry already says who runs
it. :class:`ProviderClient` is the shape anything that executes a model has to
have, structural so that Step 9's subprocess client is one without inheriting
anything across a process boundary. And the transforms OpenForecast owns —
scaling today — are executed here, on the view, between materialization and
execution.

:func:`install_default_providers` is how the built-in provider becomes visible:
its descriptors are registered into a catalog exactly as an external provider's
will be when it answers a handshake in Step 9.
"""

from openforecast.runtime.engine import (
    Engine,
    Leaf,
    leaves,
    normalize_forecast_context,
    normalize_recipe,
)
from openforecast.runtime.forecast import Forecast
from openforecast.runtime.provider import ProviderClient, ProviderRegistry
from openforecast.runtime.providers import default_providers, install_default_providers
from openforecast.runtime.transforms import TransformState
from openforecast.runtime.validation import validate_view

__all__ = [
    "Engine",
    "Forecast",
    "Leaf",
    "ProviderClient",
    "ProviderRegistry",
    "TransformState",
    "default_providers",
    "install_default_providers",
    "leaves",
    "normalize_forecast_context",
    "normalize_recipe",
    "validate_view",
]
