"""The execution engine and the providers it executes models with.

```python
engine = Engine(store=..., providers=default_providers())

handle = engine.fit("builtin/seasonal-naive", data=train, params={"season_length": 24})
forecast = engine.forecast(handle, data=context, horizon=48)
```

Four things live here. :class:`Engine` is the sequence a fit and a forecast go
through — resolve, materialize, check, execute, publish — and it is deliberately
provider-blind: nothing in it can name Nixtla or Darts, because a descriptor
already says which view to build and the provider registry already says who runs
it. :class:`ProviderRegistry` is who executes what, and
:class:`SubprocessProvider` is how a provider in its own uv environment becomes
one of them without the engine learning that it is a subprocess. And the
transforms OpenForecast owns — scaling today — are executed here, on the view,
between materialization and execution.

:func:`install_default_providers` is how models become visible: the built-in
provider's descriptors and the recorded handshake of every installed provider
environment are registered into a catalog, and nothing distinguishes them there.
"""

from openforecast.runtime.engine import (
    Engine,
    Leaf,
    leaves,
    normalize_forecast_context,
    normalize_recipe,
)
from openforecast.runtime.environments import (
    ProviderEnvironment,
    ProviderEnvironments,
    ProviderRecord,
)
from openforecast.runtime.forecast import Forecast
from openforecast.runtime.provider import ProviderClient, ProviderRegistry
from openforecast.runtime.providers import (
    default_providers,
    install_default_providers,
    installed_providers,
)
from openforecast.runtime.subprocess import SubprocessProvider
from openforecast.runtime.transforms import TransformState
from openforecast.runtime.validation import validate_view

__all__ = [
    "Engine",
    "Forecast",
    "Leaf",
    "ProviderClient",
    "ProviderEnvironment",
    "ProviderEnvironments",
    "ProviderRecord",
    "ProviderRegistry",
    "SubprocessProvider",
    "TransformState",
    "default_providers",
    "install_default_providers",
    "installed_providers",
    "leaves",
    "normalize_forecast_context",
    "normalize_recipe",
    "validate_view",
]
