"""The built-in reference provider.

```python
provider.descriptors()                       # what it advertises
provider.fit(model=..., view=..., into=...)  # trains into a directory it is given
provider.forecast(model=..., view=..., state=...)
```

It implements the same three operations an external integration will implement
over a subprocess in Step 9, against the same inputs: an execution view and a
directory. Nothing here knows whether the view was materialized from event-time
data or from real forecast vintages — that is the boundary working, and it is
the reason this provider can be used to prove the engine end to end without a
single forecasting library installed.

The state directory is handed over by the engine and is the provider's alone.
It survives the process, which is what makes a fitted model a resource rather
than a variable.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa

from openforecast.errors import UnknownModelError
from openforecast.models import ModelDescriptor, ModelRef
from openforecast.providers.builtin import seasonal_naive
from openforecast.views import FitView, ForecastView

__all__ = ["BUILTIN_PROVIDER", "PROVIDER_NAME", "PROVIDER_VERSION", "BuiltinProvider"]

#: Also the namespace of every model it advertises: ``builtin/seasonal-naive``.
PROVIDER_NAME = "builtin"

#: Bumped when a fit of the same recipe on the same data would produce a
#: different model, so that an artifact says which implementation made it.
PROVIDER_VERSION = "1.0.0"

_MODELS = (seasonal_naive.NAME,)


class BuiltinProvider:
    """Executes the models OpenForecast ships with, in this process."""

    name = PROVIDER_NAME
    version = PROVIDER_VERSION

    def descriptors(self) -> tuple[ModelDescriptor, ...]:
        """Every model this provider can execute, as the catalog holds them."""
        return (seasonal_naive.descriptor(self.name),)

    def fit(
        self,
        *,
        model: ModelRef | str,
        params: Mapping[str, Any],
        view: FitView,
        seed: int | None,
        into: Path,
    ) -> None:
        """Fit ``model`` on ``view``, persisting whatever it needs into ``into``."""
        del seed  # every model here is deterministic; the parameter is the protocol's
        self._name_of(model)
        seasonal_naive.fit(view, params, into)

    def forecast(
        self,
        *,
        model: ModelRef | str,
        params: Mapping[str, Any],
        view: ForecastView,
        output: Mapping[str, Any],
        state: Path,
    ) -> pa.Table:
        """Forecast the event times ``view`` asks about, from the state in ``state``."""
        del params  # the parameters were compiled into the state at fit time
        self._name_of(model)
        return seasonal_naive.forecast(view, output, state)

    def _name_of(self, model: ModelRef | str) -> str:
        parsed = ModelRef.parse(model)
        if parsed.namespace != self.name or parsed.name not in _MODELS:
            raise UnknownModelError(
                f"{parsed} is not a model of the {self.name!r} provider; it provides "
                f"{[f'{self.name}/{name}' for name in _MODELS]}"
            )
        return parsed.name

    def __repr__(self) -> str:
        return f"BuiltinProvider(version={self.version}, models={len(_MODELS)})"


#: The one instance. It holds no state of its own — everything a fit produces
#: lives in the artifact directory it was given.
BUILTIN_PROVIDER = BuiltinProvider()
