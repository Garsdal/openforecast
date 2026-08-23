"""``SklearnProvider``: the three calls, dispatched to an adapter.

```python
provider.descriptors()
provider.fit(model=..., params=..., view=..., seed=..., into=...)
provider.forecast(model=..., params=..., view=..., output=..., state=...)
```

The same :class:`~openforecast.providers.ProviderClient` contract the built-in
provider and the Nixtla, Darts and sktime integrations implement, which is what
lets this run in a subprocess in its own environment without the engine knowing.
There is no branching here on where the data came from, because there is nothing
to branch on: what arrives is an execution view.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa

from openforecast.models import ModelDescriptor, ModelRef
from openforecast.views import FitView, ForecastView
from openforecast_sklearn import catalog
from openforecast_sklearn._version import __version__

__all__ = ["PROVIDER_NAME", "PROVIDER_VERSION", "SklearnProvider"]

#: Also the namespace of every model advertised: ``sklearn/hist-gradient-boosting``.
PROVIDER_NAME = "sklearn"

#: The version of this distribution. Recorded in every artifact it fits, so an
#: artifact says which implementation produced it.
PROVIDER_VERSION = __version__


class SklearnProvider:
    """Executes scikit-learn estimators, in whatever process it was started in."""

    name = PROVIDER_NAME
    version = PROVIDER_VERSION

    def descriptors(self) -> tuple[ModelDescriptor, ...]:
        """Every model this integration can execute."""
        return catalog.descriptors(self.name)

    def fit(
        self,
        *,
        model: ModelRef | str,
        params: Mapping[str, Any],
        view: FitView,
        seed: int | None,
        into: Path,
    ) -> None:
        """Fit ``model`` on ``view``, persisting native state into ``into``."""
        catalog.adapter_for(model, self.name).fit(view, params, into, seed=seed)

    def forecast(
        self,
        *,
        model: ModelRef | str,
        params: Mapping[str, Any],
        view: ForecastView,
        output: Mapping[str, Any],
        state: Path,
    ) -> pa.Table:
        """Answer ``view`` from the state a previous fit wrote into ``state``."""
        del params  # the parameters were compiled into the state at fit time
        return catalog.adapter_for(model, self.name).forecast(view, output, state)

    def __repr__(self) -> str:
        return f"SklearnProvider(version={self.version}, models={len(catalog.model_names())})"
