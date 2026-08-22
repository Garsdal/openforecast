"""The user-facing client: ``of.fit`` and ``of.forecast``.

```python
import openforecast as of

model = of.fit(
    model="builtin/seasonal-naive",
    data=train,
    params={"season_length": 24},
)

forecast = of.forecast(model=model, data=context, horizon=48)
```

``of.fit`` and ``of.forecast`` delegate to a default :class:`OpenForecast`
client, which owns an artifact store in the usual place and the providers this
build ships with. That is the whole difference between the module functions and
the class: a client can be pointed at a different store — a test's, a
container's — and everything else about it is identical.

The outermost layer, by design. Everything may be imported from here and nothing
may import it, which is what leaves room for Step 16 to give the same two
methods a transport: a client that speaks HTTP to a server executing exactly
this engine will present the same surface, because the surface is defined by
what forecasting means rather than by where it runs.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from openforecast.artifacts.handle import ModelHandle
from openforecast.artifacts.store import ArtifactStore
from openforecast.models.catalog import DEFAULT_CATALOG, ModelCatalog
from openforecast.runtime.engine import Engine, ModelInput
from openforecast.runtime.forecast import Forecast
from openforecast.runtime.provider import ProviderRegistry
from openforecast.runtime.providers import install_default_providers
from openforecast.tasks.forecast import OutputSpec
from openforecast.tasks.plan import FitPlan

__all__ = ["OpenForecast", "fit", "forecast"]


class OpenForecast:
    """Fits and forecasts against one artifact store and one set of providers."""

    def __init__(
        self,
        *,
        store: str | Path | ArtifactStore | None = None,
        catalog: ModelCatalog | None = None,
        providers: ProviderRegistry | None = None,
    ) -> None:
        resolved = store if isinstance(store, ArtifactStore) else ArtifactStore(store)
        self._engine = Engine(
            store=resolved,
            catalog=catalog,
            providers=providers
            if providers is not None
            else install_default_providers(catalog if catalog is not None else DEFAULT_CATALOG),
        )

    @property
    def engine(self) -> Engine:
        return self._engine

    @property
    def models(self) -> ModelCatalog:
        """The models this client can fit."""
        return self._engine.catalog

    def fit(
        self,
        model: ModelInput,
        data: object,
        *,
        horizon: int | None = None,
        plan: FitPlan | None = None,
        name: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> ModelHandle:
        """Fit ``model`` on ``data``, returning the artifact it produced.

        The handle prints as ``local/de-price@01K...`` and is what a forecast
        takes; the unpinned ``local/de-price`` alias follows the latest fit.
        """
        return self._engine.fit(model, data, horizon=horizon, plan=plan, name=name, params=params)

    def forecast(
        self,
        model: ModelInput,
        data: object,
        *,
        horizon: int,
        output: OutputSpec | None = None,
        origin_time: str | datetime | None = None,
    ) -> Forecast:
        """Forecast ``horizon`` steps ahead of what ``data`` knows."""
        return self._engine.forecast(
            model, data, horizon=horizon, output=output, origin_time=origin_time
        )

    def __repr__(self) -> str:
        return f"OpenForecast(store={self._engine.store.root})"


_default: OpenForecast | None = None


def default_client() -> OpenForecast:
    """The client ``of.fit`` and ``of.forecast`` use.

    Built on first use rather than at import: constructing one installs
    providers and names an artifact store, and importing a library should do
    neither until it is asked to do something.
    """
    global _default  # noqa: PLW0603 - one process-wide default, built once
    if _default is None:
        _default = OpenForecast()
    return _default


def fit(
    model: ModelInput,
    data: object,
    *,
    horizon: int | None = None,
    plan: FitPlan | None = None,
    name: str | None = None,
    params: dict[str, Any] | None = None,
) -> ModelHandle:
    """Fit a model on data and publish the artifact it produced.

    ```python
    model = of.fit(
        model="builtin/seasonal-naive",
        data=train,
        params={"season_length": 24},
    )
    ```
    """
    return default_client().fit(model, data, horizon=horizon, plan=plan, name=name, params=params)


def forecast(
    model: ModelInput,
    data: object,
    *,
    horizon: int,
    output: OutputSpec | None = None,
    origin_time: str | datetime | None = None,
) -> Forecast:
    """Forecast with a fitted model.

    ```python
    forecast = of.forecast(model="local/de-price", data=context, horizon=24)
    ```

    ``model`` may be the handle a fit returned, a pinned revision, or the alias
    that follows the latest one. A reference naming a model that was never
    fitted raises ``ModelRequiresFit`` rather than quietly fitting one on the
    data the forecast was handed.
    """
    return default_client().forecast(
        model, data, horizon=horizon, output=output, origin_time=origin_time
    )
