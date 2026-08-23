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

Since Step 16 it can also be pointed somewhere else entirely:

```python
client = of.OpenForecast(transport=of.LocalTransport())
client = of.OpenForecast(transport=of.HttpTransport("http://localhost:8321"))
```

Both expose ``client.models.list()``, ``client.models.get(...)``,
``client.fit(...)`` and ``client.forecast(...)``, and the code below neither
knows nor cares which it is holding: it turns what the caller wrote into the
request models of :mod:`openforecast.server.wire`, hands them to the transport,
and turns the answer back into a :class:`~openforecast.runtime.forecast.Forecast`
or a :class:`~openforecast.artifacts.handle.ModelHandle`. There is no branch on
where the model ran, which is what makes "the same forecasting semantics
remotely" a property of the code rather than a promise.

One thing does change shape across the boundary. A fitted model is a resource
with an identity rather than a value, so a forecast names it by reference —
``local/de-price@01K...`` — and passing back the handle a fit returned means
sending that reference. Locally the artifact is on this machine; remotely it is
on the service's, and the reference means the same thing to whichever engine
owns it.

The outermost layer, by design: everything may be imported from here. The one
thing that imports it is :mod:`openforecast.evaluation`, which sits in the same
layer and is a *user* of this module rather than something beneath it — a
backtest is a loop over ``fit`` and ``forecast``, which is exactly why nothing
inside the engine knows it is being backtested.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from openforecast.artifacts.handle import ModelHandle
from openforecast.artifacts.store import ArtifactStore
from openforecast.errors import RecipeError
from openforecast.models.catalog import ModelCatalog
from openforecast.models.descriptor import ModelDescriptor
from openforecast.models.ref import ModelRef
from openforecast.recipes.nodes import Model
from openforecast.runtime.engine import Engine, ModelInput
from openforecast.runtime.forecast import Forecast
from openforecast.runtime.provider import ProviderRegistry
from openforecast.server.transport import LocalTransport, Transport
from openforecast.server.wire import (
    FitBody,
    ForecastBody,
    ForecastPayload,
    decode_table,
    encode_data,
)
from openforecast.tasks.forecast import OutputSpec
from openforecast.tasks.plan import FitPlan

__all__ = ["Models", "OpenForecast", "fit", "forecast"]


class Models:
    """The models a client can fit, however it reaches them.

    ``of.models`` is the catalog of *this* build; this is the catalog of the
    service a client is pointed at, which is the same thing when the transport
    is local. Both answer ``list()`` and ``get()``, so discovery code does not
    have to know which one it is holding.
    """

    def __init__(self, transport: Transport) -> None:
        self._transport = transport

    def list(self, *, provider: str | None = None) -> tuple[ModelDescriptor, ...]:
        """Every model, optionally only one provider's."""
        found = self._transport.models()
        if provider is None:
            return found
        return tuple(item for item in found if item.provider == provider)

    def get(self, ref: ModelRef | str) -> ModelDescriptor:
        """What one reference resolves to. No provider is started to answer it."""
        return self._transport.model(str(ref))

    def refs(self) -> tuple[ModelRef, ...]:
        """Just the references, which is what a listing is usually for."""
        return tuple(item.ref for item in self.list())

    def providers(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.provider for item in self.list()))

    def __iter__(self) -> Iterator[ModelDescriptor]:
        return iter(self.list())

    def __len__(self) -> int:
        return len(self.list())

    def __contains__(self, ref: ModelRef | str) -> bool:
        return str(ModelRef.parse(ref)) in {str(item.ref) for item in self.list()}

    def __repr__(self) -> str:
        return f"Models({len(self)} available)"


class OpenForecast:
    """Fits and forecasts, wherever its transport executes.

    ``OpenForecast()`` is local and owns a store and the providers this build
    ships with. ``OpenForecast(transport=HttpTransport(...))`` is the same
    object over a service, and the arguments naming local machinery — a store,
    a catalog, a provider registry — belong to the local transport, so passing
    both is refused rather than silently ignoring one.
    """

    def __init__(
        self,
        *,
        store: str | Path | ArtifactStore | None = None,
        catalog: ModelCatalog | None = None,
        providers: ProviderRegistry | None = None,
        transport: Transport | None = None,
    ) -> None:
        local = (store, catalog, providers)
        if transport is not None and any(item is not None for item in local):
            raise RecipeError(
                "a store, a catalog and a provider registry configure local execution, and "
                "this client was given a transport that owns them; configure them on the "
                "transport instead"
            )
        self._transport: Transport = (
            LocalTransport(store=store, catalog=catalog, providers=providers)
            if transport is None
            else transport
        )
        self._models = Models(self._transport)

    @property
    def transport(self) -> Transport:
        """Where this client executes."""
        return self._transport

    @property
    def engine(self) -> Engine:
        """The engine behind a local client.

        Raised on rather than returned as ``None`` for a remote one: there is no
        engine here, and code reaching for the store or the provider registry
        through it is code that has assumed local execution.
        """
        if isinstance(self._transport, LocalTransport):
            return self._transport.engine
        raise RecipeError(
            f"{self._transport!r} executes elsewhere, so there is no engine here; the "
            f"store, the catalog and the providers belong to the service"
        )

    @property
    def models(self) -> Models:
        """The models this client can fit."""
        return self._models

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
        return self._transport.fit(
            FitBody(
                model=_fittable(model),
                data=encode_data(data),
                horizon=horizon,
                plan=plan,
                name=name,
                params=params,
            )
        )

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
        answer = self._transport.forecast(
            ForecastBody(
                model=_fitted_ref(model),
                data=encode_data(data),
                horizon=horizon,
                output=output,
                origin_time=None if origin_time is None else _moment(origin_time),
            )
        )
        return _forecast(answer)

    def artifact(self, ref: ModelRef | str) -> ModelHandle:
        """One fitted artifact, described without loading the model behind it."""
        return self._transport.artifact(str(ref))

    def __repr__(self) -> str:
        return f"OpenForecast({self._transport!r})"


def _fittable(model: ModelInput) -> Any:
    """What ``of.fit(model=...)`` means, as the request carries it.

    A recipe travels as itself and a reference as the string it is. A fitted
    artifact is neither: it is the *result* of a fit, and refitting one means
    fitting the model it records, so it is refused here with the sentence the
    engine would have used.
    """
    if isinstance(model, ModelHandle):
        raise RecipeError(
            f"{model.ref} is a fitted artifact, not a model to fit; fit the recipe it "
            f"records, on data of your choosing"
        )
    return str(model) if isinstance(model, ModelRef) else model


def _fitted_ref(model: ModelInput) -> str:
    """The reference a forecast names, whichever way the caller wrote it.

    A fitted model is a resource with an identity, so a forecast asks for it by
    reference; the handle a fit returned *is* a pinned one. A recipe is not a
    fitted model at all, and fitting it here would forecast from a model the
    caller never trained.
    """
    if isinstance(model, ModelHandle):
        return str(model.ref)
    if isinstance(model, ModelRef | str):
        return str(model)
    if isinstance(model, Model):
        if model.params:
            # A fitted model compiled its parameters at fit time and a pretrained
            # one takes them from nowhere, so nothing downstream would read these.
            raise RecipeError(
                f"{model.ref} was given the parameters {sorted(model.params)} for a forecast, "
                f"and a forecast reads none: a fitted model already carries the ones it was "
                f"fitted with, and a pretrained one is used as it was published"
            )
        return str(model.ref)
    raise RecipeError(
        "a forecast is made with a fitted model, not with a recipe; fit the recipe "
        "and forecast with the local/... reference that comes back"
    )


def _moment(value: str | datetime) -> datetime:
    from openforecast.data.point_in_time import parse_moment

    return parse_moment(value, "origin_time")


def _forecast(payload: ForecastPayload) -> Forecast:
    """The answer, rebuilt into the object a local call returns."""
    return Forecast(
        decode_table(payload.table, "forecast"),
        origin_time=payload.origin_time,
        horizon=payload.horizon,
        targets=payload.targets,
        instance_keys=payload.instance_keys,
        model=payload.model,
    )


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
