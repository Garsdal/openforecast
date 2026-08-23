"""Where a forecast runs, as the one thing a client is configured with.

```python
client = of.OpenForecast(transport=of.LocalTransport())
client = of.OpenForecast(transport=of.HttpTransport("http://localhost:8321"))
```

Both answer the same five questions — list the models, describe one, fit,
forecast, describe a fitted artifact — and the client above them is the same
object either way. That is the whole claim of Step 16: HTTP is a projection of
what forecasting means here, not a second architecture with its own semantics.

The two implementations are deliberately asymmetric in one respect only.
:class:`LocalTransport` owns an :class:`~openforecast.runtime.engine.Engine` and
an artifact store; :class:`HttpTransport` owns a URL and knows nothing about
either, because the service at the other end is what has them. Everything else —
which recipes are legal, which model needs fitting first, what a missing feature
does — is decided by the engine in exactly one place, and a remote client
inherits those answers rather than reimplementing them.

Errors cross with the failure, not just a status code. A service refusing a fit
because the data does not declare a feature answers ``422`` with the exception
name, and :class:`HttpTransport` re-raises that exception — so
``except of.DataError`` catches the same failure whether the model ran here or
somewhere else. That is the same property the provider subprocess protocol has,
for the same reason: a caller's error handling should not depend on where the
work happened.

The HTTP client is :mod:`urllib`. A remote-only user should not have to install
a web framework to *call* a forecasting service — the framework belongs to the
service — and rule 1 makes a runtime dependency an architectural decision. The
server half is where FastAPI lives, behind the ``openforecast[server]`` extra.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from openforecast import errors
from openforecast.artifacts.handle import ModelHandle
from openforecast.artifacts.store import ArtifactStore
from openforecast.errors import OpenForecastError, ProviderError, UnknownModelError
from openforecast.models.catalog import DEFAULT_CATALOG, ModelCatalog
from openforecast.models.descriptor import ModelDescriptor
from openforecast.runtime.engine import Engine
from openforecast.runtime.provider import ProviderRegistry
from openforecast.runtime.providers import install_default_providers
from openforecast.server.wire import (
    ErrorBody,
    FitBody,
    ForecastBody,
    ForecastPayload,
    ModelListing,
    decode_data,
    encode_table,
)

__all__ = ["DEFAULT_PORT", "HttpTransport", "LocalTransport", "Transport"]

#: What ``openforecast serve`` binds to unless told otherwise.
DEFAULT_PORT = 8321

#: How long an HTTP call waits, in seconds. A fit is not a fast request, and a
#: default that expired in the middle of one would look like a service failure.
DEFAULT_TIMEOUT = 600.0

CONTENT_TYPE = "application/json"


@runtime_checkable
class Transport(Protocol):
    """The five calls a forecasting service answers.

    A structural protocol rather than a base class, for the same reason
    :class:`~openforecast.runtime.provider.ProviderClient` is one: the two
    implementations share the shape of these calls and nothing else, and one of
    them is on the far side of a network.
    """

    def models(self) -> tuple[ModelDescriptor, ...]:
        """Every model that can be fitted here."""
        ...

    def model(self, ref: str) -> ModelDescriptor:
        """What one reference resolves to, without executing anything."""
        ...

    def fit(self, body: FitBody) -> ModelHandle:
        """Fit, and return the artifact that was published."""
        ...

    def forecast(self, body: ForecastBody) -> ForecastPayload:
        """Forecast with a fitted reference, at one origin."""
        ...

    def artifact(self, ref: str) -> ModelHandle:
        """A fitted artifact, described without loading the model behind it."""
        ...


class LocalTransport:
    """Executes in this process, against a store and a set of providers.

    The default, and what ``of.fit`` has always done. It is written as a
    transport so that the HTTP one has something to be the same as: the service
    of :mod:`openforecast.server.app` is this class behind a router, which is
    what makes "the same user-facing semantics" a fact about the code rather
    than an intention.
    """

    def __init__(
        self,
        *,
        store: str | Path | ArtifactStore | None = None,
        catalog: ModelCatalog | None = None,
        providers: ProviderRegistry | None = None,
        engine: Engine | None = None,
    ) -> None:
        if engine is not None:
            self._engine = engine
            return
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

    def models(self) -> tuple[ModelDescriptor, ...]:
        return self._engine.catalog.list()

    def model(self, ref: str) -> ModelDescriptor:
        return self._engine.catalog.get(ref)

    def fit(self, body: FitBody) -> ModelHandle:
        return self._engine.fit(
            body.model,
            decode_data(body.data),
            horizon=body.horizon,
            plan=body.plan,
            name=body.name,
            params=body.params,
        )

    def forecast(self, body: ForecastBody) -> ForecastPayload:
        answer = self._engine.forecast(
            body.model,
            decode_data(body.data),
            horizon=body.horizon,
            output=body.output,
            origin_time=body.origin_time,
        )
        return ForecastPayload(
            model=answer.model,
            origin_time=answer.origin_time,
            horizon=answer.horizon,
            targets=answer.targets,
            instance_keys=answer.instance_keys,
            table=encode_table(answer.table),
        )

    def artifact(self, ref: str) -> ModelHandle:
        return self._engine.store.get(ref)

    def __repr__(self) -> str:
        return f"LocalTransport(store={self._engine.store.root})"


class HttpTransport:
    """Executes wherever ``base_url`` is serving, over ``/v1``.

    Holds no engine, no catalog and no store: what a model means, whether it
    needs fitting first and where its artifact lives are all decided by the
    service. This side encodes the request, decodes the answer, and turns an
    error envelope back into the exception it describes.
    """

    def __init__(self, base_url: str, *, timeout: float = DEFAULT_TIMEOUT) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    @property
    def base_url(self) -> str:
        return self._base_url

    def models(self) -> tuple[ModelDescriptor, ...]:
        listing = ModelListing.model_validate(self._call("GET", "/v1/models"))
        return listing.models

    def model(self, ref: str) -> ModelDescriptor:
        return ModelDescriptor.model_validate(self._call("GET", f"/v1/models/{_path(ref)}"))

    def fit(self, body: FitBody) -> ModelHandle:
        return ModelHandle.model_validate(
            self._call("POST", "/v1/fit", body.model_dump(mode="json"))
        )

    def forecast(self, body: ForecastBody) -> ForecastPayload:
        payload = self._call("POST", "/v1/forecast", body.model_dump(mode="json"))
        return ForecastPayload.model_validate(payload)

    def artifact(self, ref: str) -> ModelHandle:
        return ModelHandle.model_validate(self._call("GET", f"/v1/artifacts/{_path(ref)}"))

    # -- the wire ----------------------------------------------------------

    def _call(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        request = Request(  # noqa: S310 - the scheme is the caller's own base URL
            f"{self._base_url}{path}",
            method=method,
            data=None if body is None else json.dumps(body).encode("utf-8"),
            headers={
                "Accept": CONTENT_TYPE,
                **({} if body is None else {"Content-Type": CONTENT_TYPE}),
            },
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:  # noqa: S310
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            raise _reraise(error) from None
        except URLError as error:
            raise ProviderError(
                f"cannot reach the forecasting service at {self._base_url}: {error.reason}"
            ) from error

    def __repr__(self) -> str:
        return f"HttpTransport({self._base_url})"


def _path(ref: str) -> str:
    """A model reference in a URL path.

    ``local/de-price@01K...`` holds a slash, and the slash is *kept*: the routes
    that take a reference match the rest of the path, so a reference is one name
    in the URL as it is everywhere else. Escaping it would be read back as a
    literal ``%2F`` by some proxies and as a path separator by others.
    """
    return quote(ref, safe="/@")


def _reraise(error: HTTPError) -> OpenForecastError:
    """The exception a service's error envelope describes.

    The envelope names an OpenForecast exception, and the class is looked up in
    :mod:`openforecast.errors` rather than trusted: a service reporting a type
    this build does not have is reported as a provider failure, not resolved to
    something arbitrary. Everything that is not an envelope at all — a proxy's
    HTML error page, a gateway timeout — is a provider failure too, because the
    request never reached an engine.
    """
    raw = error.read().decode("utf-8", errors="replace")
    try:
        envelope = ErrorBody.model_validate_json(raw)
    except ValueError:
        return ProviderError(
            f"the forecasting service answered {error.code} with something that is not an "
            f"OpenForecast error: {raw[:200]}"
        )
    found = getattr(errors, envelope.error.type, None)
    if isinstance(found, type) and issubclass(found, OpenForecastError):
        return found(envelope.error.message)
    return ProviderError(
        f"the forecasting service reported {envelope.error.type}, which this build does "
        f"not know: {envelope.error.message}"
    )


def status_for(error: OpenForecastError) -> int:
    """The HTTP status one OpenForecast failure projects to.

    Three groups, and the distinction is who has to do something about it: a
    name that resolves to nothing is ``404``, a well-formed request the engine
    refuses is ``422``, and a provider that failed or answered with something
    that is not a forecast is ``502`` — the service is fine, the thing behind it
    is not.
    """
    if isinstance(error, UnknownModelError):
        return 404
    if isinstance(error, ProviderError):
        return 502
    return 422
