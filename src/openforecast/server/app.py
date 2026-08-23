"""The HTTP projection: five endpoints over one transport.

```text
GET  /v1/models
GET  /v1/models/{ref}

POST /v1/fit
POST /v1/forecast

GET  /v1/artifacts/{ref}
```

Every route is three lines, and that is the point. The router validates the
body into the Pydantic model that describes it, hands it to a
:class:`~openforecast.server.transport.Transport`, and serializes what comes
back. Nothing here decides what a recipe means, which model needs fitting first
or what to do about a missing feature — the engine below already answers all of
it, and a second answer at the HTTP layer is exactly how a projection turns into
an architecture.

Asynchronous training is deliberately not here. A fit is one request that
returns when the artifact is published, because the alternative — a job id, a
poll endpoint, a state machine — is a distributed systems design, and the thing
worth getting right first is that the *semantics* are the same remotely.

FastAPI lives behind the ``openforecast[server]`` extra. It buys routing,
request validation against the models in :mod:`openforecast.server.wire`, and an
OpenAPI document generated from those same models rather than written beside
them — which is the dependency direction rule 7 asks for, mechanically. A client
that only calls a remote service never installs it: :class:`HttpTransport` is
:mod:`urllib`, and the core install stays the three libraries the semantics are
built on.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from openforecast.artifacts.handle import ModelHandle
from openforecast.errors import OpenForecastError
from openforecast.models.descriptor import ModelDescriptor
from openforecast.server.transport import LocalTransport, Transport, status_for
from openforecast.server.wire import (
    ErrorBody,
    ErrorInfo,
    FitBody,
    ForecastBody,
    ForecastPayload,
    ModelListing,
)

__all__ = ["TITLE", "create_app"]

TITLE = "OpenForecast"

DESCRIPTION = (
    "The unified interface for forecasting. Control travels as JSON and bulk data as "
    "Arrow IPC, so a dataset crosses as the Arrow tables it already holds rather than as "
    "nested JSON rows."
)

#: Attached to every route, so a generated SDK reports the same failures the
#: Python one does rather than an untyped body.
ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    404: {"model": ErrorBody, "description": "The reference names nothing."},
    422: {"model": ErrorBody, "description": "The request is well-formed and was refused."},
    502: {"model": ErrorBody, "description": "A provider failed, or did not answer a forecast."},
}


def create_app(transport: Transport | None = None) -> FastAPI:
    """The service, over ``transport`` — by default this process's own engine."""
    resolved: Transport = LocalTransport() if transport is None else transport
    app = FastAPI(title=TITLE, description=DESCRIPTION, version=_version())

    def failed(_request: Request, error: Exception) -> JSONResponse:
        """Answer with the exception, not just a status code.

        The class name travels so that a remote client re-raises what a local
        call would have raised, which is what makes ``except of.DataError`` mean
        the same thing on both transports.
        """
        if not isinstance(error, OpenForecastError):  # pragma: no cover - not registered for
            raise error
        body = ErrorBody(error=ErrorInfo(type=type(error).__name__, message=str(error)))
        return JSONResponse(status_code=status_for(error), content=body.model_dump(mode="json"))

    # Every endpoint below is a plain ``def`` rather than ``async def``, and
    # deliberately: fitting a model is blocking work that can run for minutes,
    # and FastAPI runs a synchronous endpoint in a worker thread instead of on
    # the event loop. An ``async def`` here would stall every other request for
    # the duration of one fit.
    def list_models() -> ModelListing:
        """Every model this service can fit."""
        return ModelListing(models=resolved.models())

    def get_model(ref: str) -> ModelDescriptor:
        """What one reference resolves to. No provider is started to answer it."""
        return resolved.model(ref)

    def fit(body: FitBody) -> ModelHandle:
        """Fit, and answer with the immutable artifact that was published."""
        return resolved.fit(body)

    def forecast(body: ForecastBody) -> ForecastPayload:
        """Forecast with a fitted reference, at one origin."""
        return resolved.forecast(body)

    def get_artifact(ref: str) -> ModelHandle:
        """A fitted artifact, described without loading the model behind it."""
        return resolved.artifact(ref)

    app.add_exception_handler(OpenForecastError, failed)
    _route(app, "GET", "/v1/models", list_models, ModelListing)
    _route(app, "GET", "/v1/models/{ref:path}", get_model, ModelDescriptor)
    _route(app, "POST", "/v1/fit", fit, ModelHandle)
    _route(app, "POST", "/v1/forecast", forecast, ForecastPayload)
    _route(app, "GET", "/v1/artifacts/{ref:path}", get_artifact, ModelHandle)
    return app


def _route(
    app: FastAPI, method: str, path: str, endpoint: Callable[..., Any], answers: type[BaseModel]
) -> None:
    """Register one endpoint, with the failures every endpoint can produce.

    Registered rather than decorated so that the five routes read as the five
    lines of the API surface they are, in the order the plan lists them.
    """
    app.add_api_route(
        path,
        endpoint,
        methods=[method],
        response_model=answers,
        responses=ERROR_RESPONSES,
        summary=(endpoint.__doc__ or "").splitlines()[0],
    )


def _version() -> str:
    """The OpenForecast version, read where it is defined.

    Imported here rather than at module scope: the package imports the client,
    which reaches this layer, so naming it at import time would close a cycle.
    """
    from openforecast import __version__

    return __version__
