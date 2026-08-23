"""The request and response models the HTTP projection is generated from.

```text
POST /v1/fit       FitBody      -> ModelHandle
POST /v1/forecast  ForecastBody -> ForecastBody's answer
```

Two kinds of thing travel over this boundary and they travel differently, which
is the same split the provider protocol makes: **control is JSON, bulk data is
Arrow IPC.** A recipe, a plan, a horizon and an output specification are small,
readable and worth having in a log, so they are Pydantic models and appear in
the OpenAPI document as themselves. A training set is not, so a dataset crosses
as the Arrow tables it already holds, base64 in one opaque field rather than as
a hundred thousand nested JSON objects.

That is deliberately the *interim* arrangement. base64 costs a third of the
payload in size and the whole of it in memory, and the honest fix is a multipart
body or an uploaded Arrow object the control message points at — exactly what
the provider protocol does with a directory. What matters for now is that the
shape is already right: the encoding of the bulk channel can change without any
control model here changing, because no row of data is described by one.

Nothing in this module imports the HTTP framework. These are the *semantics* of
the remote surface, and :mod:`openforecast.server.app` is one projection of them
— which is what lets a client that only ever speaks to a remote server install
OpenForecast without a web framework in it.

A payload is decoded through the ordinary constructors, so every invariant a
frame enforces is enforced again on the far side of the network: a truncated
table fails to load rather than being fitted as a shorter history.
"""

from __future__ import annotations

import base64
import binascii
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field

from openforecast.data.forecast_context import ForecastContext
from openforecast.data.forecast_dataset import ForecastDataset
from openforecast.data.frame import TimeSeriesFrame
from openforecast.data.point_in_time import PointInTimeFrame, PointInTimeSchema
from openforecast.data.schema import TimeSeriesSchema
from openforecast.errors import DataError
from openforecast.models.descriptor import ModelDescriptor
from openforecast.recipes.nodes import Recipe
from openforecast.tasks.forecast import OutputSpec
from openforecast.tasks.plan import FitPlan

__all__ = [
    "DataKind",
    "DataPayload",
    "ErrorBody",
    "ErrorInfo",
    "FitBody",
    "ForecastBody",
    "ForecastContextPayload",
    "ForecastDatasetPayload",
    "ForecastPayload",
    "ModelListing",
    "PointInTimePayload",
    "TimeSeriesPayload",
    "decode_data",
    "encode_data",
]


class DataKind(StrEnum):
    """Which semantic dataset a payload carries.

    The four the local API accepts, named the same way: what can be fitted or
    forecast from does not change because the call went over a network.
    """

    TIME_SERIES = "time_series"
    POINT_IN_TIME = "point_in_time"
    FORECAST_DATASET = "forecast_dataset"
    FORECAST_CONTEXT = "forecast_context"


class Wire(BaseModel):
    """Frozen and closed, like every other protocol object in the package."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# -- bulk data ---------------------------------------------------------------


class TimeSeriesPayload(Wire):
    """A :class:`~openforecast.data.frame.TimeSeriesFrame` on the wire."""

    kind: Literal[DataKind.TIME_SERIES] = DataKind.TIME_SERIES
    #: The frame's declared semantics. Named ``data_schema`` rather than
    #: ``schema`` because the latter is a method on every Pydantic model.
    data_schema: TimeSeriesSchema
    #: Arrow IPC, base64. Absent tables stay absent rather than arriving empty:
    #: a frame with no future covariates is not one with an empty future table.
    history: str
    future: str | None = None
    static: str | None = None


class PointInTimePayload(Wire):
    """A :class:`~openforecast.data.point_in_time.PointInTimeFrame` on the wire."""

    kind: Literal[DataKind.POINT_IN_TIME] = DataKind.POINT_IN_TIME
    data_schema: PointInTimeSchema
    table: str


class ForecastDatasetPayload(Wire):
    """Real vintages and the outcomes they were predicting."""

    kind: Literal[DataKind.FORECAST_DATASET] = DataKind.FORECAST_DATASET
    information: PointInTimePayload
    truth: TimeSeriesPayload


class ForecastContextPayload(Wire):
    """One inference origin: a frame, and the moment it is split at."""

    kind: Literal[DataKind.FORECAST_CONTEXT] = DataKind.FORECAST_CONTEXT
    origin_time: datetime
    frame: TimeSeriesPayload


#: What ``data=`` is, discriminated on the kind it names.
DataPayload = Annotated[
    TimeSeriesPayload | PointInTimePayload | ForecastDatasetPayload | ForecastContextPayload,
    Field(discriminator="kind"),
]


def encode_data(data: object) -> DataPayload:
    """The payload for one of the four things ``fit`` and ``forecast`` accept.

    Refused rather than coerced for anything else, and with the same sentence
    the local API uses: a remote call fails for the reasons a local one does.
    """
    if isinstance(data, TimeSeriesFrame):
        return _encode_frame(data)
    if isinstance(data, PointInTimeFrame):
        return _encode_point_in_time(data)
    if isinstance(data, ForecastDataset):
        return ForecastDatasetPayload(
            information=_encode_point_in_time(data.information),
            truth=_encode_frame(data.truth),
        )
    if isinstance(data, ForecastContext):
        return ForecastContextPayload(origin_time=data.origin_time, frame=_encode_frame(data.frame))
    raise DataError(
        f"cannot send {type(data).__name__} to a forecasting service; pass a "
        f"TimeSeriesFrame, a PointInTimeFrame, a ForecastDataset, or one origin of a "
        f"ForecastDataset with dataset.at_origin(t)"
    )


def decode_data(payload: DataPayload) -> object:
    """The dataset ``payload`` carries, validated as if it had been built here."""
    if isinstance(payload, TimeSeriesPayload):
        return _decode_frame(payload)
    if isinstance(payload, PointInTimePayload):
        return _decode_point_in_time(payload)
    if isinstance(payload, ForecastDatasetPayload):
        return ForecastDataset(
            information=_decode_point_in_time(payload.information),
            truth=_decode_frame(payload.truth),
        )
    return ForecastContext(origin_time=payload.origin_time, frame=_decode_frame(payload.frame))


def _encode_frame(frame: TimeSeriesFrame) -> TimeSeriesPayload:
    return TimeSeriesPayload(
        data_schema=frame.schema,
        history=encode_table(frame.history),
        future=None if frame.future is None else encode_table(frame.future),
        static=None if frame.static is None else encode_table(frame.static),
    )


def _decode_frame(payload: TimeSeriesPayload) -> TimeSeriesFrame:
    return TimeSeriesFrame(
        history=decode_table(payload.history, "history"),
        schema=payload.data_schema,
        future=None if payload.future is None else decode_table(payload.future, "future"),
        static=None if payload.static is None else decode_table(payload.static, "static"),
    )


def _encode_point_in_time(frame: PointInTimeFrame) -> PointInTimePayload:
    return PointInTimePayload(data_schema=frame.schema, table=encode_table(frame.table))


def _decode_point_in_time(payload: PointInTimePayload) -> PointInTimeFrame:
    return PointInTimeFrame(decode_table(payload.table, "table"), payload.data_schema)


def encode_table(table: pa.Table) -> str:
    """One Arrow table as base64 Arrow IPC — the bulk channel, in a JSON field."""
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return base64.b64encode(sink.getvalue().to_pybytes()).decode("ascii")


def decode_table(payload: str, label: str) -> pa.Table:
    """Read one back, reporting a corrupt payload as data rather than as a crash."""
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as error:
        raise DataError(f"the {label} table is not valid base64: {error}") from error
    try:
        with pa.ipc.open_stream(pa.BufferReader(raw)) as reader:
            return reader.read_all()
    except pa.ArrowInvalid as error:
        raise DataError(f"the {label} table is not readable Arrow IPC: {error}") from error


# -- requests ----------------------------------------------------------------


class FitBody(Wire):
    """``POST /v1/fit`` — the arguments of ``of.fit``, named the same way.

    ``model`` is a recipe or the reference of a single model, which is the same
    union the local call accepts: ``of.fit("builtin/seasonal-naive", ...)`` is
    the short spelling of ``of.fit(of.Model("builtin/seasonal-naive"), ...)``.
    A fitted artifact is not in it, because fitting one again is a new fit and
    the recipe it records is what would be refitted.
    """

    model: Recipe | str
    data: DataPayload
    horizon: int | None = Field(default=None, ge=1)
    plan: FitPlan | None = None
    name: str | None = None
    params: dict[str, Any] | None = None


class ForecastBody(Wire):
    """``POST /v1/forecast`` — a fitted reference, one origin, one horizon.

    ``model`` is a string here where the local call also accepts the handle it
    returned. That is not a narrowing: a handle *is* a pinned reference plus the
    manifest the server already has, so sending ``local/de-price@01K...`` asks
    for the same artifact, and sending ``local/de-price`` follows the alias on
    the server that owns it.
    """

    model: str
    data: DataPayload
    horizon: int = Field(ge=1)
    output: OutputSpec | None = None
    origin_time: datetime | None = None


# -- responses ---------------------------------------------------------------


class ModelListing(Wire):
    """``GET /v1/models`` — every model the service can fit."""

    models: tuple[ModelDescriptor, ...] = ()


class ForecastPayload(Wire):
    """What a forecast is, in the one long shape it always has.

    The metadata is the control channel and ``table`` is the bulk one: the
    canonical long forecast — instance keys, ``event_time``, ``target``,
    ``kind``, ``quantile``, ``sample``, ``value`` — as Arrow IPC. A wide
    forecast is a projection the caller makes locally, so it is never what
    crosses.
    """

    model: str
    origin_time: datetime
    horizon: int
    targets: tuple[str, ...]
    instance_keys: tuple[str, ...] = ()
    table: str


class ErrorInfo(Wire):
    """A failure, in terms a caller can act on.

    ``type`` is the name of the OpenForecast exception the same failure would
    have raised in process, so a remote client re-raises what a local one would
    and a caller's ``except of.DataError`` does not depend on where the model
    ran.
    """

    type: str
    message: str


class ErrorBody(Wire):
    """The one error envelope, whatever the status code."""

    error: ErrorInfo
