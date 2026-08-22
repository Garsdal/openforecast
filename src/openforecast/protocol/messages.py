"""The provider wire protocol: one JSON object per line, in each direction.

```text
-> {"protocol_version": 1, "operation": "handshake"}
<- {"protocol_version": 1, "status": "ok", "result": {"provider": "nixtla", ...}}

-> {"protocol_version": 1, "operation": "fit", "model": "nixtla/nhits",
    "view": {"kind": "sequences", "path": "/tmp/of-.../view"}, "into": "..."}
<- {"protocol_version": 1, "status": "ok", "result": {}}
```

Two channels, and the split is deliberate. Control is JSON Lines over stdin and
stdout: small, ordered, greppable in a log. Bulk data is Arrow IPC in a
directory the message points at, because a training set does not belong inside a
line of JSON. ``stdout`` therefore carries protocol and nothing else — anything a
provider or one of its libraries prints belongs on ``stderr``, and a line of
stdout that is not a response is a protocol violation rather than noise to skip.

Every message names its ``protocol_version``. A peer speaking a different one is
refused at the handshake rather than misread: the bundles either side writes are
laid out by that number, and guessing at a layout that may have changed is
exactly the mistake worth making impossible.

This is the innermost layer, so nothing here names a view type, a descriptor or
a model class. A request points at a bundle by path and kind; the descriptors in
a handshake response are the JSON a
:class:`~openforecast.models.descriptor.ModelDescriptor` serializes to, and the
engine — which may import ``models`` — is what validates them back.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from openforecast.protocol.version import PROTOCOL_VERSION
from openforecast.protocol.vocabulary import ViewKind

__all__ = [
    "ErrorCode",
    "ErrorPayload",
    "FitRequest",
    "FitResult",
    "ForecastRequest",
    "ForecastResult",
    "HandshakeRequest",
    "HandshakeResult",
    "Operation",
    "Request",
    "Response",
    "Status",
    "ViewRef",
    "parse_request",
    "parse_response",
]


class Operation(StrEnum):
    """The three things a provider is ever asked to do."""

    HANDSHAKE = "handshake"
    FIT = "fit"
    FORECAST = "forecast"


class Status(StrEnum):
    OK = "ok"
    ERROR = "error"


class ErrorCode(StrEnum):
    """Why a request failed, in terms the caller can branch on.

    A code rather than a message, because the message is for a person and the
    code is what tells a caller whether to fix the request, install something,
    or report a bug.
    """

    #: The peer speaks a different protocol version.
    PROTOCOL_MISMATCH = "PROTOCOL_MISMATCH"
    #: The line was not a request this build can parse.
    MALFORMED_REQUEST = "MALFORMED_REQUEST"
    #: A well-formed request naming an operation the provider does not implement.
    UNSUPPORTED_OPERATION = "UNSUPPORTED_OPERATION"
    #: The provider does not advertise that model.
    UNKNOWN_MODEL = "UNKNOWN_MODEL"
    #: The model's own parameters were rejected by the model.
    INVALID_MODEL_PARAMETERS = "INVALID_MODEL_PARAMETERS"
    #: The view bundle could not be read, or is not the view that was announced.
    INVALID_VIEW = "INVALID_VIEW"
    #: The provider raised while fitting or forecasting.
    EXECUTION_FAILED = "EXECUTION_FAILED"
    #: Anything the provider did not anticipate. Always a bug somewhere.
    INTERNAL_ERROR = "INTERNAL_ERROR"


class Message(BaseModel):
    """What every message carries, in both directions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    protocol_version: int = PROTOCOL_VERSION


class ViewRef(BaseModel):
    """Where a view bundle is, and what it claims to be.

    The kind travels beside the path so a provider can refuse a view it does not
    consume without reading the bundle at all.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: ViewKind
    path: str


class HandshakeRequest(Message):
    """ "Who are you, and what do you provide?" — always the first message."""

    operation: Literal[Operation.HANDSHAKE] = Operation.HANDSHAKE


class FitRequest(Message):
    """Fit ``model`` on the bundle at ``view``, persisting state into ``into``."""

    operation: Literal[Operation.FIT] = Operation.FIT

    model: str
    params: dict[str, Any] = {}
    seed: int | None = None
    view: ViewRef
    #: The provider's own directory. Nothing else reads it.
    into: str


class ForecastRequest(Message):
    """Answer the bundle at ``view`` from ``state``, writing Arrow into ``answer``."""

    operation: Literal[Operation.FORECAST] = Operation.FORECAST

    model: str
    params: dict[str, Any] = {}
    view: ViewRef
    #: A serialized :class:`~openforecast.tasks.forecast.OutputSpec`.
    output: dict[str, Any] = {}
    #: The directory a previous fit wrote into.
    state: str
    #: Where to write the forecast, as one Arrow IPC file.
    answer: str


#: A request, discriminated on the operation it names.
Request = Annotated[
    HandshakeRequest | FitRequest | ForecastRequest,
    Field(discriminator="operation"),
]

_REQUEST = TypeAdapter[Request](Request)


class HandshakeResult(BaseModel):
    """What a provider is, and every model it advertises."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    provider_version: str
    #: Serialized model descriptors. Kept as plain JSON here because ``models/``
    #: sits above this layer; the engine validates them into descriptors.
    models: tuple[dict[str, Any], ...] = ()


class FitResult(BaseModel):
    """A fit that succeeded. Everything it produced is in the state directory."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str


class ForecastResult(BaseModel):
    """A forecast that succeeded, written to the path the request named."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    answer: str


class ErrorPayload(BaseModel):
    """A failure a caller can act on: a code, a sentence, and the specifics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: ErrorCode
    message: str
    details: dict[str, Any] = {}

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


class Response(Message):
    """The one response envelope, whatever was asked.

    A response is ``ok`` with a result or ``error`` with a payload, never both
    and never neither: a provider that answered with an empty envelope has not
    said whether it did the work.
    """

    status: Status
    result: dict[str, Any] | None = None
    error: ErrorPayload | None = None

    @model_validator(mode="after")
    def _check_envelope(self) -> Self:
        if self.status is Status.OK and self.error is not None:
            raise ValueError("a successful response carries no error payload")
        if self.status is Status.ERROR and self.error is None:
            raise ValueError("a failed response must say why it failed")
        if self.status is Status.ERROR and self.result is not None:
            raise ValueError("a failed response carries no result")
        return self

    @classmethod
    def of(cls, result: BaseModel) -> Response:
        return cls(status=Status.OK, result=result.model_dump(mode="json"))

    @classmethod
    def failed(
        cls, code: ErrorCode, message: str, details: dict[str, Any] | None = None
    ) -> Response:
        return cls(
            status=Status.ERROR,
            error=ErrorPayload(code=code, message=message, details=details or {}),
        )

    @property
    def is_ok(self) -> bool:
        return self.status is Status.OK

    @property
    def payload(self) -> dict[str, Any]:
        """The result of a successful response.

        Raising here rather than returning ``{}`` for a failure: an empty result
        and a failed request are different things, and only one of them means
        the work was done.
        """
        if self.error is not None:
            raise ValueError(f"this response is a failure: {self.error}")
        return self.result or {}


def parse_request(payload: object) -> Request:
    """Validate one decoded JSON line into the request it names."""
    return _REQUEST.validate_python(payload)


def parse_response(payload: object) -> Response:
    """Validate one decoded JSON line into a response envelope."""
    return Response.model_validate(payload)
