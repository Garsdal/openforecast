"""The provider wire protocol.

The innermost layer: it may not import any other OpenForecast subpackage, and it
knows nothing about any specific provider. What lives here is what has to be
spelled the same way on both sides of a boundary.

``ViewKind`` is named by a model's training contract in ``models/`` and by the
execution views in ``views/``, and those two layers cannot import each other.
``ForecastColumn`` is written by whoever executes a model and read by the engine
that asked for it. The request and response messages in
:mod:`openforecast.protocol.messages` are written by the engine and read by a
provider in another process, and vice versa. ``PROTOCOL_VERSION`` ties all of it
together: it is stamped into an artifact manifest, declared in every message,
and checked at the handshake, and those are one number.
"""

from openforecast.protocol.messages import (
    ErrorCode,
    ErrorPayload,
    FitRequest,
    FitResult,
    ForecastRequest,
    ForecastResult,
    HandshakeRequest,
    HandshakeResult,
    Operation,
    Request,
    Response,
    Status,
    ViewRef,
    parse_request,
    parse_response,
)
from openforecast.protocol.version import PROTOCOL_VERSION
from openforecast.protocol.vocabulary import ForecastColumn, ViewKind, forecast_columns

__all__ = [
    "PROTOCOL_VERSION",
    "ErrorCode",
    "ErrorPayload",
    "FitRequest",
    "FitResult",
    "ForecastColumn",
    "ForecastRequest",
    "ForecastResult",
    "HandshakeRequest",
    "HandshakeResult",
    "Operation",
    "Request",
    "Response",
    "Status",
    "ViewKind",
    "ViewRef",
    "forecast_columns",
    "parse_request",
    "parse_response",
]
