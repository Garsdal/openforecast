"""The wire messages themselves: what they mean, and what they refuse to mean.

These are the only objects both sides of the provider boundary construct, so
they are held to one rule above all: a message may not be ambiguous about
whether the work was done. An envelope that is neither a result nor a failure,
or that is somehow both, would be read as success by anything that checks a
field rather than the status.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from openforecast.protocol import (
    PROTOCOL_VERSION,
    ErrorCode,
    FitRequest,
    FitResult,
    ForecastRequest,
    HandshakeRequest,
    HandshakeResult,
    Operation,
    Response,
    Status,
    ViewKind,
    ViewRef,
    parse_request,
    parse_response,
)

VIEW = ViewRef(kind=ViewKind.SEQUENCES, path="/tmp/openforecast/view")


def test_a_request_is_read_back_as_the_operation_it_names() -> None:
    for request in (
        HandshakeRequest(),
        FitRequest(model="nixtla/nhits", view=VIEW, into="/tmp/state"),
        ForecastRequest(model="nixtla/nhits", view=VIEW, state="/tmp/s", answer="/tmp/a.arrow"),
    ):
        assert parse_request(json.loads(request.model_dump_json())) == request


def test_every_message_declares_the_protocol_it_speaks() -> None:
    """Bundles are laid out by that number, so it travels with every message."""
    assert HandshakeRequest().protocol_version == PROTOCOL_VERSION
    assert Response.of(FitResult(model="a/b")).protocol_version == PROTOCOL_VERSION


def test_an_operation_nobody_implements_is_not_a_request() -> None:
    with pytest.raises(ValidationError):
        parse_request({"protocol_version": PROTOCOL_VERSION, "operation": "levitate"})


def test_a_request_carrying_a_field_nobody_defined_is_refused() -> None:
    """Extra fields are refused rather than ignored: one side means something by them."""
    with pytest.raises(ValidationError):
        parse_request(
            {
                "protocol_version": PROTOCOL_VERSION,
                "operation": Operation.HANDSHAKE.value,
                "gpu": True,
            }
        )


def test_a_successful_response_carries_its_result() -> None:
    response = Response.of(HandshakeResult(provider="nixtla", provider_version="0.1.0"))

    assert response.is_ok
    assert response.status is Status.OK
    assert response.payload["provider"] == "nixtla"


def test_a_failed_response_says_why_and_has_no_result() -> None:
    response = Response.failed(ErrorCode.UNKNOWN_MODEL, "no such model", {"ref": "nixtla/x"})

    assert not response.is_ok
    assert response.error is not None
    assert response.error.code is ErrorCode.UNKNOWN_MODEL
    assert response.error.details == {"ref": "nixtla/x"}
    assert "UNKNOWN_MODEL" in str(response.error)

    with pytest.raises(ValueError, match=r"this response is a failure"):
        _ = response.payload


@pytest.mark.parametrize(
    "envelope",
    [
        {"status": "error"},
        {"status": "error", "result": {}, "error": {"code": "INTERNAL_ERROR", "message": "x"}},
        {"status": "ok", "error": {"code": "INTERNAL_ERROR", "message": "x"}},
    ],
    ids=["a failure that does not say why", "a failure with a result", "a success with an error"],
)
def test_an_envelope_that_is_ambiguous_about_success_is_refused(
    envelope: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        parse_response({"protocol_version": PROTOCOL_VERSION, **envelope})


def test_a_view_travels_as_a_kind_and_a_path_rather_than_as_data() -> None:
    """Control is JSON; a hundred thousand training sequences are not."""
    payload = json.loads(FitRequest(model="a/b", view=VIEW, into="/tmp/s").model_dump_json())

    assert payload["view"] == {"kind": "sequences", "path": "/tmp/openforecast/view"}
    assert len(json.dumps(payload)) < 300
