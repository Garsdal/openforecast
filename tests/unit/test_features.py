from __future__ import annotations

import pytest
from pydantic import ValidationError

from openforecast import FeatureAvailability, FeatureKind, FeatureSpec, SchemaError


def test_observed_feature() -> None:
    feature = FeatureSpec.observed("temperature_actual")
    assert feature.kind is FeatureKind.TEMPORAL
    assert feature.availability is FeatureAvailability.OBSERVED
    assert feature.is_temporal and feature.is_observed
    assert not feature.is_static and not feature.is_known


def test_known_feature() -> None:
    feature = FeatureSpec.known("temperature_forecast")
    assert feature.kind is FeatureKind.TEMPORAL
    assert feature.availability is FeatureAvailability.KNOWN
    assert feature.is_known and not feature.is_observed


def test_static_feature_carries_no_availability() -> None:
    feature = FeatureSpec.static("capacity")
    assert feature.kind is FeatureKind.STATIC
    assert feature.availability is None
    assert feature.is_static and not feature.is_temporal


def test_temporal_feature_requires_an_availability() -> None:
    with pytest.raises(SchemaError, match="must declare an availability"):
        FeatureSpec(name="temperature", kind=FeatureKind.TEMPORAL)


def test_temporal_is_the_default_kind_and_still_requires_availability() -> None:
    with pytest.raises(SchemaError, match="must declare an availability"):
        FeatureSpec(name="temperature")


def test_static_feature_must_not_declare_an_availability() -> None:
    with pytest.raises(SchemaError, match="must not declare an availability"):
        FeatureSpec(
            name="capacity",
            kind=FeatureKind.STATIC,
            availability=FeatureAvailability.KNOWN,
        )


def test_name_must_not_be_empty() -> None:
    with pytest.raises(SchemaError, match="must not be empty"):
        FeatureSpec.known("   ")


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        FeatureSpec(name="x", availability=FeatureAvailability.KNOWN, lag=1)  # type: ignore[call-arg]


def test_is_frozen() -> None:
    feature = FeatureSpec.known("x")
    with pytest.raises(ValidationError):
        feature.name = "y"


def test_serializes_to_the_declared_vocabulary() -> None:
    assert FeatureSpec.known("wind").model_dump() == {
        "name": "wind",
        "kind": "temporal",
        "availability": "known",
    }
