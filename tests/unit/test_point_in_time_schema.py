from __future__ import annotations

import pytest
from pydantic import ValidationError

from openforecast import (
    FeatureSpec,
    Frequency,
    FrequencyUnit,
    PointInTimeSchema,
    SchemaError,
)


def schema(**overrides: object) -> PointInTimeSchema:
    fields: dict[str, object] = {
        "origin_time": "ref_time",
        "event_time": "target_time",
        "event_frequency": "1h",
        "instance_keys": ("zone",),
        "features": (FeatureSpec.known("wind_fc"),),
    }
    fields.update(overrides)
    return PointInTimeSchema.model_validate(fields)


def test_frequency_strings_are_parsed_into_native_semantics() -> None:
    parsed = schema(event_frequency="15m", origin_frequency="6h")
    assert parsed.event_frequency == Frequency(unit=FrequencyUnit.MINUTE, step=15)
    assert parsed.origin_frequency == Frequency(unit=FrequencyUnit.HOUR, step=6)


def test_origin_frequency_is_optional() -> None:
    """Vintages are often irregular; declaring a grid opts into validating it."""
    assert schema().origin_frequency is None


def test_the_two_time_axes_must_be_different_columns() -> None:
    with pytest.raises(SchemaError, match="must be different columns"):
        schema(event_time="ref_time")


@pytest.mark.parametrize("axis", ["origin_time", "event_time"])
def test_a_time_axis_cannot_also_be_a_feature(axis: str) -> None:
    with pytest.raises(SchemaError, match="cannot also be a feature"):
        schema(**{axis: "wind_fc"})


@pytest.mark.parametrize("axis", ["origin_time", "event_time"])
def test_a_time_axis_cannot_also_be_an_instance_key(axis: str) -> None:
    with pytest.raises(SchemaError, match="cannot also be an instance key"):
        schema(**{axis: "zone"})


def test_an_instance_key_cannot_also_be_a_feature() -> None:
    with pytest.raises(SchemaError, match="both an instance key and a feature"):
        schema(features=(FeatureSpec.known("zone"),))


def test_duplicate_feature_names_are_rejected() -> None:
    with pytest.raises(SchemaError, match="duplicate feature names"):
        schema(features=(FeatureSpec.known("wind_fc"), FeatureSpec.observed("wind_fc")))


def test_duplicate_instance_keys_are_rejected() -> None:
    with pytest.raises(SchemaError, match="duplicate instance key names"):
        schema(instance_keys=("zone", "zone"))


def test_a_schema_without_features_carries_no_information() -> None:
    with pytest.raises(SchemaError, match="at least one feature"):
        schema(features=())


def test_static_features_belong_to_the_truth_frame() -> None:
    """A value that varies with neither axis has no place in a vintage table."""
    with pytest.raises(SchemaError, match="cannot hold static features"):
        schema(features=(FeatureSpec.known("wind_fc"), FeatureSpec.static("capacity")))


def test_empty_column_names_are_rejected() -> None:
    with pytest.raises(SchemaError, match="must not be empty"):
        schema(origin_time="  ")


def test_there_are_no_targets_on_a_point_in_time_schema() -> None:
    """Information and outcome are separate; targets live on the truth frame."""
    assert "targets" not in PointInTimeSchema.model_fields
    with pytest.raises(ValidationError):
        schema(targets=("price",))


def test_feature_groups_are_derived_from_the_specs() -> None:
    parsed = schema(
        features=(
            FeatureSpec.observed("load_actual"),
            FeatureSpec.known("wind_fc"),
            FeatureSpec.known("solar_fc"),
        )
    )
    assert parsed.feature_names == ("load_actual", "wind_fc", "solar_fc")
    assert [feature.name for feature in parsed.observed_features] == ["load_actual"]
    assert [feature.name for feature in parsed.known_features] == ["wind_fc", "solar_fc"]
    assert parsed.has_observed_features
    assert parsed.has_known_features


def test_the_canonical_layout_puts_the_keys_first_then_observed_then_known() -> None:
    parsed = schema(
        features=(
            FeatureSpec.known("wind_fc"),
            FeatureSpec.observed("load_actual"),
        )
    )
    assert parsed.columns == ("zone", "ref_time", "target_time", "load_actual", "wind_fc")
    assert parsed.key_columns == ("zone", "ref_time", "target_time")


def test_is_panel_is_derived_from_the_instance_keys() -> None:
    assert schema().is_panel
    assert not schema(instance_keys=()).is_panel


def test_with_features_revalidates() -> None:
    parsed = schema()
    grown = parsed.with_features(FeatureSpec.known("lead_time"))
    assert grown.feature_names == ("wind_fc", "lead_time")
    with pytest.raises(SchemaError, match="duplicate feature names"):
        parsed.with_features(FeatureSpec.known("wind_fc"))


def test_the_schema_is_frozen_and_rejects_unknown_fields() -> None:
    parsed = schema()
    with pytest.raises(ValidationError):
        parsed.origin_time = "other"
    with pytest.raises(ValidationError):
        schema(lead_time="lead")


def test_round_trips_through_json() -> None:
    parsed = schema(origin_frequency="1h", features=(FeatureSpec.observed("load_actual"),))
    assert PointInTimeSchema.model_validate_json(parsed.model_dump_json()) == parsed
