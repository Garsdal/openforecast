from __future__ import annotations

import pytest

from openforecast import (
    FeatureSpec,
    Frequency,
    FrequencyUnit,
    SchemaError,
    TimeSeriesSchema,
)


def schema(**overrides: object) -> TimeSeriesSchema:
    fields: dict[str, object] = {
        "time": "timestamp",
        "frequency": "1h",
        "targets": ("load",),
    }
    fields.update(overrides)
    return TimeSeriesSchema(**fields)  # type: ignore[arg-type]


def test_frequency_accepts_a_convenience_string_and_stores_native_semantics() -> None:
    assert schema(frequency="15m").frequency == Frequency(unit=FrequencyUnit.MINUTE, step=15)


def test_single_univariate_shape() -> None:
    single = schema()
    assert not single.is_panel
    assert single.is_univariate
    assert not single.is_multivariate
    assert single.target_count == 1


def test_panel_multivariate_shape() -> None:
    panel = schema(instance_keys=("country",), targets=("load", "price"))
    assert panel.is_panel
    assert panel.is_multivariate
    assert not panel.is_univariate
    assert panel.target_count == 2


def test_feature_groups_are_derived_from_the_specs() -> None:
    described = schema(
        features=(
            FeatureSpec.observed("temperature_actual"),
            FeatureSpec.known("temperature_forecast"),
            FeatureSpec.static("capacity"),
        )
    )
    assert described.feature_names == ("temperature_actual", "temperature_forecast", "capacity")
    assert [feature.name for feature in described.observed_features] == ["temperature_actual"]
    assert [feature.name for feature in described.known_features] == ["temperature_forecast"]
    assert [feature.name for feature in described.static_features] == ["capacity"]
    assert [feature.name for feature in described.temporal_features] == [
        "temperature_actual",
        "temperature_forecast",
    ]
    assert described.has_observed_features
    assert described.has_known_features
    assert described.has_static_features


def test_a_schema_without_features_has_no_feature_groups() -> None:
    bare = schema()
    assert bare.features == ()
    assert not bare.has_observed_features
    assert not bare.has_known_features
    assert not bare.has_static_features


def test_canonical_column_layouts() -> None:
    described = schema(
        instance_keys=("country", "zone"),
        targets=("load", "price"),
        features=(
            FeatureSpec.known("temperature_forecast"),
            FeatureSpec.observed("temperature_actual"),
            FeatureSpec.static("capacity"),
        ),
    )
    assert described.history_columns == (
        "country",
        "zone",
        "timestamp",
        "load",
        "price",
        "temperature_actual",
        "temperature_forecast",
    )
    assert described.future_columns == ("country", "zone", "timestamp", "temperature_forecast")
    assert described.static_columns == ("country", "zone", "capacity")


def test_at_least_one_target_is_required() -> None:
    with pytest.raises(SchemaError, match="at least one target"):
        schema(targets=())


def test_targets_must_be_unique() -> None:
    with pytest.raises(SchemaError, match="duplicate target names"):
        schema(targets=("load", "load"))


def test_features_must_be_unique() -> None:
    with pytest.raises(SchemaError, match="duplicate feature names"):
        schema(features=(FeatureSpec.known("wind"), FeatureSpec.observed("wind")))


def test_instance_keys_must_be_unique() -> None:
    with pytest.raises(SchemaError, match="duplicate instance key names"):
        schema(instance_keys=("country", "country"))


def test_a_target_cannot_also_be_a_feature() -> None:
    with pytest.raises(SchemaError, match="both a target and a feature"):
        schema(targets=("load",), features=(FeatureSpec.known("load"),))


@pytest.mark.parametrize(
    "overrides",
    [
        {"targets": ("timestamp",)},
        {"features": (FeatureSpec.known("timestamp"),)},
        {"instance_keys": ("timestamp",)},
    ],
)
def test_the_time_column_cannot_hold_another_role(overrides: dict[str, object]) -> None:
    with pytest.raises(SchemaError, match="time column"):
        schema(**overrides)


def test_an_instance_key_cannot_also_be_a_target_or_feature() -> None:
    with pytest.raises(SchemaError, match="instance key and a target or feature"):
        schema(instance_keys=("load",))


def test_column_names_must_not_be_empty() -> None:
    with pytest.raises(SchemaError, match="must not be empty"):
        schema(targets=("load", " "))


def test_json_round_trip() -> None:
    described = schema(
        instance_keys=("country",),
        targets=("load", "price"),
        frequency="15m",
        features=(FeatureSpec.observed("temperature_actual"), FeatureSpec.static("capacity")),
    )
    assert TimeSeriesSchema.model_validate_json(described.model_dump_json()) == described


def test_is_frozen() -> None:
    described = schema()
    with pytest.raises(Exception, match="frozen"):
        described.time = "other"
