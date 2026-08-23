from __future__ import annotations

from typing import Literal

import pytest

from openforecast.errors import RecipeError
from openforecast.providers.native import (
    checked,
    class_slug,
    named,
    parameters_from_signature,
    schema_of,
)

DEFAULT_LAYERS = (8, 8)


class NativeEstimator:
    def __init__(
        self,
        *,
        count: int,
        rate: float = 0.1,
        mode: Literal["fast", "exact"] = "fast",
        layers: tuple[int, ...] = DEFAULT_LAYERS,
        callback: object | None = None,
        random_state: int | None = None,
    ) -> None:
        pass


def test_a_signature_becomes_one_closed_parameter_schema() -> None:
    discovery = parameters_from_signature(
        NativeEstimator,
        exclude=("random_state",),
    )
    declared = named(discovery.parameters)

    assert discovery.is_constructible
    assert tuple(declared) == ("count", "rate", "mode", "layers")
    assert schema_of(declared) == {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "count": {
                "type": "integer",
                "description": "Native 'count' constructor parameter.",
            },
            "rate": {
                "type": "number",
                "description": "Native 'rate' constructor parameter.",
                "default": 0.1,
            },
            "mode": {
                "description": "Native 'mode' constructor parameter.",
                "enum": ["fast", "exact"],
                "default": "fast",
            },
            "layers": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Native 'layers' constructor parameter.",
                "default": list(DEFAULT_LAYERS),
            },
        },
        "required": ["count"],
    }


def test_the_same_declaration_validates_before_native_construction() -> None:
    declared = named(parameters_from_signature(NativeEstimator).parameters)

    assert checked(
        {"count": 2, "mode": "exact", "layers": [4]}, declared, "native"
    ) == {"count": 2, "mode": "exact", "layers": [4]}
    with pytest.raises(RecipeError, match="requires parameter"):
        checked({}, declared, "native")
    with pytest.raises(RecipeError, match="takes no parameter"):
        checked({"count": 2, "typo": True}, declared, "native")
    with pytest.raises(RecipeError, match="takes mode"):
        checked({"count": 2, "mode": "wrong"}, declared, "native")


def test_an_opaque_required_object_marks_a_model_unavailable_to_reflection() -> None:
    class MetaEstimator:
        def __init__(self, estimator: object) -> None:
            pass

    discovery = parameters_from_signature(MetaEstimator)

    assert not discovery.is_constructible
    assert discovery.unsupported_required == ("estimator",)


@pytest.mark.parametrize(
    ("native", "suffixes", "expected"),
    [
        ("RandomForestRegressor", ("Regressor",), "random-forest"),
        ("ThetaForecaster", ("Forecaster",), "theta"),
        ("HistGradientBoostingRegressor", ("Regressor",), "hist-gradient-boosting"),
    ],
)
def test_native_class_names_have_stable_model_slugs(
    native: str, suffixes: tuple[str, ...], expected: str
) -> None:
    assert class_slug(native, suffixes=suffixes) == expected
