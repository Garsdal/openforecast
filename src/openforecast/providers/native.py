"""Small, framework-neutral building blocks for reflective provider catalogs.

Forecasting libraries already standardize model construction inside their own
ecosystems.  Integrations should reuse that contract instead of copying every
constructor parameter into OpenForecast by hand.  This module contains only
the provider-authoring mechanics needed to do that: turn a native constructor
signature into JSON Schema, validate the JSON parameter bag against the same
declaration, and produce stable model-name slugs.

It deliberately knows nothing about sklearn, Darts, sktime, Nixtla or Chronos.
Each integration decides which classes satisfy its execution protocol, which
parameters name OpenForecast-owned concepts, and which capabilities are safe to
advertise.  Reflection removes repetition; it does not guess semantics.
"""

from __future__ import annotations

import inspect
import json
import re
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Union, cast, get_args, get_origin, get_type_hints

from openforecast.errors import RecipeError

__all__ = [
    "Parameter",
    "ParameterDiscovery",
    "checked",
    "class_slug",
    "named",
    "parameters_from_signature",
    "schema_of",
]

_JSON_TYPES: Mapping[type, str] = {
    bool: "boolean",
    int: "integer",
    float: "number",
    str: "string",
}

_MISSING = object()


@dataclass(frozen=True)
class Parameter:
    """One JSON-representable native constructor parameter.

    The first six fields preserve the compact declaration used by the existing
    integrations.  ``json_schema`` and ``required`` are what reflective
    catalogs add for container-valued and required native parameters.
    """

    name: str
    kind: type[int] | type[bool] | type[str] | type[float] | None
    description: str
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[Any, ...] = ()
    json_schema: Mapping[str, Any] | None = None
    required: bool = False
    default: Any = _MISSING

    @property
    def json_type(self) -> str:
        if self.kind is None:
            return "JSON value"
        return _JSON_TYPES[self.kind]

    def schema(self) -> dict[str, Any]:
        schema = dict(self.json_schema or {})
        if self.kind is not None:
            schema.setdefault("type", self.json_type)
        schema.setdefault("description", self.description)
        if self.minimum is not None:
            schema["minimum"] = self.minimum
        if self.maximum is not None:
            schema["maximum"] = self.maximum
        if self.choices:
            schema["enum"] = list(self.choices)
        if self.default is not _MISSING and _is_json(self.default):
            schema["default"] = self.default
        return schema

    def check(self, value: object, model: str) -> None:
        schema = self.schema()
        expected = schema.get("type")
        if expected is not None and not _matches_json_type(value, expected):
            raise RecipeError(
                f"{model} takes {self.name} as {expected} ({self.description}); got {value!r}"
            )
        numeric = value if isinstance(value, int | float) and not isinstance(value, bool) else None
        if self.minimum is not None and numeric is not None and numeric < self.minimum:
            raise RecipeError(
                f"{model} takes {self.name} of at least {self.minimum}; got {value!r}"
            )
        if self.maximum is not None and numeric is not None and numeric > self.maximum:
            raise RecipeError(
                f"{model} takes {self.name} of at most {self.maximum}; got {value!r}"
            )
        if self.choices and value not in self.choices:
            raise RecipeError(f"{model} takes {self.name} in {list(self.choices)}; got {value!r}")


@dataclass(frozen=True)
class ParameterDiscovery:
    """The parameters reflection can expose and required ones it cannot encode."""

    parameters: tuple[Parameter, ...]
    unsupported_required: tuple[str, ...] = ()

    @property
    def is_constructible(self) -> bool:
        return not self.unsupported_required


def parameters_from_signature(
    constructor: Any,
    *,
    exclude: Sequence[str] = (),
    descriptions: Mapping[str, str] | None = None,
) -> ParameterDiscovery:
    """Infer the JSON-safe portion of a native constructor's public signature.

    Optional opaque Python objects are omitted: callbacks, estimators, losses
    and similar values cannot survive a recipe or the provider protocol.
    A required opaque object makes the model non-constructible through the
    generic path and is reported to the integration so it can skip that class.
    """
    signature_source = constructor.__init__ if inspect.isclass(constructor) else constructor
    try:
        signature = inspect.signature(signature_source)
    except (TypeError, ValueError):
        return ParameterDiscovery((), ("<signature>",))

    excluded = set(exclude)
    prose = descriptions or {}
    hint_source = signature_source
    try:
        hints = get_type_hints(hint_source)
    except (NameError, TypeError):
        hints = {}
    found: list[Parameter] = []
    unsupported: list[str] = []
    for name, native in signature.parameters.items():
        if name in excluded or name in {"self", "cls"}:
            continue
        if native.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue
        required = native.default is inspect.Parameter.empty
        default = _MISSING if required else native.default
        schema = _schema_for(hints.get(name, native.annotation), default)
        if schema is None:
            if required:
                unsupported.append(name)
            continue
        kind = _kind_for_schema(schema)
        description = prose.get(name, f"Native {name!r} constructor parameter.")
        choices = tuple(schema.pop("enum", ()))
        found.append(
            Parameter(
                name=name,
                kind=kind,
                description=description,
                choices=choices,
                json_schema=schema,
                required=required,
                default=default,
            )
        )
    return ParameterDiscovery(tuple(found), tuple(unsupported))


def schema_of(parameters: Mapping[str, Parameter]) -> dict[str, Any]:
    """The JSON Schema for a model's whole native parameter bag."""
    required = [name for name, parameter in parameters.items() if parameter.required]
    schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {name: parameter.schema() for name, parameter in parameters.items()},
    }
    if required:
        schema["required"] = required
    # Descriptor schemas cross JSON transports. Normalize tuples and enum
    # values now so in-process and subprocess handshakes compare identically.
    return json.loads(json.dumps(schema, allow_nan=False))


def checked(
    params: Mapping[str, Any], declared: Mapping[str, Parameter], model: str
) -> dict[str, Any]:
    """Validate a JSON parameter bag against its reflected declaration."""
    unknown = sorted(set(params) - set(declared))
    if unknown:
        raise RecipeError(f"{model} takes no parameter {unknown}; it takes {sorted(declared)}")
    missing = sorted(
        name
        for name, parameter in declared.items()
        if parameter.required and name not in params
    )
    if missing:
        raise RecipeError(f"{model} requires parameter {missing}")
    for name, value in params.items():
        declared[name].check(value, model)
    return dict(params)


def named(parameters: Sequence[Parameter]) -> dict[str, Parameter]:
    return {parameter.name: parameter for parameter in parameters}


def class_slug(name: str, *, suffixes: Sequence[str] = ()) -> str:
    """Turn a native class name into a stable lowercase model-name segment."""
    stem = name
    for suffix in suffixes:
        if stem.endswith(suffix) and len(stem) > len(suffix):
            stem = stem[: -len(suffix)]
            break
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", stem)
    words = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1-\2", words)
    return words.replace("_", "-").lower()


def _schema_for(annotation: Any, default: Any) -> dict[str, Any] | None:
    if annotation is inspect.Parameter.empty or isinstance(annotation, str):
        return _schema_from_default(default)
    if annotation is Any:
        return _schema_from_default(default)
    if annotation in _JSON_TYPES:
        return {"type": _JSON_TYPES[annotation]}
    if inspect.isclass(annotation) and issubclass(annotation, Enum):
        values = [member.value for member in annotation]
        return {"enum": values} if all(_is_json(value) for value in values) else None

    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Literal:
        return {"enum": list(args)} if all(_is_json(value) for value in args) else None
    if origin in {Union, types.UnionType}:
        alternatives = [_schema_for(arg, _MISSING) for arg in args if arg is not type(None)]
        alternatives = [schema for schema in alternatives if schema is not None]
        if len(alternatives) == 1:
            return alternatives[0]
        return {"anyOf": alternatives} if alternatives else _schema_from_default(default)
    if origin in {list, tuple, set, frozenset, Sequence}:
        item = _schema_for(args[0], _MISSING) if args else {}
        return {"type": "array", "items": item or {}}
    if origin in {dict, Mapping}:
        value = _schema_for(args[1], _MISSING) if len(args) > 1 else {}
        return {"type": "object", "additionalProperties": value or {}}
    return _schema_from_default(default)


def _schema_from_default(default: Any) -> dict[str, Any] | None:
    if default is _MISSING:
        return None
    if default is None:
        return None
    if isinstance(default, bool):
        return {"type": "boolean"}
    if isinstance(default, int):
        return {"type": "integer"}
    if isinstance(default, float):
        return {"type": "number"}
    if isinstance(default, str):
        return {"type": "string"}
    if isinstance(default, (list, tuple)):
        values = list(cast(Sequence[Any], default))
        if _is_json(values):
            return {"type": "array"}
    if isinstance(default, dict) and _is_json(default):
        return {"type": "object"}
    return None


def _kind_for_schema(schema: Mapping[str, Any]) -> type | None:
    reverse = {value: key for key, value in _JSON_TYPES.items()}
    schema_type = schema.get("type")
    return reverse.get(schema_type) if isinstance(schema_type, str) else None


def _matches_json_type(value: object, expected: object) -> bool:
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "array":
        return isinstance(value, list | tuple)
    if expected == "object":
        return isinstance(value, Mapping)
    return True


def _is_json(value: Any) -> bool:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return False
    return True
