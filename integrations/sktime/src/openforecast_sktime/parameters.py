"""A native model's parameters, declared once as both a schema and a check.

```python
Parameter("sp", int, "Steps in one seasonal period.", minimum=1)
```

Every sktime forecaster takes a different bag of keyword arguments, and a caller
needs two things from an integration about each of them: what may be passed, and
why what they passed was refused. Declaring the parameter once and deriving both
from it is what stops the JSON Schema in the descriptor and the validation in
the adapter from drifting apart — a parameter that is documented is one that is
accepted, and a parameter that is refused is refused by name.

Shared by the adapters rather than living with one of them: the declaration is
about how OpenForecast exposes a native parameter, which is the same question
whether the model behind it is fitted per series or pooled across a panel.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from openforecast.errors import RecipeError

__all__ = ["Parameter", "checked", "named", "schema_of"]

_JSON_TYPES: Mapping[type, str] = {
    bool: "boolean",
    int: "integer",
    float: "number",
    str: "string",
}


@dataclass(frozen=True)
class Parameter:
    """One parameter of a native model, as both a schema and a check.

    Declared once so that the JSON Schema a caller reads and the validation a
    caller hits cannot disagree: an unknown parameter is refused by name and a
    wrongly typed one is refused with the type it should have been.
    """

    name: str
    kind: type[int] | type[bool] | type[str] | type[float]
    description: str
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()

    @property
    def json_type(self) -> str:
        return _JSON_TYPES[self.kind]

    def schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {"type": self.json_type, "description": self.description}
        if self.minimum is not None:
            schema["minimum"] = self.minimum
        if self.maximum is not None:
            schema["maximum"] = self.maximum
        if self.choices:
            schema["enum"] = list(self.choices)
        return schema

    def check(self, value: object, model: str) -> None:
        """Refuse a value this parameter cannot take, naming what it can."""
        # ``bool`` is an ``int`` in Python and is not one here: a model taking a
        # count and given ``True`` has been handed the wrong thing. An ``int``
        # where a ``float`` is declared is the one widening that is safe.
        accepted: tuple[type, ...] = (int, float) if self.kind is float else (self.kind,)
        wrong_type = not isinstance(value, accepted) or (
            self.kind is not bool and isinstance(value, bool)
        )
        if wrong_type:
            raise RecipeError(
                f"{model} takes {self.name} as {self.json_type} ({self.description}); got {value!r}"
            )
        numeric = value if isinstance(value, int | float) else None
        if self.minimum is not None and numeric is not None and numeric < self.minimum:
            raise RecipeError(
                f"{model} takes {self.name} of at least {self.minimum}; got {value!r}"
            )
        if self.maximum is not None and numeric is not None and numeric > self.maximum:
            raise RecipeError(f"{model} takes {self.name} of at most {self.maximum}; got {value!r}")
        if self.choices and value not in self.choices:
            raise RecipeError(f"{model} takes {self.name} in {list(self.choices)}; got {value!r}")


def schema_of(parameters: Mapping[str, Parameter]) -> dict[str, Any]:
    """The JSON Schema of a model's whole parameter bag."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {name: parameter.schema() for name, parameter in parameters.items()},
    }


def checked(
    params: Mapping[str, Any], declared: Mapping[str, Parameter], model: str
) -> dict[str, Any]:
    """``params``, once every one of them is a parameter ``model`` actually takes.

    An unknown name is refused rather than forwarded: a typo that reaches a
    native constructor as an ignored keyword is a parameter the caller believes
    took effect and did not.
    """
    unknown = sorted(set(params) - set(declared))
    if unknown:
        raise RecipeError(f"{model} takes no parameter {unknown}; it takes {sorted(declared)}")
    for name, value in params.items():
        declared[name].check(value, model)
    return dict(params)


def named(parameters: Sequence[Parameter]) -> dict[str, Parameter]:
    """The declared parameters, by name, in the order they were declared."""
    return {parameter.name: parameter for parameter in parameters}
