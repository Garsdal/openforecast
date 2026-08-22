"""Which forecast origins a fit learns from.

```python
of.AllOrigins(stride=1)
of.LatestOrigin()
of.AtOrigin(timestamp)
of.OriginsBetween(start, end, stride=12)
```

The same four selections mean the same thing for both semantic sources, which is
what keeps point-in-time out of the public API surface:

```text
TimeSeriesFrame   the origins that can be simulated by cutting windows
ForecastDataset   the vintages that actually exist
```

Each selection is its own type rather than one model with a mode flag and four
optional fields, because the fields are not shared: a stride is meaningless for
``LatestOrigin``, and a range that ``AtOrigin`` would have to reject cannot be
written in the first place. They form a discriminated union, so a plan
round-trips through JSON as ``{"mode": "between", ...}`` without the reader
having to guess which shape it is holding.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from openforecast.data._arrow import summarize
from openforecast.data.point_in_time import resolve_origin
from openforecast.errors import DataError, RecipeError

__all__ = [
    "AllOrigins",
    "AtOrigin",
    "LatestOrigin",
    "OriginMode",
    "OriginSelection",
    "OriginsBetween",
]


class OriginMode(StrEnum):
    """The discriminator of the four selections, and their wire spelling."""

    ALL = "all"
    LATEST = "latest"
    AT = "at"
    BETWEEN = "between"


class _Selection(BaseModel, ABC):
    """What the four share: immutability, and how to resolve themselves.

    The ``mode`` tag is declared by each subclass rather than here, because it
    is the discriminator: the whole point is that every subclass narrows it to
    exactly one value.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    @abstractmethod
    def select(self, available: Sequence[datetime]) -> tuple[datetime, ...]:
        """The chosen origins, in ascending order."""
        raise NotImplementedError

    @staticmethod
    def _ordered(available: Sequence[datetime]) -> list[datetime]:
        ordered = sorted(set(available))
        if not ordered:
            raise DataError("the data holds no forecast origins")
        return ordered


class AllOrigins(_Selection):
    """Every origin the data holds, optionally thinned by ``stride``.

    A stride of 12 on hourly vintages trains on two origins a day. It thins the
    samples rather than the horizon each sample covers, so the sequences stay
    exactly as long — this is a way to spend less compute on highly overlapping
    origins, not a way to change what a sample means.
    """

    mode: Literal[OriginMode.ALL] = OriginMode.ALL
    stride: int = Field(default=1, ge=1)

    def select(self, available: Sequence[datetime]) -> tuple[datetime, ...]:
        return tuple(self._ordered(available)[:: self.stride])


class LatestOrigin(_Selection):
    """Only the freshest origin — the one production inference would run at."""

    mode: Literal[OriginMode.LATEST] = OriginMode.LATEST

    def select(self, available: Sequence[datetime]) -> tuple[datetime, ...]:
        return (self._ordered(available)[-1],)


class AtOrigin(_Selection):
    """Exactly one named origin.

    Matched exactly rather than approximately. Answering for 10:00 when 11:00
    was asked for would train the model on a vintage the caller never named.
    """

    mode: Literal[OriginMode.AT] = OriginMode.AT
    origin: datetime

    def __init__(self, origin: datetime | str | None = None, /, **data: Any) -> None:
        """``AtOrigin(t)`` as well as ``AtOrigin(origin=t)``."""
        super().__init__(**_positional({"origin": origin}, data))

    def select(self, available: Sequence[datetime]) -> tuple[datetime, ...]:
        return (resolve_origin(self.origin, self._ordered(available)),)


class OriginsBetween(_Selection):
    """Every origin in a closed interval, optionally thinned by ``stride``.

    The bounds are inclusive, and an interval the data does not cover is an
    error rather than an empty fit: it almost always means the caller is
    reasoning about a different dataset than the one they passed.
    """

    mode: Literal[OriginMode.BETWEEN] = OriginMode.BETWEEN
    start: datetime
    end: datetime
    stride: int = Field(default=1, ge=1)

    def __init__(
        self,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
        /,
        **data: Any,
    ) -> None:
        """``OriginsBetween(start, end, stride=12)`` as well as all-keyword form."""
        super().__init__(**_positional({"start": start, "end": end}, data))

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        if self.end < self.start:
            raise RecipeError(
                f"the range of origins ends before it starts: "
                f"{self.start.isoformat()} .. {self.end.isoformat()}"
            )
        return self

    def select(self, available: Sequence[datetime]) -> tuple[datetime, ...]:
        ordered = self._ordered(available)
        within = [moment for moment in ordered if self.start <= moment <= self.end]
        if not within:
            raise DataError(
                f"no origin between {self.start.isoformat()} and {self.end.isoformat()}; "
                f"available: {summarize(ordered)}"
            )
        return tuple(within[:: self.stride])


#: Any of the four. Annotated with the discriminator so that a serialized
#: selection deserializes back into the same type it was written from.
OriginSelection = Annotated[
    AllOrigins | LatestOrigin | AtOrigin | OriginsBetween,
    Field(discriminator="mode"),
]


def _positional(named: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    """Fold positional arguments into the keyword ones, refusing to shadow.

    ``AtOrigin(t, origin=u)`` names one origin twice and cannot be resolved by
    picking a winner, so it is rejected rather than silently favouring either.
    """
    for name, value in named.items():
        if value is None:
            continue
        if name in data:
            raise RecipeError(f"{name} was given both positionally and by keyword")
        data[name] = value
    return data
