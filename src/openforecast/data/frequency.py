"""Sampling frequency as an OpenForecast-native concept.

A frequency is a unit plus a step count. Convenience strings such as ``"15m"``
parse into that representation; they are never stored, so no provider's spelling
of a frequency becomes OpenForecast's.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from openforecast.errors import FrequencyError

__all__ = ["Frequency", "FrequencyUnit"]


class FrequencyUnit(StrEnum):
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


# Units with a duration that does not depend on where on the calendar it falls.
_FIXED_DURATIONS: dict[FrequencyUnit, timedelta] = {
    FrequencyUnit.SECOND: timedelta(seconds=1),
    FrequencyUnit.MINUTE: timedelta(minutes=1),
    FrequencyUnit.HOUR: timedelta(hours=1),
    FrequencyUnit.DAY: timedelta(days=1),
    FrequencyUnit.WEEK: timedelta(weeks=1),
}

# Note that ``m`` means minutes and months are spelled ``mo``. Pandas resolves
# the same ambiguity the other way round; OpenForecast prefers the reading that
# is right far more often over compatibility with one library's convention.
_ALIASES: dict[str, FrequencyUnit] = {
    "s": FrequencyUnit.SECOND,
    "sec": FrequencyUnit.SECOND,
    "secs": FrequencyUnit.SECOND,
    "second": FrequencyUnit.SECOND,
    "seconds": FrequencyUnit.SECOND,
    "m": FrequencyUnit.MINUTE,
    "min": FrequencyUnit.MINUTE,
    "mins": FrequencyUnit.MINUTE,
    "minute": FrequencyUnit.MINUTE,
    "minutes": FrequencyUnit.MINUTE,
    "h": FrequencyUnit.HOUR,
    "hr": FrequencyUnit.HOUR,
    "hrs": FrequencyUnit.HOUR,
    "hour": FrequencyUnit.HOUR,
    "hours": FrequencyUnit.HOUR,
    "d": FrequencyUnit.DAY,
    "day": FrequencyUnit.DAY,
    "days": FrequencyUnit.DAY,
    "w": FrequencyUnit.WEEK,
    "week": FrequencyUnit.WEEK,
    "weeks": FrequencyUnit.WEEK,
    "mo": FrequencyUnit.MONTH,
    "mon": FrequencyUnit.MONTH,
    "mth": FrequencyUnit.MONTH,
    "month": FrequencyUnit.MONTH,
    "months": FrequencyUnit.MONTH,
}

_ABBREVIATIONS: dict[FrequencyUnit, str] = {
    FrequencyUnit.SECOND: "s",
    FrequencyUnit.MINUTE: "m",
    FrequencyUnit.HOUR: "h",
    FrequencyUnit.DAY: "d",
    FrequencyUnit.WEEK: "w",
    FrequencyUnit.MONTH: "mo",
}

_TOKEN = re.compile(r"^\s*(\d*)\s*([A-Za-z]+)\s*$")


class Frequency(BaseModel):
    """How far apart consecutive steps of a time axis are."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    unit: FrequencyUnit
    step: int = Field(default=1, ge=1)

    @classmethod
    def parse(cls, value: str | Frequency) -> Frequency:
        """Build a frequency from ``"15m"``, ``"1h"``, ``"2 weeks"``, ``"mo"``.

        A :class:`Frequency` passes through unchanged so that callers can accept
        either form without branching.
        """
        if isinstance(value, Frequency):
            return value
        match = _TOKEN.match(value)
        if match is None:
            raise FrequencyError(
                f"cannot parse frequency {value!r}; expected a count and a unit, e.g. '15m', '1h'"
            )
        digits, alias = match.groups()
        unit = _ALIASES.get(alias.lower())
        if unit is None:
            raise FrequencyError(
                f"unknown frequency unit {alias!r} in {value!r}; "
                f"known units: {', '.join(sorted(_ALIASES))}"
            )
        step = int(digits) if digits else 1
        if step < 1:
            raise FrequencyError(f"frequency step must be at least 1, got {value!r}")
        return cls(unit=unit, step=step)

    @field_validator("unit", mode="before")
    @classmethod
    def _accept_aliases(cls, value: object) -> object:
        """``Frequency(unit="h")`` should mean the same as ``Frequency.parse("h")``."""
        if isinstance(value, str) and value not in FrequencyUnit.__members__.values():
            return _ALIASES.get(value.lower(), value)
        return value

    @property
    def is_calendar(self) -> bool:
        """True when the step's duration depends on the calendar (months)."""
        return self.unit not in _FIXED_DURATIONS

    def as_timedelta(self) -> timedelta:
        """The step as a fixed duration.

        Raises :class:`FrequencyError` for calendar units, which have no single
        duration — use :meth:`steps_between` or :meth:`shift` instead.
        """
        base = _FIXED_DURATIONS.get(self.unit)
        if base is None:
            raise FrequencyError(f"{self} has no fixed duration; it is a calendar frequency")
        return base * self.step

    def steps_between(self, start: datetime, end: datetime) -> int | None:
        """How many whole steps separate ``start`` and ``end``.

        Returns ``None`` when ``end`` does not sit on the grid anchored at
        ``start``, which is how off-grid timestamps are detected.
        """
        if self.unit is FrequencyUnit.MONTH:
            if (end.day, end.time(), end.tzinfo) != (start.day, start.time(), start.tzinfo):
                return None
            months = (end.year - start.year) * 12 + (end.month - start.month)
            if months % self.step:
                return None
            return months // self.step
        delta = end - start
        period = self.as_timedelta()
        if delta % period:
            return None
        return delta // period

    def shift(self, moment: datetime, steps: int = 1) -> datetime:
        """``moment`` moved by ``steps`` steps, forwards or backwards."""
        if self.unit is not FrequencyUnit.MONTH:
            return moment + self.as_timedelta() * steps
        total = (moment.year * 12 + moment.month - 1) + steps * self.step
        year, month = divmod(total, 12)
        return moment.replace(year=year, month=month + 1)

    def __str__(self) -> str:
        return f"{self.step}{_ABBREVIATIONS[self.unit]}"
