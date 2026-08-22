from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from openforecast import Frequency, FrequencyError, FrequencyUnit


@pytest.mark.parametrize(
    ("text", "unit", "step"),
    [
        ("15m", FrequencyUnit.MINUTE, 15),
        ("1h", FrequencyUnit.HOUR, 1),
        ("h", FrequencyUnit.HOUR, 1),
        ("30s", FrequencyUnit.SECOND, 30),
        ("2 weeks", FrequencyUnit.WEEK, 2),
        (" 7d ", FrequencyUnit.DAY, 7),
        ("mo", FrequencyUnit.MONTH, 1),
        ("3months", FrequencyUnit.MONTH, 3),
        ("1H", FrequencyUnit.HOUR, 1),
        ("15MIN", FrequencyUnit.MINUTE, 15),
    ],
)
def test_parse_accepts_convenience_strings(text: str, unit: FrequencyUnit, step: int) -> None:
    assert Frequency.parse(text) == Frequency(unit=unit, step=step)


def test_lowercase_m_is_minutes_and_mo_is_months() -> None:
    """OpenForecast resolves the pandas ``M`` ambiguity towards the common case."""
    assert Frequency.parse("M").unit is FrequencyUnit.MINUTE
    assert Frequency.parse("1mo").unit is FrequencyUnit.MONTH


def test_parse_passes_a_frequency_through() -> None:
    frequency = Frequency(unit=FrequencyUnit.HOUR, step=6)
    assert Frequency.parse(frequency) is frequency


@pytest.mark.parametrize("text", ["", "1", "  ", "1.5h", "h1", "3 fortnights", "-1h", "0h", "0"])
def test_parse_rejects_nonsense(text: str) -> None:
    with pytest.raises(FrequencyError):
        Frequency.parse(text)


def test_constructor_accepts_unit_aliases() -> None:
    assert Frequency(unit="h").unit is FrequencyUnit.HOUR  # type: ignore[arg-type]
    assert Frequency(unit="minutes", step=5).unit is FrequencyUnit.MINUTE  # type: ignore[arg-type]


def test_step_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Frequency(unit=FrequencyUnit.HOUR, step=0)


def test_is_frozen() -> None:
    frequency = Frequency.parse("1h")
    with pytest.raises(ValidationError):
        frequency.step = 2


@pytest.mark.parametrize("text", ["30s", "15m", "1h", "6h", "1d", "2w", "1mo", "3mo"])
def test_str_round_trips_through_parse(text: str) -> None:
    frequency = Frequency.parse(text)
    assert Frequency.parse(str(frequency)) == frequency


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("30s", timedelta(seconds=30)),
        ("15m", timedelta(minutes=15)),
        ("1h", timedelta(hours=1)),
        ("1d", timedelta(days=1)),
        ("2w", timedelta(weeks=2)),
    ],
)
def test_fixed_units_have_a_duration(text: str, expected: timedelta) -> None:
    frequency = Frequency.parse(text)
    assert not frequency.is_calendar
    assert frequency.as_timedelta() == expected


def test_months_are_a_calendar_frequency() -> None:
    frequency = Frequency.parse("1mo")
    assert frequency.is_calendar
    with pytest.raises(FrequencyError):
        frequency.as_timedelta()


def test_steps_between_counts_whole_steps() -> None:
    frequency = Frequency.parse("15m")
    start = datetime(2026, 1, 1)
    assert frequency.steps_between(start, start) == 0
    assert frequency.steps_between(start, start + timedelta(hours=1)) == 4


def test_steps_between_rejects_off_grid_moments() -> None:
    frequency = Frequency.parse("15m")
    start = datetime(2026, 1, 1)
    assert frequency.steps_between(start, start + timedelta(minutes=7)) is None


def test_steps_between_handles_months() -> None:
    frequency = Frequency.parse("3mo")
    start = datetime(2026, 1, 15, 6, 30)
    assert frequency.steps_between(start, datetime(2026, 4, 15, 6, 30)) == 1
    assert frequency.steps_between(start, datetime(2026, 3, 15, 6, 30)) is None
    assert frequency.steps_between(start, datetime(2026, 4, 16, 6, 30)) is None
    assert frequency.steps_between(start, datetime(2026, 4, 15, 7, 30)) is None


def test_steps_between_is_timezone_aware() -> None:
    frequency = Frequency.parse("1h")
    start = datetime(2026, 1, 1, tzinfo=UTC)
    assert frequency.steps_between(start, datetime(2026, 1, 1, 5, tzinfo=UTC)) == 5


@pytest.mark.parametrize(
    ("text", "steps", "expected"),
    [
        ("1h", 3, datetime(2026, 1, 1, 3)),
        ("1h", -1, datetime(2025, 12, 31, 23)),
        ("15m", 2, datetime(2026, 1, 1, 0, 30)),
        ("1mo", 1, datetime(2026, 2, 1)),
        ("1mo", 12, datetime(2027, 1, 1)),
        ("2mo", -1, datetime(2025, 11, 1)),
    ],
)
def test_shift_moves_along_the_grid(text: str, steps: int, expected: datetime) -> None:
    assert Frequency.parse(text).shift(datetime(2026, 1, 1), steps) == expected


def test_shift_and_steps_between_are_inverses() -> None:
    for text in ("30s", "15m", "1h", "1d", "1w", "1mo", "4mo"):
        frequency = Frequency.parse(text)
        start = datetime(2026, 3, 1, 12)
        for steps in range(1, 13):
            assert frequency.steps_between(start, frequency.shift(start, steps)) == steps
