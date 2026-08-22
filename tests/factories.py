"""Deterministic sample data for the semantic-model tests.

The timestamps are built from ``datetime`` arithmetic rather than from
:class:`~openforecast.data.frequency.Frequency`, so a bug in the frequency
primitive cannot make the frame tests agree with it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

START = datetime(2026, 1, 1, 0, 0, 0)
HOUR = timedelta(hours=1)

NAN = math.nan


def timestamps(periods: int, *, start: datetime = START, step: timedelta = HOUR) -> list[datetime]:
    return [start + step * index for index in range(periods)]


def history(
    *,
    instances: Sequence[str] = ("DE",),
    instance_key: str | None = None,
    periods: int = 6,
    targets: Sequence[str] = ("load",),
    observed: Sequence[str] = (),
    known: Sequence[str] = (),
    static: Mapping[str, Mapping[str, float]] | None = None,
    start: datetime = START,
    step: timedelta = HOUR,
    time: str = "timestamp",
) -> pd.DataFrame:
    """A long history frame, one row per instance and event time.

    ``instance_key`` is the column holding ``instances``; leaving it ``None``
    builds a single (non-panel) series from the first instance only.
    """
    moments = timestamps(periods, start=start, step=step)
    keyed = instances if instance_key is not None else instances[:1]
    rows: list[dict[str, Any]] = []
    for instance_index, instance in enumerate(keyed):
        for period, moment in enumerate(moments):
            row: dict[str, Any] = {time: moment}
            if instance_key is not None:
                row[instance_key] = instance
            offset = instance_index * 1000 + period
            for target_index, target in enumerate(targets):
                row[target] = float(offset + target_index * 100)
            for name_index, name in enumerate(observed):
                row[name] = float(offset) / 2 + name_index
            for name_index, name in enumerate(known):
                row[name] = float(offset) / 4 + name_index
            for name, per_instance in (static or {}).items():
                row[name] = per_instance[instance]
            rows.append(row)
    return pd.DataFrame(rows)


def future(
    *,
    instances: Sequence[str] = ("DE",),
    instance_key: str | None = None,
    periods: int = 3,
    known: Sequence[str] = (),
    after: int = 6,
    start: datetime = START,
    step: timedelta = HOUR,
    time: str = "timestamp",
) -> pd.DataFrame:
    """Known temporal features for the ``periods`` steps after ``after`` steps."""
    moments = timestamps(after + periods, start=start, step=step)[after:]
    keyed = instances if instance_key is not None else instances[:1]
    rows: list[dict[str, Any]] = []
    for instance_index, instance in enumerate(keyed):
        for period, moment in enumerate(moments):
            row: dict[str, Any] = {time: moment}
            if instance_key is not None:
                row[instance_key] = instance
            for name_index, name in enumerate(known):
                row[name] = float(instance_index * 1000 + after + period) / 4 + name_index
            rows.append(row)
    return pd.DataFrame(rows)


def point_in_time(
    *,
    instances: Sequence[str] = ("DE",),
    instance_key: str | None = None,
    origins: int = 3,
    horizon: int = 4,
    targets: Sequence[str] = ("price",),
    known: Sequence[str] = ("wind_fc",),
    observed: Sequence[str] = (),
    static: Mapping[str, Mapping[str, float]] | None = None,
    start: datetime = START,
    step: timedelta = HOUR,
    origin_time: str = "ref_time",
    event_time: str = "target_time",
) -> pd.DataFrame:
    """A long ``(ref_time, target_time)`` frame, the shape production pipelines emit.

    Each origin covers ``horizon`` event times starting at the origin itself, so
    later origins see event times that earlier ones already covered. Known
    features vary by vintage — that is what distinguishes point-in-time data
    from event-time data — while targets depend only on the event time, so the
    truth is consistent. Observed features are ``NaN`` past their own origin,
    since nobody could have measured them yet.
    """
    keyed = instances if instance_key is not None else instances[:1]
    rows: list[dict[str, Any]] = []
    for instance_index, instance in enumerate(keyed):
        for origin_index in range(origins):
            origin = start + step * origin_index
            for lead in range(horizon):
                event = origin + step * lead
                event_index = origin_index + lead
                row: dict[str, Any] = {origin_time: origin, event_time: event}
                if instance_key is not None:
                    row[instance_key] = instance
                for target_index, target in enumerate(targets):
                    row[target] = float(instance_index * 1000 + event_index * 10 + target_index)
                for name_index, name in enumerate(known):
                    row[name] = float(origin_index * 100 + event_index) + name_index
                for name_index, name in enumerate(observed):
                    row[name] = NAN if lead else float(event_index) + name_index
                for name, per_instance in (static or {}).items():
                    row[name] = per_instance[instance]
                rows.append(row)
    return pd.DataFrame(rows)
