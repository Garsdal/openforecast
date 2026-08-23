"""The data these tests are written against, and the client that executes it.

Every value is a function of the coordinates it sits at, and the target is a
straight line — ``instance * 1000 + event * 10``. That is deliberate: a line is
something an extrapolating statistical model continues closely, so an assertion
can be arithmetic rather than a recorded number, and a forecast that came back
for the wrong event times or the wrong instance is visible in the value itself.

These builders are the integration's own, so the tests here run against a
published install as well as a checkout. The conformance suite in
``tests/test_conformance.py`` is the part that comes from the OpenForecast
repository. They are deliberately the same builders the Nixtla integration
writes its tests against — the point of Step 13 is that the same data, the same
plans and the same assertions hold for a different library, so the fixtures are
the same too.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from openforecast_darts import DartsProvider

import openforecast as of
from openforecast.models.catalog import ModelCatalog
from openforecast.runtime.provider import ProviderClient, ProviderRegistry

START = datetime(2026, 1, 1)
HOUR = timedelta(hours=1)
FREQUENCY = "1h"

ZONE = "zone"
TIME = "timestamp"
ORIGIN_TIME = "ref_time"
EVENT_TIME = "target_time"
TARGET = "load"
KNOWN = "temp_fc"
OBSERVED = "temp"
STATIC = "capacity"

ZONES = ("DE", "FR", "NL")

PROVIDER = DartsProvider()
THETA = "darts/theta"
TIDE = "darts/tide"
NHITS = "darts/nhits"

#: Enough optimization to prove the wiring, and no more. These tests assert what
#: the model was *given* and how the answer is *labeled*; how well a neural
#: network fits six-step windows is not something a unit test can assert without
#: becoming a benchmark, and paying for a hundred epochs to find out would make
#: the suite slow for nothing.
FAST = {"n_epochs": 1}


def at(step: int) -> datetime:
    return START + HOUR * step


def target_value(instance: int, event: int) -> float:
    """A straight line per instance, which a trend model continues closely."""
    return float(instance * 1_000 + event * 10)


def known_value(instance: int, event: int, origin: int | None = None) -> float:
    """A value knowable ahead of its event time, tagged with its vintage."""
    base = float(instance * 100 + event) + 0.5
    return base if origin is None else base + 10_000.0 * (origin + 1)


def observed_value(instance: int, event: int) -> float:
    """A measurement, knowable only once its event time has passed."""
    return float(instance * 100 + event) + 0.25


def static_value(instance: int) -> float:
    """A feature with no time axis, constant within an instance."""
    return float(100 * (instance + 1))


def event_time_frame(
    *,
    instances: int = 1,
    periods: int = 24,
    future_periods: int = 0,
    known: bool = False,
    observed: bool = False,
    static: bool = False,
) -> of.TimeSeriesFrame:
    """An ordinary event-time frame, optionally with a feature of each role."""
    history: list[dict[str, Any]] = []
    future: list[dict[str, Any]] = []
    for index, zone in enumerate(ZONES[:instances]):
        for event in range(periods + future_periods):
            row: dict[str, Any] = {TIME: at(event)}
            if instances > 1:
                row[ZONE] = zone
            if known:
                row[KNOWN] = known_value(index, event)
            if static:
                row[STATIC] = static_value(index)
            if event < periods:
                row[TARGET] = target_value(index, event)
                if observed:
                    row[OBSERVED] = observed_value(index, event)
                history.append(row)
            else:
                future.append(row)
    return of.TimeSeriesFrame.from_pandas(
        history=pd.DataFrame(history),
        future=pd.DataFrame(future) if future else None,
        time=TIME,
        frequency=FREQUENCY,
        instance_keys=[ZONE] if instances > 1 else [],
        targets=[TARGET],
        known_features=[KNOWN] if known else [],
        observed_features=[OBSERVED] if observed else [],
        static_features=[STATIC] if static else [],
    )


def point_in_time_dataset(
    *,
    instances: int = 1,
    origins: int = 6,
    first_origin: int = 2,
    horizon: int = 3,
    known: bool = True,
    observed: bool = False,
    static: bool = False,
) -> of.ForecastDataset:
    """Real vintages, each describing everything up to its own origin and beyond.

    Every vintage reaches back to the first event time, because a series view at
    a selected origin is one *complete* series and is cut from a single vintage.
    The known values carry the origin that issued them, so a value from another
    vintage is identifiable rather than merely suspicious.

    An observed feature stops at its own origin, because that is what "observed"
    means; a vintage that carried one for an event time ahead of itself would be
    rejected by the semantic model before any of this got near a provider.
    """
    rows: list[dict[str, Any]] = []
    for index, zone in enumerate(ZONES[:instances]):
        for step in range(origins):
            origin = first_origin + step
            for event in range(origin + horizon + 1):
                row: dict[str, Any] = {ORIGIN_TIME: at(origin), EVENT_TIME: at(event)}
                if instances > 1:
                    row[ZONE] = zone
                row[TARGET] = target_value(index, event)
                if known:
                    row[KNOWN] = known_value(index, event, origin)
                if observed:
                    row[OBSERVED] = observed_value(index, event) if event <= origin else math.nan
                if static:
                    row[STATIC] = static_value(index)
                rows.append(row)
    return of.ForecastDataset.from_pandas(
        pd.DataFrame(rows),
        origin_time=ORIGIN_TIME,
        event_time=EVENT_TIME,
        event_frequency=FREQUENCY,
        origin_frequency=FREQUENCY,
        instance_keys=[ZONE] if instances > 1 else [],
        targets=[TARGET],
        known_features=[KNOWN] if known else [],
        observed_features=[OBSERVED] if observed else [],
        static_features=[STATIC] if static else [],
    )


def client(store: str | Path, provider: ProviderClient | None = None) -> of.OpenForecast:
    """A client that can execute this integration's models, and nothing else."""
    executor = provider if provider is not None else PROVIDER
    return of.OpenForecast(
        store=store,
        catalog=ModelCatalog(list(executor.descriptors())),
        providers=ProviderRegistry([executor]),
    )


def values(forecast: of.Forecast) -> list[float]:
    """The forecast values, in the order the table holds them."""
    column: list[float] = forecast.table.column("value").to_pylist()
    return column
