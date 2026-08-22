"""The golden semantic datasets the conformance suite is written against.

```text
single_univariate          one series, one target
single_multivariate        one series, two targets
panel_univariate           three series, one target
panel_multivariate         three series, two targets

pit_panel_univariate       three series of real vintages, one target
pit_panel_multivariate     three series of real vintages, two targets
pit_missingness            a feed that starts publishing partway through
pit_varying_vintages       every value names the origin that produced it
pit_known_future           vintages describing event times ahead of themselves
pit_observed_features      measurements that stop at their own origin
```

Every value is a function of the coordinates it sits at, so a materialized view
can be checked against arithmetic rather than against a recorded blob:

```text
target        instance, event time
observed      instance, event time
known         instance, event time, and the origin that issued it
```

That last term is the whole point of the point-in-time datasets. A known value
carries the origin that produced it, so :func:`origin_of` recovers it from any
number found anywhere downstream, and a vintage that leaked into a view built at
another origin is *identifiable* rather than merely suspicious.

The two builders — :func:`event_time` and :func:`point_in_time` — are shared
with the provider conformance harness, which parameterizes them by what a model
declares it can consume. The golden datasets are the fixed configurations of
them that the view and point-in-time tests are written against.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import pyarrow as pa

import openforecast as of

#: Any of the two semantic sources a fit view can be materialized from.
SemanticDataset = of.TimeSeriesFrame | of.ForecastDataset

START = datetime(2026, 1, 1)
HOUR = timedelta(hours=1)
FREQUENCY = "1h"

#: The column names the golden datasets use. Deliberately not the ones the views
#: use for themselves: a caller's columns keep the caller's names.
ZONE = "zone"
TIME = "timestamp"
ORIGIN_TIME = "ref_time"
EVENT_TIME = "target_time"
CAPACITY = "capacity"

ZONES = ("DE", "FR", "NL")

NAN = math.nan

#: A value no feed would publish, so a leaked vintage is unmistakable.
POISON = 999999.0

#: How :func:`known_value` encodes the origin that issued a value.
VINTAGE_SCALE = 10_000


def at(step: int) -> datetime:
    """The event time ``step`` hours after the start of every golden dataset."""
    return START + HOUR * step


def zones(count: int) -> tuple[str, ...]:
    if count > len(ZONES):
        raise ValueError(f"the golden datasets name {len(ZONES)} instances, not {count}")
    return ZONES[:count]


# -- values -----------------------------------------------------------------


def target_value(instance: int, event: int, target: int = 0) -> float:
    """What happened, which is a fact about an event time and has no vintage."""
    return float(instance * 1_000 + event * 10 + target)


def observed_value(instance: int, event: int, feature: int = 0) -> float:
    """A measurement, knowable only once its event time has passed."""
    return float(instance * 1_000 + event) + 0.25 + feature


def known_value(instance: int, event: int, origin: int | None = None, feature: int = 0) -> float:
    """A value knowable ahead of its event time, tagged with the vintage that issued it.

    ``origin=None`` is an event-time frame, which holds one value per event time
    and no vintages at all; :func:`origin_of` reports ``-1`` for those.
    """
    base = float(instance * 1_000 + event) + 0.5 + feature
    return base if origin is None else base + VINTAGE_SCALE * (origin + 1)


def origin_of(value: float) -> int:
    """The origin step that issued ``value``, or ``-1`` if it carries no vintage."""
    return int(value) // VINTAGE_SCALE - 1


def column(table: pa.Table, name: str) -> list[Any]:
    """One column of an Arrow table, as plain Python."""
    values: list[Any] = table.column(name).to_pylist()
    return values


def is_missing(value: float | None) -> bool:
    """Null and ``NaN`` are the two spellings of "no value here".

    Which one a value arrives as depends on how the caller's frame was
    converted, and neither is a value. What matters — and what the conformance
    suite asserts — is that a missing value stays missing.
    """
    return value is None or math.isnan(value)


def capacity_value(instance: int) -> float:
    """A static feature: constant within an instance, so it has no time axis."""
    return float(100 * (instance + 1))


# -- builders ---------------------------------------------------------------


def event_time(
    *,
    instances: int = 1,
    targets: Sequence[str] = ("load",),
    periods: int = 24,
    observed: Sequence[str] = (),
    known: Sequence[str] = (),
    static: bool = False,
    future_periods: int = 0,
    gaps: Sequence[int] = (),
) -> of.TimeSeriesFrame:
    """An ordinary event-time frame: one row per instance and event time.

    ``instances=1`` builds a single series with no instance keys at all, rather
    than a panel of one — the two are different shapes, and a model may support
    one and not the other.

    ``gaps`` names steps whose targets were never measured. The rows are still
    there — a gap is a fact about the data, not an absence of it.
    """
    keyed = instances > 1
    rows: list[dict[str, Any]] = []
    for index, zone in enumerate(zones(instances)):
        for step in range(periods):
            row: dict[str, Any] = {TIME: at(step)}
            if keyed:
                row[ZONE] = zone
            for position, target in enumerate(targets):
                row[target] = NAN if step in gaps else target_value(index, step, position)
            for position, name in enumerate(observed):
                row[name] = observed_value(index, step, position)
            for position, name in enumerate(known):
                row[name] = known_value(index, step, feature=position)
            if static:
                row[CAPACITY] = capacity_value(index)
            rows.append(row)

    future: list[dict[str, Any]] = []
    for index, zone in enumerate(zones(instances)):
        for step in range(periods, periods + future_periods):
            row = {TIME: at(step)}
            if keyed:
                row[ZONE] = zone
            for position, name in enumerate(known):
                row[name] = known_value(index, step, feature=position)
            future.append(row)

    return of.TimeSeriesFrame.from_pandas(
        history=pd.DataFrame(rows),
        future=pd.DataFrame(future) if future and known else None,
        time=TIME,
        frequency=FREQUENCY,
        instance_keys=[ZONE] if keyed else [],
        targets=list(targets),
        observed_features=list(observed),
        known_features=list(known),
        static_features=[CAPACITY] if static else [],
    )


def point_in_time(
    *,
    instances: int = 1,
    targets: Sequence[str] = ("price",),
    origins: int = 6,
    context: int = 3,
    horizon: int = 3,
    observed: Sequence[str] = (),
    known: Sequence[str] = ("wind_fc",),
    static: bool = False,
    stable: bool = False,
    cumulative: bool = False,
) -> of.ForecastDataset:
    """Real forecast vintages, paired with what actually happened.

    Origin ``k`` sits at event step ``context - 1 + k`` and describes the
    ``context`` event times ending at itself plus the ``horizon`` after it, so
    the earliest origin already carries a full context window. ``cumulative``
    widens every vintage back to the first event time instead, which is what a
    view holding one complete series at a selected origin needs.

    ``stable`` drops the vintage term from the known values, making every
    vintage of an event time identical — the construction the event-time
    equivalence test compares against an event-time frame.
    """
    keyed = instances > 1
    rows: list[dict[str, Any]] = []
    for index, zone in enumerate(zones(instances)):
        for step in range(origins):
            origin = context - 1 + step
            first = 0 if cumulative else origin - context + 1
            for event in range(first, origin + horizon + 1):
                row: dict[str, Any] = {ORIGIN_TIME: at(origin), EVENT_TIME: at(event)}
                if keyed:
                    row[ZONE] = zone
                for position, target in enumerate(targets):
                    row[target] = target_value(index, event, position)
                for position, name in enumerate(known):
                    row[name] = known_value(
                        index, event, None if stable else origin, feature=position
                    )
                for position, name in enumerate(observed):
                    row[name] = observed_value(index, event, position) if event <= origin else NAN
                if static:
                    row[CAPACITY] = capacity_value(index)
                rows.append(row)
    return forecast_dataset(
        rows,
        targets=targets,
        observed=observed,
        known=known,
        static=static,
        keyed=keyed,
    )


def forecast_dataset(
    rows: Sequence[Mapping[str, Any]],
    *,
    targets: Sequence[str] = ("price",),
    observed: Sequence[str] = (),
    known: Sequence[str] = ("wind_fc",),
    static: bool = False,
    keyed: bool = False,
) -> of.ForecastDataset:
    """The golden ``(ref_time, target_time)`` split, for hand-written rows."""
    return of.ForecastDataset.from_pandas(
        pd.DataFrame(list(rows)),
        origin_time=ORIGIN_TIME,
        event_time=EVENT_TIME,
        event_frequency=FREQUENCY,
        origin_frequency=FREQUENCY,
        instance_keys=[ZONE] if keyed else [],
        targets=list(targets),
        observed_features=list(observed),
        known_features=list(known),
        static_features=[CAPACITY] if static else [],
    )


# -- the golden datasets ----------------------------------------------------


def single_univariate() -> of.TimeSeriesFrame:
    """One series, one target, one feature of each temporal role."""
    return event_time(
        instances=1, targets=("load",), observed=("temp",), known=("temp_fc",), future_periods=6
    )


def single_multivariate() -> of.TimeSeriesFrame:
    """Two targets on one time axis: a multivariate model's training unit."""
    return event_time(
        instances=1,
        targets=("load", "wind"),
        observed=("temp",),
        known=("temp_fc",),
        future_periods=6,
    )


def panel_univariate() -> of.TimeSeriesFrame:
    """Three series with a static feature, which only a panel can distinguish by."""
    return event_time(
        instances=3,
        targets=("load",),
        observed=("temp",),
        known=("temp_fc",),
        static=True,
        future_periods=6,
    )


def panel_multivariate() -> of.TimeSeriesFrame:
    return event_time(
        instances=3,
        targets=("load", "wind"),
        observed=("temp",),
        known=("temp_fc",),
        static=True,
        future_periods=6,
    )


def pit_panel_univariate() -> of.ForecastDataset:
    """Three series of real vintages: the point-in-time counterpart of the panel."""
    return point_in_time(instances=3, targets=("price",), static=True)


def pit_panel_multivariate() -> of.ForecastDataset:
    return point_in_time(instances=3, targets=("price", "volume"), static=True)


#: The event time :func:`pit_missingness` withholds, the origin whose vintage
#: finally carries it, and the value that arrives then.
MISSING_EVENT = 12
AVAILABLE_ORIGIN = 10
AVAILABLE_VALUE = 42.0


def pit_missingness() -> of.ForecastDataset:
    """A feed that starts publishing partway through, for one event time.

    ```text
    origin 08 -> NaN
    origin 09 -> NaN
    origin 10 -> 42
    ```

    all describing event time 12. Every other cell carries an ordinary value, so
    a test can tell "this feed was unavailable" apart from "this dataset is
    empty".
    """
    rows: list[dict[str, Any]] = []
    for origin in (8, 9, AVAILABLE_ORIGIN):
        for event in range(8, 13):
            if event != MISSING_EVENT:
                wind = known_value(0, event, origin)
            else:
                wind = AVAILABLE_VALUE if origin == AVAILABLE_ORIGIN else NAN
            rows.append(
                {
                    ORIGIN_TIME: at(origin),
                    EVENT_TIME: at(event),
                    "price": target_value(0, event),
                    "wind_fc": wind,
                }
            )
    return forecast_dataset(rows)


def pit_varying_vintages() -> of.ForecastDataset:
    """Two series whose every known value names the origin that issued it."""
    return point_in_time(instances=2, origins=6, context=3, horizon=3)


def pit_known_future() -> of.ForecastDataset:
    """Vintages that mostly describe event times ahead of their own origin."""
    return point_in_time(instances=2, origins=6, context=1, horizon=6)


def pit_observed_features() -> of.ForecastDataset:
    """Measurements that stop at their own origin, beside forecasts that do not."""
    return point_in_time(instances=2, observed=("temp",), known=("temp_fc",))


#: The event time every vintage of :func:`leakage_sentinel` disagrees about, and
#: what each origin published for it.
SENTINEL_EVENT = 12
SENTINEL_VALUES: Mapping[int, float] = {8: 10.0, 9: 20.0, 10: POISON}


def leakage_sentinel() -> of.ForecastDataset:
    """Three vintages of one event time, the last of them poisoned.

    ```text
    origin 08 -> target 12 -> wind = 10
    origin 09 -> target 12 -> wind = 20
    origin 10 -> target 12 -> wind = 999999
    ```

    Materializing origin 09 must reach the 20 and must not reach the 999999,
    whatever view is asked for. It is the crudest possible leakage test and the
    one that would catch a planner joining on event time alone.
    """
    rows = [
        {
            ORIGIN_TIME: at(origin),
            EVENT_TIME: at(event),
            "price": target_value(0, event),
            "wind_fc": (
                SENTINEL_VALUES[origin]
                if event == SENTINEL_EVENT
                else known_value(0, event, origin)
            ),
        }
        for origin in SENTINEL_VALUES
        for event in range(8, 13)
    ]
    return forecast_dataset(rows)


#: Every golden dataset, by the name the conformance suite refers to it by.
GOLDEN_DATASETS: Mapping[str, Callable[[], SemanticDataset]] = {
    "single_univariate": single_univariate,
    "single_multivariate": single_multivariate,
    "panel_univariate": panel_univariate,
    "panel_multivariate": panel_multivariate,
    "pit_panel_univariate": pit_panel_univariate,
    "pit_panel_multivariate": pit_panel_multivariate,
    "pit_missingness": pit_missingness,
    "pit_varying_vintages": pit_varying_vintages,
    "pit_known_future": pit_known_future,
    "pit_observed_features": pit_observed_features,
}


def golden(name: str) -> SemanticDataset:
    """The golden dataset called ``name``, built fresh."""
    build = GOLDEN_DATASETS.get(name)
    if build is None:
        raise KeyError(f"no golden dataset named {name!r}; known: {sorted(GOLDEN_DATASETS)}")
    return build()
