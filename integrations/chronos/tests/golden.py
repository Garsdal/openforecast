"""Deterministic inference views, built without any of OpenForecast's planners.

The conformance suite materializes its views through the engine, which is the
right way to check that the boundary holds. What it cannot check is the
translation *inside* the boundary — which array became the target, which became
a past covariate, and whether either was read in the order the context is in —
because a stand-in pipeline that answered the right shape from the wrong numbers
would pass every one of those cases.

So these views are constructed here by hand, with values a test can name.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pyarrow as pa

from openforecast.data.features import FeatureSpec
from openforecast.views import EVENT_TIME, ForecastView
from openforecast.views.forecast import ForecastViewMetadata

START = datetime(2026, 1, 1)
HOUR = timedelta(hours=1)

#: Long enough to be a context and short enough to read in a failure message.
HISTORY = 6
HORIZON = 3


def moments(count: int, *, after: int = 0) -> list[datetime]:
    return [START + HOUR * (after + step) for step in range(count)]


def view(
    *,
    instances: tuple[str, ...] = (),
    observed: bool = False,
    known: bool = False,
    history: int = HISTORY,
    horizon: int = HORIZON,
) -> ForecastView:
    """One inference origin, with the feature roles a test asks for.

    ``instances`` empty is a single series with no instance key, which is the
    shape most of these tests want; naming zones makes it a panel.
    """
    keys = ("zone",) if instances else ()
    named = instances or ("",)
    features: list[FeatureSpec] = []
    if observed:
        features.append(FeatureSpec.observed("temp"))
    if known:
        features.append(FeatureSpec.known("temp_fc"))

    metadata = ForecastViewMetadata(
        frequency="1h",
        targets=("load",),
        instance_keys=keys,
        features=tuple(features),
        horizon=horizon,
    )
    past = moments(history)
    ahead = moments(horizon, after=history)

    return ForecastView(
        origin_time=past[-1],
        history=_table(named, keys, past, {"load": _values, **_roles(observed, known, past=True)}),
        future=_table(named, keys, ahead, _roles(False, known, past=False), offset=history),
        metadata=metadata,
    )


def _roles(observed: bool, known: bool, *, past: bool) -> dict[str, Any]:
    columns: dict[str, Any] = {}
    if observed and past:
        columns["temp"] = _observed
    if known:
        columns["temp_fc"] = _forecast
    return columns


def _values(instance: int, step: int) -> float:
    """The target: readable as ``instance`` in the tens and ``step`` in the units."""
    return float(instance * 100 + step)


def _observed(instance: int, step: int) -> float:
    return float(instance * 100 + step) / 2


def _forecast(instance: int, step: int) -> float:
    return float(instance * 100 + step) / 4


def _table(
    instances: tuple[str, ...],
    keys: tuple[str, ...],
    times: list[datetime],
    columns: dict[str, Any],
    *,
    offset: int = 0,
) -> pa.Table:
    """``offset`` continues the step counter, so a future value follows its history."""
    rows: dict[str, list[Any]] = {name: [] for name in (*keys, EVENT_TIME, *columns)}
    for index, instance in enumerate(instances):
        for position, moment in enumerate(times):
            step = offset + position
            for name in keys:
                rows[name].append(instance)
            rows[EVENT_TIME].append(moment)
            for name, produce in columns.items():
                rows[name].append(produce(index, step))
    built: dict[str, pa.Array[Any]] = {
        name: pa.array(values, type=pa.timestamp("us") if name == EVENT_TIME else None)
        for name, values in rows.items()
    }
    return pa.table(built)
