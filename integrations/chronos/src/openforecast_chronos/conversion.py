"""The translation layer: a ``ForecastView`` in, a canonical forecast out.

```text
ForecastView  ->  one Chronos input dict per instance
predictions   ->  the canonical forecast columns, as Arrow — point or quantile
```

Smaller than the other integrations' conversion modules, and for the reason the
zero-shot lifecycle is worth having: there is no training side. A fitted
integration converts twice — a fit view into whatever the library trains on, and
a forecast view into whatever it predicts from — and the two have to agree about
column order, feature roles and instance identity, which is where the bugs live.
Here there is one direction and one moment.

Two properties of the mapping are worth stating.

**A feature role becomes a covariate slot, and the roles are not symmetric.** A
Chronos input carries ``past_covariates`` for what is known up to the origin and
``future_covariates`` for what is known beyond it, and the library requires the
second to be a subset of the first. That lines up exactly with OpenForecast's
roles: a *known* feature is in both tables, so it appears in both slots; an
*observed* feature exists only in the history, so it appears in the past slot
alone. No role is invented and none is dropped, which is what makes "the model
saw what was knowable at this origin" a fact about this file.

**Order is the contract, and it is this module's, not the tables'.** A pipeline
answers with an array per instance and no statement about what it is about, so
the instances are taken in the view's own order and the event times in ascending
order, and the answer is labeled from those two — never from however a transport
happened to lay the rows out.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from openforecast.errors import ProviderError
from openforecast.views import (
    EVENT_TIME,
    ForecastColumn,
    ForecastView,
    forecast_columns,
)

__all__ = [
    "ChronosInput",
    "InstanceKey",
    "answer",
    "inputs_for",
    "single_target",
]

#: An instance of the caller's data, as the tuple of its key columns.
InstanceKey = tuple[Any, ...]

#: What a Chronos pipeline computes in. Casting once, here, is what keeps "the
#: missing values reach the model unchanged" checkable: a null in an Arrow column
#: and a ``NaN`` in a float column are two spellings of the same absence, and
#: ``float64`` holds both as ``NaN``.
DOUBLE = pa.float64()


@dataclass(frozen=True)
class ChronosInput:
    """One instance, in the shape ``Chronos2Pipeline.predict`` takes.

    A mapping rather than a bare array because the covariates travel with the
    target: the library reads all three keys out of one dict per series, and
    every dict in a batch has to declare the same covariate names.
    """

    target: np.ndarray[Any, Any]
    past_covariates: dict[str, np.ndarray[Any, Any]]
    future_covariates: dict[str, np.ndarray[Any, Any]]

    def as_mapping(self) -> dict[str, Any]:
        """What the pipeline is handed, with the empty slots left out.

        An empty covariate dict is not the same as no covariates: passing one
        would declare a schema of nothing, and the library validates the schemas
        of a batch against each other.
        """
        payload: dict[str, Any] = {"target": self.target}
        if self.past_covariates:
            payload["past_covariates"] = dict(self.past_covariates)
        if self.future_covariates:
            payload["future_covariates"] = dict(self.future_covariates)
        return payload


def single_target(targets: Sequence[str]) -> str:
    """The one target being forecast, or a refusal counting what was given.

    Chronos-2 can forecast several variates jointly, and this integration does
    not expose that yet: a multivariate answer has to say which variate each
    number is about, and the descriptor declares ``multivariate=False`` so that
    the engine refuses the request before a pipeline is loaded. This is the
    provider-side half of the same statement.
    """
    if len(targets) != 1:
        raise ProviderError(
            f"this provider forecasts one target at a time and was given {len(targets)}: "
            f"{list(targets)}"
        )
    return targets[0]


# -- what the model is handed -------------------------------------------------


def inputs_for(view: ForecastView, target: str) -> tuple[ChronosInput, ...]:
    """One input per instance, in the view's own instance order.

    The history is ordered by event time per instance here rather than trusted
    to arrive that way: a context is a sequence, and a sequence read out of order
    is a different question asked confidently.
    """
    metadata = view.metadata
    keys = metadata.instance_keys
    known = tuple(feature.name for feature in metadata.known_features)
    past = tuple(name for name in metadata.temporal_feature_names)

    history = _by_instance(view.history, keys, (target, *past))
    future = _by_instance(view.future, keys, known)
    horizon = len(view.event_times)

    found: list[ChronosInput] = []
    for instance in view.instances:
        columns = history[instance]
        ahead = future.get(instance, {})
        _require_length(ahead, horizon, instance, "future")
        found.append(
            ChronosInput(
                target=columns[target],
                past_covariates={name: columns[name] for name in past},
                future_covariates={name: ahead[name] for name in known},
            )
        )
    return tuple(found)


def _by_instance(
    table: pa.Table, keys: Sequence[str], columns: Sequence[str]
) -> dict[InstanceKey, dict[str, np.ndarray[Any, Any]]]:
    """``columns`` of ``table``, per instance, ascending by event time."""
    if not columns:
        return {instance: {} for instance in _key_rows(table, keys)}
    ordered = table.sort_by([(name, "ascending") for name in (*keys, EVENT_TIME)])
    rows = _key_rows(ordered, keys)
    values = {name: _numeric(ordered, name) for name in columns}

    found: dict[InstanceKey, dict[str, list[float]]] = {}
    for position, instance in enumerate(rows):
        holder = found.setdefault(instance, {name: [] for name in columns})
        for name in columns:
            holder[name].append(values[name][position])
    return {
        instance: {name: np.asarray(series, dtype=float) for name, series in holder.items()}
        for instance, holder in found.items()
    }


def _require_length(
    columns: Mapping[str, np.ndarray[Any, Any]], horizon: int, instance: InstanceKey, label: str
) -> None:
    for name, values in columns.items():
        if values.shape[0] != horizon:
            raise ProviderError(
                f"the {label} of {name} holds {values.shape[0]} steps for instance {instance} "
                f"and the horizon is {horizon}"
            )


def _numeric(table: pa.Table, name: str) -> list[float]:
    """One column as floats, missing values intact.

    Chronos reads a ``NaN`` as an unobserved step, which is what makes the
    descriptor's ``NATIVE`` declaration true, so nothing is filled in here. A
    column with no numeric reading is refused rather than encoded: an encoding
    this integration invented would be one the caller cannot see.
    """
    column = table.column(name)
    try:
        cast = pc.cast(column, DOUBLE)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as error:
        raise ProviderError(
            f"Chronos takes numeric series and column {name!r} holds {column.type}: {error}. "
            f"Encode it before it reaches OpenForecast, or drop it from the data"
        ) from error
    values: list[float] = cast.to_pylist()
    return [float("nan") if value is None else float(value) for value in values]


# -- what comes back ----------------------------------------------------------


def answer(
    view: ForecastView,
    predicted: Sequence[Sequence[Sequence[float]]],
    *,
    target: str,
    levels: Sequence[float | None],
) -> pa.Table:
    """The canonical long forecast, from the numbers the pipeline returned.

    ``predicted`` is one entry per instance, in the order :func:`inputs_for`
    built them, and each entry is ``horizon x len(levels)``. ``levels`` is what
    each of those columns is: ``(None,)`` for a point forecast, and the requested
    levels in ascending order for a quantile one. Spelled that way rather than
    with a flag because it is the same labeling job either way, and doing it
    twice is how two answers of one provider drift apart.

    The lengths are checked rather than trusted. A pipeline that answered a
    shorter horizon produces a table that looks exactly like a correct one, and
    the engine's own check would report it as a provider that answered a
    different question — which is true, but says less than this does.
    """
    instances = view.instances
    if len(predicted) != len(instances):
        raise ProviderError(
            f"Chronos was asked about {len(instances)} instances and answered "
            f"{len(predicted)} of them"
        )
    event_times = view.event_times
    is_point = all(level is None for level in levels)

    keys: list[InstanceKey] = []
    times: list[datetime] = []
    quantiles: list[float | None] = []
    values: list[float] = []
    for instance, steps in zip(instances, predicted, strict=True):
        if len(steps) != len(event_times):
            raise ProviderError(
                f"Chronos answered {len(steps)} steps for instance {instance} and "
                f"{len(event_times)} were asked for"
            )
        for moment, row in zip(event_times, steps, strict=True):
            if len(row) != len(levels):
                raise ProviderError(
                    f"Chronos answered {len(row)} values per step and {len(levels)} were asked for"
                )
            for level, value in zip(levels, row, strict=True):
                keys.append(instance)
                times.append(moment)
                quantiles.append(level)
                values.append(float(value))

    instance_keys = view.metadata.instance_keys
    columns: dict[str, pa.Array[Any]] = {
        name: pa.array([key[index] for key in keys], type=view.future.column(name).type)
        for index, name in enumerate(instance_keys)
    }
    count = len(values)
    columns[ForecastColumn.EVENT_TIME.value] = pa.array(
        times, type=view.future.column(EVENT_TIME).type
    )
    columns[ForecastColumn.TARGET.value] = pa.array([target] * count, type=pa.string())
    columns[ForecastColumn.KIND.value] = pa.array(
        ["point" if is_point else "quantile"] * count, type=pa.string()
    )
    columns[ForecastColumn.QUANTILE.value] = pa.array(quantiles, type=pa.float64())
    columns[ForecastColumn.SAMPLE.value] = pa.nulls(count, type=pa.int64())
    columns[ForecastColumn.VALUE.value] = pa.array(values, type=pa.float64())
    return pa.table({name: columns[name] for name in forecast_columns(instance_keys)})


def _key_rows(table: pa.Table, instance_keys: Sequence[str]) -> list[InstanceKey]:
    if not instance_keys:
        return [()] * table.num_rows
    columns = [table.column(name).to_pylist() for name in instance_keys]
    return list(zip(*columns, strict=True))
