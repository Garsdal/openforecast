"""``builtin/seasonal-naive``: the forecast is what happened one season ago.

```text
season_length = 24

y(T + 1)   = y(T + 1 - 24)
y(T + 25)  = y(T + 1 - 24)
```

The oldest forecasting method that is still hard to beat, and the reference
implementation of everything a provider has to do: declare a contract, consume
one execution view, persist state that outlives the process, and answer a
:class:`~openforecast.views.ForecastView` in the canonical forecast columns.

It is a *local* model — each series is fitted on its own — so its training unit
is one complete time series and its view is a ``SeriesView``. What it persists
is the last season of each series, keyed by the instance that series belongs to,
because a forecast has to come back labeled with the instance it is about.

Missing values pass straight through. A season-ago value that was missing yields
a missing forecast, which is what the model actually knows; imputing it here
would be OpenForecast inventing an observation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa

from openforecast.errors import DataError, ProviderError, RecipeError
from openforecast.models import (
    FeatureCapabilities,
    InstanceCapabilities,
    MissingValueSupport,
    ModelCapabilities,
    ModelDescriptor,
    ModelLifecycle,
    ModelRef,
    OutputCapabilities,
    TargetCapabilities,
    TrainingContract,
)
from openforecast.protocol import ForecastColumn, forecast_columns
from openforecast.views import (
    EVENT_TIME,
    SERIES_ID,
    FitView,
    ForecastView,
    Frequency,
    SeriesView,
)

__all__ = ["NAME", "descriptor", "fit", "forecast"]

NAME = "seasonal-naive"
REF = "builtin/seasonal-naive"

#: The provider's whole persisted state for one fitted artifact.
STATE_FILENAME = "seasonal-naive.json"

_DEFAULT_SEASON_LENGTH = 1


def descriptor(provider: str) -> ModelDescriptor:
    """What the catalog and the engine are told about this model.

    Deliberately complete: the engine materializes data, checks it and refuses an
    unanswerable request entirely from this, without starting anything.
    """
    return ModelDescriptor(
        ref=ModelRef.parse(REF),
        provider=provider,
        display_name="Seasonal naive",
        lifecycle=ModelLifecycle.trainable(),
        training=TrainingContract.series(),
        capabilities=ModelCapabilities(
            instances=InstanceCapabilities(single=True, panel=True),
            targets=TargetCapabilities(univariate=True, multivariate=True),
            # It can be given any feature role and conditions on none of them:
            # the forecast is a value from its own past, which is what makes it
            # the baseline. Declaring them accepted rather than unsupported is
            # what lets it be fitted on the same data as the model it is the
            # baseline for — including point-in-time data, which always carries
            # the features whose vintages are the point of it.
            features=FeatureCapabilities(observed=True, known=True, static=True),
            outputs=OutputCapabilities(point=True),
            missing_values=MissingValueSupport.NATIVE,
        ),
        parameters_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "season_length": {
                    "type": "integer",
                    "minimum": 1,
                    "default": _DEFAULT_SEASON_LENGTH,
                    "description": "Steps of the data's frequency in one season.",
                }
            },
        },
    )


# -- fit --------------------------------------------------------------------


def fit(view: FitView, params: Mapping[str, Any], into: Path) -> None:
    """Remember the last season of every series, keyed by its instance."""
    if not isinstance(view, SeriesView):
        raise ProviderError(
            f"{REF} trains on one complete time series, so it cannot be fitted from a "
            f"{view.kind} view"
        )
    season_length = _season_length(params)
    schema = view.schema
    keys = _instance_keys(view)
    times: list[Any] = view.temporal.column(EVENT_TIME).to_pylist()
    series_ids: list[Any] = view.temporal.column(SERIES_ID).to_pylist()
    targets = {name: view.temporal.column(name).to_pylist() for name in schema.targets}

    series: list[dict[str, Any]] = []
    for series_id in view.series_ids:
        positions = sorted(
            (index for index, value in enumerate(series_ids) if value == series_id),
            key=lambda index: times[index],
        )
        if len(positions) < season_length:
            raise DataError(
                f"{REF} with season_length={season_length} needs {season_length} "
                f"observations of every series; instance {keys[series_id]} has "
                f"{len(positions)}. Shorten the season, or fit on more history"
            )
        tail = positions[-season_length:]
        series.append(
            {
                "key": list(keys[series_id]),
                "last_event_time": times[tail[-1]].isoformat(),
                "tail": {
                    name: [values[index] for index in tail] for name, values in targets.items()
                },
            }
        )

    _write_state(
        into / STATE_FILENAME,
        {
            "model": NAME,
            "season_length": season_length,
            "instance_keys": list(schema.instance_keys),
            "targets": list(schema.targets),
            "frequency": str(schema.frequency),
            "series": series,
        },
    )


# -- forecast ---------------------------------------------------------------


def forecast(view: ForecastView, output: Mapping[str, Any], state: Path) -> pa.Table:
    """The season-ago value of every event time the view asks about."""
    kind = output.get("kind", ForecastColumn.KIND.value)
    if kind != "point":
        raise ProviderError(f"{REF} produces point forecasts, not {kind}")
    persisted = _read_state(state / STATE_FILENAME)
    season_length = int(persisted["season_length"])
    tails: dict[tuple[Any, ...], dict[str, Any]] = {
        tuple(entry["key"]): entry for entry in persisted["series"]
    }

    metadata = view.metadata
    instance_keys = metadata.instance_keys
    future = view.future
    columns = [future.column(name).to_pylist() for name in instance_keys]
    rows = list(zip(*columns, strict=True)) if instance_keys else [()] * future.num_rows
    moments: list[Any] = future.column(EVENT_TIME).to_pylist()

    keys: list[tuple[Any, ...]] = []
    times: list[datetime] = []
    targets: list[str] = []
    values: list[float | None] = []
    for instance, moment in zip(rows, moments, strict=True):
        entry = tails.get(instance)
        if entry is None:
            raise DataError(
                f"{REF} is fitted per series, so it has no model for instance {instance}; "
                f"it was fitted on {sorted(str(key) for key in tails)}"
            )
        position = _phase(entry, moment, season_length, metadata.frequency)
        for target in metadata.targets:
            keys.append(instance)
            times.append(moment)
            targets.append(target)
            values.append(entry["tail"][target][position])

    return _forecast_table(view, keys, times, targets, values)


def _phase(
    entry: Mapping[str, Any], moment: datetime, season_length: int, frequency: Frequency
) -> int:
    """Which remembered step ``moment`` repeats."""
    last = datetime.fromisoformat(entry["last_event_time"])
    ahead = frequency.steps_between(last, moment)
    if ahead is None or ahead < 1:
        raise DataError(
            f"{REF} forecasts steps after the end of the series it remembers; "
            f"{moment.isoformat()} is not a {frequency} step after {last.isoformat()}"
        )
    return (ahead - 1) % season_length


def _forecast_table(
    view: ForecastView,
    keys: Sequence[tuple[Any, ...]],
    times: Sequence[datetime],
    targets: Sequence[str],
    values: Sequence[float | None],
) -> pa.Table:
    """The canonical long forecast: one row per instance, event time and target."""
    instance_keys = view.metadata.instance_keys
    columns: dict[str, pa.Array[Any]] = {
        name: pa.array([key[index] for key in keys], type=view.future.column(name).type)
        for index, name in enumerate(instance_keys)
    }
    columns[ForecastColumn.EVENT_TIME.value] = pa.array(
        times, type=view.future.column(EVENT_TIME).type
    )
    columns[ForecastColumn.TARGET.value] = pa.array(targets, type=pa.string())
    columns[ForecastColumn.KIND.value] = pa.array(["point"] * len(values), type=pa.string())
    columns[ForecastColumn.QUANTILE.value] = pa.nulls(len(values), type=pa.float64())
    columns[ForecastColumn.SAMPLE.value] = pa.nulls(len(values), type=pa.int64())
    columns[ForecastColumn.VALUE.value] = pa.array(values, type=pa.float64())
    return pa.table({name: columns[name] for name in forecast_columns(instance_keys)})


# -- state ------------------------------------------------------------------


def _season_length(params: Mapping[str, Any]) -> int:
    unknown = sorted(set(params) - {"season_length"})
    if unknown:
        raise RecipeError(f"{REF} takes no parameter {unknown}; it takes season_length")
    season_length = params.get("season_length", _DEFAULT_SEASON_LENGTH)
    if not isinstance(season_length, int) or isinstance(season_length, bool) or season_length < 1:
        raise RecipeError(
            f"season_length counts steps of the data's frequency in one season, so it is a "
            f"positive integer; got {season_length!r}"
        )
    return season_length


def _instance_keys(view: SeriesView) -> dict[str, tuple[Any, ...]]:
    """``series_id -> the instance it belongs to``, from the view's key table."""
    ids: list[Any] = view.series.column(SERIES_ID).to_pylist()
    columns = [view.series.column(name).to_pylist() for name in view.schema.instance_keys]
    rows = list(zip(*columns, strict=True)) if columns else [()] * len(ids)
    return dict(zip(ids, rows, strict=True))


def _write_state(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        encoded = json.dumps(payload, indent=2)
    except TypeError as error:  # an instance key no JSON document can hold
        raise ProviderError(f"{REF} cannot persist this instance key: {error}") from error
    path.write_text(encoded + "\n", encoding="utf-8")


def _read_state(path: Path) -> Mapping[str, Any]:
    try:
        persisted: Any = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ProviderError(f"{REF} has no fitted state at {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ProviderError(
            f"the fitted state of {REF} at {path} is not valid JSON: {error}"
        ) from error
    if not isinstance(persisted, dict) or persisted.get("model") != NAME:  # pyright: ignore[reportUnknownMemberType]
        raise ProviderError(f"{path} does not hold the fitted state of {REF}")
    return persisted  # pyright: ignore[reportUnknownVariableType]
