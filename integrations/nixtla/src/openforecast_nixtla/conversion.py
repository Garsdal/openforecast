"""The translation layer: execution views in, Nixtla's long frame out.

```text
SeriesView    ->  unique_id, ds, y, exogenous columns
ForecastView  ->  unique_id, ds, exogenous columns   (the future half)
predictions   ->  the canonical forecast columns, as Arrow
```

This module is the only place in the integration where ``unique_id``, ``ds`` and
``y`` appear, and they are constructed here rather than received: a
``SeriesView`` names its series by an opaque ``series_id`` and its columns by
whatever the caller called them, so the Nixtla spelling is something this
integration puts on the data on its way in and takes off again on its way out.
Rule 6 of ARCHITECTURE.md is what makes that worth a module of its own.

Two details are worth stating.

**The instance keys are not in the frame.** Nixtla identifies a series by one
string, and OpenForecast identifies it by the caller's key columns — possibly
several, possibly not strings. So the ``series_id`` the view already assigns is
used as ``unique_id`` and the mapping back to instance keys is persisted beside
the fitted model. A forecast comes back labeled with the instance it is about
because that mapping was written down at fit time, not because a key survived a
round trip through somebody's dataframe.

**The event times come from the view, not from the library.** Predicting ``h``
steps makes a library extend its own time axis using its own spelling of the
frequency; the view already names the exact event times being asked about. So
the answer is labeled from :attr:`ForecastView.event_times` and the library's
``ds`` is used only to order the values it returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
import pyarrow as pa

from openforecast.errors import DataError, ProviderError
from openforecast.views import (
    EVENT_TIME,
    SERIES_ID,
    ForecastColumn,
    ForecastView,
    Frequency,
    FrequencyUnit,
    SeriesView,
    forecast_columns,
)

__all__ = [
    "PANEL_ID",
    "TARGET",
    "TIME",
    "TrainingFrame",
    "answer",
    "future_frame",
    "pandas_frequency",
    "training_frame",
]

#: Nixtla's names for the three columns every one of its libraries expects.
#: Legal here and nowhere else in OpenForecast.
PANEL_ID = "unique_id"
TIME = "ds"
TARGET = "y"

#: OpenForecast's frequency units in pandas' spelling. Weeks are ``7D`` rather
#: than ``W`` deliberately: pandas anchors a weekly offset to a weekday, and a
#: weekly series does not have to start on the one pandas would pick.
_PANDAS_UNITS: dict[FrequencyUnit, str] = {
    FrequencyUnit.SECOND: "s",
    FrequencyUnit.MINUTE: "min",
    FrequencyUnit.HOUR: "h",
    FrequencyUnit.DAY: "D",
    FrequencyUnit.WEEK: "D",
    FrequencyUnit.MONTH: "MS",
}

_WEEK_DAYS = 7


def pandas_frequency(frequency: Frequency) -> str:
    """``Frequency`` in the offset alias a Nixtla library will accept."""
    unit = _PANDAS_UNITS.get(frequency.unit)
    if unit is None:  # pragma: no cover - every unit is mapped above
        raise ProviderError(f"no pandas offset alias for the frequency {frequency}")
    step = frequency.step * (_WEEK_DAYS if frequency.unit is FrequencyUnit.WEEK else 1)
    return f"{step}{unit}"


@dataclass(frozen=True)
class TrainingFrame:
    """A ``SeriesView`` in the representation a Nixtla library trains on."""

    #: ``unique_id, ds, y`` and one column per exogenous feature.
    frame: pd.DataFrame
    #: The caller's name for the one target being modeled.
    target: str
    #: The known features handed to the model as exogenous regressors.
    exogenous: tuple[str, ...]
    #: ``unique_id -> the instance key that series belongs to``.
    instances: dict[str, tuple[Any, ...]]
    #: ``unique_id -> the last event time it was fitted on``.
    last_event_times: dict[str, datetime]
    frequency: str


def training_frame(view: SeriesView) -> TrainingFrame:
    """Build the long frame, and everything the forecast side needs remembered."""
    schema = view.schema
    if len(schema.targets) != 1:
        raise ProviderError(
            f"this provider fits one target at a time and was given {len(schema.targets)}: "
            f"{list(schema.targets)}"
        )
    target = schema.targets[0]
    exogenous = tuple(feature.name for feature in schema.known_features)
    unsupported = tuple(
        feature.name for feature in schema.features if feature.name not in exogenous
    )
    if unsupported:
        raise ProviderError(
            f"a Nixtla statistical model conditions on values known ahead of their event "
            f"time; it was given the features {list(unsupported)}"
        )

    columns = [SERIES_ID, EVENT_TIME, target, *exogenous]
    frame = view.temporal.select(columns).to_pandas()
    frame = frame.rename(columns={SERIES_ID: PANEL_ID, EVENT_TIME: TIME, target: TARGET})
    frame = frame.sort_values([PANEL_ID, TIME], ignore_index=True)

    return TrainingFrame(
        frame=frame,
        target=target,
        exogenous=exogenous,
        instances=_instances(view),
        last_event_times={
            str(series_id): moment.to_pydatetime()
            for series_id, moment in frame.groupby(PANEL_ID)[TIME].max().items()
        },
        frequency=pandas_frequency(schema.frequency),
    )


def future_frame(
    view: ForecastView, unique_ids: dict[tuple[Any, ...], str], exogenous: tuple[str, ...]
) -> pd.DataFrame | None:
    """The exogenous values for the event times being forecast, or ``None``.

    ``None`` when the fitted model has no exogenous regressors, which is what a
    Nixtla library expects rather than an empty frame.
    """
    if not exogenous:
        return None
    metadata = view.metadata
    missing = sorted(set(exogenous) - set(view.future.column_names))
    if missing:
        raise DataError(
            f"this model was fitted with the exogenous features {list(exogenous)} and the "
            f"forecast view carries no {missing}"
        )
    frame = view.future.select([*metadata.instance_keys, EVENT_TIME, *exogenous]).to_pandas()
    keys = _key_rows(frame, metadata.instance_keys)
    frame[PANEL_ID] = [_unique_id(unique_ids, key) for key in keys]
    frame = frame.rename(columns={EVENT_TIME: TIME})
    frame = frame[[PANEL_ID, TIME, *exogenous]]
    return frame.sort_values([PANEL_ID, TIME], ignore_index=True)


def answer(
    view: ForecastView,
    unique_ids: dict[tuple[Any, ...], str],
    predictions: pd.DataFrame,
    column: str,
    target: str,
) -> pa.Table:
    """The canonical long forecast, from what the library returned.

    The values are ordered by event time per series and labeled with the event
    times the view asked about — never with the ones the library derived from its
    own reading of the frequency.
    """
    if column not in predictions.columns:
        raise ProviderError(
            f"the fitted model answered with the columns {list(predictions.columns)} and "
            f"{column!r} is not among them"
        )
    event_times = view.event_times
    ordered = predictions.sort_values([PANEL_ID, TIME])
    by_series = {
        str(series_id): [float(value) for value in group[column]]
        for series_id, group in ordered.groupby(PANEL_ID)
    }

    instance_keys = view.metadata.instance_keys
    keys: list[tuple[Any, ...]] = []
    times: list[datetime] = []
    values: list[float] = []
    for instance in view.instances:
        series_id = _unique_id(unique_ids, instance)
        predicted = by_series.get(series_id, [])
        if len(predicted) != len(event_times):
            raise ProviderError(
                f"the fitted model answered {len(predicted)} steps for instance {instance} "
                f"and {len(event_times)} were asked for"
            )
        keys.extend([instance] * len(event_times))
        times.extend(event_times)
        values.extend(predicted)

    columns: dict[str, pa.Array[Any]] = {
        name: pa.array([key[index] for key in keys], type=view.future.column(name).type)
        for index, name in enumerate(instance_keys)
    }
    columns[ForecastColumn.EVENT_TIME.value] = pa.array(
        times, type=view.future.column(EVENT_TIME).type
    )
    columns[ForecastColumn.TARGET.value] = pa.array([target] * len(values), type=pa.string())
    columns[ForecastColumn.KIND.value] = pa.array(["point"] * len(values), type=pa.string())
    columns[ForecastColumn.QUANTILE.value] = pa.nulls(len(values), type=pa.float64())
    columns[ForecastColumn.SAMPLE.value] = pa.nulls(len(values), type=pa.int64())
    columns[ForecastColumn.VALUE.value] = pa.array(values, type=pa.float64())
    return pa.table({name: columns[name] for name in forecast_columns(instance_keys)})


def _instances(view: SeriesView) -> dict[str, tuple[Any, ...]]:
    """``series_id -> the instance it belongs to``, from the view's key table."""
    ids: list[Any] = view.series.column(SERIES_ID).to_pylist()
    columns = [view.series.column(name).to_pylist() for name in view.schema.instance_keys]
    rows = list(zip(*columns, strict=True)) if columns else [()] * len(ids)
    return {str(series_id): row for series_id, row in zip(ids, rows, strict=True)}


def _key_rows(frame: pd.DataFrame, instance_keys: tuple[str, ...]) -> list[tuple[Any, ...]]:
    if not instance_keys:
        return [()] * len(frame)
    columns = [list(frame[name]) for name in instance_keys]
    return list(zip(*columns, strict=True))


def _unique_id(unique_ids: dict[tuple[Any, ...], str], instance: tuple[Any, ...]) -> str:
    series_id = unique_ids.get(instance)
    if series_id is None:
        raise DataError(
            f"this model is fitted per series, so it has no model for instance {instance}; "
            f"it was fitted on {sorted(str(key) for key in unique_ids)}"
        )
    return series_id
