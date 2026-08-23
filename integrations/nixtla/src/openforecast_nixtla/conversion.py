"""The translation layer: execution views in, Nixtla's long frame out.

```text
SeriesView    ->  unique_id, ds, y, exogenous columns
SequenceView  ->  unique_id, ds, y, hist/futr/stat exogenous columns
ForecastView  ->  unique_id, ds, y  +  the future half, separately
predictions   ->  the canonical forecast columns, as Arrow — point or quantile
```

This module is the only place in the integration where ``unique_id``, ``ds``,
``y`` and the three ``*_exog_list`` roles appear, and they are constructed here
rather than received: a view names its training units by an opaque identifier
and its columns by whatever the caller called them, so the Nixtla spelling is
something this integration puts on the data on its way in and takes off again on
its way out. Rule 6 of ARCHITECTURE.md is what makes that worth a module of its
own.

Three details are worth stating.

**The instance keys are not in the frame.** Nixtla identifies a series by one
string, and OpenForecast identifies it by the caller's key columns — possibly
several, possibly not strings. So the identifier the view already assigns is
used as ``unique_id`` and the mapping back to instance keys is persisted beside
the fitted model, or rebuilt for the one origin a forecast is made at. A
forecast comes back labeled with the instance it is about because that mapping
was written down, not because a key survived a round trip through somebody's
dataframe.

**A sequence sample is a series to Nixtla.** A ``SequenceView`` already holds one
``context + horizon`` window per ``instance × origin`` under its own
``sample_id``, so the mapping is ``sample_id -> unique_id`` and nothing else.
That is what keeps a global model from learning across two forecast origins: the
library is handed windows of exactly the model's own ``input_size + h``, so
there is no second window for it to slide onto.

**The event times come from the view, not from the library.** Predicting ``h``
steps makes a library extend its own time axis using its own spelling of the
frequency; the view already names the exact event times being asked about. So
the answer is labeled from :attr:`ForecastView.event_times` and the library's
``ds`` is used only to order the values it returned.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
import pyarrow as pa

from openforecast.errors import DataError, ProviderError
from openforecast.views import (
    EVENT_TIME,
    SAMPLE_ID,
    SERIES_ID,
    ForecastColumn,
    ForecastView,
    Frequency,
    FrequencyUnit,
    SequenceView,
    SeriesView,
    forecast_columns,
)

__all__ = [
    "PANEL_ID",
    "TARGET",
    "TIME",
    "ForecastFrames",
    "SequenceFrames",
    "TrainingFrame",
    "answer",
    "forecast_frames",
    "future_frame",
    "pandas_frequency",
    "quantile_answer",
    "sequence_frames",
    "single_target",
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


def single_target(targets: Sequence[str]) -> str:
    """The one target being modeled, or a refusal counting what was given."""
    if len(targets) != 1:
        raise ProviderError(
            f"this provider fits one target at a time and was given {len(targets)}: {list(targets)}"
        )
    return targets[0]


def training_frame(view: SeriesView) -> TrainingFrame:
    """Build the long frame, and everything the forecast side needs remembered."""
    schema = view.schema
    target = single_target(schema.targets)
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


# -- sequences ---------------------------------------------------------------


@dataclass(frozen=True)
class SequenceFrames:
    """A ``SequenceView`` in the representation a global Nixtla model trains on."""

    #: ``unique_id, ds, y`` and one column per temporal covariate. One
    #: ``unique_id`` per sample, holding exactly ``context + horizon`` rows.
    frame: pd.DataFrame
    #: ``unique_id`` and the static covariates, or ``None`` when there are none.
    static: pd.DataFrame | None
    #: The caller's name for the one target being modeled.
    target: str
    #: The three covariate roles, in Nixtla's vocabulary.
    hist_exog: tuple[str, ...]
    futr_exog: tuple[str, ...]
    stat_exog: tuple[str, ...]
    frequency: str


def sequence_frames(view: SequenceView) -> SequenceFrames:
    """Build the long frame of one window per sample, and the static frame beside it.

    The three OpenForecast feature roles become the three Nixtla covariate
    lists, which is the whole of the mapping: an observed feature has no value
    past the forecast origin and is therefore historical, a known feature does
    and is therefore future, and a static feature has no time axis at all.
    """
    schema = view.schema
    target = single_target(schema.targets)
    hist_exog = tuple(feature.name for feature in schema.observed_features)
    futr_exog = tuple(feature.name for feature in schema.known_features)
    stat_exog = schema.static_feature_names

    columns = [SAMPLE_ID, EVENT_TIME, target, *hist_exog, *futr_exog]
    frame = view.temporal.select(columns).to_pandas()
    frame = frame.rename(columns={SAMPLE_ID: PANEL_ID, EVENT_TIME: TIME, target: TARGET})
    frame = frame.sort_values([PANEL_ID, TIME], ignore_index=True)

    static = None
    if stat_exog:
        if view.static is None:  # pragma: no cover - the view refuses to be built this way
            raise ProviderError(
                f"this view declares the static features {list(stat_exog)} and holds no "
                f"static table"
            )
        static = view.static.select([SAMPLE_ID, *stat_exog]).to_pandas()
        static = static.rename(columns={SAMPLE_ID: PANEL_ID})

    return SequenceFrames(
        frame=frame,
        static=static,
        target=target,
        hist_exog=hist_exog,
        futr_exog=futr_exog,
        stat_exog=stat_exog,
        frequency=pandas_frequency(schema.frequency),
    )


@dataclass(frozen=True)
class ForecastFrames:
    """One inference origin, split the way a global Nixtla model is asked."""

    #: The context window: ``unique_id, ds, y`` and the temporal covariates.
    history: pd.DataFrame
    #: The event times being forecast and the covariates known for them, or
    #: ``None`` when the fitted model conditions on no future covariate.
    future: pd.DataFrame | None
    static: pd.DataFrame | None
    #: ``instance key -> unique_id``, so the answer can be labeled again.
    unique_ids: dict[tuple[Any, ...], str]


def forecast_frames(
    view: ForecastView,
    *,
    target: str,
    hist_exog: Sequence[str],
    futr_exog: Sequence[str],
    stat_exog: Sequence[str],
) -> ForecastFrames:
    """The context, the future and the statics of one origin, in Nixtla's spelling.

    The identifiers are minted here rather than remembered from the fit. A
    global model has one set of shared parameters and can be asked about an
    instance it never saw, so what a ``unique_id`` has to be is unique within
    this one call — tying it to a training sample would forbid exactly the
    generalization the model was fitted for.
    """
    metadata = view.metadata
    instance_keys = metadata.instance_keys
    unique_ids = {instance: f"instance-{index}" for index, instance in enumerate(view.instances)}

    history = _labeled(
        view.history.select([*instance_keys, EVENT_TIME, target, *hist_exog, *futr_exog]),
        instance_keys,
        unique_ids,
    )
    history = history.rename(columns={target: TARGET})
    history = history[[PANEL_ID, TIME, TARGET, *hist_exog, *futr_exog]]

    future = None
    if futr_exog:
        absent = sorted(set(futr_exog) - set(view.future.column_names))
        if absent:
            raise DataError(
                f"this model was fitted with the known features {list(futr_exog)} and the "
                f"forecast view carries no {absent}"
            )
        future = _labeled(
            view.future.select([*instance_keys, EVENT_TIME, *futr_exog]), instance_keys, unique_ids
        )
        future = future[[PANEL_ID, TIME, *futr_exog]]

    static = None
    if stat_exog:
        if view.static is None:
            raise DataError(
                f"this model was fitted with the static features {list(stat_exog)} and the "
                f"forecast view carries none"
            )
        absent = sorted(set(stat_exog) - set(view.static.column_names))
        if absent:
            raise DataError(
                f"this model was fitted with the static features {list(stat_exog)} and the "
                f"forecast view carries no {absent}"
            )
        static = view.static.select([*instance_keys, *stat_exog]).to_pandas()
        keys = _key_rows(static, instance_keys)
        static[PANEL_ID] = [_unique_id(unique_ids, key) for key in keys]
        static = static[[PANEL_ID, *stat_exog]]

    return ForecastFrames(
        history=history.sort_values([PANEL_ID, TIME], ignore_index=True),
        future=None if future is None else future.sort_values([PANEL_ID, TIME], ignore_index=True),
        static=static,
        unique_ids=unique_ids,
    )


def _labeled(
    table: pa.Table, instance_keys: Sequence[str], unique_ids: dict[tuple[Any, ...], str]
) -> pd.DataFrame:
    """An Arrow table as a long frame keyed by ``unique_id`` and ``ds``."""
    frame = table.to_pandas()
    keys = _key_rows(frame, tuple(instance_keys))
    frame[PANEL_ID] = [_unique_id(unique_ids, key) for key in keys]
    return frame.rename(columns={EVENT_TIME: TIME})


def answer(
    view: ForecastView,
    unique_ids: dict[tuple[Any, ...], str],
    predictions: pd.DataFrame,
    column: str,
    target: str,
) -> pa.Table:
    """The canonical long point forecast, from what the library returned.

    The values are ordered by event time per series and labeled with the event
    times the view asked about — never with the ones the library derived from its
    own reading of the frequency.
    """
    return _long_answer(view, unique_ids, predictions, target, ((None, column),))


def quantile_answer(
    view: ForecastView,
    unique_ids: dict[tuple[Any, ...], str],
    predictions: pd.DataFrame,
    columns: Sequence[tuple[float, str]],
    target: str,
) -> pa.Table:
    """The canonical long quantile forecast: one row per level, ascending.

    ``columns`` says which returned column holds which quantile level, which is
    the whole of the translation — a StatsForecast interval bound is a quantile
    of the predictive distribution under another name, and naming the mapping
    here rather than in the adapter keeps the library's column spellings in the
    one module rule 6 confines them to.
    """
    return _long_answer(view, unique_ids, predictions, target, columns)


def _long_answer(
    view: ForecastView,
    unique_ids: dict[tuple[Any, ...], str],
    predictions: pd.DataFrame,
    target: str,
    parts: Sequence[tuple[float | None, str]],
) -> pa.Table:
    """One row per instance, event time and requested part of the distribution.

    ``parts`` is ``(quantile level or None, column)``: a point forecast is one
    part holding no level, and a quantile forecast is one part per level. Written
    once for both because everything except the ``kind`` and ``quantile`` columns
    is the same labeling job, and doing it twice is how two answers of one
    provider drift apart.
    """
    missing = [column for _, column in parts if column not in predictions.columns]
    if missing:
        raise ProviderError(
            f"the fitted model answered with the columns {list(predictions.columns)} and "
            f"{missing} are not among them"
        )
    event_times = view.event_times
    ordered = predictions.sort_values([PANEL_ID, TIME])
    by_series = {
        str(series_id): {column: [float(value) for value in group[column]] for _, column in parts}
        for series_id, group in ordered.groupby(PANEL_ID)
    }

    instance_keys = view.metadata.instance_keys
    keys: list[tuple[Any, ...]] = []
    times: list[datetime] = []
    levels: list[float | None] = []
    values: list[float] = []
    for instance in view.instances:
        series_id = _unique_id(unique_ids, instance)
        answered = by_series.get(series_id, {})
        for position, moment in enumerate(event_times):
            for level, column in parts:
                predicted = answered.get(column, [])
                if len(predicted) != len(event_times):
                    raise ProviderError(
                        f"the fitted model answered {len(predicted)} steps for instance "
                        f"{instance} and {len(event_times)} were asked for"
                    )
                keys.append(instance)
                times.append(moment)
                levels.append(level)
                values.append(predicted[position])

    is_point = all(level is None for level, _ in parts)
    columns: dict[str, pa.Array[Any]] = {
        name: pa.array([key[index] for key in keys], type=view.future.column(name).type)
        for index, name in enumerate(instance_keys)
    }
    columns[ForecastColumn.EVENT_TIME.value] = pa.array(
        times, type=view.future.column(EVENT_TIME).type
    )
    columns[ForecastColumn.TARGET.value] = pa.array([target] * len(values), type=pa.string())
    columns[ForecastColumn.KIND.value] = pa.array(
        ["point" if is_point else "quantile"] * len(values), type=pa.string()
    )
    columns[ForecastColumn.QUANTILE.value] = pa.array(levels, type=pa.float64())
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
