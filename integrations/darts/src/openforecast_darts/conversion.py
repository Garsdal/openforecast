"""The translation layer: execution views in, Darts ``TimeSeries`` objects out.

```text
SeriesView    ->  one TimeSeries per series
SequenceView  ->  one TimeSeries per sample, plus past/future/static covariates
ForecastView  ->  one context TimeSeries per instance, plus the covariates
predictions   ->  the canonical forecast columns, as Arrow
```

This module is the only place in the integration where Darts' own vocabulary
appears, and it is constructed here rather than received. That vocabulary is
not Nixtla's, which is the point of the whole step:

```text
Nixtla                          Darts
unique_id, ds, y column names   a TimeSeries object per series
hist/futr/stat_exog_list        past/future/static covariates
input_size, h                   input_chunk_length, output_chunk_length
```

A view names its training units by an opaque identifier and its columns by
whatever the caller called them, so both spellings are something an integration
puts on the data on its way in and takes off again on its way out. Rule 6 of
ARCHITECTURE.md is what makes that worth a module of its own.

Three details are worth stating.

**A sample is a series to Darts, and its identity is its position.** Darts has
no identifier column at all: a panel is a *list* of ``TimeSeries``, and the
answer comes back as a list in the same order. So the mapping from the view's
``sample_id`` or instance key to a position in that list is what this module
maintains, and the list order is the only thing that labels a prediction. That
is a different bookkeeping problem from Nixtla's ``unique_id`` and it is solved
in the same place, which is what makes the two integrations swappable.

**A sample of exactly ``context + horizon`` steps is one training window.**
Darts slides a window along every series it is given, so handing it one long
series per instance would make it learn from sequences nobody described. A
``SequenceView`` already holds one window per ``instance × origin``, and a
series of exactly ``input_chunk_length + output_chunk_length`` steps admits
exactly one window — so the number of windows Darts can cut is the number of
samples and no more.

**The event times come from the view, not from the library.** Predicting ``n``
steps makes Darts extend its own time axis from the frequency it was given; the
view already names the exact event times being asked about. So the answer is
labeled from :attr:`ForecastView.event_times` and the library's index is used
only to order the values it returned.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pandas as pd
import pyarrow as pa

from openforecast.errors import DataError, ProviderError
from openforecast.models import FeatureCapabilities
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

if TYPE_CHECKING:  # ``darts`` is imported inside the calls that need it
    from darts import TimeSeries

__all__ = [
    "ForecastSeries",
    "SequenceSeries",
    "TrainingSeries",
    "answer",
    "forecast_series",
    "pandas_frequency",
    "sequence_series",
    "single_target",
    "training_series",
]

#: An instance of the caller's data, as the tuple of its key columns.
InstanceKey = tuple[Any, ...]

#: OpenForecast's frequency units in pandas' spelling, which is what Darts takes
#: as the frequency of a ``TimeSeries``. Weeks are ``7D`` rather than ``W``
#: deliberately: pandas anchors a weekly offset to a weekday, and a weekly
#: series does not have to start on the one pandas would pick.
_PANDAS_UNITS: dict[FrequencyUnit, str] = {
    FrequencyUnit.SECOND: "s",
    FrequencyUnit.MINUTE: "min",
    FrequencyUnit.HOUR: "h",
    FrequencyUnit.DAY: "D",
    FrequencyUnit.WEEK: "D",
    FrequencyUnit.MONTH: "MS",
}

_WEEK_DAYS = 7

#: The precision a ``TimeSeries`` carries, which is a decision this module has to
#: make rather than inherit. Darts derives the dtype of its tensors from the
#: series it is given, so a double-precision series builds a double-precision
#: network — slower everywhere and unsupported on some accelerators, Apple's
#: among them. Single precision is what a neural model trains in; the answer that
#: comes back out is widened again, because the canonical forecast column is
#: ``float64`` whatever the model computed in.
SINGLE = "float32"

#: What everything else keeps: a statistical model is fitted on the caller's
#: values at the precision they arrived in.
DOUBLE = "float64"


def pandas_frequency(frequency: Frequency) -> str:
    """``Frequency`` in the offset alias a Darts ``TimeSeries`` will accept."""
    unit = _PANDAS_UNITS.get(frequency.unit)
    if unit is None:  # pragma: no cover - every unit is mapped above
        raise ProviderError(f"no pandas offset alias for the frequency {frequency}")
    step = frequency.step * (_WEEK_DAYS if frequency.unit is FrequencyUnit.WEEK else 1)
    return f"{step}{unit}"


def single_target(targets: Sequence[str]) -> str:
    """The one target being modeled, or a refusal counting what was given."""
    if len(targets) != 1:
        raise ProviderError(
            f"this provider fits one target at a time and was given {len(targets)}: {list(targets)}"
        )
    return targets[0]


# -- series ------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingSeries:
    """A ``SeriesView`` as the one ``TimeSeries`` per series Darts fits on."""

    #: ``series_id -> the whole series, as Darts holds it``. One fitted model each.
    series: dict[str, TimeSeries]
    #: The caller's name for the one target being modeled.
    target: str
    #: ``series_id -> the instance key that series belongs to``.
    instances: dict[str, InstanceKey]
    #: ``series_id -> the last event time it was fitted on``.
    last_event_times: dict[str, datetime]
    frequency: str


def training_series(view: SeriesView) -> TrainingSeries:
    """Build one ``TimeSeries`` per series, and what the forecast side must remember.

    The local models exposed here condition on the target's own past and take no
    covariate, so a feature in the view is a fit the descriptor said could not
    happen; it is refused by name rather than silently dropped.
    """
    schema = view.schema
    target = single_target(schema.targets)
    features = tuple(feature.name for feature in schema.features)
    if features:
        raise ProviderError(
            f"a local Darts model exposed here conditions on the target's own past; it was "
            f"given the features {list(features)}"
        )

    frame = view.temporal.select([SERIES_ID, EVENT_TIME, target]).to_pandas()
    frame = frame.sort_values([SERIES_ID, EVENT_TIME])
    frequency = pandas_frequency(schema.frequency)
    windows = _by_identifier(frame, SERIES_ID)

    return TrainingSeries(
        series={
            series_id: _time_series(window, [target], frequency)
            for series_id, window in windows.items()
        },
        target=target,
        instances=_instances(view),
        last_event_times={
            series_id: _last_event_time(window) for series_id, window in windows.items()
        },
        frequency=frequency,
    )


# -- sequences ---------------------------------------------------------------


@dataclass(frozen=True)
class SequenceSeries:
    """A ``SequenceView`` as the lists of ``TimeSeries`` a global Darts model fits on.

    Every list is in the same order and the same length, because that order is
    the only thing that ties a covariate to the series it belongs to.
    """

    #: One target series per sample, holding exactly ``context + horizon`` steps,
    #: carrying the sample's static features as Darts' static covariates.
    targets: list[TimeSeries]
    #: The observed features, or ``None`` when there are none — which is what
    #: Darts expects rather than a list of empty series.
    past_covariates: list[TimeSeries] | None
    #: The known features, over the whole window.
    future_covariates: list[TimeSeries] | None
    #: The caller's name for the one target being modeled.
    target: str
    #: The three feature roles, in the order the lists above hold them.
    observed: tuple[str, ...]
    known: tuple[str, ...]
    static: tuple[str, ...]
    frequency: str
    #: The samples, in the order the lists hold them.
    sample_ids: tuple[str, ...]


def sequence_series(view: SequenceView, *, features: FeatureCapabilities) -> SequenceSeries:
    """Build one window per sample, with the covariates the model declared it takes.

    The three OpenForecast feature roles become Darts' three covariate kinds,
    which is the whole of the mapping: an observed feature has no value past the
    forecast origin and is therefore a past covariate, a known feature does and
    is therefore a future covariate, and a static feature has no time axis at
    all and rides on the target series.

    Which of the three a given model accepts is not the same question for every
    Darts model — its own ``supports_*_covariates`` says so — and the descriptor
    is what states it. A role that reaches here anyway is refused by name.
    """
    schema = view.schema
    target = single_target(schema.targets)
    observed = tuple(feature.name for feature in schema.observed_features)
    known = tuple(feature.name for feature in schema.known_features)
    static = schema.static_feature_names
    _require_accepted_roles(features, observed=observed, known=known, static=static)

    frequency = pandas_frequency(schema.frequency)
    frame = view.temporal.select([SAMPLE_ID, EVENT_TIME, target, *observed, *known]).to_pandas()
    frame = frame.sort_values([SAMPLE_ID, EVENT_TIME])
    windows = _by_identifier(frame, SAMPLE_ID)
    statics = _static_rows(view.static, SAMPLE_ID, static)

    sample_ids = view.sample_ids
    targets: list[TimeSeries] = []
    past: list[TimeSeries] = []
    future: list[TimeSeries] = []
    for sample_id in sample_ids:
        window = windows[sample_id]
        targets.append(
            _time_series(window, [target], frequency, static=statics.get(sample_id), dtype=SINGLE)
        )
        if observed:
            past.append(_time_series(window, observed, frequency, dtype=SINGLE))
        if known:
            future.append(_time_series(window, known, frequency, dtype=SINGLE))

    return SequenceSeries(
        targets=targets,
        past_covariates=past or None,
        future_covariates=future or None,
        target=target,
        observed=observed,
        known=known,
        static=static,
        frequency=frequency,
        sample_ids=sample_ids,
    )


# -- inference ---------------------------------------------------------------


@dataclass(frozen=True)
class ForecastSeries:
    """One inference origin, as the lists a Darts model is asked to continue."""

    #: The context window of every instance, one ``TimeSeries`` each.
    history: list[TimeSeries]
    past_covariates: list[TimeSeries] | None
    #: The known features over the context *and* the horizon, because a future
    #: covariate has to reach past the origin to be one.
    future_covariates: list[TimeSeries] | None
    #: The instances, in the order the lists hold them — which is the only thing
    #: that will label the predictions Darts returns.
    instances: tuple[InstanceKey, ...]


def forecast_series(
    view: ForecastView,
    *,
    target: str,
    observed: Sequence[str],
    known: Sequence[str],
    static: Sequence[str],
) -> ForecastSeries:
    """The context and the covariates of one origin, as Darts is asked for them.

    Nothing is remembered from the fit. A global model has one set of shared
    parameters and can be asked about an instance it never saw, so what the
    position of a series in these lists has to be is consistent within this one
    call — tying it to a training sample would forbid exactly the generalization
    the model was fitted for.
    """
    metadata = view.metadata
    keys = metadata.instance_keys
    frequency = pandas_frequency(metadata.frequency)
    instances = view.instances

    history_frame = view.history.select([*keys, EVENT_TIME, target, *observed, *known]).to_pandas()
    history_rows = _by_instance(history_frame, keys, instances)
    future_frame = None
    if known:
        _require_columns(view.future.column_names, known, "known features")
        future_frame = view.future.select([*keys, EVENT_TIME, *known]).to_pandas()
        future_rows = _by_instance(future_frame, keys, instances)
    statics = _instance_statics(view, static)

    history: list[TimeSeries] = []
    past: list[TimeSeries] = []
    future: list[TimeSeries] = []
    for instance in instances:
        rows = history_rows[instance]
        history.append(
            _time_series(rows, [target], frequency, static=statics.get(instance), dtype=SINGLE)
        )
        if observed:
            past.append(_time_series(rows, observed, frequency, dtype=SINGLE))
        if known:
            span = pd.concat(
                [rows[[EVENT_TIME, *known]], future_rows[instance][[EVENT_TIME, *known]]]
            )
            future.append(_time_series(span, known, frequency, dtype=SINGLE))

    return ForecastSeries(
        history=history,
        past_covariates=past or None,
        future_covariates=future or None,
        instances=instances,
    )


def answer(
    view: ForecastView,
    predictions: Sequence[TimeSeries],
    *,
    instances: Sequence[InstanceKey],
    target: str,
) -> pa.Table:
    """The canonical long forecast, from the series Darts returned.

    The values are ordered by the time index Darts gave them and labeled with
    the event times the view asked about — never with the ones the library
    derived from its own reading of the frequency.
    """
    if len(predictions) != len(instances):
        raise ProviderError(
            f"the fitted model answered about {len(predictions)} series and "
            f"{len(instances)} were asked about"
        )
    event_times = view.event_times
    keys: list[InstanceKey] = []
    times: list[datetime] = []
    values: list[float] = []
    for instance, prediction in zip(instances, predictions, strict=True):
        predicted = _predicted_values(prediction)
        if len(predicted) != len(event_times):
            raise ProviderError(
                f"the fitted model answered {len(predicted)} steps for instance {instance} "
                f"and {len(event_times)} were asked for"
            )
        keys.extend([instance] * len(event_times))
        times.extend(event_times)
        values.extend(predicted)

    instance_keys = view.metadata.instance_keys
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


# -- the pieces --------------------------------------------------------------


def _time_series(
    frame: pd.DataFrame,
    columns: Sequence[str],
    frequency: str,
    static: pd.DataFrame | None = None,
    dtype: str = DOUBLE,
) -> TimeSeries:
    """``columns`` of ``frame``, on its event-time axis, as Darts holds a series."""
    from darts import TimeSeries

    values = _numeric(frame, columns, dtype)
    values.index = pd.DatetimeIndex(frame[EVENT_TIME])
    try:
        series = TimeSeries.from_dataframe(values, freq=frequency)
    except Exception as error:
        # A frame Darts refuses to read as a series — a gap on the frequency
        # grid, most likely — is data the caller can act on.
        raise ProviderError(
            f"these event times do not form a series on a {frequency} grid: "
            f"{type(error).__name__}: {error}"
        ) from error
    return series if static is None else series.with_static_covariates(static)


def _numeric(frame: pd.DataFrame, columns: Sequence[str], dtype: str = DOUBLE) -> pd.DataFrame:
    """``columns``, as the floating-point values a Darts model consumes."""
    try:
        return frame[list(columns)].astype(dtype)
    except (TypeError, ValueError) as error:
        raise ProviderError(
            f"a Darts model takes numeric values and {list(columns)} holds something else: {error}"
        ) from error


def _by_identifier(frame: pd.DataFrame, identifier: str) -> dict[str, pd.DataFrame]:
    """The rows of ``frame``, grouped by the view's own opaque identifier."""
    return {
        str(key): rows
        for key, rows in frame.groupby(identifier, sort=False)  # pyright: ignore[reportUnknownArgumentType]
    }


def _by_instance(
    frame: pd.DataFrame, instance_keys: Sequence[str], instances: Sequence[InstanceKey]
) -> dict[InstanceKey, pd.DataFrame]:
    """The rows of ``frame``, grouped by the caller's instance key columns."""
    rows = _key_rows(frame, tuple(instance_keys))
    positions: dict[InstanceKey, list[int]] = {instance: [] for instance in instances}
    for position, key in enumerate(rows):
        if key in positions:
            positions[key].append(position)
    grouped: dict[InstanceKey, pd.DataFrame] = {}
    for instance, found in positions.items():
        if not found:
            raise DataError(f"this forecast view holds no rows for instance {instance}")
        grouped[instance] = frame.iloc[found].sort_values(EVENT_TIME)
    return grouped


def _static_rows(
    static: pa.Table | None, identifier: str, names: Sequence[str]
) -> dict[str, pd.DataFrame]:
    """``identifier -> the one-row frame Darts takes as static covariates``."""
    if not names:
        return {}
    if static is None:  # pragma: no cover - the view refuses to be built this way
        raise ProviderError(
            f"this view declares the static features {list(names)} and holds no static table"
        )
    frame = static.select([identifier, *names]).to_pandas()
    return {
        str(row[identifier]): _numeric(frame.iloc[[position]], names, SINGLE).reset_index(drop=True)
        for position, row in enumerate(frame.to_dict("records"))
    }


def _instance_statics(view: ForecastView, names: Sequence[str]) -> dict[InstanceKey, pd.DataFrame]:
    """The static covariates of every instance in a forecast view."""
    if not names:
        return {}
    if view.static is None:
        raise DataError(
            f"this model was fitted with the static features {list(names)} and the forecast "
            f"view carries none"
        )
    _require_columns(view.static.column_names, names, "static features")
    keys = view.metadata.instance_keys
    frame = view.static.select([*keys, *names]).to_pandas()
    rows = _key_rows(frame, tuple(keys))
    return {
        key: _numeric(frame.iloc[[position]], names, SINGLE).reset_index(drop=True)
        for position, key in enumerate(rows)
    }


def _require_columns(present: Sequence[str], wanted: Sequence[str], role: str) -> None:
    absent = sorted(set(wanted) - set(present))
    if absent:
        raise DataError(
            f"this model was fitted with the {role} {list(wanted)} and the forecast view "
            f"carries no {absent}"
        )


def _require_accepted_roles(
    features: FeatureCapabilities,
    *,
    observed: Sequence[str],
    known: Sequence[str],
    static: Sequence[str],
) -> None:
    """A feature role the native model has no covariate for is refused by name."""
    roles = (
        ("observed", observed, features.observed),
        ("known", known, features.known),
        ("static", static, features.static),
    )
    refused = sorted(name for _, names, accepted in roles if not accepted for name in names)
    if refused:
        accepted = sorted(role for role, _, supported in roles if supported)
        raise ProviderError(
            f"this model takes no covariate for the features {refused}; the feature roles "
            f"it accepts are {accepted}"
        )


def _predicted_values(prediction: TimeSeries) -> list[float]:
    """One univariate Darts series as plain floats, in time order."""
    frame = prediction.to_dataframe().sort_index()
    if frame.shape[1] != 1:
        raise ProviderError(
            f"the fitted model answered with {frame.shape[1]} components and one target "
            f"was asked about"
        )
    return [float(value) for value in frame.iloc[:, 0]]


def _instances(view: SeriesView) -> dict[str, InstanceKey]:
    """``series_id -> the instance it belongs to``, from the view's key table."""
    ids: list[Any] = view.series.column(SERIES_ID).to_pylist()
    columns = [view.series.column(name).to_pylist() for name in view.schema.instance_keys]
    rows = list(zip(*columns, strict=True)) if columns else [()] * len(ids)
    return {str(series_id): row for series_id, row in zip(ids, rows, strict=True)}


def _last_event_time(frame: pd.DataFrame) -> datetime:
    moment: Any = frame[EVENT_TIME].max()
    return pd.Timestamp(moment).to_pydatetime()


def _key_rows(frame: pd.DataFrame, instance_keys: tuple[str, ...]) -> list[InstanceKey]:
    if not instance_keys:
        return [()] * len(frame)
    columns = [list(frame[name]) for name in instance_keys]
    return list(zip(*columns, strict=True))


def unique_instances(mapping: Mapping[str, InstanceKey]) -> tuple[InstanceKey, ...]:
    """The instances a per-series fit covers, in a stable order."""
    return tuple(dict.fromkeys(mapping.values()))
