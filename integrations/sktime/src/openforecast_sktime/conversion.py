"""The translation layer: execution views in, sktime's pandas containers out.

```text
SeriesView    ->  one pd.Series per series, on a DatetimeIndex
SequenceView  ->  one panel DataFrame, MultiIndex (sample_id, event_time)
ForecastView  ->  the same panel, keyed by instance instead of by sample
predictions   ->  the canonical forecast columns, as Arrow
```

This module is the only place in the integration where sktime's own vocabulary
appears, and it is constructed here rather than received. That vocabulary is a
third one, which is the point of the step:

```text
Nixtla                     Darts                      sktime
unique_id, ds, y columns   a TimeSeries per series    a pandas MultiIndex level
hist/futr/stat_exog_list   past/future covariates     one exogenous X frame
input_size, h              input_chunk_length         window_length, fh
```

A view names its training units by an opaque identifier and its columns by
whatever the caller called them, so every one of those spellings is something an
integration puts on the data on its way in and takes off again on its way out.
Rule 6 of ARCHITECTURE.md is what makes that worth a module of its own.

Three details are worth stating.

**A panel is a MultiIndex, and its outer level is the training unit.** sktime's
explicit panel format is exactly what a ``SequenceView`` already holds: one
identifier per training unit, one event-time axis inside it. So the mapping is
a ``set_index`` rather than a reshape, and it is the same mapping at inference
with the instance in place of the sample — which is what makes an instance the
model never saw forecastable at all.

**A sample of exactly ``context + horizon`` steps is one training window.** A
pooled reducer slides a window along every series it is given, so handing it one
long series per instance would make it learn from sequences nobody described. A
``SequenceView`` already holds one window per ``instance × origin``.

**The event times come from the view, not from the library.** A relative
``ForecastingHorizon`` makes sktime derive its own event times from the cutoff
and the inferred frequency; the view already names the exact event times being
asked about. So the answer is labeled from :attr:`ForecastView.event_times`, and
what the library returned is checked against them rather than trusted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

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

__all__ = [
    "ForecastPanel",
    "InstanceKey",
    "PANEL_LEVELS",
    "SequencePanel",
    "TrainingSeries",
    "answer",
    "forecast_panel",
    "instance_labels",
    "pandas_frequency",
    "sequence_panel",
    "single_target",
    "training_series",
]

#: An instance of the caller's data, as the tuple of its key columns.
InstanceKey = tuple[Any, ...]

#: The index levels of every panel this module builds. sktime reads the outer
#: level as the series identifier and the inner one as the time axis; keeping
#: the names identical between fit and inference is what lets a fitted panel
#: model be handed a series it has never seen.
PANEL_LEVELS = (SAMPLE_ID, EVENT_TIME)

#: OpenForecast's frequency units in pandas' spelling, which is what sktime
#: infers a forecasting horizon from. Weeks are ``7D`` rather than ``W``
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

#: What a fitted model computes in. Every model here is a scikit-learn regressor
#: or a statsmodels fit under the hood, and both work in double precision, so
#: the caller's values are neither narrowed on the way in nor widened on the way
#: out — the canonical forecast column is ``float64`` and so is the arithmetic.
DOUBLE = "float64"


def pandas_frequency(frequency: Frequency) -> str:
    """``Frequency`` in the offset alias a pandas index will accept."""
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


def instance_labels(instances: Sequence[InstanceKey]) -> tuple[str, ...]:
    """A panel level value per instance, in the order the view holds them.

    Positional rather than derived from the caller's key: an instance key is the
    caller's data, of whatever type they chose, and a pandas index level is not
    the place to find out that two of them stringify the same. The position is
    what maps a returned row back to the instance it is about.
    """
    return tuple(f"instance-{position}" for position in range(len(instances)))


# -- series ------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingSeries:
    """A ``SeriesView`` as the one ``pd.Series`` per series sktime fits on."""

    #: ``series_id -> the whole series``. One fitted forecaster each.
    series: dict[str, pd.Series[float]]
    #: The caller's name for the one target being modeled.
    target: str
    #: ``series_id -> the instance key that series belongs to``.
    instances: dict[str, InstanceKey]
    #: ``series_id -> the last event time it was fitted on``.
    last_event_times: dict[str, datetime]
    frequency: str


def training_series(view: SeriesView) -> TrainingSeries:
    """Build one series per training unit, and what the forecast side must remember.

    The local forecasters exposed here condition on the target's own past and
    take no exogenous column, so a feature in the view is a fit the descriptor
    said could not happen; it is refused by name rather than silently dropped.
    """
    schema = view.schema
    target = single_target(schema.targets)
    features = tuple(feature.name for feature in schema.features)
    if features:
        raise ProviderError(
            f"a local sktime forecaster exposed here conditions on the target's own past; "
            f"it was given the features {list(features)}"
        )

    frame = view.temporal.select([SERIES_ID, EVENT_TIME, target]).to_pandas()
    frame = frame.sort_values([SERIES_ID, EVENT_TIME])
    frequency = pandas_frequency(schema.frequency)
    windows = {
        str(key): rows
        for key, rows in frame.groupby(SERIES_ID, sort=False)  # pyright: ignore[reportUnknownArgumentType]
    }

    return TrainingSeries(
        series={
            series_id: _series(window, target, frequency) for series_id, window in windows.items()
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
class SequencePanel:
    """A ``SequenceView`` as the one panel a pooled sktime forecaster fits on."""

    #: One column, the target, indexed by ``(sample_id, event_time)``.
    y: pd.DataFrame
    #: The exogenous columns over the whole window, or ``None`` when the model
    #: was given no feature — which is what sktime expects rather than an empty
    #: frame.
    X: pd.DataFrame | None
    #: The caller's name for the one target being modeled.
    target: str
    #: The two feature roles this model consumes, in the order ``X`` holds them.
    known: tuple[str, ...]
    static: tuple[str, ...]
    frequency: str
    #: The samples, in the order the panel holds them.
    sample_ids: tuple[str, ...]


def sequence_panel(view: SequenceView, *, features: FeatureCapabilities) -> SequencePanel:
    """Build the panel of every sample, with the columns the model declared it takes.

    The OpenForecast feature roles become one exogenous frame, which is all
    sktime has: a value is either in ``X`` at an event time or it is not. That
    makes the *known* role the one a panel forecaster can consume — ``X`` has to
    reach past the forecast origin, and an observed feature has no value there —
    and a static feature a column that is constant within its sample.

    Which roles a given model accepts is a
    :class:`~openforecast.models.FeatureCapabilities` declaration, and a role
    that reaches here anyway is refused by name.
    """
    schema = view.schema
    target = single_target(schema.targets)
    observed = tuple(feature.name for feature in schema.observed_features)
    known = tuple(feature.name for feature in schema.known_features)
    static = schema.static_feature_names
    _require_accepted_roles(features, observed=observed, known=known, static=static)

    frequency = pandas_frequency(schema.frequency)
    frame = view.temporal.select([SAMPLE_ID, EVENT_TIME, target, *known]).to_pandas()
    if static:
        frame = _with_statics(frame, _static_table(view.static, SAMPLE_ID, static), static)
    panel = _panel(frame, SAMPLE_ID, frequency)

    exogenous = (*known, *static)
    return SequencePanel(
        y=_numeric(panel, [target]),
        X=_numeric(panel, exogenous) if exogenous else None,
        target=target,
        known=known,
        static=static,
        frequency=frequency,
        sample_ids=view.sample_ids,
    )


# -- inference ---------------------------------------------------------------


@dataclass(frozen=True)
class ForecastPanel:
    """One inference origin, as the panel a fitted model is asked to continue."""

    #: The context window of every instance, keyed by its panel label.
    y: pd.DataFrame
    #: The exogenous columns over the context *and* the horizon, because a value
    #: sktime conditions a forecast on has to exist at the event time forecast.
    X: pd.DataFrame | None
    #: ``panel label -> the instance it stands for``, in the view's own order.
    instances: dict[str, InstanceKey]


def forecast_panel(
    view: ForecastView,
    *,
    target: str,
    known: Sequence[str],
    static: Sequence[str],
) -> ForecastPanel:
    """The context and the exogenous columns of one origin, as sktime is asked for them.

    Nothing is remembered from the fit. A pooled model has one set of shared
    parameters and can be asked about an instance it never saw, so the panel
    label an instance gets has to be consistent within this one call only —
    tying it to a training sample would forbid exactly the generalization the
    model was fitted for.
    """
    metadata = view.metadata
    keys = metadata.instance_keys
    frequency = pandas_frequency(metadata.frequency)
    instances = view.instances
    labels = instance_labels(instances)
    positions = dict(zip(instances, labels, strict=True))

    history = view.history.select([*keys, EVENT_TIME, target, *known]).to_pandas()
    history = _labeled(history, keys, positions, "history")
    if known:
        _require_columns(view.future.column_names, known, "known features")
    future = view.future.select([*keys, EVENT_TIME, *known]).to_pandas()
    future = _labeled(future, keys, positions, "future")

    span = pd.concat([history[[SAMPLE_ID, EVENT_TIME, *known]], future])
    if static:
        statics = _instance_statics(view, static)
        span = _with_statics(span, statics, static)

    exogenous = (*known, *static)
    return ForecastPanel(
        y=_numeric(_panel(history, SAMPLE_ID, frequency), [target]),
        X=_numeric(_panel(span, SAMPLE_ID, frequency), exogenous) if exogenous else None,
        instances={label: instance for instance, label in positions.items()},
    )


def answer(
    view: ForecastView,
    predicted: pd.DataFrame,
    *,
    instances: Mapping[str, InstanceKey],
    target: str,
) -> pa.Table:
    """The canonical long forecast, from the panel sktime returned.

    A pooled forecaster answers about every series it holds, which after an
    update is the training panel as well as the instances asked about — so the
    rows are selected by panel label rather than assumed, and every instance the
    view asked about has to be among them.

    The values are ordered by the event times the view asked about and labeled
    with them, never with the ones the library derived from its own reading of
    the frequency.
    """
    event_times = view.event_times
    frame = predicted.sort_index()
    if frame.shape[1] != 1:
        raise ProviderError(
            f"the fitted model answered with {frame.shape[1]} columns and one target "
            f"was asked about"
        )

    keys: list[InstanceKey] = []
    times: list[datetime] = []
    values: list[float] = []
    for label, instance in instances.items():
        if label not in set(frame.index.get_level_values(0)):
            raise ProviderError(
                f"the fitted model answered about {frame.index.get_level_values(0).nunique()} "
                f"series and instance {instance} is not among them"
            )
        rows = frame.xs(label, level=0)
        answered = {
            pd.Timestamp(moment).to_pydatetime(): position
            for position, moment in enumerate(rows.index)
        }
        missing = [moment for moment in event_times if moment not in answered]
        if missing:
            raise ProviderError(
                f"the fitted model answered {len(rows)} steps for instance {instance} and "
                f"the {len(event_times)} event times asked about are not among them"
            )
        keys.extend([instance] * len(event_times))
        times.extend(event_times)
        values.extend(float(rows.iloc[answered[moment], 0]) for moment in event_times)

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


def _series(frame: pd.DataFrame, target: str, frequency: str) -> pd.Series[float]:
    """One series of a ``SeriesView``, on the event-time axis sktime reads."""
    values = _numeric(frame.set_index(_time_index(frame, frequency)), [target])
    return values[target]


def _panel(frame: pd.DataFrame, identifier: str, frequency: str) -> pd.DataFrame:
    """``frame`` as sktime's explicit panel: a MultiIndex of unit and event time."""
    ordered = frame.sort_values([identifier, EVENT_TIME])
    index = pd.MultiIndex.from_arrays(
        [ordered[identifier], pd.DatetimeIndex(ordered[EVENT_TIME], freq=None)],
        names=PANEL_LEVELS,
    )
    panel = ordered.drop(columns=[identifier, EVENT_TIME]).set_index(index)
    if panel.index.has_duplicates:
        raise DataError(
            f"a panel holds one row per unit and event time, and this view holds "
            f"{int(panel.index.duplicated().sum())} repeated pairs; the {frequency} grid "
            f"a forecaster reads cannot be recovered from them"
        )
    return panel


def _time_index(frame: pd.DataFrame, frequency: str) -> pd.DatetimeIndex:
    """The event-time axis of one series, carrying the frequency sktime asks for."""
    moments = pd.DatetimeIndex(frame[EVENT_TIME])
    try:
        return pd.DatetimeIndex(moments, freq=frequency)
    except ValueError as error:
        # An index pandas refuses to stamp with the frequency — a gap on the
        # grid, most likely — is data the caller can act on.
        raise ProviderError(
            f"these event times do not form a series on a {frequency} grid: {error}"
        ) from error


def _numeric(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """``columns``, as the floating-point values a fitted regressor consumes."""
    try:
        return frame[list(columns)].astype(DOUBLE)
    except (TypeError, ValueError) as error:
        raise ProviderError(
            f"an sktime forecaster takes numeric values and {list(columns)} holds "
            f"something else: {error}"
        ) from error


def _labeled(
    frame: pd.DataFrame,
    instance_keys: Sequence[str],
    positions: Mapping[InstanceKey, str],
    table: str,
) -> pd.DataFrame:
    """``frame`` with its instance key columns replaced by one panel label."""
    keys = _key_rows(frame, tuple(instance_keys))
    labels = [positions.get(key) for key in keys]
    unknown = sorted({str(key) for key, label in zip(keys, labels, strict=True) if label is None})
    if unknown:  # pragma: no cover - the view is built from its own instances
        raise DataError(f"the {table} of this forecast view holds unknown instances {unknown}")
    relabeled = frame.drop(columns=list(instance_keys)).copy()
    relabeled[SAMPLE_ID] = labels
    return relabeled


def _with_statics(
    frame: pd.DataFrame, values: Mapping[str, Mapping[str, Any]], names: Sequence[str]
) -> pd.DataFrame:
    """The static features, broadcast to every row of the unit they belong to.

    A static feature has no time axis and sktime's exogenous frame has nothing
    else, so "constant within its series" is how one is spelled here. That is a
    translation rather than a fabrication: every row of a unit carries the value
    the view recorded once for it.
    """
    joined = frame.copy()
    for name in names:
        joined[name] = [values[str(unit)][name] for unit in frame[SAMPLE_ID]]
    return joined


def _static_table(
    static: pa.Table | None, identifier: str, names: Sequence[str]
) -> Mapping[str, Mapping[str, Any]]:
    """``identifier -> its static values``, from a fit view's static table."""
    if static is None:  # pragma: no cover - the view refuses to be built this way
        raise ProviderError(
            f"this view declares the static features {list(names)} and holds no static table"
        )
    frame = static.select([identifier, *names]).to_pandas()
    return {
        str(row[identifier]): {name: row[name] for name in names}
        for row in frame.to_dict("records")
    }


def _instance_statics(view: ForecastView, names: Sequence[str]) -> dict[str, dict[str, Any]]:
    """The static features of every instance in a forecast view, by panel label."""
    if view.static is None:
        raise DataError(
            f"this model was fitted with the static features {list(names)} and the forecast "
            f"view carries none"
        )
    _require_columns(view.static.column_names, names, "static features")
    keys = view.metadata.instance_keys
    frame = view.static.select([*keys, *names]).to_pandas()
    positions = dict(zip(view.instances, instance_labels(view.instances), strict=True))
    rows = _key_rows(frame, tuple(keys))
    found: dict[str, dict[str, Any]] = {}
    for position, key in enumerate(rows):
        label = positions.get(key)
        if label is not None:
            found[label] = {name: frame.iloc[position][name] for name in names}
    return found


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
    """A feature role the native model has no column for is refused by name."""
    roles = (
        ("observed", observed, features.observed),
        ("known", known, features.known),
        ("static", static, features.static),
    )
    refused = sorted(name for _, names, accepted in roles if not accepted for name in names)
    if refused:
        accepted = sorted(role for role, _, supported in roles if supported)
        raise ProviderError(
            f"this model takes no exogenous column for the features {refused}; the feature "
            f"roles it accepts are {accepted}"
        )


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
