"""``ViewPlanner``: the one place that knows where the data came from.

```text
TimeSeriesFrame  ─┐                        ┌─ SeriesView
ForecastDataset  ─┼─►  ViewPlanner  ───────┼─ SequenceView
ForecastContext  ─┘                        ├─ TabularView
                                           └─ ForecastView
```

Every branch on "is this event-time or point-in-time data" that would otherwise
appear in every integration lives here instead, exactly once. The mapping the
planner implements:

```text
                     Series      Sequences    Tabular

TimeSeriesFrame       yes         yes          yes
ForecastDataset       selected    yes          yes
ForecastContext       forecast    forecast     forecast
```

For a ``TimeSeriesFrame``, historical forecast origins are *simulated*: the
window is cut out of one freshest series, so the feature values at each origin
are today's. For a ``ForecastDataset`` the origins are *observed* — each sample
carries the values that actually existed at its origin. The resulting views are
the same type, and
:class:`~openforecast.views.provenance.OriginFidelity` is what tells them apart.

``ViewRequest`` is the (contract, fit plan, task) triple, flattened: a model's
``TrainingContract`` supplies ``kind``, a ``FitPlan`` the origins and the context
length, and a ``ForecastTask`` the horizon. :meth:`ViewRequest.for_contract` is
that translation, so the engine of Step 8 reads a descriptor and a plan and has
nothing left to decide.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any, Self

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field, model_validator

from openforecast.data._arrow import InstanceKey, build_table, column_values, key_rows, summarize
from openforecast.data.features import FeatureSpec
from openforecast.data.forecast_context import ForecastContext
from openforecast.data.frequency import Frequency
from openforecast.errors import DataError, OriginScopeError, RecipeError, SchemaError
from openforecast.models.contract import TrainingContract
from openforecast.tasks.forecast import ForecastTask
from openforecast.tasks.origins import AllOrigins, OriginMode, OriginSelection
from openforecast.tasks.plan import FitPlan
from openforecast.views._sources import Cell, Source, Vintage, source_for
from openforecast.views.base import (
    CONTEXT_END,
    CONTEXT_START,
    EVENT_TIME,
    FORECAST_END,
    FORECAST_START,
    HORIZON_STEP,
    ORIGIN_TIME,
    ROW_ID,
    SAMPLE_ID,
    SERIES_ID,
    ViewKind,
    opaque_id,
)
from openforecast.views.forecast import ForecastView, ForecastViewMetadata
from openforecast.views.provenance import SourceKind, ViewProvenance
from openforecast.views.sequences import SequenceView, SequenceViewSchema
from openforecast.views.series import SeriesView, SeriesViewSchema
from openforecast.views.tabular import TabularView, TabularViewSchema

__all__ = ["FitView", "ViewPlanner", "ViewRequest"]

#: What ``fit_view`` returns. A provider is handed exactly one of these.
FitView = SeriesView | SequenceView | TabularView

Column = tuple[list[Any], pa.DataType]


class ViewRequest(BaseModel):
    """What to materialize: which view, how far, and from which origins.

    A ``SeriesView`` binds neither a context length nor a horizon — a local
    forecaster is asked for a horizon at inference time — so those fields are
    unused for ``kind=series`` and rejected rather than quietly ignored.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: ViewKind
    horizon: int | None = Field(default=None, ge=1)
    context: int | None = Field(default=None, ge=1)
    origins: OriginSelection = AllOrigins()

    @model_validator(mode="after")
    def _check_requirements(self) -> Self:
        if self.kind is ViewKind.SEQUENCES and self.context is None:
            raise SchemaError(
                "a sequence view needs a context length: it is what makes one "
                "context -> horizon sequence a training sample"
            )
        if self.kind is not ViewKind.SERIES and self.horizon is None:
            raise SchemaError(f"a {self.kind} view needs a horizon")
        if self.kind is ViewKind.SERIES and (self.horizon is not None or self.context is not None):
            raise SchemaError(
                "a series view binds neither a context length nor a horizon; "
                "one complete time series is the training unit"
            )
        if self.kind is ViewKind.TABULAR and self.context is not None:
            raise SchemaError(
                "a tabular view binds no context length; lagged features are declared "
                "on the recipe, not on the view"
            )
        return self

    @classmethod
    def for_contract(
        cls,
        contract: TrainingContract,
        *,
        plan: FitPlan | None = None,
        task: ForecastTask | None = None,
        shared_plan: bool = False,
    ) -> ViewRequest:
        """What a model's contract, a fit plan and a forecast task jointly ask for.

        Purely a translation: the contract says which view, the plan says which
        origins and how much context, the task says how far ahead. Whether the
        materialized result is data the model accepts is a capability question,
        and the engine asks it of the view rather than of the request.

        A field the requested view does not bind is an error rather than
        something quietly dropped — a ``WindowPlan`` handed to a series model was
        written by someone expecting it to have an effect.

        ``shared_plan`` says the plan was written for several models at once, as
        an ensemble's is. A ``WindowPlan`` then does have an effect, on whichever
        member sizes a context window, and a member that binds none is not the
        one it was addressed to; that the plan reaches *someone* is checked
        against the whole recipe rather than one contract at a time.
        """
        plan = FitPlan() if plan is None else plan
        if contract.view is ViewKind.SERIES:
            if plan.window is not None and not shared_plan:
                raise RecipeError(
                    "a series model sizes no context window: it trains on one complete "
                    "time series, so a WindowPlan would have no effect. Drop it, or fit "
                    "a model that learns from sequences"
                )
            return cls(kind=ViewKind.SERIES, origins=plan.origins)
        if task is None:
            raise RecipeError(
                f"a {contract.view} view needs a horizon: its training samples are "
                f"bounded by one, so a ForecastTask is required to materialize it"
            )
        if contract.context_required and plan.window is None:
            raise RecipeError(
                "this model learns from context -> horizon sequences and cannot be "
                "given a default context length; state one with "
                "of.FitPlan(window=of.WindowPlan(context=...))"
            )
        if contract.view is ViewKind.TABULAR and plan.window is not None and not shared_plan:
            raise RecipeError(
                "a tabular view binds no context length; lagged features are declared "
                "on the recipe, as of.Reduction(lags=[...])"
            )
        return cls(
            kind=contract.view,
            horizon=task.horizon,
            context=plan.context if contract.view is ViewKind.SEQUENCES else None,
            origins=plan.origins,
        )

    @property
    def required_horizon(self) -> int:
        if self.horizon is None:
            raise SchemaError(f"a {self.kind} view has no horizon")
        return self.horizon

    @property
    def required_context(self) -> int:
        if self.context is None:
            raise SchemaError(f"a {self.kind} view has no context length")
        return self.context


class ViewPlanner:
    """Materializes semantic source data into provider-neutral execution views."""

    def fit_view(self, data: object, request: ViewRequest) -> FitView:
        """The training view ``request`` asks for, built from ``data``.

        ``data`` is a ``TimeSeriesFrame`` or a ``ForecastDataset``. A
        ``ForecastContext`` is a single inference origin and materializes with
        :meth:`forecast_view` instead.
        """
        if request.kind is ViewKind.FORECAST:
            raise SchemaError("a forecast view is materialized by forecast_view, not fit_view")
        source = source_for(data)
        if request.kind is ViewKind.SERIES:
            return _series_view(source, request)
        if request.kind is ViewKind.SEQUENCES:
            return _sequence_view(source, request)
        return _tabular_view(source, request)

    def forecast_view(self, context: ForecastContext, request: ViewRequest) -> ForecastView:
        """The inference view of one origin.

        ``request.context`` trims the history to the steps the model was trained
        to expect; leaving it unset keeps the whole history, which is what a
        local forecaster fitting at inference time needs.
        """
        if not isinstance(context, ForecastContext):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise DataError(
                f"a forecast view is materialized from a ForecastContext, got "
                f"{type(context).__name__}; take one origin of your data with at_origin(t)"
            )
        return _forecast_view(context, request)


# -- series ----------------------------------------------------------------


def _series_view(source: Source, request: ViewRequest) -> SeriesView:
    origin = _single_origin(source, request.origins)
    vintage = source.vintage(origin)
    temporal_features = source.temporal_features

    ids: dict[InstanceKey, str] = {}
    temporal: dict[str, list[Any]] = {name: [] for name in (SERIES_ID, EVENT_TIME)}
    for name in (*source.targets, *(feature.name for feature in temporal_features)):
        temporal[name] = []

    for instance in source.instances:
        moments = [
            moment for moment in source.target_times(instance) if origin is None or moment <= origin
        ]
        if not moments:
            continue
        series_id = opaque_id(instance, origin)
        ids[instance] = series_id
        for moment in moments:
            cell = (instance, moment)
            temporal[SERIES_ID].append(series_id)
            temporal[EVENT_TIME].append(moment)
            for target in source.targets:
                temporal[target].append(vintage.target(cell, target))
            for feature in temporal_features:
                temporal[feature.name].append(vintage.feature(cell, feature))

    if not ids:
        raise DataError(
            f"no instance has an event time at or before "
            f"{'the end of its history' if origin is None else origin.isoformat()}"
        )

    schema = SeriesViewSchema(
        frequency=source.frequency,
        targets=source.targets,
        instance_keys=source.instance_keys,
        origin_time=origin,
        features=source.features,
    )
    columns: dict[str, Column] = {
        SERIES_ID: (temporal[SERIES_ID], pa.string()),
        EVENT_TIME: (temporal[EVENT_TIME], source.time_type),
    }
    for name in schema.temporal_columns[2:]:
        columns[name] = (temporal[name], source.column_type(name))

    return SeriesView(
        temporal=build_table(columns),
        series=_key_table(SERIES_ID, ids, source),
        schema=schema,
        provenance=_provenance(source),
        static=_static_table(SERIES_ID, ids, source),
    )


def _single_origin(source: Source, selection: OriginSelection) -> datetime | None:
    """The one origin a series can carry, or ``None`` for a whole event-time series.

    An event-time frame *is* one series per instance, so ``AllOrigins`` needs no
    origin at all. A point-in-time dataset holds many vintages, and there is no
    honest way to flatten several of them onto one time axis.
    """
    available = source.origins()
    if selection.mode is OriginMode.ALL:
        if source.kind is SourceKind.TIME_SERIES:
            return None
        if len(available) == 1:
            return available[0]
        raise OriginScopeError(
            f"a series view holds one forecast origin, but the selection covers "
            f"{len(available)} vintages; select one with OriginSelection.at(t) or "
            f"OriginSelection.latest()"
        )
    chosen = selection.select(available)
    if len(chosen) != 1:
        raise OriginScopeError(
            f"a series view holds one forecast origin, but the selection covers "
            f"{len(chosen)} of them: {summarize(chosen)}"
        )
    return chosen[0]


# -- sequences -------------------------------------------------------------


def _sequence_view(source: Source, request: ViewRequest) -> SequenceView:
    context = request.required_context
    horizon = request.required_horizon
    frequency = source.frequency
    temporal_features = source.temporal_features

    temporal: dict[str, list[Any]] = {name: [] for name in (SAMPLE_ID, EVENT_TIME)}
    for name in (*source.targets, *(feature.name for feature in temporal_features)):
        temporal[name] = []
    bounds: dict[str, list[Any]] = {
        name: []
        for name in (
            SAMPLE_ID,
            ORIGIN_TIME,
            CONTEXT_START,
            CONTEXT_END,
            FORECAST_START,
            FORECAST_END,
        )
    }
    sample_instances: list[InstanceKey] = []

    for origin in request.origins.select(source.origins()):
        vintage = source.vintage(origin)
        window = _window(frequency, origin, context, horizon)
        for instance in source.instances:
            if not _covers(vintage, instance, window, context, bool(temporal_features)):
                continue
            sample_id = opaque_id(instance, origin)
            sample_instances.append(instance)
            bounds[SAMPLE_ID].append(sample_id)
            bounds[ORIGIN_TIME].append(origin)
            bounds[CONTEXT_START].append(window[0])
            bounds[CONTEXT_END].append(origin)
            bounds[FORECAST_START].append(window[context])
            bounds[FORECAST_END].append(window[-1])
            for moment in window:
                cell = (instance, moment)
                temporal[SAMPLE_ID].append(sample_id)
                temporal[EVENT_TIME].append(moment)
                for target in source.targets:
                    temporal[target].append(vintage.target(cell, target))
                for feature in temporal_features:
                    temporal[feature.name].append(vintage.feature(cell, feature))

    if not bounds[SAMPLE_ID]:
        raise DataError(
            f"no origin has {context} context steps and {horizon} forecast steps of "
            f"outcomes around it, so no training sequence could be built; shorten the "
            f"context or the horizon, or select origins the data covers"
        )

    schema = SequenceViewSchema(
        frequency=frequency,
        context=context,
        horizon=horizon,
        targets=source.targets,
        instance_keys=source.instance_keys,
        features=source.features,
    )
    columns: dict[str, Column] = {
        SAMPLE_ID: (temporal[SAMPLE_ID], pa.string()),
        EVENT_TIME: (temporal[EVENT_TIME], source.time_type),
    }
    for name in schema.temporal_columns[2:]:
        columns[name] = (temporal[name], source.column_type(name))

    samples: dict[str, Column] = {SAMPLE_ID: (bounds[SAMPLE_ID], pa.string())}
    for index, name in enumerate(source.instance_keys):
        samples[name] = (
            [instance[index] for instance in sample_instances],
            source.column_type(name),
        )
    for name in (ORIGIN_TIME, CONTEXT_START, CONTEXT_END, FORECAST_START, FORECAST_END):
        samples[name] = (bounds[name], source.time_type)

    return SequenceView(
        temporal=build_table(columns),
        samples=build_table(samples),
        schema=schema,
        provenance=_provenance(source),
        static=_static_table(
            SAMPLE_ID, list(zip(sample_instances, bounds[SAMPLE_ID], strict=True)), source
        ),
    )


def _window(
    frequency: Frequency, origin: datetime, context: int, horizon: int
) -> tuple[datetime, ...]:
    """The ``context + horizon`` event times of one sample, in order."""
    return (
        *(frequency.shift(origin, -step) for step in reversed(range(context))),
        *(frequency.shift(origin, step) for step in range(1, horizon + 1)),
    )


def _covers(
    vintage: Vintage,
    instance: InstanceKey,
    window: Sequence[datetime],
    context: int,
    needs_information: bool,
) -> bool:
    """Whether this instance and origin can carry a complete sample.

    Every step needs an outcome — the context to condition on, the forecast
    window to learn from — and, when the data has features at all, the vintage
    has to actually describe the forecast window. A partial sample is not
    padded: a sequence model would read the padding as data.
    """
    cells = [(instance, moment) for moment in window]
    if not all(vintage.has_target(cell) for cell in cells):
        return False
    if not needs_information:
        return True
    return all(vintage.has_information(cell) for cell in cells[context:])


# -- tabular ---------------------------------------------------------------


def _tabular_view(source: Source, request: ViewRequest) -> TabularView:
    horizon = request.required_horizon
    frequency = source.frequency
    known = [feature for feature in source.features if feature.is_known]
    statics = list(source.static_features)
    features = tuple((*known, *statics))

    if not features:
        # Every other view has the target's own history to learn from; a
        # supervised row does not, because it *is* one event time. So a tabular
        # view with no feature columns is not a small view, it is an empty design
        # matrix, and the row-alignment error it would fail with downstream says
        # nothing about what is missing.
        dropped = [feature.name for feature in source.features if feature.is_observed]
        raise DataError(
            "a supervised row is built from what was knowable at its origin, and this data "
            "declares no known or static feature to put in one"
            + (
                f"; the observed features {dropped} have no value at an event time after "
                f"the origin, so carry what they hold as a known feature instead"
                if dropped
                else ""
            )
        )

    x: dict[str, list[Any]] = {feature.name: [] for feature in features}
    y: dict[str, list[Any]] = {target: [] for target in source.targets}
    keys: dict[str, list[Any]] = {
        name: [] for name in (ROW_ID, ORIGIN_TIME, EVENT_TIME, HORIZON_STEP)
    }
    row_instances: list[InstanceKey] = []

    for origin in request.origins.select(source.origins()):
        vintage = source.vintage(origin)
        for instance in source.instances:
            for step in range(1, horizon + 1):
                moment = frequency.shift(origin, step)
                cell: Cell = (instance, moment)
                if not vintage.has_target(cell):
                    continue
                if known and not vintage.has_information(cell):
                    continue
                keys[ROW_ID].append(opaque_id(instance, origin, moment))
                keys[ORIGIN_TIME].append(origin)
                keys[EVENT_TIME].append(moment)
                keys[HORIZON_STEP].append(step)
                row_instances.append(instance)
                for target in source.targets:
                    y[target].append(vintage.target(cell, target))
                for feature in known:
                    x[feature.name].append(vintage.feature(cell, feature))
                for feature in statics:
                    x[feature.name].append(source.static_value(instance, feature.name))

    if not keys[ROW_ID]:
        raise DataError(
            f"no origin has an outcome within {horizon} steps of it, so no supervised "
            f"row could be built; select origins the data covers"
        )

    schema = TabularViewSchema(
        frequency=frequency,
        horizon=horizon,
        targets=source.targets,
        instance_keys=source.instance_keys,
        features=features,
    )
    key_columns: dict[str, Column] = {ROW_ID: (keys[ROW_ID], pa.string())}
    for index, name in enumerate(source.instance_keys):
        key_columns[name] = (
            [instance[index] for instance in row_instances],
            source.column_type(name),
        )
    key_columns[ORIGIN_TIME] = (keys[ORIGIN_TIME], source.time_type)
    key_columns[EVENT_TIME] = (keys[EVENT_TIME], source.time_type)
    key_columns[HORIZON_STEP] = (keys[HORIZON_STEP], pa.int64())

    return TabularView(
        X=build_table({name: (x[name], source.column_type(name)) for name in schema.x_columns}),
        y=build_table({name: (y[name], source.column_type(name)) for name in schema.y_columns}),
        keys=build_table(key_columns),
        schema=schema,
        provenance=_provenance(source),
    )


# -- forecast --------------------------------------------------------------


def _forecast_view(context: ForecastContext, request: ViewRequest) -> ForecastView:
    horizon = request.required_horizon
    schema = context.schema
    frequency = schema.frequency
    origin = context.origin_time
    instance_keys = schema.instance_keys
    time_type = context.history.column(schema.time).type

    history = _rename_time(context.history, schema.time)
    if request.context is not None:
        history = _trim(history, context, request.context)

    known = [feature.name for feature in schema.known_features]
    future_rows = context.future
    lookup = (
        None
        if future_rows is None
        else {
            cell: position
            for position, cell in enumerate(
                zip(
                    key_rows(future_rows, instance_keys),
                    column_values(future_rows, schema.time),
                    strict=True,
                )
            )
        }
    )
    available: dict[str, list[Any]] = (
        {} if future_rows is None else {name: column_values(future_rows, name) for name in known}
    )

    moments = [frequency.shift(origin, step) for step in range(1, horizon + 1)]
    future: dict[str, list[Any]] = {name: [] for name in (EVENT_TIME, *known)}
    instances: list[InstanceKey] = []
    for instance in context.instances:
        for moment in moments:
            instances.append(instance)
            future[EVENT_TIME].append(moment)
            position = None if lookup is None else lookup.get((instance, moment))
            for name in known:
                future[name].append(None if position is None else available[name][position])

    columns: dict[str, Column] = {}
    for index, name in enumerate(instance_keys):
        columns[name] = (
            [instance[index] for instance in instances],
            context.history.column(name).type,
        )
    columns[EVENT_TIME] = (future[EVENT_TIME], time_type)
    for name in known:
        columns[name] = (future[name], _future_type(context, name))

    metadata = ForecastViewMetadata(
        frequency=frequency,
        horizon=horizon,
        targets=schema.targets,
        instance_keys=instance_keys,
        context=request.context,
        features=schema.features,
    )
    return ForecastView(
        origin_time=origin,
        history=history,
        future=build_table(columns),
        metadata=metadata,
        static=context.static,
    )


def _rename_time(table: pa.Table, time: str) -> pa.Table:
    """Give the event-time axis the name every view uses for it."""
    if time == EVENT_TIME:
        return table
    names = [EVENT_TIME if name == time else name for name in table.column_names]
    return table.rename_columns(names)


def _trim(history: pa.Table, context: ForecastContext, steps: int) -> pa.Table:
    """Keep exactly the ``steps`` context rows ending at the origin, per instance.

    A short history is an error rather than a padded window: a model trained on
    168 steps that is handed 40 is not being asked the question it learned.
    """
    frequency = context.schema.frequency
    origin = context.origin_time
    wanted = {frequency.shift(origin, -step) for step in range(steps)}
    cells = list(
        zip(
            key_rows(history, context.schema.instance_keys),
            column_values(history, EVENT_TIME),
            strict=True,
        )
    )
    mask = [moment in wanted for _, moment in cells]
    for instance in context.instances:
        held = {
            moment
            for (key, moment), keep in zip(cells, mask, strict=True)
            if keep and key == instance
        }
        missing = sorted(wanted - held)
        if missing:
            raise DataError(
                f"history is {len(missing)} steps short of the {steps} context steps ending "
                f"at {origin.isoformat()}{f' for instance {instance}' if instance else ''}: "
                f"{summarize(missing)}"
            )
    return history.filter(pa.array(mask))


def _future_type(context: ForecastContext, name: str) -> pa.DataType:
    if context.future is not None and name in context.future.column_names:
        return context.future.column(name).type
    return context.history.column(name).type


# -- shared ----------------------------------------------------------------


def _provenance(source: Source) -> ViewProvenance:
    return ViewProvenance(source=source.kind, origin_fidelity=source.fidelity)


def _key_table(id_column: str, ids: dict[InstanceKey, str], source: Source) -> pa.Table:
    """``<id> -> instance keys``, one row per training unit."""
    columns: dict[str, Column] = {id_column: (list(ids.values()), pa.string())}
    for index, name in enumerate(source.instance_keys):
        columns[name] = ([instance[index] for instance in ids], source.column_type(name))
    return build_table(columns)


def _static_table(
    id_column: str, ids: dict[InstanceKey, str] | Sequence[tuple[InstanceKey, str]], source: Source
) -> pa.Table | None:
    """Static features re-keyed by the view's own identifier."""
    statics: tuple[FeatureSpec, ...] = source.static_features
    if not statics:
        return None
    pairs = list(ids.items()) if isinstance(ids, dict) else list(ids)
    columns: dict[str, Column] = {id_column: ([value for _, value in pairs], pa.string())}
    for feature in statics:
        columns[feature.name] = (
            [source.static_value(instance, feature.name) for instance, _ in pairs],
            source.column_type(feature.name),
        )
    return build_table(columns)
