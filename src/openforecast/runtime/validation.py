"""Checking a materialized view against what a model declared it can consume.

The engine asks these questions *after* materializing and *before* starting a
provider, which is the only place they can all be answered: the view is what the
model will actually be given, and the descriptor is what the model says it can
take. A mismatch here is a declaration meeting data, so it is reported as such —
naming the model, the property and the data — rather than surfacing later as a
stack trace from inside somebody's library.

The missing-value check is the one with teeth. Point-in-time data is full of
real missing values, and a model that cannot consume them has exactly two
honest outcomes: the caller writes an explicit ``of.Impute`` step, which is
recorded in the artifact, or the request is refused. Filling them in here would
be the silent imputation the architecture forbids.

Since Step 23 the same questions are asked of a *forecast* view, for the models
that never have a fit view. A pretrained model is checked once, at the only
moment it is ever handed data — and it is the same four questions, over the
tables an inference origin holds, so that "this model cannot be given a panel"
means one thing whichever lifecycle the model has.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import pyarrow as pa

from openforecast.data.features import FeatureSpec
from openforecast.errors import DataError, UnsupportedDataShape, UnsupportedFeature
from openforecast.models.capabilities import MissingValueSupport
from openforecast.models.descriptor import ModelDescriptor
from openforecast.recipes.transforms import Impute, Transform
from openforecast.views.base import EVENT_TIME
from openforecast.views.forecast import ForecastView
from openforecast.views.planner import FitView
from openforecast.views.sequences import SequenceView
from openforecast.views.series import SeriesView
from openforecast.views.tabular import TabularView

__all__ = ["validate_forecast_view", "validate_view", "view_tables"]


def validate_view(
    view: FitView, descriptor: ModelDescriptor, transforms: Sequence[Transform] = ()
) -> None:
    """Refuse a view the model has not said it can be given.

    ``transforms`` are the recipe steps that will have been applied by the time
    the provider sees the data, which is what lets a model requiring an explicit
    imputation be fitted on data that has missing values.
    """
    schema = view.schema
    _check_shape(
        descriptor, schema.instance_keys, schema.targets, schema.features, verb="fitted on"
    )
    _validate_missing_values(lambda: _missing_columns(view), descriptor, transforms)


def validate_forecast_view(view: ForecastView, descriptor: ModelDescriptor) -> None:
    """The same questions, of the one view a pretrained model is ever handed.

    A fitted model was checked against the data it learned from, and a forecast
    from it is checked against *that* — :func:`_check_data_schema` in the engine
    — because the fit is what the artifact promises to answer. A model that was
    never fitted has no such promise behind it, so the declaration meets data
    here instead, and it is the only place it can.

    Transforms are not a parameter: they belong to a recipe, a recipe is fitted,
    and this path never fits anything.
    """
    metadata = view.metadata
    _check_shape(
        descriptor, metadata.instance_keys, metadata.targets, metadata.features, verb="given"
    )
    _validate_missing_values(lambda: _missing_forecast_columns(view), descriptor, ())


def _check_shape(
    descriptor: ModelDescriptor,
    instance_keys: Sequence[str],
    targets: Sequence[str],
    features: Sequence[FeatureSpec],
    *,
    verb: str,
) -> None:
    """How many series, how many targets, and which feature roles."""
    capabilities = descriptor.capabilities
    if not capabilities.instances.supports(is_panel=bool(instance_keys)):
        shape = "a panel" if instance_keys else "a single series"
        raise UnsupportedDataShape(
            f"{descriptor.ref} cannot be {verb} {shape}; it declares "
            f"single={capabilities.instances.single}, panel={capabilities.instances.panel}",
            model=str(descriptor.ref),
            instance_keys=list(instance_keys),
            single=capabilities.instances.single,
            panel=capabilities.instances.panel,
        )
    if not capabilities.targets.supports(len(targets)):
        raise UnsupportedDataShape(
            f"{descriptor.ref} cannot be {verb} {len(targets)} targets "
            f"{list(targets)}; it declares univariate="
            f"{capabilities.targets.univariate}, multivariate={capabilities.targets.multivariate}",
            model=str(descriptor.ref),
            targets=list(targets),
            univariate=capabilities.targets.univariate,
            multivariate=capabilities.targets.multivariate,
        )
    unsupported = capabilities.features.unsupported(features)
    if unsupported:
        raise UnsupportedFeature(
            f"{descriptor.ref} cannot be given the features {list(unsupported)}; it declares "
            f"observed={capabilities.features.observed}, known={capabilities.features.known}, "
            f"static={capabilities.features.static}. Drop them from the data, or fit a model "
            f"that consumes them",
            model=str(descriptor.ref),
            features=list(unsupported),
        )


def _validate_missing_values(
    found: Callable[[], set[str]],
    descriptor: ModelDescriptor,
    transforms: Sequence[Transform],
) -> None:
    """``found`` is a thunk: a tolerant model does not pay to scan the tables."""
    capabilities = descriptor.capabilities
    if capabilities.tolerates_missing_values:
        return
    columns = sorted(found())
    if not columns:
        return
    if capabilities.requires_missing_value_transform and any(
        isinstance(transform, Impute) for transform in transforms
    ):
        return
    remedy = (
        "state how they should be filled, with of.Impute(columns=..., method=...) in a "
        "pipeline, so that the artifact records it"
        if capabilities.missing_values is MissingValueSupport.REQUIRES_TRANSFORM
        else "fit a model that consumes missing values, or remove them from the data"
    )
    raise DataError(
        f"{descriptor.ref} declares missing_values={capabilities.missing_values} and the "
        f"materialized view has missing values in {columns}; {remedy}"
    )


def view_tables(view: FitView) -> tuple[pa.Table, ...]:
    """The tables holding a view's values, whatever the view calls them.

    The key tables — which series, which sample, which row — are deliberately
    left out: they are identifiers and bounds OpenForecast wrote itself, so a
    null in one would be a bug in the planner rather than a missing observation.
    """
    if isinstance(view, SeriesView):
        return (view.temporal,) + ((view.static,) if view.static is not None else ())
    if isinstance(view, SequenceView):
        return (view.temporal,) + ((view.static,) if view.static is not None else ())
    return (view.X, view.y)


def _missing_columns(view: FitView) -> set[str]:
    """Every value column of the view that holds a null or a NaN."""
    identifiers = _identifiers(view)
    return {
        name
        for table in view_tables(view)
        for name in table.column_names
        if name not in identifiers and _holds_missing(table.column(name))
    }


def _missing_forecast_columns(view: ForecastView) -> set[str]:
    """The same question of an inference origin's three tables.

    The instance keys and the event time are left out for the reason the fit
    views leave their identifiers out: they say which row this is rather than
    what was observed, so a null in one is a broken view rather than an absent
    measurement — and the view's own constructor is what refuses that.
    """
    identifiers = frozenset({*view.metadata.instance_keys, EVENT_TIME})
    tables = (view.history, view.future, *((view.static,) if view.static is not None else ()))
    return {
        name
        for table in tables
        for name in table.column_names
        if name not in identifiers and _holds_missing(table.column(name))
    }


def _identifiers(view: FitView) -> frozenset[str]:
    schema: Any = view.schema
    if isinstance(view, TabularView):
        return frozenset(schema.keys_columns)
    return frozenset(schema.temporal_columns[:2])


def _holds_missing(column: pa.ChunkedArray[Any]) -> bool:
    if column.null_count:
        return True
    if not pa.types.is_floating(column.type):
        return False
    values: Iterable[Any] = column.to_pylist()
    return any(value is not None and math.isnan(value) for value in values)
