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
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import pyarrow as pa

from openforecast.errors import DataError
from openforecast.models.capabilities import MissingValueSupport, ModelCapabilities
from openforecast.models.descriptor import ModelDescriptor
from openforecast.recipes.transforms import Impute, Transform
from openforecast.views.planner import FitView
from openforecast.views.sequences import SequenceView
from openforecast.views.series import SeriesView
from openforecast.views.tabular import TabularView

__all__ = ["validate_view", "view_tables"]


def validate_view(
    view: FitView, descriptor: ModelDescriptor, transforms: Sequence[Transform] = ()
) -> None:
    """Refuse a view the model has not said it can be given.

    ``transforms`` are the recipe steps that will have been applied by the time
    the provider sees the data, which is what lets a model requiring an explicit
    imputation be fitted on data that has missing values.
    """
    capabilities = descriptor.capabilities
    schema = view.schema
    if not capabilities.instances.supports(is_panel=bool(schema.instance_keys)):
        shape = "a panel" if schema.instance_keys else "a single series"
        raise DataError(
            f"{descriptor.ref} cannot be fitted on {shape}; it declares "
            f"single={capabilities.instances.single}, panel={capabilities.instances.panel}"
        )
    if not capabilities.targets.supports(len(schema.targets)):
        raise DataError(
            f"{descriptor.ref} cannot be fitted on {len(schema.targets)} targets "
            f"{list(schema.targets)}; it declares univariate="
            f"{capabilities.targets.univariate}, multivariate={capabilities.targets.multivariate}"
        )
    unsupported = capabilities.features.unsupported(schema.features)
    if unsupported:
        raise DataError(
            f"{descriptor.ref} cannot be given the features {list(unsupported)}; it declares "
            f"observed={capabilities.features.observed}, known={capabilities.features.known}, "
            f"static={capabilities.features.static}. Drop them from the data, or fit a model "
            f"that consumes them"
        )
    _validate_missing_values(view, descriptor, capabilities, transforms)


def _validate_missing_values(
    view: FitView,
    descriptor: ModelDescriptor,
    capabilities: ModelCapabilities,
    transforms: Sequence[Transform],
) -> None:
    if capabilities.tolerates_missing_values:
        return
    columns = sorted(_missing_columns(view))
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
