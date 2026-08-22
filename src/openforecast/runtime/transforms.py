"""Executing the transforms a recipe declares, on the views a provider consumes.

```python
of.Pipeline(steps=[
    of.StandardScaler(columns="targets"),
    of.Model("builtin/seasonal-naive", params={"season_length": 24}),
])
```

A transform is OpenForecast's, not a provider's: it happens to the view between
materialization and execution, and it happens again — from the statistics that
were *fitted*, never recomputed — to the history a forecast is made from. The
forecast then comes back on the scale the caller's data was on, because a model
that answers in standard deviations has not answered the question.

Fitting the statistics once and persisting them is the whole point. Scaling a
forecast context by its own mean would leak whatever that context happens to
contain into the answer, and the difference is invisible in the output.

Only the scaler executes today. The point-in-time features and the explicit
missing-value transforms of Step 6 are part of the recipe protocol and are
recorded in the artifact; asking for one to be executed says so rather than
silently doing nothing, since a pipeline that quietly skipped a step would look
exactly like one that ran it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
from pydantic import BaseModel, ConfigDict

from openforecast.data._arrow import InstanceKey, column_values, key_rows
from openforecast.errors import RecipeError, UnsupportedPlanError
from openforecast.protocol.vocabulary import ForecastColumn
from openforecast.recipes.base import ColumnSet, ColumnTransform
from openforecast.recipes.transforms import StandardScaler, Transform
from openforecast.views.base import SAMPLE_ID, SERIES_ID
from openforecast.views.forecast import ForecastView
from openforecast.views.planner import FitView
from openforecast.views.sequences import SequenceView
from openforecast.views.series import SeriesView

__all__ = [
    "STATE_FILENAME",
    "TransformState",
    "apply_to_forecast_view",
    "fit_transforms",
    "invert_forecast",
    "read_state",
    "write_state",
]

#: Where a leaf's fitted transform statistics live inside its artifact.
STATE_FILENAME = "transforms.json"

#: A constant column has nothing to scale by, and dividing by zero would turn a
#: perfectly ordinary flat series into NaNs.
_UNIT_SCALE = 1.0


class ColumnStats(BaseModel):
    """What one column was centered and scaled by."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mean: float
    std: float


class InstanceStats(BaseModel):
    """The statistics of one instance, or of the whole view when ``key`` is ``None``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: tuple[Any, ...] | None = None
    columns: dict[str, ColumnStats]


class ScalerState(BaseModel):
    """A fitted :class:`~openforecast.recipes.StandardScaler`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["standard_scaler"] = "standard_scaler"
    per_instance: bool
    columns: tuple[str, ...]
    stats: tuple[InstanceStats, ...]

    def stats_for(self, instance: InstanceKey) -> InstanceStats | None:
        lookup = {
            tuple(entry.key) if entry.key is not None else None: entry for entry in self.stats
        }
        return lookup.get(tuple(instance) if self.per_instance else None)


class TransformState(BaseModel):
    """Every fitted transform of one leaf, in the order they were applied."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    steps: tuple[ScalerState, ...] = ()


# -- fit --------------------------------------------------------------------


def fit_transforms(
    view: FitView, transforms: Sequence[Transform]
) -> tuple[FitView, TransformState]:
    """Apply ``transforms`` to ``view``, keeping the statistics they were fitted with."""
    steps: list[ScalerState] = []
    for transform in transforms:
        scaler = _require_executable(transform)
        columns = _resolve_columns(scaler, view)
        state = _fit_scaler(view, scaler, columns)
        view = _scale_fit_view(view, state)
        steps.append(state)
    return view, TransformState(steps=tuple(steps))


def _require_executable(transform: Transform) -> StandardScaler:
    if isinstance(transform, StandardScaler):
        return transform
    raise UnsupportedPlanError(
        f"{type(transform).__name__} is part of the recipe protocol but is not executable "
        f"yet; it is recorded in the artifact, and a pipeline that silently skipped it "
        f"would be indistinguishable from one that ran it"
    )


def _resolve_columns(transform: ColumnTransform, view: FitView) -> tuple[str, ...]:
    """Which columns of the view this transform applies to."""
    schema: Any = view.schema
    role = transform.column_set
    if role is ColumnSet.TARGETS:
        columns: tuple[str, ...] = tuple(schema.targets)
    elif role is ColumnSet.FEATURES:
        columns = tuple(schema.temporal_feature_names)
    else:
        columns = transform.explicit_columns or ()
    table = _values_table(view)
    unknown = [name for name in columns if name not in table.column_names]
    if unknown:
        raise RecipeError(
            f"{type(transform).__name__} names columns the materialized view does not "
            f"hold: {unknown}; it holds {table.column_names}"
        )
    if not columns:
        raise RecipeError(
            f"{type(transform).__name__} applies to no column of this data; the view "
            f"holds {table.column_names}"
        )
    return columns


def _fit_scaler(view: FitView, transform: StandardScaler, columns: Sequence[str]) -> ScalerState:
    table = _values_table(view)
    instances = _row_instances(view)
    grouped: dict[InstanceKey | None, list[int]] = {}
    for position, instance in enumerate(instances):
        grouped.setdefault(instance if transform.per_instance else None, []).append(position)

    stats = tuple(
        InstanceStats(
            key=None if key is None else tuple(key),
            columns={
                name: _column_stats(column_values(table, name), positions) for name in columns
            },
        )
        for key, positions in grouped.items()
    )
    return ScalerState(per_instance=transform.per_instance, columns=tuple(columns), stats=stats)


def _column_stats(values: Sequence[Any], positions: Iterable[int]) -> ColumnStats:
    """Mean and standard deviation over what is actually there.

    Missing values are skipped rather than treated as zeros: a gap is not an
    observation of zero, and counting it as one would move the mean.
    """
    present = [
        float(values[position])
        for position in positions
        if values[position] is not None and not _is_nan(values[position])
    ]
    if not present:
        return ColumnStats(mean=0.0, std=_UNIT_SCALE)
    mean = sum(present) / len(present)
    variance = sum((value - mean) ** 2 for value in present) / len(present)
    deviation = math.sqrt(variance)
    return ColumnStats(mean=mean, std=deviation if deviation > 0 else _UNIT_SCALE)


# -- application ------------------------------------------------------------


def apply_to_forecast_view(view: ForecastView, state: TransformState) -> ForecastView:
    """Scale an inference view the way the training view was scaled."""
    for step in state.steps:
        view = _scale_forecast_view(view, step)
    return view


def invert_forecast(
    table: pa.Table, instance_keys: Sequence[str], state: TransformState
) -> pa.Table:
    """Put a forecast back on the scale the caller's data was on.

    Applied in reverse, since the last transform fitted is the first one that has
    to be undone.
    """
    for step in reversed(state.steps):
        table = _unscale_forecast(table, instance_keys, step)
    return table


def _scale_fit_view(view: FitView, state: ScalerState) -> FitView:
    table = _values_table(view)
    scaled = _scale_table(table, _row_instances(view), state)
    if isinstance(view, SeriesView):
        return SeriesView(
            temporal=scaled,
            series=view.series,
            schema=view.schema,
            provenance=view.provenance,
            static=view.static,
        )
    assert isinstance(view, SequenceView)  # _values_table refused anything else
    return SequenceView(
        temporal=scaled,
        samples=view.samples,
        schema=view.schema,
        provenance=view.provenance,
        static=view.static,
    )


def _scale_forecast_view(view: ForecastView, state: ScalerState) -> ForecastView:
    metadata = view.metadata
    keys = metadata.instance_keys
    return ForecastView(
        origin_time=view.origin_time,
        history=_scale_table(view.history, key_rows(view.history, keys), state),
        future=_scale_table(view.future, key_rows(view.future, keys), state),
        metadata=metadata,
        static=view.static,
    )


def _scale_table(table: pa.Table, instances: Sequence[InstanceKey], state: ScalerState) -> pa.Table:
    """Center and scale every column of ``table`` the state has statistics for."""
    result = table
    for name in state.columns:
        if name not in table.column_names:
            continue
        values = column_values(table, name)
        scaled = [
            _scale(value, _stats_of(state, instances[position], name))
            for position, value in enumerate(values)
        ]
        result = result.set_column(
            result.column_names.index(name), name, pa.array(scaled, type=pa.float64())
        )
    return result


def _unscale_forecast(
    table: pa.Table, instance_keys: Sequence[str], state: ScalerState
) -> pa.Table:
    instances = key_rows(table, instance_keys)
    targets: list[str] = column_values(table, ForecastColumn.TARGET.value)
    values: list[Any] = column_values(table, ForecastColumn.VALUE.value)
    restored = [
        _unscale(value, _stats_of(state, instances[position], targets[position]))
        for position, value in enumerate(values)
    ]
    return table.set_column(
        table.column_names.index(ForecastColumn.VALUE.value),
        ForecastColumn.VALUE.value,
        pa.array(restored, type=pa.float64()),
    )


def _stats_of(state: ScalerState, instance: InstanceKey, column: str) -> ColumnStats | None:
    entry = state.stats_for(instance)
    return None if entry is None else entry.columns.get(column)


def _scale(value: Any, stats: ColumnStats | None) -> float | None:
    if stats is None or value is None or _is_nan(value):
        return None if value is None or _is_nan(value) else float(value)
    return (float(value) - stats.mean) / stats.std


def _unscale(value: Any, stats: ColumnStats | None) -> float | None:
    if value is None or _is_nan(value):
        return None
    if stats is None:
        return float(value)
    return float(value) * stats.std + stats.mean


# -- view plumbing ----------------------------------------------------------


def _values_table(view: FitView) -> pa.Table:
    """The table a transform acts on, and the reason a tabular one cannot yet.

    A ``TabularView``'s values are split across ``X`` and ``y``, which is a
    different transformation to write and nothing executes one today — the
    reduction models that consume tabular rows arrive in Step 14.
    """
    if isinstance(view, SeriesView | SequenceView):
        return view.temporal
    raise UnsupportedPlanError(
        "transforms on a tabular view are not executable yet; the reduction models "
        "that consume one arrive with their execution path"
    )


def _row_instances(view: FitView) -> list[InstanceKey]:
    """The instance every row of the values table belongs to.

    The views key their rows by an opaque identifier and hold the instance in a
    separate table, precisely so that nothing downstream can condition on it.
    Scaling per instance is OpenForecast's own semantics, so it does the join.
    """
    table = _values_table(view)
    if isinstance(view, SeriesView):
        identifier, keys = SERIES_ID, view.series
    else:
        assert isinstance(view, SequenceView)
        identifier, keys = SAMPLE_ID, view.samples
    instance_keys = view.schema.instance_keys
    mapping = dict(zip(column_values(keys, identifier), key_rows(keys, instance_keys), strict=True))
    return [mapping[value] for value in column_values(table, identifier)]


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


# -- persistence ------------------------------------------------------------


def write_state(path: Path, state: TransformState) -> None:
    path.write_text(json.dumps(state.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")


def read_state(path: Path) -> TransformState:
    """The transforms fitted for one leaf; nothing fitted means nothing applied."""
    if not path.is_file():
        return TransformState()
    return TransformState.model_validate(json.loads(path.read_text(encoding="utf-8")))
