"""Shape and property tests for the event-time model.

The strategies build tables that are correct by construction and then perturb
exactly one thing, so a failure names the invariant that broke rather than the
example that happened to trip over it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pyarrow as pa
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from openforecast import DataError, Frequency, TimeSeriesFrame

ANCHOR = datetime(2026, 1, 1)


@dataclass(frozen=True)
class Shape:
    """A frame's layout, with no bearing on the values it holds.

    ``instances=0`` means a single series with no instance keys; anything else
    is a panel keyed by ``zone``.
    """

    instances: int
    targets: int
    periods: int
    observed: int
    known: int
    static: int
    frequency: Frequency

    @property
    def instance_count(self) -> int:
        return max(self.instances, 1)

    @property
    def target_names(self) -> list[str]:
        return [f"target_{index}" for index in range(self.targets)]

    @property
    def observed_names(self) -> list[str]:
        return [f"observed_{index}" for index in range(self.observed)]

    @property
    def known_names(self) -> list[str]:
        return [f"known_{index}" for index in range(self.known)]

    @property
    def static_names(self) -> list[str]:
        return [f"static_{index}" for index in range(self.static)]


# The search space is over layouts, not over floating point values: every
# invariant here is about columns, keys and grids.
shapes = st.builds(
    Shape,
    instances=st.integers(min_value=0, max_value=3),
    targets=st.integers(min_value=1, max_value=3),
    periods=st.integers(min_value=1, max_value=12),
    observed=st.integers(min_value=0, max_value=2),
    known=st.integers(min_value=0, max_value=2),
    static=st.integers(min_value=0, max_value=2),
    frequency=st.sampled_from(["30s", "15m", "1h", "6h", "1d", "1w", "1mo", "3mo"]).map(
        Frequency.parse
    ),
)

multi_period_shapes = shapes.filter(lambda shape: shape.periods >= 2)


def build(shape: Shape) -> TimeSeriesFrame:
    """Materialize a frame that satisfies ``shape`` by construction."""
    temporal = (*shape.target_names, *shape.observed_names, *shape.known_names)
    columns: dict[str, list[Any]] = {
        name: [] for name in ("timestamp", *temporal, *shape.static_names)
    }
    if shape.instances:
        columns["zone"] = []

    for instance_index in range(shape.instance_count):
        for period in range(shape.periods):
            if shape.instances:
                columns["zone"].append(f"z{instance_index}")
            columns["timestamp"].append(shape.frequency.shift(ANCHOR, period))
            value = float(instance_index * 100 + period)
            for offset, name in enumerate(temporal):
                columns[name].append(value + offset)
            for offset, name in enumerate(shape.static_names):
                columns[name].append(float(instance_index + offset))

    return TimeSeriesFrame.from_arrow(
        pa.table(columns),
        time="timestamp",
        frequency=shape.frequency,
        instance_keys=["zone"] if shape.instances else [],
        targets=shape.target_names,
        observed_features=shape.observed_names,
        known_features=shape.known_names,
        static_features=shape.static_names,
    )


@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
@given(shapes)
def test_any_shape_round_trips_through_arrow(shape: Shape) -> None:
    frame = build(shape)
    # A plain temporary directory rather than tmp_path: a function-scoped
    # fixture would be shared by every generated example.
    with TemporaryDirectory() as directory:
        target = Path(directory) / "frame"
        frame.write(target)
        assert TimeSeriesFrame.read(target) == frame


@given(shapes)
def test_layout_and_derived_properties_agree_with_the_shape(shape: Shape) -> None:
    frame = build(shape)
    schema = frame.schema

    assert schema.is_panel == bool(shape.instances)
    assert schema.target_count == shape.targets
    assert schema.is_univariate == (shape.targets == 1)
    assert schema.is_multivariate == (shape.targets > 1)
    assert schema.has_observed_features == bool(shape.observed)
    assert schema.has_known_features == bool(shape.known)
    assert schema.has_static_features == bool(shape.static)

    assert frame.history.column_names == list(schema.history_columns)
    assert frame.history.num_rows == shape.instance_count * shape.periods
    assert len(frame.instances) == shape.instance_count
    if shape.static:
        assert frame.static is not None
        assert frame.static.num_rows == shape.instance_count
    else:
        assert frame.static is None


@given(shapes)
def test_static_features_never_reach_the_history_table(shape: Shape) -> None:
    frame = build(shape)
    assert set(shape.static_names).isdisjoint(frame.history.column_names)


@settings(max_examples=40)
@given(shapes, st.data())
def test_a_duplicated_row_is_always_rejected(shape: Shape, data: st.DataObject) -> None:
    frame = build(shape)
    history = frame.history
    index = data.draw(st.integers(min_value=0, max_value=history.num_rows - 1))
    duplicated = pa.concat_tables([history, history.slice(index, 1)])
    with pytest.raises(DataError, match="duplicate instance/time rows"):
        TimeSeriesFrame(history=duplicated, schema=frame.schema, static=frame.static)


@settings(max_examples=40)
# A single-row instance is its own grid anchor, so there is nothing to be off
# of; the perturbation only has meaning from two timestamps upwards.
@given(multi_period_shapes, st.data())
def test_an_off_grid_timestamp_is_always_rejected(shape: Shape, data: st.DataObject) -> None:
    frame = build(shape)
    history = frame.history
    index = data.draw(st.integers(min_value=0, max_value=history.num_rows - 1))

    times: list[Any] = history.column("timestamp").to_pylist()
    # Half a step is off-grid for every fixed frequency; for calendar
    # frequencies, moving the day of month is the equivalent perturbation.
    if shape.frequency.is_calendar:
        times[index] = times[index].replace(day=times[index].day + 1)
    else:
        times[index] = times[index] + shape.frequency.as_timedelta() / 2

    perturbed = history.set_column(
        history.column_names.index("timestamp"),
        "timestamp",
        pa.array(times, type=pa.timestamp("us")),
    )
    with pytest.raises(DataError, match="do not sit on the"):
        TimeSeriesFrame(history=perturbed, schema=frame.schema, static=frame.static)
