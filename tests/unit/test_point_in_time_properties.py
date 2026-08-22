"""Shape and property tests for the point-in-time model.

Every generated dataset is poisoned the same way: the value of a feature
encodes the origin that produced it. Any materialization can therefore be
checked against the origin it claims to describe, rather than against a fixture
somebody has to keep in step with the code.
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

from openforecast import DataError, ForecastDataset, Frequency, PointInTimeFrame

ANCHOR = datetime(2026, 1, 1)


@dataclass(frozen=True)
class Shape:
    """A point-in-time layout, with no bearing on the values it holds.

    ``instances=0`` means a single series with no instance keys; anything else
    is a panel keyed by ``zone``. Each origin covers ``horizon`` event times
    starting at the origin itself, so consecutive vintages overlap.
    """

    instances: int
    origins: int
    horizon: int
    known: int
    frequency: Frequency

    @property
    def instance_count(self) -> int:
        return max(self.instances, 1)

    @property
    def instance_keys(self) -> list[str]:
        return ["zone"] if self.instances else []

    @property
    def known_names(self) -> list[str]:
        return [f"known_{index}" for index in range(self.known)]

    @property
    def rows(self) -> int:
        return self.instance_count * self.origins * self.horizon

    def origin_at(self, index: int) -> datetime:
        return self.frequency.shift(ANCHOR, index)


shapes = st.builds(
    Shape,
    instances=st.integers(min_value=0, max_value=3),
    origins=st.integers(min_value=1, max_value=5),
    horizon=st.integers(min_value=1, max_value=5),
    known=st.integers(min_value=1, max_value=3),
    frequency=st.sampled_from(["15m", "1h", "6h", "1d", "1w", "1mo"]).map(Frequency.parse),
)

# A single event time per instance is its own grid anchor, so there is nothing
# to be off of; perturbing a timestamp only has meaning from two of them up.
multi_event_shapes = shapes.filter(lambda shape: shape.origins + shape.horizon >= 3)


def vintage_value(origin_index: int, event_index: int, offset: int = 0) -> float:
    """A value that names the vintage it came from, to the last decimal."""
    return origin_index * 1000.0 + event_index * 10.0 + offset


def source(shape: Shape) -> pa.Table:
    columns: dict[str, list[Any]] = {
        name: [] for name in ("ref_time", "target_time", "price", *shape.known_names)
    }
    if shape.instances:
        columns["zone"] = []

    for instance_index in range(shape.instance_count):
        for origin_index in range(shape.origins):
            for lead in range(shape.horizon):
                event_index = origin_index + lead
                if shape.instances:
                    columns["zone"].append(f"z{instance_index}")
                columns["ref_time"].append(shape.origin_at(origin_index))
                columns["target_time"].append(shape.origin_at(event_index))
                # The target depends on the event alone, so every vintage agrees.
                columns["price"].append(float(instance_index * 100 + event_index))
                for offset, name in enumerate(shape.known_names):
                    columns[name].append(vintage_value(origin_index, event_index, offset))
    return pa.table(columns)


def build(shape: Shape) -> ForecastDataset:
    return ForecastDataset.from_arrow(
        source(shape),
        origin_time="ref_time",
        event_time="target_time",
        targets=["price"],
        event_frequency=shape.frequency,
        origin_frequency=shape.frequency,
        instance_keys=shape.instance_keys,
        known_features=shape.known_names,
    )


@given(shapes)
def test_the_layout_follows_the_shape(shape: Shape) -> None:
    dataset = build(shape)
    information = dataset.information

    assert information.table.num_rows == shape.rows
    assert information.table.column_names == list(information.schema.columns)
    assert len(information.origins) == shape.origins
    assert len(dataset.instances) == shape.instance_count
    assert information.schema.is_panel == bool(shape.instances)


@given(shapes)
def test_truth_holds_exactly_one_row_per_instance_and_event(shape: Shape) -> None:
    dataset = build(shape)
    events = shape.origins + shape.horizon - 1
    assert dataset.truth.history.num_rows == shape.instance_count * events


@given(shapes)
def test_a_vintage_contains_only_its_own_values(shape: Shape) -> None:
    """The property the whole model exists for: no origin sees another's data."""
    dataset = build(shape)
    for origin_index in range(shape.origins):
        vintage = dataset.information.at_origin(shape.origin_at(origin_index))
        expected = {
            vintage_value(origin_index, origin_index + lead, offset)
            for lead in range(shape.horizon)
            for offset in range(shape.known)
        }
        found = {
            value for name in shape.known_names for value in vintage.table.column(name).to_pylist()
        }
        assert found == expected


@given(shapes)
def test_a_context_never_sees_a_later_vintage(shape: Shape) -> None:
    dataset = build(shape)
    for origin_index in range(shape.origins):
        later = {
            vintage_value(other, other + lead, offset)
            for other in range(origin_index + 1, shape.origins)
            for lead in range(shape.horizon)
            for offset in range(shape.known)
        }
        try:
            context = dataset.at_origin(shape.origin_at(origin_index))
        except DataError:
            # The first origins of a horizon-1 dataset have no history yet.
            continue
        for table in (context.history, context.future):
            if table is None:
                continue
            for name in shape.known_names:
                assert later.isdisjoint(table.column(name).to_pylist())


@given(shapes)
def test_a_context_splits_cleanly_at_its_origin(shape: Shape) -> None:
    dataset = build(shape)
    origin = shape.origin_at(shape.origins - 1)
    context = dataset.at_origin(origin)

    past: list[Any] = context.history.column("target_time").to_pylist()
    assert past and max(past) <= origin
    if context.future is not None:
        upcoming: list[Any] = context.future.column("target_time").to_pylist()
        assert min(upcoming) > origin


@given(shapes)
def test_lead_time_is_the_distance_between_the_two_axes(shape: Shape) -> None:
    frame = build(shape).information.with_lead_time(shape.frequency)
    origins: list[Any] = frame.table.column("ref_time").to_pylist()
    events: list[Any] = frame.table.column("target_time").to_pylist()
    leads: list[Any] = frame.table.column("lead_time").to_pylist()

    for origin, event, lead in zip(origins, events, leads, strict=True):
        assert lead == shape.frequency.steps_between(origin, event)
        assert 0 <= lead < shape.horizon


@settings(max_examples=40, suppress_health_check=[HealthCheck.too_slow])
@given(shapes)
def test_any_shape_round_trips_through_arrow(shape: Shape) -> None:
    dataset = build(shape)
    # A plain temporary directory rather than tmp_path: a function-scoped
    # fixture would be shared by every generated example.
    with TemporaryDirectory() as directory:
        target = Path(directory) / "dataset"
        dataset.write(target)
        assert ForecastDataset.read(target) == dataset


@settings(max_examples=40)
@given(shapes, st.data())
def test_a_duplicated_vintage_row_is_always_rejected(shape: Shape, data: st.DataObject) -> None:
    frame = build(shape).information
    index = data.draw(st.integers(min_value=0, max_value=frame.table.num_rows - 1))
    duplicated = pa.concat_tables([frame.table, frame.table.slice(index, 1)])
    with pytest.raises(DataError, match="duplicate instance/origin/event rows"):
        PointInTimeFrame(duplicated, frame.schema)


@settings(max_examples=40)
@given(multi_event_shapes, st.data())
def test_an_off_grid_event_time_is_always_rejected(shape: Shape, data: st.DataObject) -> None:
    frame = build(shape).information
    table = frame.table
    index = data.draw(st.integers(min_value=0, max_value=table.num_rows - 1))

    times: list[Any] = table.column("target_time").to_pylist()
    # Half a step is off-grid for every fixed frequency; for calendar
    # frequencies, moving the day of month is the equivalent perturbation.
    if shape.frequency.is_calendar:
        times[index] = times[index].replace(day=times[index].day + 1)
    else:
        times[index] = times[index] + shape.frequency.as_timedelta() / 2

    perturbed = table.set_column(
        table.column_names.index("target_time"),
        "target_time",
        pa.array(times, type=pa.timestamp("us")),
    )
    with pytest.raises(DataError, match="do not sit on the"):
        PointInTimeFrame(perturbed, frame.schema)
