"""Uniform read access to the semantic sources a view can be built from.

The whole point of the views package is that exactly one place in OpenForecast
knows the difference between an event-time frame and a point-in-time dataset.
This is that place. Everything downstream — the three fit views, the forecast
view — asks the same four questions of either source:

```text
which origins exist
which instances exist
what was the target at this instance and event time
what did the vintage of this origin say about it
```

An event-time frame answers the last question with its newest values and an
observed feature masked past the origin, which is precisely what "simulated
origin" means. A point-in-time dataset answers it from the vintage itself.
Neither answer is repaired: where a source says nothing, the value is missing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import pyarrow as pa

from openforecast.data._arrow import InstanceKey, column_values, group_times, key_rows
from openforecast.data.features import FeatureSpec
from openforecast.data.forecast_dataset import ForecastDataset
from openforecast.data.frame import TimeSeriesFrame
from openforecast.data.frequency import Frequency
from openforecast.errors import DataError
from openforecast.views.provenance import OriginFidelity, SourceKind

__all__ = ["Cell", "Source", "Vintage", "source_for"]

#: One instance at one event time — the coordinate every lookup is keyed by.
Cell = tuple[InstanceKey, datetime]


class RowIndex:
    """``(instance, event time) -> row``, with cached column access."""

    def __init__(self, table: pa.Table, instance_keys: Sequence[str], time: str) -> None:
        self._table = table
        cells = zip(key_rows(table, instance_keys), column_values(table, time), strict=True)
        self._rows: dict[Cell, int] = {cell: position for position, cell in enumerate(cells)}
        self._columns: dict[str, list[Any]] = {}

    def has(self, cell: Cell) -> bool:
        return cell in self._rows

    def value(self, cell: Cell, name: str) -> Any:
        position = self._rows.get(cell)
        if position is None:
            return None
        return self._column(name)[position]

    def holds(self, name: str) -> bool:
        return name in self._table.column_names

    def column_type(self, name: str) -> pa.DataType:
        return self._table.column(name).type

    def _column(self, name: str) -> list[Any]:
        if name not in self._columns:
            self._columns[name] = column_values(self._table, name)
        return self._columns[name]


class KeyIndex:
    """``instance -> row``, for the per-instance static table."""

    def __init__(self, table: pa.Table, instance_keys: Sequence[str]) -> None:
        self._table = table
        self._rows = {key: position for position, key in enumerate(key_rows(table, instance_keys))}
        self._columns: dict[str, list[Any]] = {}

    def value(self, instance: InstanceKey, name: str) -> Any:
        position = self._rows.get(instance)
        if position is None:
            return None
        if name not in self._columns:
            self._columns[name] = column_values(self._table, name)
        return self._columns[name][position]

    def holds(self, name: str) -> bool:
        return name in self._table.column_names

    def column_type(self, name: str) -> pa.DataType:
        return self._table.column(name).type


class Vintage:
    """What one forecast origin could see.

    ``truth`` answers what happened, which is a fact about an event time and has
    no vintage. ``information`` answers what was knowable, in priority order —
    for an event-time frame that is history then future, for a point-in-time
    dataset it is the single vintage table.
    """

    def __init__(
        self,
        origin: datetime | None,
        truth: RowIndex,
        information: Sequence[RowIndex],
    ) -> None:
        self._origin = origin
        self._truth = truth
        self._information = tuple(information)

    @property
    def origin(self) -> datetime | None:
        return self._origin

    def has_target(self, cell: Cell) -> bool:
        return self._truth.has(cell)

    def target(self, cell: Cell, name: str) -> Any:
        return self._truth.value(cell, name)

    def has_information(self, cell: Cell) -> bool:
        return any(index.has(cell) for index in self._information)

    def feature(self, cell: Cell, feature: FeatureSpec) -> Any:
        """The value of ``feature`` at ``cell``, as of this origin.

        An observed feature past the origin is masked rather than read. On a
        point-in-time source it is already missing there — the semantic model
        rejects anything else — so the mask is what makes an event-time source
        behave identically instead of handing over a value nobody had yet.
        """
        if feature.is_observed and self._origin is not None and cell[1] > self._origin:
            return None
        for index in self._information:
            if index.has(cell):
                return index.value(cell, feature.name)
        return None


class Source(ABC):
    """A semantic dataset, read the way a view materializer needs to read it."""

    kind: SourceKind
    fidelity: OriginFidelity

    @property
    @abstractmethod
    def instance_keys(self) -> tuple[str, ...]: ...

    @property
    @abstractmethod
    def frequency(self) -> Frequency: ...

    @property
    @abstractmethod
    def targets(self) -> tuple[str, ...]: ...

    @property
    @abstractmethod
    def features(self) -> tuple[FeatureSpec, ...]: ...

    @property
    @abstractmethod
    def instances(self) -> tuple[InstanceKey, ...]: ...

    @property
    @abstractmethod
    def time_type(self) -> pa.DataType:
        """The Arrow type of the event-time axis, reused for every time column."""

    @abstractmethod
    def origins(self) -> tuple[datetime, ...]:
        """Every origin a sample could be built at, in ascending order."""

    @abstractmethod
    def target_times(self, instance: InstanceKey) -> tuple[datetime, ...]:
        """The event times this instance has an outcome for, in ascending order."""

    @abstractmethod
    def vintage(self, origin: datetime | None) -> Vintage: ...

    @abstractmethod
    def column_type(self, name: str) -> pa.DataType: ...

    @abstractmethod
    def static_value(self, instance: InstanceKey, name: str) -> Any: ...

    @property
    def static_features(self) -> tuple[FeatureSpec, ...]:
        return tuple(feature for feature in self.features if feature.is_static)

    @property
    def temporal_features(self) -> tuple[FeatureSpec, ...]:
        return tuple(feature for feature in self.features if feature.is_temporal)


class TimeSeriesSource(Source):
    """An ordinary event-time frame. Its origins are simulated by construction."""

    kind = SourceKind.TIME_SERIES
    fidelity = OriginFidelity.SIMULATED

    def __init__(self, frame: TimeSeriesFrame) -> None:
        self._frame = frame
        schema = frame.schema
        self._history = RowIndex(frame.history, schema.instance_keys, schema.time)
        self._future = (
            None
            if frame.future is None
            else RowIndex(frame.future, schema.instance_keys, schema.time)
        )
        self._static = (
            None if frame.static is None else KeyIndex(frame.static, schema.instance_keys)
        )
        self._times = group_times(
            key_rows(frame.history, schema.instance_keys),
            column_values(frame.history, schema.time),
        )

    @property
    def instance_keys(self) -> tuple[str, ...]:
        return self._frame.schema.instance_keys

    @property
    def frequency(self) -> Frequency:
        return self._frame.schema.frequency

    @property
    def targets(self) -> tuple[str, ...]:
        return self._frame.schema.targets

    @property
    def features(self) -> tuple[FeatureSpec, ...]:
        return self._frame.schema.features

    @property
    def instances(self) -> tuple[InstanceKey, ...]:
        return self._frame.instances

    @property
    def time_type(self) -> pa.DataType:
        return self._frame.history.column(self._frame.schema.time).type

    def origins(self) -> tuple[datetime, ...]:
        """Any event time in the history can act as a simulated origin."""
        return tuple(sorted(set(column_values(self._frame.history, self._frame.schema.time))))

    def target_times(self, instance: InstanceKey) -> tuple[datetime, ...]:
        return tuple(sorted(self._times.get(instance, [])))

    def vintage(self, origin: datetime | None) -> Vintage:
        information = [self._history] if self._future is None else [self._history, self._future]
        return Vintage(origin, truth=self._history, information=information)

    def column_type(self, name: str) -> pa.DataType:
        for index in (self._history, self._future, self._static):
            if index is not None and index.holds(name):
                return index.column_type(name)
        raise DataError(f"no column {name!r} in this TimeSeriesFrame")

    def static_value(self, instance: InstanceKey, name: str) -> Any:
        return None if self._static is None else self._static.value(instance, name)


class ForecastDatasetSource(Source):
    """Real forecast vintages. Its origins are observed by construction."""

    kind = SourceKind.FORECAST_DATASET
    fidelity = OriginFidelity.OBSERVED

    def __init__(self, dataset: ForecastDataset) -> None:
        self._dataset = dataset
        truth_schema = dataset.truth.schema
        self._truth = RowIndex(dataset.truth.history, truth_schema.instance_keys, truth_schema.time)
        self._static = (
            None
            if dataset.truth.static is None
            else KeyIndex(dataset.truth.static, truth_schema.instance_keys)
        )
        self._vintages: dict[datetime, RowIndex] = {}
        self._times = group_times(
            key_rows(dataset.truth.history, truth_schema.instance_keys),
            column_values(dataset.truth.history, truth_schema.time),
        )

    @property
    def instance_keys(self) -> tuple[str, ...]:
        return self._dataset.information.schema.instance_keys

    @property
    def frequency(self) -> Frequency:
        return self._dataset.information.schema.event_frequency

    @property
    def targets(self) -> tuple[str, ...]:
        return self._dataset.truth.schema.targets

    @property
    def features(self) -> tuple[FeatureSpec, ...]:
        """The vintage's temporal features, plus the truth frame's static ones.

        Static features cannot vary with an origin, which is exactly why the
        point-in-time frame refuses to hold them; they live on the truth side and
        are knowable at every origin.
        """
        return (
            *self._dataset.information.schema.features,
            *self._dataset.truth.schema.static_features,
        )

    @property
    def instances(self) -> tuple[InstanceKey, ...]:
        return self._dataset.instances

    @property
    def time_type(self) -> pa.DataType:
        truth_schema = self._dataset.truth.schema
        return self._dataset.truth.history.column(truth_schema.time).type

    def origins(self) -> tuple[datetime, ...]:
        return self._dataset.origins

    def target_times(self, instance: InstanceKey) -> tuple[datetime, ...]:
        return tuple(sorted(self._times.get(instance, [])))

    def vintage(self, origin: datetime | None) -> Vintage:
        if origin is None:
            raise DataError(
                "a point-in-time dataset holds several vintages, so materializing it "
                "requires choosing an origin"
            )
        if origin not in self._vintages:
            schema = self._dataset.information.schema
            self._vintages[origin] = RowIndex(
                self._dataset.information.at_origin(origin).table,
                schema.instance_keys,
                schema.event_time,
            )
        return Vintage(origin, truth=self._truth, information=[self._vintages[origin]])

    def column_type(self, name: str) -> pa.DataType:
        information = self._dataset.information.table
        if name in information.column_names:
            return information.column(name).type
        if self._truth.holds(name):
            return self._truth.column_type(name)
        if self._static is not None and self._static.holds(name):
            return self._static.column_type(name)
        raise DataError(f"no column {name!r} in this ForecastDataset")

    def static_value(self, instance: InstanceKey, name: str) -> Any:
        return None if self._static is None else self._static.value(instance, name)


def source_for(data: object) -> Source:
    """The reader for ``data``, or an error naming what can be materialized."""
    if isinstance(data, TimeSeriesFrame):
        return TimeSeriesSource(data)
    if isinstance(data, ForecastDataset):
        return ForecastDatasetSource(data)
    raise DataError(
        f"cannot build an execution view from {type(data).__name__}; "
        f"fit views are materialized from a TimeSeriesFrame or a ForecastDataset, "
        f"and a ForecastContext materializes into a ForecastView"
    )
