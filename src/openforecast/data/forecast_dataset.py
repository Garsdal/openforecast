"""``ForecastDataset``: what was knowable, paired with what actually happened.

```text
information   PointInTimeFrame   every vintage, exactly as it was issued
truth         TimeSeriesFrame    the realized outcome, once per event time
```

Keeping the two apart is what makes point-in-time training honest. A feature
value belongs to the origin that produced it and to no other; an outcome
belongs to an event time and has no vintage at all. Collapsing them into one
table is how leakage gets in.

The ``(ref_time, target_time)`` tables that production forecasting pipelines
already produce carry both at once — the label is repeated on every vintage of
the same event time. :meth:`ForecastDataset.from_pandas` splits that apart, and
raises :class:`~openforecast.errors.InconsistentTruthError` rather than choosing
for you when the repeated labels disagree.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa

from openforecast.data._arrow import (
    InstanceKey,
    build_table,
    canonical_value,
    column_type,
    column_values,
    is_missing,
    key_rows,
    summarize,
    table_from_pandas,
)
from openforecast.data.features import FeatureSpec
from openforecast.data.forecast_context import ForecastContext
from openforecast.data.frame import TimeSeriesFrame, extract_static
from openforecast.data.frequency import Frequency
from openforecast.data.point_in_time import (
    PointInTimeFrame,
    point_in_time_schema,
    resolve_origin,
)
from openforecast.data.schema import TimeSeriesSchema
from openforecast.errors import DataError, InconsistentTruthError

__all__ = ["ForecastDataset"]

INFORMATION_DIRNAME = "information"
TRUTH_DIRNAME = "truth"


class ForecastDataset:
    """Point-in-time information and the outcomes it was trying to predict."""

    def __init__(self, information: PointInTimeFrame, truth: TimeSeriesFrame) -> None:
        _require_matching_axes(information, truth)
        self._information = information
        self._truth = truth

    # -- accessors ---------------------------------------------------------

    @property
    def information(self) -> PointInTimeFrame:
        return self._information

    @property
    def truth(self) -> TimeSeriesFrame:
        return self._truth

    @property
    def origins(self) -> tuple[datetime, ...]:
        """Every distinct origin time in the information, in ascending order."""
        return self._information.origins

    @property
    def instances(self) -> tuple[InstanceKey, ...]:
        return self._information.instances

    @property
    def targets(self) -> tuple[str, ...]:
        return self._truth.schema.targets

    # -- one origin --------------------------------------------------------

    def at_origin(self, origin_time: str | datetime) -> ForecastContext:
        """Everything, and only what, was knowable at ``origin_time``.

        The history is the target realizations up to the origin alongside the
        feature values of *that* vintage; the future is the same vintage's known
        features for later event times. No other vintage contributes, so a
        feature value that was revised at 10:00 cannot appear in the context of
        the 09:00 origin.

        Target realizations up to the origin are treated as available at it. If
        a target arrives with a reporting lag, model that lag by carrying the
        target as an observed feature in ``information``, where its vintages are
        explicit.
        """
        moment = resolve_origin(origin_time, self.origins)
        vintage = self._information.at_origin(moment)
        information_schema = self._information.schema
        truth_schema = self._truth.schema
        event_time = truth_schema.time

        vintage_values = _row_lookup(
            vintage.table, information_schema.instance_keys, information_schema.event_time
        )
        truth_values = _row_lookup(self._truth.history, truth_schema.instance_keys, event_time)

        past, upcoming = _split_event_times(vintage_values, truth_values, moment)
        if not past:
            raise DataError(
                f"nothing is knowable at origin {moment.isoformat()}: neither the vintage nor "
                f"the truth holds an event time at or before it"
            )

        feature_names = information_schema.feature_names
        known_names = tuple(feature.name for feature in information_schema.known_features)
        history = _assemble(
            rows=past,
            instance_keys=truth_schema.instance_keys,
            event_time=event_time,
            sources=(
                (self._truth.history, truth_schema.targets, truth_values),
                (vintage.table, feature_names, vintage_values),
            ),
            key_source=self._truth.history,
        )
        future = _assemble(
            rows=upcoming,
            instance_keys=truth_schema.instance_keys,
            event_time=event_time,
            sources=((vintage.table, known_names, vintage_values),),
            key_source=self._truth.history,
        )

        schema = TimeSeriesSchema(
            time=event_time,
            frequency=information_schema.event_frequency,
            instance_keys=truth_schema.instance_keys,
            targets=truth_schema.targets,
            features=(*information_schema.features, *truth_schema.static_features),
        )
        return ForecastContext(
            origin_time=moment,
            frame=TimeSeriesFrame(
                history=history,
                schema=schema,
                future=future if future.num_rows else None,
                static=self._truth.static,
            ),
        )

    # -- construction ------------------------------------------------------

    @classmethod
    def from_arrow(
        cls,
        table: pa.Table,
        *,
        origin_time: str,
        event_time: str,
        targets: Sequence[str],
        event_frequency: str | Frequency,
        origin_frequency: str | Frequency | None = None,
        instance_keys: Sequence[str] = (),
        observed_features: Sequence[str] = (),
        known_features: Sequence[str] = (),
        static_features: Sequence[str] = (),
    ) -> ForecastDataset:
        """Split one long ``(origin_time, event_time)`` table into information and truth.

        Target columns are lifted out into ``truth``, one row per instance and
        event time. Repeated labels must agree: missing values are ignored,
        since a label that is not yet known says nothing, but two different
        realizations of the same event are a contradiction in the source data
        and raise :class:`~openforecast.errors.InconsistentTruthError`.
        """
        information_schema = point_in_time_schema(
            origin_time=origin_time,
            event_time=event_time,
            event_frequency=event_frequency,
            origin_frequency=origin_frequency,
            instance_keys=instance_keys,
            observed_features=observed_features,
            known_features=known_features,
        )
        truth_schema = TimeSeriesSchema(
            time=event_time,
            frequency=Frequency.parse(event_frequency),
            instance_keys=tuple(instance_keys),
            targets=tuple(targets),
            features=tuple(FeatureSpec.static(name) for name in static_features),
        )
        missing = [
            name
            for name in (*truth_schema.targets, *truth_schema.static_columns)
            if name not in table.column_names
        ]
        if missing:
            raise DataError(
                f"table is missing declared columns {missing}; present: {table.column_names}"
            )
        return cls(
            information=PointInTimeFrame(table, information_schema),
            truth=TimeSeriesFrame(
                history=_extract_truth(table, truth_schema),
                schema=truth_schema,
                static=extract_static(table, truth_schema)
                if truth_schema.has_static_features
                else None,
            ),
        )

    @classmethod
    def from_pandas(
        cls,
        table: Any,
        *,
        origin_time: str,
        event_time: str,
        targets: Sequence[str],
        event_frequency: str | Frequency,
        origin_frequency: str | Frequency | None = None,
        instance_keys: Sequence[str] = (),
        observed_features: Sequence[str] = (),
        known_features: Sequence[str] = (),
        static_features: Sequence[str] = (),
    ) -> ForecastDataset:
        """Same as :meth:`from_arrow`, for a pandas ``DataFrame`` input."""
        return cls.from_arrow(
            table_from_pandas(table, "table"),
            origin_time=origin_time,
            event_time=event_time,
            targets=targets,
            event_frequency=event_frequency,
            origin_frequency=origin_frequency,
            instance_keys=instance_keys,
            observed_features=observed_features,
            known_features=known_features,
            static_features=static_features,
        )

    # -- serialization -----------------------------------------------------

    def write(self, path: str | Path) -> Path:
        """Write the information and truth frames into subdirectories of ``path``."""
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        self._information.write(directory / INFORMATION_DIRNAME)
        self._truth.write(directory / TRUTH_DIRNAME)
        return directory

    @classmethod
    def read(cls, path: str | Path) -> ForecastDataset:
        """Read back a directory written by :meth:`write`, re-validating it."""
        directory = Path(path)
        for required in (INFORMATION_DIRNAME, TRUTH_DIRNAME):
            if not (directory / required).is_dir():
                raise DataError(f"{directory} is not a ForecastDataset: {required}/ is missing")
        return cls(
            information=PointInTimeFrame.read(directory / INFORMATION_DIRNAME),
            truth=TimeSeriesFrame.read(directory / TRUTH_DIRNAME),
        )

    # -- dunder ------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ForecastDataset):
            return NotImplemented
        return self._information == other._information and self._truth == other._truth

    def __repr__(self) -> str:
        return (
            f"ForecastDataset(instances={len(self.instances)}, "
            f"origins={len(self.origins)}, "
            f"targets={list(self.targets)}, "
            f"information_rows={self._information.table.num_rows}, "
            f"truth_rows={self._truth.history.num_rows})"
        )


# -- validation ------------------------------------------------------------


def _require_matching_axes(information: PointInTimeFrame, truth: TimeSeriesFrame) -> None:
    """The two frames must describe the same instances on the same event axis."""
    if information.schema.instance_keys != truth.schema.instance_keys:
        raise DataError(
            f"information and truth must share their instance keys, got "
            f"{list(information.schema.instance_keys)} and {list(truth.schema.instance_keys)}"
        )
    if information.schema.event_time != truth.schema.time:
        raise DataError(
            f"information event time {information.schema.event_time!r} and truth time "
            f"{truth.schema.time!r} must be the same column: both name the event axis"
        )
    if information.schema.event_frequency != truth.schema.frequency:
        raise DataError(
            f"information event frequency {information.schema.event_frequency} and truth "
            f"frequency {truth.schema.frequency} must agree"
        )
    shared = set(truth.schema.targets) & set(information.schema.feature_names)
    if shared:
        raise DataError(
            f"{sorted(shared)} is both a truth target and an information feature; "
            f"what was knowable and what happened must be distinguishable by column name"
        )
    unknown = sorted(set(truth.instances) - set(information.instances), key=repr)
    if unknown:
        raise DataError(
            f"truth holds instances absent from information: {summarize(unknown)}; "
            f"there is no origin at which they could be forecast"
        )


# -- truth extraction ------------------------------------------------------

TruthKey = tuple[InstanceKey, datetime]


def _extract_truth(table: pa.Table, schema: TimeSeriesSchema) -> pa.Table:
    """One row per instance and event time, reconciled across the vintages."""
    keys = key_rows(table, schema.instance_keys)
    events: list[datetime] = column_values(table, schema.time)
    rows = list(zip(keys, events, strict=True))
    ordered = list(dict.fromkeys(rows))

    values: dict[str, list[Any]] = {}
    for target in schema.targets:
        # canonical value -> first original value, per (instance, event time).
        seen: dict[TruthKey, dict[Any, Any]] = {row: {} for row in ordered}
        for row, value in zip(rows, column_values(table, target), strict=True):
            if not is_missing(value):
                seen[row].setdefault(canonical_value(value), value)
        conflicts = [
            f"{_describe(row)}: {sorted(distinct.values(), key=repr)}"
            for row, distinct in seen.items()
            if len(distinct) > 1
        ]
        if conflicts:
            raise InconsistentTruthError(
                f"target {target!r} disagrees between vintages of the same event time: "
                f"{summarize(conflicts)}; only one of them can be what happened, and "
                f"OpenForecast will not choose"
            )
        values[target] = [next(iter(seen[row].values()), None) for row in ordered]

    columns: dict[str, tuple[list[Any], pa.DataType]] = {}
    for index, name in enumerate(schema.instance_keys):
        columns[name] = ([row[0][index] for row in ordered], column_type(table, name))
    columns[schema.time] = ([row[1] for row in ordered], column_type(table, schema.time))
    for target in schema.targets:
        columns[target] = (values[target], column_type(table, target))
    return build_table(columns)


def _describe(row: TruthKey) -> str:
    key, event = row
    instance = f"{key} " if key else ""
    return f"{instance}{event.isoformat()}"


# -- context assembly ------------------------------------------------------

RowLookup = dict[TruthKey, int]


def _row_lookup(table: pa.Table, instance_keys: Sequence[str], time: str) -> RowLookup:
    """``(instance, event time) -> row index``, for tables already known unique."""
    keys = key_rows(table, instance_keys)
    events: list[datetime] = column_values(table, time)
    return {row: index for index, row in enumerate(zip(keys, events, strict=True))}


def _split_event_times(
    vintage: RowLookup, truth: RowLookup, origin: datetime
) -> tuple[list[TruthKey], list[TruthKey]]:
    """Rows at or before the origin, and the vintage's rows after it.

    The past side is the union of both sources: a vintage typically describes
    event times ahead of its own origin, and the target history behind it comes
    from the truth. The future side is the vintage alone — the truth of an event
    that has not happened yet was not knowable at the origin.
    """
    past = sorted(
        {row for row in (*truth, *vintage) if row[1] <= origin},
        key=lambda row: (repr(row[0]), row[1]),
    )
    upcoming = sorted(
        (row for row in vintage if row[1] > origin), key=lambda row: (repr(row[0]), row[1])
    )
    return past, upcoming


def _assemble(
    *,
    rows: Sequence[TruthKey],
    instance_keys: tuple[str, ...],
    event_time: str,
    sources: Sequence[tuple[pa.Table, Sequence[str], RowLookup]],
    key_source: pa.Table,
) -> pa.Table:
    """Gather ``rows`` from several tables, leaving a null where a source has no row."""
    columns: dict[str, tuple[list[Any], pa.DataType]] = {}
    for index, name in enumerate(instance_keys):
        columns[name] = ([row[0][index] for row in rows], column_type(key_source, name))
    columns[event_time] = ([row[1] for row in rows], column_type(key_source, event_time))
    for table, names, lookup in sources:
        for name in names:
            available = column_values(table, name)
            positions = [lookup.get(row) for row in rows]
            columns[name] = (
                [None if position is None else available[position] for position in positions],
                column_type(table, name),
            )
    return build_table(columns)
