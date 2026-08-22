"""The point-in-time semantic model: ``instance x origin_time x event_time x variable``.

```text
origin_time = when the information was available
event_time  = what time the information refers to
```

A :class:`PointInTimeFrame` holds *what was knowable*, vintage by vintage. The
same event time appears once per origin, and the values may differ between
origins — that difference is the whole point, so nothing here collapses,
deduplicates or forward-fills it.

```text
zone origin_time event_time wind_fc load_fc
DE   08:00       12:00      10.1    54.2
DE   09:00       12:00      11.7    54.8
DE   10:00       12:00      12.4    55.1
```

Lead time is deliberately not a stored column: it is ``event_time -
origin_time``, and materializing it by default would make every consumer
wonder which of the three columns is authoritative. Ask for it with
:meth:`PointInTimeFrame.with_lead_time` when a model actually needs it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Self

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from openforecast.data._arrow import (
    InstanceKey,
    canonicalize,
    column_values,
    group_times,
    is_missing,
    key_rows,
    read_table,
    require_no_nulls,
    require_table,
    require_timestamps,
    require_unique,
    summarize,
    table_from_pandas,
    validate_grid,
    write_table,
)
from openforecast.data.features import FeatureSpec
from openforecast.data.frequency import Frequency
from openforecast.data.schema import reject_duplicate_names
from openforecast.errors import DataError, SchemaError

__all__ = ["PointInTimeFrame", "PointInTimeSchema"]

SCHEMA_FILENAME = "schema.json"
TABLE_FILENAME = "table.arrow"

DEFAULT_LEAD_TIME_NAME = "lead_time"


class PointInTimeSchema(BaseModel):
    """What the columns of a :class:`PointInTimeFrame` mean.

    There are no targets here. A point-in-time frame describes information, not
    outcomes; what actually happened lives in the ``truth`` side of a
    :class:`~openforecast.data.forecast_dataset.ForecastDataset`.

    ``origin_frequency`` is optional because vintages are often irregular — a
    day-ahead run at 10:00 and an intraday run at 14:00 sit on no single grid.
    Declaring it opts into validating that they do.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    origin_time: str
    event_time: str

    event_frequency: Frequency
    origin_frequency: Frequency | None = None

    instance_keys: tuple[str, ...] = ()
    features: tuple[FeatureSpec, ...]

    @field_validator("event_frequency", "origin_frequency", mode="before")
    @classmethod
    def _parse_frequency(cls, value: object) -> object:
        """Accept ``event_frequency="1h"`` and store the native representation."""
        return Frequency.parse(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def _check_names(self) -> Self:
        if not self.features:
            raise SchemaError(
                "a point-in-time schema must declare at least one feature; "
                "an origin and an event time on their own carry no information"
            )

        reject_duplicate_names(self.instance_keys, "instance key")
        reject_duplicate_names(self.feature_names, "feature")

        for name in (self.origin_time, self.event_time, *self.instance_keys, *self.feature_names):
            if not name.strip():
                raise SchemaError("column names must not be empty")

        if self.origin_time == self.event_time:
            raise SchemaError(
                f"origin_time and event_time must be different columns, both are "
                f"{self.origin_time!r}; when information was available and what it refers to "
                f"are different axes"
            )
        for role, names in (
            ("a feature", self.feature_names),
            ("an instance key", self.instance_keys),
        ):
            for axis, name in (("origin_time", self.origin_time), ("event_time", self.event_time)):
                if name in names:
                    raise SchemaError(f"the {axis} column {name!r} cannot also be {role}")
        keyed = set(self.instance_keys) & set(self.feature_names)
        if keyed:
            raise SchemaError(
                f"a column cannot be both an instance key and a feature: {sorted(keyed)}"
            )

        static = [feature.name for feature in self.features if feature.is_static]
        if static:
            raise SchemaError(
                f"a point-in-time frame cannot hold static features {static}; "
                f"a value that varies with neither origin nor event time belongs to the "
                f"static table of the truth frame"
            )
        return self

    # -- feature groups ----------------------------------------------------

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(feature.name for feature in self.features)

    @property
    def observed_features(self) -> tuple[FeatureSpec, ...]:
        return tuple(feature for feature in self.features if feature.is_observed)

    @property
    def known_features(self) -> tuple[FeatureSpec, ...]:
        return tuple(feature for feature in self.features if feature.is_known)

    @property
    def has_observed_features(self) -> bool:
        return bool(self.observed_features)

    @property
    def has_known_features(self) -> bool:
        return bool(self.known_features)

    # -- derived shape -----------------------------------------------------

    @property
    def is_panel(self) -> bool:
        """Several instances share this schema, identified by ``instance_keys``."""
        return bool(self.instance_keys)

    # -- canonical table layout --------------------------------------------

    @property
    def columns(self) -> tuple[str, ...]:
        """``instance keys, origin time, event time, observed features, known features``."""
        return (
            *self.instance_keys,
            self.origin_time,
            self.event_time,
            *(feature.name for feature in self.observed_features),
            *(feature.name for feature in self.known_features),
        )

    @property
    def key_columns(self) -> tuple[str, ...]:
        """The columns that identify a row: ``instance keys, origin time, event time``."""
        return (*self.instance_keys, self.origin_time, self.event_time)

    def with_features(self, *added: FeatureSpec) -> PointInTimeSchema:
        """A copy declaring ``added`` in addition to the current features.

        Re-validated rather than copied, so a name that collides with an
        existing column is rejected here rather than surfacing later.
        """
        return PointInTimeSchema(
            origin_time=self.origin_time,
            event_time=self.event_time,
            event_frequency=self.event_frequency,
            origin_frequency=self.origin_frequency,
            instance_keys=self.instance_keys,
            features=(*self.features, *added),
        )


class PointInTimeFrame:
    """One Arrow table keyed by ``(instance keys..., origin_time, event_time)``.

    NaNs and nulls are preserved exactly. An availability that improves between
    vintages — ``NaN``, ``NaN``, ``42`` — is information about the data feed,
    and imputing it away would destroy the very thing point-in-time training is
    for.
    """

    def __init__(self, table: pa.Table, schema: PointInTimeSchema) -> None:
        self._schema = schema
        self._table = canonicalize(require_table(table, "table"), schema.columns, "table")

        require_no_nulls(self._table, schema.instance_keys, "table", "instance key")
        origin_zone = require_timestamps(self._table, schema.origin_time, "table", "origin time")
        event_zone = require_timestamps(self._table, schema.event_time, "table", "event time")
        if origin_zone != event_zone:
            raise DataError(
                f"origin time {schema.origin_time!r} and event time {schema.event_time!r} must "
                f"share a time zone, got {origin_zone!r} and {event_zone!r}; the two axes are "
                f"compared to each other on every lead time"
            )

        require_unique(
            self._table,
            schema.key_columns,
            "table",
            what="instance/origin/event",
            hint="each event time may appear once per instance and origin",
        )

        keys = key_rows(self._table, schema.instance_keys)
        origins: list[datetime] = column_values(self._table, schema.origin_time)
        events: list[datetime] = column_values(self._table, schema.event_time)

        validate_grid(group_times(keys, events), schema.event_frequency, "event time")
        if schema.origin_frequency is not None:
            validate_grid(group_times(keys, origins), schema.origin_frequency, "origin time")
        _reject_observed_values_after_origin(self._table, schema, origins, events)

    # -- accessors ---------------------------------------------------------

    @property
    def schema(self) -> PointInTimeSchema:
        return self._schema

    @property
    def table(self) -> pa.Table:
        return self._table

    @property
    def instances(self) -> tuple[InstanceKey, ...]:
        """The distinct instance keys present, in first-seen order."""
        return tuple(dict.fromkeys(key_rows(self._table, self._schema.instance_keys)))

    @property
    def origins(self) -> tuple[datetime, ...]:
        """Every distinct origin time, in ascending order."""
        return tuple(sorted(set(column_values(self._table, self._schema.origin_time))))

    @property
    def event_times(self) -> tuple[datetime, ...]:
        """Every distinct event time, in ascending order."""
        return tuple(sorted(set(column_values(self._table, self._schema.event_time))))

    # -- vintages ----------------------------------------------------------

    def at_origin(self, origin_time: str | datetime) -> PointInTimeFrame:
        """The single vintage issued at ``origin_time``.

        Nothing from a later origin can appear in the result, which is what
        makes it safe to hand downstream: a newer vintage is not information
        that existed at ``origin_time``.
        """
        moment = resolve_origin(origin_time, self.origins)
        mask = [value == moment for value in column_values(self._table, self._schema.origin_time)]
        return PointInTimeFrame(self._table.filter(pa.array(mask)), self._schema)

    def with_lead_time(
        self, unit: str | Frequency = "hour", *, name: str = DEFAULT_LEAD_TIME_NAME
    ) -> PointInTimeFrame:
        """A copy carrying ``event_time - origin_time`` as a known feature.

        The lead is measured in whole ``unit`` steps and may be negative, since
        a vintage can describe event times before its own origin. A lead that is
        not a whole number of steps is an error rather than a rounded value.
        """
        if name in self._table.column_names:
            raise DataError(
                f"cannot add lead time as {name!r}: the table already has that column; "
                f"pass name= to choose another"
            )
        frequency = Frequency.parse(unit)
        origins: list[datetime] = column_values(self._table, self._schema.origin_time)
        events: list[datetime] = column_values(self._table, self._schema.event_time)

        leads: list[int] = []
        offenders: list[str] = []
        for origin, event in zip(origins, events, strict=True):
            steps = frequency.steps_between(origin, event)
            if steps is None:
                offenders.append(f"{origin.isoformat()} -> {event.isoformat()}")
                steps = 0
            leads.append(steps)
        if offenders:
            raise DataError(
                f"{len(offenders)} lead times are not a whole number of {frequency} steps: "
                f"{summarize(offenders)}; ask for a finer unit rather than a rounded lead"
            )

        table = self._table.append_column(name, pa.array(leads, type=pa.int64()))
        return PointInTimeFrame(table, self._schema.with_features(FeatureSpec.known(name)))

    # -- construction ------------------------------------------------------

    @classmethod
    def from_arrow(
        cls,
        table: pa.Table,
        *,
        origin_time: str,
        event_time: str,
        event_frequency: str | Frequency,
        origin_frequency: str | Frequency | None = None,
        instance_keys: Sequence[str] = (),
        observed_features: Sequence[str] = (),
        known_features: Sequence[str] = (),
    ) -> PointInTimeFrame:
        """Build a frame from an Arrow table and a column-role description."""
        return cls(
            table,
            point_in_time_schema(
                origin_time=origin_time,
                event_time=event_time,
                event_frequency=event_frequency,
                origin_frequency=origin_frequency,
                instance_keys=instance_keys,
                observed_features=observed_features,
                known_features=known_features,
            ),
        )

    @classmethod
    def from_pandas(
        cls,
        table: Any,
        *,
        origin_time: str,
        event_time: str,
        event_frequency: str | Frequency,
        origin_frequency: str | Frequency | None = None,
        instance_keys: Sequence[str] = (),
        observed_features: Sequence[str] = (),
        known_features: Sequence[str] = (),
    ) -> PointInTimeFrame:
        """Same as :meth:`from_arrow`, for a pandas ``DataFrame`` input."""
        return cls.from_arrow(
            table_from_pandas(table, "table"),
            origin_time=origin_time,
            event_time=event_time,
            event_frequency=event_frequency,
            origin_frequency=origin_frequency,
            instance_keys=instance_keys,
            observed_features=observed_features,
            known_features=known_features,
        )

    # -- serialization -----------------------------------------------------

    def write(self, path: str | Path) -> Path:
        """Write ``schema.json`` and ``table.arrow`` into ``path``."""
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / SCHEMA_FILENAME).write_text(
            self._schema.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        write_table(directory / TABLE_FILENAME, self._table)
        return directory

    @classmethod
    def read(cls, path: str | Path) -> PointInTimeFrame:
        """Read back a directory written by :meth:`write`, re-validating it."""
        directory = Path(path)
        schema_path = directory / SCHEMA_FILENAME
        table_path = directory / TABLE_FILENAME
        for required in (schema_path, table_path):
            if not required.is_file():
                raise DataError(
                    f"{directory} is not a PointInTimeFrame: {required.name} is missing"
                )
        schema = PointInTimeSchema.model_validate(
            json.loads(schema_path.read_text(encoding="utf-8"))
        )
        return cls(read_table(table_path), schema)

    # -- dunder ------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PointInTimeFrame):
            return NotImplemented
        return self._schema == other._schema and bool(self._table.equals(other._table))

    def __repr__(self) -> str:
        shape = "panel" if self._schema.is_panel else "single"
        return (
            f"PointInTimeFrame({shape}, "
            f"event_frequency={self._schema.event_frequency}, "
            f"instances={len(self.instances)}, "
            f"origins={len(self.origins)}, "
            f"rows={self._table.num_rows})"
        )


def point_in_time_schema(
    *,
    origin_time: str,
    event_time: str,
    event_frequency: str | Frequency,
    origin_frequency: str | Frequency | None = None,
    instance_keys: Sequence[str] = (),
    observed_features: Sequence[str] = (),
    known_features: Sequence[str] = (),
) -> PointInTimeSchema:
    """Build a :class:`PointInTimeSchema` from lists of column names by role."""
    return PointInTimeSchema(
        origin_time=origin_time,
        event_time=event_time,
        event_frequency=Frequency.parse(event_frequency),
        origin_frequency=None if origin_frequency is None else Frequency.parse(origin_frequency),
        instance_keys=tuple(instance_keys),
        features=(
            *(FeatureSpec.observed(name) for name in observed_features),
            *(FeatureSpec.known(name) for name in known_features),
        ),
    )


def resolve_origin(origin_time: str | datetime, available: Sequence[datetime]) -> datetime:
    """Match ``origin_time`` against the origins that exist, exactly.

    Nearest-origin fallback is deliberately absent: silently answering for
    10:00 when 11:00 was asked for would hand back information from the wrong
    vintage.
    """
    moment = parse_moment(origin_time, "origin_time")
    if moment not in available:
        raise DataError(
            f"no origin {moment.isoformat()} in this data; "
            f"{len(available)} origins are available: {summarize(available)}"
        )
    return moment


def parse_moment(value: str | datetime, label: str) -> datetime:
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise DataError(f"cannot parse {label} {value!r} as an ISO 8601 timestamp") from error


def _reject_observed_values_after_origin(
    table: pa.Table,
    schema: PointInTimeSchema,
    origins: Sequence[datetime],
    events: Sequence[datetime],
) -> None:
    """An observed feature is unknowable past its own origin, so it must be missing there.

    A measured value present for an event time after the origin that supposedly
    produced it is leakage in its purest form: at that origin, nobody could
    have had it.
    """
    after = [
        index
        for index, (origin, event) in enumerate(zip(origins, events, strict=True))
        if event > origin
    ]
    if not after:
        return
    for feature in schema.observed_features:
        values = column_values(table, feature.name)
        offenders = [
            f"origin {origins[index].isoformat()} event {events[index].isoformat()} "
            f"= {values[index]!r}"
            for index in after
            if not is_missing(values[index])
        ]
        if offenders:
            raise DataError(
                f"observed feature {feature.name!r} has {len(offenders)} values for event times "
                f"after their origin: {summarize(offenders)}; an observed value is not knowable "
                f"before it happens, so declare the column as a known feature if it really is"
            )
