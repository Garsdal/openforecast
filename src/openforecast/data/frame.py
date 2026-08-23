"""``TimeSeriesFrame``: ordinary event-time data in canonical Arrow layout.

Three tables, one schema:

```text
history   instance keys, event time, targets, observed features, known features
future    instance keys, event time, known temporal features
static    instance keys, static features
```

Everything here validates rather than repairs. Duplicate rows, off-grid
timestamps, targets leaking into the future table and static features that vary
within an instance are all errors, because each of them silently changes what
the data means.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa

from openforecast.data._arrow import (
    InstanceKey,
    build_table,
    canonical_value,
    canonicalize,
    column_type,
    column_values,
    group_times,
    key_rows,
    read_table,
    require_no_nulls,
    require_table,
    require_timestamps,
    require_unique,
    summarize,
    table_from_pandas,
    tables_equal,
    validate_grid,
    write_table,
)
from openforecast.data.features import FeatureSpec
from openforecast.data.frequency import Frequency
from openforecast.data.point_in_time import parse_moment
from openforecast.data.schema import TimeSeriesSchema
from openforecast.errors import DataError

__all__ = ["TimeSeriesFrame"]

SCHEMA_FILENAME = "schema.json"
HISTORY_FILENAME = "history.arrow"
FUTURE_FILENAME = "future.arrow"
STATIC_FILENAME = "static.arrow"


class TimeSeriesFrame:
    """Event-time time-series data validated against a :class:`TimeSeriesSchema`.

    The tables are stored in canonical column order. Columns of the input that
    the schema does not declare are dropped; columns the schema does declare
    must be present.
    """

    def __init__(
        self,
        history: pa.Table,
        schema: TimeSeriesSchema,
        future: pa.Table | None = None,
        static: pa.Table | None = None,
    ) -> None:
        self._schema = schema
        history = require_table(history, "history")
        future = None if future is None else require_table(future, "future")
        static = None if static is None else require_table(static, "static")

        self._history = canonicalize(history, schema.history_columns, "history")
        _reject_target_and_observed_columns(future, schema)
        self._future = (
            None if future is None else canonicalize(future, schema.future_columns, "future")
        )
        self._static = _resolve_static(static, schema)

        history_times = _instance_times(self._history, schema, "history")
        validate_grid(history_times, schema.frequency, "history")
        if self._future is not None:
            future_times = _instance_times(self._future, schema, "future")
            _reject_unknown_instances(future_times, history_times, schema)
            validate_grid(future_times, schema.frequency, "future", anchors=history_times)
        if self._static is not None:
            _validate_static_rows(self._static, schema, set(history_times))

    # -- accessors ---------------------------------------------------------

    @property
    def schema(self) -> TimeSeriesSchema:
        return self._schema

    @property
    def history(self) -> pa.Table:
        return self._history

    @property
    def future(self) -> pa.Table | None:
        return self._future

    @property
    def static(self) -> pa.Table | None:
        return self._static

    @property
    def instances(self) -> tuple[InstanceKey, ...]:
        """The distinct instance keys present in ``history``, in first-seen order."""
        return tuple(dict.fromkeys(key_rows(self._history, self._schema.instance_keys)))

    # -- one moment --------------------------------------------------------

    def up_to(self, moment: str | datetime) -> TimeSeriesFrame:
        """This frame as it would have looked at ``moment``.

        The history is truncated to the event times at or before it, which is
        what makes a historical origin usable as a fit: a model evaluated at
        ``moment`` must not have been trained on what happened afterwards.

        Known features are not discarded with it. A known feature is one whose
        future values are knowable in advance — that is the role's whole
        meaning — so the values the truncated rows held for later event times
        move into the future table, where a forecast at ``moment`` can condition
        on them. Observed features and targets do not move: neither was knowable
        then, and the future table refuses to carry either.

        Origins here are therefore *simulated*, in the sense
        :class:`~openforecast.views.provenance.OriginFidelity` means: the values
        are today's, cut back to the shape the past had. Real vintages are what
        :meth:`~openforecast.data.forecast_dataset.ForecastDataset.up_to` is for.
        """
        when = parse_moment(moment, "moment")
        times: list[datetime] = column_values(self._history, self._schema.time)
        kept = [event <= when for event in times]
        if not any(kept):
            raise DataError(
                f"nothing in this frame is at or before {when.isoformat()}; its history "
                f"begins at {min(times).isoformat()}"
            )
        history = self._history.filter(pa.array(kept))
        instances = set(key_rows(history, self._schema.instance_keys))
        static = self._static
        return TimeSeriesFrame(
            history=history,
            schema=self._schema,
            future=self._knowable_after(when, instances),
            static=None if static is None else static_for(static, self._schema, instances),
        )

    def _knowable_after(self, moment: datetime, instances: set[InstanceKey]) -> pa.Table | None:
        """The known-feature values for event times after ``moment``.

        Gathered from the discarded history rows and from the future table
        alike, since the two are the same statement about the same feature; a
        cell held by both takes the history's value, which is the frame's own
        record of it.
        """
        schema = self._schema
        if not schema.has_known_features:
            return None
        columns = schema.future_columns
        values: dict[str, list[Any]] = {name: [] for name in columns}
        seen: set[tuple[InstanceKey, datetime]] = set()
        for table in (self._history, self._future):
            if table is None:
                continue
            keys = key_rows(table, schema.instance_keys)
            times: list[datetime] = column_values(table, schema.time)
            available = {name: column_values(table, name) for name in columns}
            for index, cell in enumerate(zip(keys, times, strict=True)):
                if cell[1] <= moment or cell in seen or cell[0] not in instances:
                    continue
                seen.add(cell)
                for name in columns:
                    values[name].append(available[name][index])
        if not seen:
            return None
        return build_table(
            {name: (values[name], column_type(self._history, name)) for name in columns}
        )

    # -- construction ------------------------------------------------------

    @classmethod
    def from_arrow(
        cls,
        history: pa.Table,
        *,
        time: str,
        frequency: str | Frequency,
        targets: Sequence[str],
        instance_keys: Sequence[str] = (),
        observed_features: Sequence[str] = (),
        known_features: Sequence[str] = (),
        static_features: Sequence[str] = (),
        future: pa.Table | None = None,
        static: pa.Table | None = None,
    ) -> TimeSeriesFrame:
        """Build a frame from Arrow tables and a column-role description.

        When ``static_features`` are declared but no ``static`` table is given,
        the static table is extracted from ``history``, which requires each
        static column to hold exactly one value per instance.
        """
        schema = TimeSeriesSchema(
            time=time,
            frequency=Frequency.parse(frequency),
            instance_keys=tuple(instance_keys),
            targets=tuple(targets),
            features=(
                *(FeatureSpec.observed(name) for name in observed_features),
                *(FeatureSpec.known(name) for name in known_features),
                *(FeatureSpec.static(name) for name in static_features),
            ),
        )
        if static is None and schema.has_static_features:
            static = extract_static(history, schema)
        return cls(history=history, schema=schema, future=future, static=static)

    @classmethod
    def from_pandas(
        cls,
        history: Any,
        *,
        time: str,
        frequency: str | Frequency,
        targets: Sequence[str],
        instance_keys: Sequence[str] = (),
        observed_features: Sequence[str] = (),
        known_features: Sequence[str] = (),
        static_features: Sequence[str] = (),
        future: Any | None = None,
        static: Any | None = None,
    ) -> TimeSeriesFrame:
        """Same as :meth:`from_arrow`, for pandas ``DataFrame`` inputs.

        ``pandas`` is not an OpenForecast dependency; the frames are converted
        by ``pyarrow`` and never stored in pandas form.
        """
        return cls.from_arrow(
            table_from_pandas(history, "history"),
            time=time,
            frequency=frequency,
            targets=targets,
            instance_keys=instance_keys,
            observed_features=observed_features,
            known_features=known_features,
            static_features=static_features,
            future=None if future is None else table_from_pandas(future, "future"),
            static=None if static is None else table_from_pandas(static, "static"),
        )

    # -- serialization -----------------------------------------------------

    def write(self, path: str | Path) -> Path:
        """Write ``schema.json`` and one Arrow IPC file per table into ``path``."""
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / SCHEMA_FILENAME).write_text(
            self._schema.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        write_table(directory / HISTORY_FILENAME, self._history)
        for filename, table in (
            (FUTURE_FILENAME, self._future),
            (STATIC_FILENAME, self._static),
        ):
            target = directory / filename
            if table is None:
                # An absent table must not be read back from a previous write.
                target.unlink(missing_ok=True)
            else:
                write_table(target, table)
        return directory

    @classmethod
    def read(cls, path: str | Path) -> TimeSeriesFrame:
        """Read back a directory written by :meth:`write`, re-validating it."""
        directory = Path(path)
        schema_path = directory / SCHEMA_FILENAME
        history_path = directory / HISTORY_FILENAME
        for required in (schema_path, history_path):
            if not required.is_file():
                raise DataError(f"{directory} is not a TimeSeriesFrame: {required.name} is missing")
        schema = TimeSeriesSchema.model_validate(
            json.loads(schema_path.read_text(encoding="utf-8"))
        )
        future_path = directory / FUTURE_FILENAME
        static_path = directory / STATIC_FILENAME
        return cls(
            history=read_table(history_path),
            schema=schema,
            future=read_table(future_path) if future_path.is_file() else None,
            static=read_table(static_path) if static_path.is_file() else None,
        )

    # -- dunder ------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TimeSeriesFrame):
            return NotImplemented
        return (
            self._schema == other._schema
            and bool(self._history.equals(other._history))
            and tables_equal(self._future, other._future)
            and tables_equal(self._static, other._static)
        )

    def __repr__(self) -> str:
        shape = "panel" if self._schema.is_panel else "single"
        variate = "univariate" if self._schema.is_univariate else "multivariate"
        return (
            f"TimeSeriesFrame({shape} {variate}, "
            f"frequency={self._schema.frequency}, "
            f"instances={len(self.instances)}, "
            f"history_rows={self._history.num_rows}, "
            f"future_rows={0 if self._future is None else self._future.num_rows})"
        )


# -- validation ------------------------------------------------------------


def _reject_target_and_observed_columns(future: pa.Table | None, schema: TimeSeriesSchema) -> None:
    """The future table describes what is knowable, so it holds neither.

    A target in the future table is the label being predicted; an observed
    feature there is a value that will not exist at forecast time. Both are
    leakage, so they are rejected rather than dropped.
    """
    if future is None:
        return
    present = set(future.column_names)
    targets = sorted(present & set(schema.targets))
    if targets:
        raise DataError(
            f"future must not contain target columns {targets}; "
            f"the future table describes what is known, not what is being predicted"
        )
    observed = sorted(present & {feature.name for feature in schema.observed_features})
    if observed:
        raise DataError(
            f"future must not contain observed-only features {observed}; "
            f"declare them as known features if their future values really are knowable"
        )


def _resolve_static(static: pa.Table | None, schema: TimeSeriesSchema) -> pa.Table | None:
    if static is None:
        if schema.has_static_features:
            raise DataError(
                f"schema declares static features "
                f"{[feature.name for feature in schema.static_features]} but no static table "
                f"was provided"
            )
        return None
    if not schema.has_static_features:
        raise DataError("a static table was provided but the schema declares no static features")
    return canonicalize(static, schema.static_columns, "static")


def _instance_times(
    table: pa.Table, schema: TimeSeriesSchema, label: str
) -> dict[InstanceKey, list[datetime]]:
    """Group event times by instance, rejecting nulls and duplicate rows."""
    require_no_nulls(table, schema.instance_keys, label, "instance key")
    require_timestamps(table, schema.time, label)
    require_unique(
        table,
        (*schema.instance_keys, schema.time),
        label,
        what="instance/time",
        hint="each event time may appear once per instance",
    )
    return group_times(key_rows(table, schema.instance_keys), column_values(table, schema.time))


def _reject_unknown_instances(
    future_times: dict[InstanceKey, list[datetime]],
    history_times: dict[InstanceKey, list[datetime]],
    schema: TimeSeriesSchema,
) -> None:
    unknown = sorted(set(future_times) - set(history_times), key=repr)
    if unknown:
        raise DataError(
            f"future contains instances absent from history: {summarize(unknown)}; "
            f"instance keys are {list(schema.instance_keys)}"
        )


def _validate_static_rows(
    static: pa.Table, schema: TimeSeriesSchema, instances: set[InstanceKey]
) -> None:
    require_no_nulls(static, schema.instance_keys, "static", "instance key")

    keys = key_rows(static, schema.instance_keys)
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    if duplicates:
        raise DataError(
            f"static must hold exactly one row per instance but repeats {summarize(duplicates)}"
        )
    missing = sorted(instances - set(keys), key=repr)
    if missing:
        raise DataError(f"static is missing rows for instances {summarize(missing)}")
    extra = sorted(set(keys) - instances, key=repr)
    if extra:
        raise DataError(f"static holds rows for instances absent from history: {summarize(extra)}")


def static_for(static: pa.Table, schema: TimeSeriesSchema, instances: set[InstanceKey]) -> pa.Table:
    """The static rows of ``instances`` and no others.

    A static table holds exactly one row per instance in the history, so
    truncating a frame past the whole of some instance's history has to drop its
    static row with it rather than leave one describing nothing.
    """
    keys = key_rows(static, schema.instance_keys)
    return static.filter(pa.array([key in instances for key in keys]))


def extract_static(history: pa.Table, schema: TimeSeriesSchema) -> pa.Table:
    """Lift declared static features out of a wide table.

    A static feature that varies within an instance is not static, so this
    raises instead of picking a value. Repeated rows per instance are fine,
    which is what lets a point-in-time table — many origins per instance — be a
    source of static features too.
    """
    names = [feature.name for feature in schema.static_features]
    missing = [name for name in names if name not in history.column_names]
    if missing:
        raise DataError(
            f"static features {missing} are neither in history nor in a static table; "
            f"pass static= explicitly if they live in their own frame"
        )

    keys = key_rows(history, schema.instance_keys)
    ordered = list(dict.fromkeys(keys))
    values: dict[str, list[Any]] = {}
    for name in names:
        # canonical value -> first original value, per instance.
        by_instance: dict[InstanceKey, dict[Any, Any]] = defaultdict(dict)
        for key, value in zip(keys, column_values(history, name), strict=True):
            by_instance[key].setdefault(canonical_value(value), value)
        conflicts = [
            f"{key}: {sorted(seen.values(), key=repr)}"
            for key, seen in by_instance.items()
            if len(seen) > 1
        ]
        if conflicts:
            raise DataError(
                f"static feature {name!r} varies within an instance, so it is not static: "
                f"{summarize(conflicts)}"
            )
        values[name] = [next(iter(by_instance[key].values())) for key in ordered]

    columns: dict[str, pa.Array[Any]] = {}
    for index, key_name in enumerate(schema.instance_keys):
        columns[key_name] = pa.array(
            [key[index] for key in ordered], type=column_type(history, key_name)
        )
    for name in names:
        columns[name] = pa.array(values[name], type=column_type(history, name))
    return pa.table(columns)
