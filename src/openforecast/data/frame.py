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
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa

from openforecast.data.features import FeatureSpec
from openforecast.data.frequency import Frequency
from openforecast.data.schema import TimeSeriesSchema
from openforecast.errors import DataError

__all__ = ["TimeSeriesFrame"]

SCHEMA_FILENAME = "schema.json"
HISTORY_FILENAME = "history.arrow"
FUTURE_FILENAME = "future.arrow"
STATIC_FILENAME = "static.arrow"

# How many offending values an error message quotes before it truncates.
_MAX_REPORTED = 5

InstanceKey = tuple[Any, ...]


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
        history = _require_table(history, "history")
        future = None if future is None else _require_table(future, "future")
        static = None if static is None else _require_table(static, "static")

        self._history = _canonicalize(history, schema.history_columns, "history")
        _reject_target_and_observed_columns(future, schema)
        self._future = (
            None if future is None else _canonicalize(future, schema.future_columns, "future")
        )
        self._static = _resolve_static(static, schema)

        history_times = _instance_times(self._history, schema, "history")
        _validate_grid(history_times, schema.frequency, "history")
        if self._future is not None:
            future_times = _instance_times(self._future, schema, "future")
            _reject_unknown_instances(future_times, history_times, schema)
            _validate_grid(future_times, schema.frequency, "future", anchors=history_times)
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
        return tuple(dict.fromkeys(_key_rows(self._history, self._schema.instance_keys)))

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
            static = _extract_static(history, schema)
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
            _table_from_pandas(history, "history"),
            time=time,
            frequency=frequency,
            targets=targets,
            instance_keys=instance_keys,
            observed_features=observed_features,
            known_features=known_features,
            static_features=static_features,
            future=None if future is None else _table_from_pandas(future, "future"),
            static=None if static is None else _table_from_pandas(static, "static"),
        )

    # -- serialization -----------------------------------------------------

    def write(self, path: str | Path) -> Path:
        """Write ``schema.json`` and one Arrow IPC file per table into ``path``."""
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / SCHEMA_FILENAME).write_text(
            self._schema.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        _write_table(directory / HISTORY_FILENAME, self._history)
        for filename, table in (
            (FUTURE_FILENAME, self._future),
            (STATIC_FILENAME, self._static),
        ):
            target = directory / filename
            if table is None:
                # An absent table must not be read back from a previous write.
                target.unlink(missing_ok=True)
            else:
                _write_table(target, table)
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
            history=_read_table(history_path),
            schema=schema,
            future=_read_table(future_path) if future_path.is_file() else None,
            static=_read_table(static_path) if static_path.is_file() else None,
        )

    # -- dunder ------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TimeSeriesFrame):
            return NotImplemented
        return (
            self._schema == other._schema
            and bool(self._history.equals(other._history))
            and _tables_equal(self._future, other._future)
            and _tables_equal(self._static, other._static)
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


# -- table plumbing --------------------------------------------------------


def _table_from_pandas(frame: Any, label: str) -> pa.Table:
    try:
        return pa.Table.from_pandas(frame, preserve_index=False)
    except (TypeError, AttributeError) as error:  # not a DataFrame at all
        raise DataError(f"{label} is not a pandas DataFrame: {error}") from error


def _require_table(table: pa.Table, label: str) -> pa.Table:
    """Guard the boundary: everything downstream may assume a real Arrow table."""
    if not isinstance(table, pa.Table):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise DataError(f"{label} must be a pyarrow.Table, got {type(table).__name__}")
    return table


def _canonicalize(table: pa.Table, columns: tuple[str, ...], label: str) -> pa.Table:
    missing = [name for name in columns if name not in table.column_names]
    if missing:
        raise DataError(
            f"{label} is missing declared columns {missing}; present: {table.column_names}"
        )
    return table.select(list(columns))


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
    return _canonicalize(static, schema.static_columns, "static")


def _write_table(path: Path, table: pa.Table) -> None:
    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)


def _read_table(path: Path) -> pa.Table:
    with pa.OSFile(str(path), "rb") as source:
        return pa.ipc.open_file(source).read_all()


def _tables_equal(left: pa.Table | None, right: pa.Table | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return bool(left.equals(right))


# -- validation ------------------------------------------------------------


def _column_values(table: pa.Table, name: str) -> list[Any]:
    return table.column(name).to_pylist()


def _key_rows(table: pa.Table, instance_keys: tuple[str, ...]) -> list[InstanceKey]:
    """One tuple per row identifying its instance; ``()`` when there are no keys."""
    if not instance_keys:
        return [()] * table.num_rows
    columns = [_column_values(table, name) for name in instance_keys]
    return list(zip(*columns, strict=True))


def _instance_times(
    table: pa.Table, schema: TimeSeriesSchema, label: str
) -> dict[InstanceKey, list[datetime]]:
    """Group event times by instance, rejecting nulls and duplicate rows."""
    for name in schema.instance_keys:
        if table.column(name).null_count:
            raise DataError(f"{label} has null values in instance key {name!r}")
    time_column = table.column(schema.time)
    if not pa.types.is_timestamp(time_column.type):
        raise DataError(
            f"{label} column {schema.time!r} must be a timestamp, got {time_column.type}"
        )
    if time_column.null_count:
        raise DataError(f"{label} has null values in time column {schema.time!r}")

    keys = _key_rows(table, schema.instance_keys)
    times: list[datetime] = _column_values(table, schema.time)
    duplicates = [row for row, count in Counter(zip(keys, times, strict=True)).items() if count > 1]
    if duplicates:
        raise DataError(
            f"{label} has {len(duplicates)} duplicate instance/time rows: "
            f"{_summarize(duplicates)}; each event time may appear once per instance"
        )

    grouped: dict[InstanceKey, list[datetime]] = defaultdict(list)
    for key, moment in zip(keys, times, strict=True):
        grouped[key].append(moment)
    return dict(grouped)


def _validate_grid(
    grouped: dict[InstanceKey, list[datetime]],
    frequency: Frequency,
    label: str,
    anchors: dict[InstanceKey, list[datetime]] | None = None,
) -> None:
    """Every timestamp must sit on the frequency grid of its instance.

    Gaps are allowed — a missing observation is information, and filling it in
    would be exactly the silent repair the architecture forbids. Timestamps
    *between* grid points are not, because they mean the declared frequency is
    wrong. ``anchors`` lets the future table be checked against the history
    grid rather than its own, so the two cannot be a half-step apart.
    """
    for key, times in grouped.items():
        reference = min(anchors[key]) if anchors is not None else min(times)
        offenders = [
            moment for moment in times if frequency.steps_between(reference, moment) is None
        ]
        if offenders:
            instance = f" for instance {key}" if key else ""
            raise DataError(
                f"{label} has {len(offenders)} timestamps{instance} that do not sit on the "
                f"{frequency} grid anchored at {reference.isoformat()}: {_summarize(offenders)}"
            )


def _reject_unknown_instances(
    future_times: dict[InstanceKey, list[datetime]],
    history_times: dict[InstanceKey, list[datetime]],
    schema: TimeSeriesSchema,
) -> None:
    unknown = sorted(set(future_times) - set(history_times), key=repr)
    if unknown:
        raise DataError(
            f"future contains instances absent from history: {_summarize(unknown)}; "
            f"instance keys are {list(schema.instance_keys)}"
        )


def _validate_static_rows(
    static: pa.Table, schema: TimeSeriesSchema, instances: set[InstanceKey]
) -> None:
    for name in schema.instance_keys:
        if static.column(name).null_count:
            raise DataError(f"static has null values in instance key {name!r}")

    keys = _key_rows(static, schema.instance_keys)
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    if duplicates:
        raise DataError(
            f"static must hold exactly one row per instance but repeats {_summarize(duplicates)}"
        )
    missing = sorted(instances - set(keys), key=repr)
    if missing:
        raise DataError(f"static is missing rows for instances {_summarize(missing)}")
    extra = sorted(set(keys) - instances, key=repr)
    if extra:
        raise DataError(f"static holds rows for instances absent from history: {_summarize(extra)}")


def _extract_static(history: pa.Table, schema: TimeSeriesSchema) -> pa.Table:
    """Lift declared static features out of a wide history table.

    A static feature that varies within an instance is not static, so this
    raises instead of picking a value.
    """
    names = [feature.name for feature in schema.static_features]
    missing = [name for name in names if name not in history.column_names]
    if missing:
        raise DataError(
            f"static features {missing} are neither in history nor in a static table; "
            f"pass static= explicitly if they live in their own frame"
        )

    keys = _key_rows(history, schema.instance_keys)
    ordered = list(dict.fromkeys(keys))
    values: dict[str, list[Any]] = {}
    for name in names:
        # canonical value -> first original value, per instance.
        by_instance: dict[InstanceKey, dict[Any, Any]] = defaultdict(dict)
        for key, value in zip(keys, _column_values(history, name), strict=True):
            by_instance[key].setdefault(_canonical(value), value)
        conflicts = [
            f"{key}: {sorted(seen.values(), key=repr)}"
            for key, seen in by_instance.items()
            if len(seen) > 1
        ]
        if conflicts:
            raise DataError(
                f"static feature {name!r} varies within an instance, so it is not static: "
                f"{_summarize(conflicts)}"
            )
        values[name] = [next(iter(by_instance[key].values())) for key in ordered]

    columns: dict[str, pa.Array[Any]] = {}
    for index, key_name in enumerate(schema.instance_keys):
        columns[key_name] = pa.array(
            [key[index] for key in ordered], type=history.column(key_name).type
        )
    for name in names:
        columns[name] = pa.array(values[name], type=history.column(name).type)
    return pa.table(columns)


_NOT_A_NUMBER = object()


def _canonical(value: Any) -> Any:
    """A grouping key that treats every NaN as the same missing value.

    ``float('nan') != float('nan')``, so grouping on the raw values would report
    a column of NaNs as varying within its instance.
    """
    if isinstance(value, float) and math.isnan(value):
        return _NOT_A_NUMBER
    return value


def _summarize(items: Iterable[Any]) -> str:
    listed = list(items)
    shown = ", ".join(repr(item) for item in listed[:_MAX_REPORTED])
    if len(listed) > _MAX_REPORTED:
        shown += f", ... (+{len(listed) - _MAX_REPORTED} more)"
    return shown
