"""Arrow plumbing shared by the semantic frames.

Private to :mod:`openforecast.data`. Everything here is a validation or a
mechanical table operation — nothing in this module decides what data *means*,
which is why both the event-time and the point-in-time frame can use it without
either one leaking its vocabulary into the other.

The validations raise :class:`~openforecast.errors.DataError` rather than
repairing anything: no deduplication, no snapping to the grid, no imputation.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa

from openforecast.data.frequency import Frequency
from openforecast.errors import DataError

#: The values of the instance-key columns for one row; ``()`` when unkeyed.
InstanceKey = tuple[Any, ...]

# How many offending values an error message quotes before it truncates.
MAX_REPORTED = 5


# -- table plumbing --------------------------------------------------------


def table_from_pandas(frame: Any, label: str) -> pa.Table:
    try:
        return pa.Table.from_pandas(frame, preserve_index=False)
    except (TypeError, AttributeError) as error:  # not a DataFrame at all
        raise DataError(f"{label} is not a pandas DataFrame: {error}") from error


def require_table(table: pa.Table, label: str) -> pa.Table:
    """Guard the boundary: everything downstream may assume a real Arrow table."""
    if not isinstance(table, pa.Table):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise DataError(f"{label} must be a pyarrow.Table, got {type(table).__name__}")
    return table


def canonicalize(table: pa.Table, columns: Sequence[str], label: str) -> pa.Table:
    """Select ``columns`` in canonical order, requiring every one to be present.

    Undeclared columns are dropped; declared ones missing from the table are an
    error, because the schema is the description of what the data is.
    """
    missing = [name for name in columns if name not in table.column_names]
    if missing:
        raise DataError(
            f"{label} is missing declared columns {missing}; present: {table.column_names}"
        )
    return table.select(list(columns))


def build_table(columns: dict[str, tuple[list[Any], pa.DataType]]) -> pa.Table:
    """Assemble a table from ``name -> (values, type)``, preserving nulls and NaNs."""
    return pa.table({name: pa.array(values, type=kind) for name, (values, kind) in columns.items()})


def column_values(table: pa.Table, name: str) -> list[Any]:
    return table.column(name).to_pylist()


def column_type(table: pa.Table, name: str) -> pa.DataType:
    return table.column(name).type


def key_rows(table: pa.Table, columns: Sequence[str]) -> list[InstanceKey]:
    """One tuple per row holding its ``columns`` values; ``()`` when there are none."""
    if not columns:
        return [()] * table.num_rows
    values = [column_values(table, name) for name in columns]
    return list(zip(*values, strict=True))


def write_table(path: Path, table: pa.Table) -> None:
    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)


def read_table(path: Path) -> pa.Table:
    with pa.OSFile(str(path), "rb") as source:
        return pa.ipc.open_file(source).read_all()


def tables_equal(left: pa.Table | None, right: pa.Table | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return bool(left.equals(right))


# -- validation ------------------------------------------------------------


def require_no_nulls(table: pa.Table, columns: Sequence[str], label: str, role: str) -> None:
    for name in columns:
        if table.column(name).null_count:
            raise DataError(f"{label} has null values in {role} {name!r}")


def require_timestamps(table: pa.Table, name: str, label: str, role: str = "time column") -> Any:
    """Require a non-null timestamp column, returning its time zone (or ``None``)."""
    column = table.column(name)
    if not pa.types.is_timestamp(column.type):
        raise DataError(f"{label} column {name!r} must be a timestamp, got {column.type}")
    if column.null_count:
        raise DataError(f"{label} has null values in {role} {name!r}")
    return getattr(column.type, "tz", None)


def require_unique(
    table: pa.Table, columns: Sequence[str], label: str, *, what: str, hint: str
) -> list[InstanceKey]:
    """Reject repeated ``columns`` combinations, returning them row by row."""
    rows = key_rows(table, columns)
    duplicates = [row for row, count in Counter(rows).items() if count > 1]
    if duplicates:
        raise DataError(
            f"{label} has {len(duplicates)} duplicate {what} rows: {summarize(duplicates)}; {hint}"
        )
    return rows


def group_times(
    keys: Sequence[InstanceKey], times: Sequence[datetime]
) -> dict[InstanceKey, list[datetime]]:
    grouped: dict[InstanceKey, list[datetime]] = defaultdict(list)
    for key, moment in zip(keys, times, strict=True):
        grouped[key].append(moment)
    return dict(grouped)


def validate_grid(
    grouped: dict[InstanceKey, list[datetime]],
    frequency: Frequency,
    label: str,
    anchors: dict[InstanceKey, list[datetime]] | None = None,
) -> None:
    """Every timestamp must sit on the frequency grid of its instance.

    Gaps are allowed — a missing observation is information, and filling it in
    would be exactly the silent repair the architecture forbids. Timestamps
    *between* grid points are not, because they mean the declared frequency is
    wrong. ``anchors`` lets one axis be checked against another's grid rather
    than its own, so the two cannot be a half-step apart.
    """
    for key, times in grouped.items():
        anchoring = times if anchors is None else anchors.get(key, times)
        reference = min(anchoring)
        offenders = [
            moment for moment in times if frequency.steps_between(reference, moment) is None
        ]
        if offenders:
            instance = f" for instance {key}" if key else ""
            raise DataError(
                f"{label} has {len(offenders)} timestamps{instance} that do not sit on the "
                f"{frequency} grid anchored at {reference.isoformat()}: {summarize(offenders)}"
            )


_NOT_A_NUMBER = object()


def canonical_value(value: Any) -> Any:
    """A grouping key that treats every NaN and null as the same missing value.

    ``float('nan') != float('nan')``, so grouping on raw values would report a
    column of NaNs as holding several distinct values.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return _NOT_A_NUMBER
    return value


def is_missing(value: Any) -> bool:
    """True for a null or a NaN — the two spellings of "no value here"."""
    return canonical_value(value) is _NOT_A_NUMBER


def summarize(items: Iterable[Any]) -> str:
    listed = list(items)
    shown = ", ".join(repr(item) for item in listed[:MAX_REPORTED])
    if len(listed) > MAX_REPORTED:
        shown += f", ... (+{len(listed) - MAX_REPORTED} more)"
    return shown
