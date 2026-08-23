"""What a command prints: one JSON document, or one aligned table.

```text
openforecast models list           MODEL  NAME  FIT  VIEW  OUTPUTS
openforecast models list --json    {"models": [...]}
```

Every information-producing command supports both, which is Step 26.3, and the
two are the same facts rather than two answers — the table is a projection of
the document. Shared here so that adding a command is adding a projection and
not a second opinion about how a null prints.

Everything written through this module goes to the stream the command was handed,
which is stdout. That is Step 26.4's half of the contract: stdout is the
requested output and nothing else, so ``--json`` really can be piped into ``jq``.
Logs, progress and warnings are stderr's, and no function here writes there.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import IO, Any, cast

import pyarrow as pa

__all__ = ["cell", "dump", "jsonable", "rows_as_table", "rows_of", "table"]

#: What a null prints as in a table. JSON has ``null``; a column of blanks in a
#: terminal is a column you cannot tell from a column of empty strings.
MISSING = "-"


def dump(payload: object, out: IO[str]) -> None:
    """One JSON document, indented, on the stream that carries the answer."""
    print(json.dumps(jsonable(payload), indent=2), file=out)


def jsonable(value: Any) -> Any:
    """The same value in types :func:`json.dumps` writes without a hook.

    Moments become ISO 8601 strings and paths become their text, because those
    are what a caller reading the JSON expects to compare and to pass back in.
    A NaN is written as ``null``: it is the absence of a measurement, and
    ``NaN`` is not JSON.
    """
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        mapping = cast(Mapping[Any, Any], value)
        return {str(key): jsonable(item) for key, item in mapping.items()}
    if isinstance(value, str | bytes) or not isinstance(value, Iterable):
        return value
    return [jsonable(item) for item in cast(Iterable[Any], value)]


def rows_of(source: pa.Table, *, limit: int | None = None) -> list[dict[str, Any]]:
    """An Arrow table as JSON-ready rows, in the order it holds them.

    ``limit`` truncates, for the human rendering of a forecast that is thousands
    of rows long. The JSON one is never truncated: a document that quietly held
    the first twenty rows would be a document a script cannot trust.
    """
    table_ = source if limit is None else source.slice(0, limit)
    return [jsonable(row) for row in table_.to_pylist()]


def cell(value: object) -> str:
    """One value, as a table cell: a moment as ISO 8601, a null as ``-``."""
    if value is None:
        return MISSING
    if isinstance(value, float):
        return MISSING if not math.isfinite(value) else f"{value:g}"
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def table(header: Sequence[str], rows: Sequence[Sequence[str]], out: IO[str]) -> None:
    """Columns aligned to their widest cell, header included."""
    if not rows:
        print("  ".join(header), file=out)
        return
    widths = [max(len(row[index]) for row in (header, *rows)) for index in range(len(header))]
    for row in (header, *rows):
        cells = (text.ljust(width) for text, width in zip(row, widths, strict=True))
        print("  ".join(cells).rstrip(), file=out)


def rows_as_table(rows: Sequence[Mapping[str, Any]], out: IO[str]) -> None:
    """A list of rows as a table, taking the columns from the first one."""
    if not rows:
        return
    header = list(rows[0])
    table(
        [name.upper() for name in header],
        [[cell(row.get(name)) for name in header] for row in rows],
        out,
    )
