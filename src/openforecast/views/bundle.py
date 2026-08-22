"""Execution views on disk, as a provider in another process receives them.

```text
series/            sequences/         tabular/           forecast/
    schema.json        schema.json        schema.json        schema.json
    provenance.json    provenance.json    provenance.json    origin.json
    temporal.arrow     temporal.arrow     x.arrow            history.arrow
    series.arrow       samples.arrow      y.arrow            future.arrow
    static.arrow       static.arrow       keys.arrow         static.arrow
```

Control messages are small and travel as JSON; a view is bulk data and travels
as Arrow IPC in a directory beside it. That split is the reason a request stays
readable in a log while a hundred thousand training sequences do not have to be
base64 in the middle of it.

A bundle is written by the engine and read by the provider, so it is the *same*
representation the in-process provider is handed — the tables are the view's own
tables, not a flattened copy of them. Reading one reconstructs the view through
its ordinary constructor, which means every invariant the view enforces is
enforced again on the far side of the process boundary: a bundle that was
truncated in transit fails to load rather than training on a short window.

``schema.json`` names the view kind, so :func:`read_view` needs to be told
nothing about what it is opening.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
from pydantic import BaseModel, ConfigDict

from openforecast.data._arrow import read_table, write_table
from openforecast.errors import DataError
from openforecast.protocol.vocabulary import ViewKind
from openforecast.views.forecast import ForecastView, ForecastViewMetadata
from openforecast.views.planner import FitView
from openforecast.views.provenance import ViewProvenance
from openforecast.views.sequences import SequenceView, SequenceViewSchema
from openforecast.views.series import SeriesView, SeriesViewSchema
from openforecast.views.tabular import TabularView, TabularViewSchema

__all__ = [
    "ORIGIN_FILENAME",
    "PROVENANCE_FILENAME",
    "SCHEMA_FILENAME",
    "read_answer",
    "read_fit_view",
    "read_forecast_view",
    "read_view",
    "write_answer",
    "write_view",
]

SCHEMA_FILENAME = "schema.json"
PROVENANCE_FILENAME = "provenance.json"
#: A forecast view carries no provenance — it is one origin, not a training
#: set — but it does carry the origin, and nothing else in the bundle says it.
ORIGIN_FILENAME = "origin.json"

TEMPORAL_FILENAME = "temporal.arrow"
SERIES_FILENAME = "series.arrow"
SAMPLES_FILENAME = "samples.arrow"
STATIC_FILENAME = "static.arrow"
X_FILENAME = "x.arrow"
Y_FILENAME = "y.arrow"
KEYS_FILENAME = "keys.arrow"
HISTORY_FILENAME = "history.arrow"
FUTURE_FILENAME = "future.arrow"


class ForecastOrigin(BaseModel):
    """``origin.json``: the one moment a forecast bundle describes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    origin_time: datetime


def write_view(view: FitView | ForecastView, path: str | Path) -> Path:
    """Write ``view`` into ``path`` as a bundle, and return the directory."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    if isinstance(view, ForecastView):
        _write_json(directory / SCHEMA_FILENAME, view.metadata)
        _write_json(directory / ORIGIN_FILENAME, ForecastOrigin(origin_time=view.origin_time))
        tables = {
            HISTORY_FILENAME: view.history,
            FUTURE_FILENAME: view.future,
            STATIC_FILENAME: view.static,
        }
    else:
        _write_json(directory / SCHEMA_FILENAME, view.schema)
        _write_json(directory / PROVENANCE_FILENAME, view.provenance)
        tables = _fit_tables(view)
    for filename, table in tables.items():
        target = directory / filename
        if table is None:
            # An absent table must not be read back from an earlier write.
            target.unlink(missing_ok=True)
        else:
            write_table(target, table)
    return directory


def _fit_tables(view: FitView) -> dict[str, pa.Table | None]:
    if isinstance(view, SeriesView):
        return {
            TEMPORAL_FILENAME: view.temporal,
            SERIES_FILENAME: view.series,
            STATIC_FILENAME: view.static,
        }
    if isinstance(view, SequenceView):
        return {
            TEMPORAL_FILENAME: view.temporal,
            SAMPLES_FILENAME: view.samples,
            STATIC_FILENAME: view.static,
        }
    return {X_FILENAME: view.X, Y_FILENAME: view.y, KEYS_FILENAME: view.keys}


def read_view(path: str | Path) -> FitView | ForecastView:
    """Read whichever view ``path`` holds, dispatching on the kind it declares."""
    directory = Path(path)
    payload = _read_schema(directory)
    kind = _declared_kind(payload, directory)
    if kind is ViewKind.FORECAST:
        return _read_forecast(directory, payload)
    return _read_fit(directory, payload, kind)


def read_fit_view(path: str | Path) -> FitView:
    """Read a training bundle, refusing a forecast one."""
    view = read_view(path)
    if isinstance(view, ForecastView):
        raise DataError(f"{path} holds a forecast view, and a training view was expected")
    return view


def read_forecast_view(path: str | Path) -> ForecastView:
    """Read a forecast bundle, refusing a training one."""
    view = read_view(path)
    if not isinstance(view, ForecastView):
        raise DataError(f"{path} holds a {view.kind} view, and a forecast view was expected")
    return view


def _read_fit(directory: Path, payload: dict[str, Any], kind: ViewKind) -> FitView:
    provenance = ViewProvenance.model_validate(
        _read_json(directory / PROVENANCE_FILENAME, directory)
    )
    if kind is ViewKind.SERIES:
        return SeriesView(
            temporal=_read_arrow(directory / TEMPORAL_FILENAME, directory),
            series=_read_arrow(directory / SERIES_FILENAME, directory),
            schema=SeriesViewSchema.model_validate(payload),
            provenance=provenance,
            static=_read_optional(directory / STATIC_FILENAME),
        )
    if kind is ViewKind.SEQUENCES:
        return SequenceView(
            temporal=_read_arrow(directory / TEMPORAL_FILENAME, directory),
            samples=_read_arrow(directory / SAMPLES_FILENAME, directory),
            schema=SequenceViewSchema.model_validate(payload),
            provenance=provenance,
            static=_read_optional(directory / STATIC_FILENAME),
        )
    return TabularView(
        X=_read_arrow(directory / X_FILENAME, directory),
        y=_read_arrow(directory / Y_FILENAME, directory),
        keys=_read_arrow(directory / KEYS_FILENAME, directory),
        schema=TabularViewSchema.model_validate(payload),
        provenance=provenance,
    )


def _read_forecast(directory: Path, payload: dict[str, Any]) -> ForecastView:
    origin = ForecastOrigin.model_validate(_read_json(directory / ORIGIN_FILENAME, directory))
    return ForecastView(
        origin_time=origin.origin_time,
        history=_read_arrow(directory / HISTORY_FILENAME, directory),
        future=_read_arrow(directory / FUTURE_FILENAME, directory),
        metadata=ForecastViewMetadata.model_validate(payload),
        static=_read_optional(directory / STATIC_FILENAME),
    )


def write_answer(answer: pa.Table, path: str | Path) -> Path:
    """Write a forecast to ``path`` as one Arrow IPC file.

    The answer half of the boundary. It lives beside the view bundles for the
    same reason they do: a provider may import :mod:`openforecast.views` and not
    the Arrow plumbing of the semantic layer, and both directions of bulk data
    should be written by one piece of code.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_table(target, answer)
    return target


def read_answer(path: str | Path) -> pa.Table:
    """Read a forecast a provider wrote with :func:`write_answer`."""
    target = Path(path)
    if not target.is_file():
        raise DataError(f"no forecast was written to {target}")
    return read_table(target)


def _declared_kind(payload: dict[str, Any], directory: Path) -> ViewKind:
    declared = payload.get("kind")
    try:
        return ViewKind(declared)
    except ValueError:
        raise DataError(
            f"{directory / SCHEMA_FILENAME} declares kind {declared!r}, which is not one of "
            f"{[kind.value for kind in ViewKind]}"
        ) from None


def _read_schema(directory: Path) -> dict[str, Any]:
    payload = _read_json(directory / SCHEMA_FILENAME, directory)
    if not isinstance(payload, dict):
        raise DataError(f"{directory / SCHEMA_FILENAME} is not a view schema object")
    return payload  # pyright: ignore[reportUnknownVariableType]


def _read_json(path: Path, directory: Path) -> Any:
    if not path.is_file():
        raise DataError(f"{directory} is not a view bundle: {path.name} is missing")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DataError(f"{path} is not valid JSON: {error}") from error


def _read_arrow(path: Path, directory: Path) -> pa.Table:
    if not path.is_file():
        raise DataError(f"{directory} is not a view bundle: {path.name} is missing")
    return read_table(path)


def _read_optional(path: Path) -> pa.Table | None:
    return read_table(path) if path.is_file() else None


def _write_json(path: Path, payload: BaseModel) -> None:
    path.write_text(payload.model_dump_json(indent=2) + "\n", encoding="utf-8")
