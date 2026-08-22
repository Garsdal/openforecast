"""``SeriesView``: one complete time series per training unit.

The execution view of classical local forecasters — ARIMA, ETS, Theta. Each
series is fitted on its own, so the view is a long table keyed by an opaque
``series_id`` plus a small table mapping those ids back to instance keys.

```text
temporal   series_id, event_time, targets, observed features, known features
series     series_id, instance keys
static     series_id, static features
```

A series carries exactly one forecast origin, because a single time axis cannot
express two. That is why point-in-time data reaches this view only at a selected
origin: the vintage of that origin becomes the series, and no other vintage
contributes. Where that vintage says nothing — most vintages describe event
times ahead of themselves, not behind — the feature values are missing, which is
what was true at the origin.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

import pyarrow as pa

from openforecast.data._arrow import column_values, tables_equal
from openforecast.errors import DataError
from openforecast.views.base import (
    EVENT_TIME,
    SERIES_ID,
    ViewKind,
    ViewSchema,
    prepare,
    require_matching_ids,
    require_rows,
)
from openforecast.views.provenance import ViewProvenance

__all__ = ["SeriesView", "SeriesViewSchema"]


class SeriesViewSchema(ViewSchema):
    """What the columns of a :class:`SeriesView` mean."""

    expected_kind: ClassVar[ViewKind] = ViewKind.SERIES
    reserved: ClassVar[tuple[str, ...]] = (SERIES_ID, EVENT_TIME)

    kind: ViewKind = ViewKind.SERIES
    #: The origin the series was cut at, when one was selected. ``None`` for an
    #: ordinary event-time series, which is its own newest vintage.
    origin_time: datetime | None = None

    @property
    def temporal_columns(self) -> tuple[str, ...]:
        """``series_id, event_time, targets, observed features, known features``."""
        return (SERIES_ID, EVENT_TIME, *self.targets, *self.temporal_feature_names)

    @property
    def series_columns(self) -> tuple[str, ...]:
        """``series_id, instance keys``."""
        return (SERIES_ID, *self.instance_keys)

    @property
    def static_columns(self) -> tuple[str, ...]:
        """``series_id, static features``."""
        return (SERIES_ID, *self.static_feature_names)


class SeriesView:
    """One long temporal table holding one or more complete series."""

    def __init__(
        self,
        temporal: pa.Table,
        series: pa.Table,
        schema: SeriesViewSchema,
        provenance: ViewProvenance,
        static: pa.Table | None = None,
    ) -> None:
        self._schema = schema
        self._provenance = provenance
        self._temporal = require_rows(
            prepare(temporal, schema.temporal_columns, "temporal"), "temporal"
        )
        self._series = require_rows(prepare(series, schema.series_columns, "series"), "series")
        self._ids = require_matching_ids(
            self._temporal, self._series, SERIES_ID, "temporal", "series"
        )
        self._static = _resolve_static(static, schema, self._ids)

    # -- accessors ---------------------------------------------------------

    @property
    def kind(self) -> ViewKind:
        return self._schema.kind

    @property
    def schema(self) -> SeriesViewSchema:
        return self._schema

    @property
    def provenance(self) -> ViewProvenance:
        return self._provenance

    @property
    def temporal(self) -> pa.Table:
        return self._temporal

    @property
    def series(self) -> pa.Table:
        """``series_id -> instance keys``: how a forecast is labeled again."""
        return self._series

    @property
    def static(self) -> pa.Table | None:
        return self._static

    @property
    def series_ids(self) -> tuple[str, ...]:
        return self._ids

    # -- dunder ------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SeriesView):
            return NotImplemented
        return (
            self._schema == other._schema
            and self._provenance == other._provenance
            and bool(self._temporal.equals(other._temporal))
            and bool(self._series.equals(other._series))
            and tables_equal(self._static, other._static)
        )

    def __repr__(self) -> str:
        return (
            f"SeriesView(series={len(self._ids)}, "
            f"rows={self._temporal.num_rows}, "
            f"targets={list(self._schema.targets)}, "
            f"origin_fidelity={self._provenance.origin_fidelity})"
        )


def _resolve_static(
    static: pa.Table | None, schema: SeriesViewSchema, ids: tuple[str, ...]
) -> pa.Table | None:
    if static is None:
        if schema.has_static_features:
            raise DataError(
                f"schema declares static features {list(schema.static_feature_names)} "
                f"but no static table was provided"
            )
        return None
    if not schema.has_static_features:
        raise DataError("a static table was provided but the schema declares no static features")
    table = prepare(static, schema.static_columns, "static")
    keys: list[str] = column_values(table, SERIES_ID)
    if sorted(keys) != sorted(ids):
        raise DataError(
            f"static must hold exactly one row per series; it holds {len(keys)} rows "
            f"for {len(ids)} series"
        )
    return table
