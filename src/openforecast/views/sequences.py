"""``SequenceView``: many context → horizon sequences per training unit.

The execution view of global models — NHiTS, TFT, PatchTST, the global Darts
models. One ``instance × origin`` pair is one training sample: a fixed-length
context window ending at the origin, followed by a fixed-length forecast window
after it.

```text
temporal   sample_id, event_time, targets, observed features, known features
samples    sample_id, instance keys, origin_time, context_start, context_end,
           forecast_start, forecast_end
static     sample_id, static features
```

The sample boundaries are declared in ``samples`` and enforced against
``temporal``: every sample holds exactly ``context + horizon`` rows on the
frequency grid, and its origin is the last context step. That is what stops a
provider from accidentally learning across two origins — the invariant lives in
the view rather than in each integration's conversion code.

An ordinary event-time frame and a point-in-time dataset produce the *same* type
here. The only thing that distinguishes them is
:class:`~openforecast.views.provenance.OriginFidelity`.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

import pyarrow as pa
from pydantic import Field

from openforecast.data._arrow import column_values, summarize, tables_equal
from openforecast.errors import DataError
from openforecast.views.base import (
    CONTEXT_END,
    CONTEXT_START,
    EVENT_TIME,
    FORECAST_END,
    FORECAST_START,
    ORIGIN_TIME,
    SAMPLE_ID,
    ViewKind,
    ViewSchema,
    group_positions,
    prepare,
    require_matching_ids,
    require_rows,
)
from openforecast.views.provenance import ViewProvenance

__all__ = ["SequenceView", "SequenceViewSchema"]


class SequenceViewSchema(ViewSchema):
    """What the columns of a :class:`SequenceView` mean, and how long a sample is."""

    expected_kind: ClassVar[ViewKind] = ViewKind.SEQUENCES
    reserved: ClassVar[tuple[str, ...]] = (
        SAMPLE_ID,
        EVENT_TIME,
        ORIGIN_TIME,
        CONTEXT_START,
        CONTEXT_END,
        FORECAST_START,
        FORECAST_END,
    )

    kind: ViewKind = ViewKind.SEQUENCES
    context: int = Field(ge=1)
    horizon: int = Field(ge=1)

    @property
    def length(self) -> int:
        """How many event times one sample spans."""
        return self.context + self.horizon

    @property
    def temporal_columns(self) -> tuple[str, ...]:
        """``sample_id, event_time, targets, observed features, known features``."""
        return (SAMPLE_ID, EVENT_TIME, *self.targets, *self.temporal_feature_names)

    @property
    def samples_columns(self) -> tuple[str, ...]:
        """``sample_id, instance keys, origin_time, and the four window bounds``."""
        return (
            SAMPLE_ID,
            *self.instance_keys,
            ORIGIN_TIME,
            CONTEXT_START,
            CONTEXT_END,
            FORECAST_START,
            FORECAST_END,
        )

    @property
    def static_columns(self) -> tuple[str, ...]:
        """``sample_id, static features``."""
        return (SAMPLE_ID, *self.static_feature_names)


class SequenceView:
    """Forecast-conditioned sequences, one per ``instance × origin``."""

    def __init__(
        self,
        temporal: pa.Table,
        samples: pa.Table,
        schema: SequenceViewSchema,
        provenance: ViewProvenance,
        static: pa.Table | None = None,
    ) -> None:
        self._schema = schema
        self._provenance = provenance
        self._temporal = require_rows(
            prepare(temporal, schema.temporal_columns, "temporal"), "temporal"
        )
        self._samples = require_rows(prepare(samples, schema.samples_columns, "samples"), "samples")
        self._ids = require_matching_ids(
            self._temporal, self._samples, SAMPLE_ID, "temporal", "samples"
        )
        self._static = _resolve_static(static, schema, self._ids)
        _validate_windows(self._temporal, self._samples, schema)

    # -- accessors ---------------------------------------------------------

    @property
    def kind(self) -> ViewKind:
        return self._schema.kind

    @property
    def schema(self) -> SequenceViewSchema:
        return self._schema

    @property
    def provenance(self) -> ViewProvenance:
        return self._provenance

    @property
    def temporal(self) -> pa.Table:
        return self._temporal

    @property
    def samples(self) -> pa.Table:
        """``sample_id -> instance keys and window bounds``, one row per sample."""
        return self._samples

    @property
    def static(self) -> pa.Table | None:
        return self._static

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return self._ids

    @property
    def origins(self) -> tuple[datetime, ...]:
        """The distinct origins the samples were built at, in ascending order."""
        return tuple(sorted(set(column_values(self._samples, ORIGIN_TIME))))

    # -- dunder ------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SequenceView):
            return NotImplemented
        return (
            self._schema == other._schema
            and self._provenance == other._provenance
            and bool(self._temporal.equals(other._temporal))
            and bool(self._samples.equals(other._samples))
            and tables_equal(self._static, other._static)
        )

    def __repr__(self) -> str:
        return (
            f"SequenceView(samples={len(self._ids)}, "
            f"origins={len(self.origins)}, "
            f"context={self._schema.context}, horizon={self._schema.horizon}, "
            f"origin_fidelity={self._provenance.origin_fidelity})"
        )


def _resolve_static(
    static: pa.Table | None, schema: SequenceViewSchema, ids: tuple[str, ...]
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
    keys: list[str] = column_values(table, SAMPLE_ID)
    if sorted(keys) != sorted(ids):
        raise DataError(
            f"static must hold exactly one row per sample; it holds {len(keys)} rows "
            f"for {len(ids)} samples"
        )
    return table


def _validate_windows(temporal: pa.Table, samples: pa.Table, schema: SequenceViewSchema) -> None:
    """Every sample must cover exactly its declared window, step by step.

    This is the one-sequence invariant. A sample that is short a step, holds a
    duplicate event time or spans two origins would train the model on a
    sequence nobody described, and no provider can detect that afterwards.
    """
    frequency = schema.frequency
    rows = group_positions(column_values(temporal, SAMPLE_ID))
    event_times: list[datetime] = column_values(temporal, EVENT_TIME)

    bounds = zip(
        column_values(samples, SAMPLE_ID),
        column_values(samples, ORIGIN_TIME),
        column_values(samples, CONTEXT_START),
        column_values(samples, CONTEXT_END),
        column_values(samples, FORECAST_START),
        column_values(samples, FORECAST_END),
        strict=True,
    )
    for sample_id, origin, context_start, context_end, forecast_start, forecast_end in bounds:
        expected = [frequency.shift(context_start, step) for step in range(schema.length)]
        declared = (origin, context_end, forecast_start, forecast_end)
        wanted = (
            expected[schema.context - 1],
            expected[schema.context - 1],
            expected[schema.context],
            expected[-1],
        )
        if declared != wanted:
            raise DataError(
                f"sample {sample_id} declares bounds that do not describe "
                f"{schema.context} context and {schema.horizon} forecast steps of "
                f"{frequency} from {context_start.isoformat()}: "
                f"origin/context_end/forecast_start/forecast_end are {declared}, "
                f"expected {wanted}"
            )
        actual = [event_times[position] for position in rows[sample_id]]
        if actual != expected:
            raise DataError(
                f"sample {sample_id} holds {len(actual)} event times, expected the "
                f"{schema.length} steps {summarize(expected)}; got {summarize(actual)}"
            )
