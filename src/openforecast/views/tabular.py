"""``TabularView``: one supervised row per target to be predicted.

The execution view of reduction models — LightGBM, XGBoost, CatBoost. The
training unit is a single row: what was knowable at an origin about one event
time, and what that event time turned out to be.

```text
X      feature columns, knowable at the origin
y      target columns, the realized outcome
keys   row_id, instance keys, origin_time, event_time, horizon_step
```

The three tables are row-aligned: row *i* of ``X`` is described by row *i* of
``keys`` and labeled by row *i* of ``y``. Keeping the keys out of ``X`` is what
stops an estimator from being handed a timestamp as a feature by accident.

This is exactly the shape point-in-time gradient-boosting pipelines already
train on: one row per ``instance × origin × event time``, with the feature
values of *that* origin rather than the newest ones.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar, Self

import pyarrow as pa
from pydantic import Field, model_validator

from openforecast.data._arrow import column_values, require_unique, summarize
from openforecast.errors import DataError, SchemaError
from openforecast.views.base import (
    EVENT_TIME,
    HORIZON_STEP,
    ORIGIN_TIME,
    ROW_ID,
    ViewKind,
    ViewSchema,
    prepare,
    require_rows,
)
from openforecast.views.provenance import ViewProvenance

__all__ = ["TabularView", "TabularViewSchema"]


class TabularViewSchema(ViewSchema):
    """What the columns of a :class:`TabularView` mean."""

    expected_kind: ClassVar[ViewKind] = ViewKind.TABULAR
    reserved: ClassVar[tuple[str, ...]] = (ROW_ID, ORIGIN_TIME, EVENT_TIME, HORIZON_STEP)

    kind: ViewKind = ViewKind.TABULAR
    horizon: int = Field(ge=1)

    @model_validator(mode="after")
    def _reject_observed_features(self) -> Self:
        if self.has_observed_features:
            raise SchemaError(
                f"a tabular row describes an event time after its origin, so the observed "
                f"features {[feature.name for feature in self.observed_features]} have no "
                f"value there; carry the information they hold as an explicit lag feature "
                f"instead"
            )
        return self

    @property
    def x_columns(self) -> tuple[str, ...]:
        """``known features, static features`` — everything knowable at the origin."""
        return (
            *(feature.name for feature in self.known_features),
            *self.static_feature_names,
        )

    @property
    def y_columns(self) -> tuple[str, ...]:
        return self.targets

    @property
    def keys_columns(self) -> tuple[str, ...]:
        """``row_id, instance keys, origin_time, event_time, horizon_step``."""
        return (ROW_ID, *self.instance_keys, ORIGIN_TIME, EVENT_TIME, HORIZON_STEP)


class TabularView:
    """Row-aligned features, labels and keys."""

    def __init__(
        self,
        # Capitalized because the design matrix is X everywhere it is discussed.
        X: pa.Table,
        y: pa.Table,
        keys: pa.Table,
        schema: TabularViewSchema,
        provenance: ViewProvenance,
    ) -> None:
        self._schema = schema
        self._provenance = provenance
        self._x = prepare(X, schema.x_columns, "X")
        self._y = require_rows(prepare(y, schema.y_columns, "y"), "y")
        self._keys = require_rows(prepare(keys, schema.keys_columns, "keys"), "keys")
        _validate_alignment(self._x, self._y, self._keys)
        _validate_keys(self._keys, schema)

    # -- accessors ---------------------------------------------------------

    @property
    def kind(self) -> ViewKind:
        return self._schema.kind

    @property
    def schema(self) -> TabularViewSchema:
        return self._schema

    @property
    def provenance(self) -> ViewProvenance:
        return self._provenance

    @property
    def X(self) -> pa.Table:
        return self._x

    @property
    def y(self) -> pa.Table:
        return self._y

    @property
    def keys(self) -> pa.Table:
        return self._keys

    @property
    def num_rows(self) -> int:
        return self._keys.num_rows

    @property
    def origins(self) -> tuple[datetime, ...]:
        """The distinct origins the rows were built at, in ascending order."""
        return tuple(sorted(set(column_values(self._keys, ORIGIN_TIME))))

    # -- dunder ------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TabularView):
            return NotImplemented
        return (
            self._schema == other._schema
            and self._provenance == other._provenance
            and bool(self._x.equals(other._x))
            and bool(self._y.equals(other._y))
            and bool(self._keys.equals(other._keys))
        )

    def __repr__(self) -> str:
        return (
            f"TabularView(rows={self.num_rows}, "
            f"features={len(self._schema.x_columns)}, "
            f"targets={list(self._schema.targets)}, "
            f"origin_fidelity={self._provenance.origin_fidelity})"
        )


def _validate_alignment(x: pa.Table, y: pa.Table, keys: pa.Table) -> None:
    counts = {"X": x.num_rows, "y": y.num_rows, "keys": keys.num_rows}
    if len(set(counts.values())) > 1:
        raise DataError(
            f"X, y and keys are row-aligned, so they must hold the same number of rows: {counts}"
        )


def _validate_keys(keys: pa.Table, schema: TabularViewSchema) -> None:
    """A key row must place its label where its horizon step says it is."""
    require_unique(
        keys,
        (ROW_ID,),
        "keys",
        what="row_id",
        hint="a row id identifies exactly one supervised row",
    )
    origins: list[datetime] = column_values(keys, ORIGIN_TIME)
    events: list[datetime] = column_values(keys, EVENT_TIME)
    steps: list[int] = column_values(keys, HORIZON_STEP)

    offenders = [
        f"origin {origin.isoformat()} event {event.isoformat()} step {step}"
        for origin, event, step in zip(origins, events, steps, strict=True)
        if not 1 <= step <= schema.horizon or schema.frequency.steps_between(origin, event) != step
    ]
    if offenders:
        raise DataError(
            f"{len(offenders)} key rows have a horizon step that is not the number of "
            f"{schema.frequency} steps from the origin to the event time, or lies outside "
            f"1..{schema.horizon}: {summarize(offenders)}"
        )
