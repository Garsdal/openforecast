"""Vocabulary shared by the execution views.

Views are provider-neutral, so they are keyed by OpenForecast's own column
names rather than by the caller's. ``series_id``, ``sample_id``, ``row_id``,
``event_time`` and ``origin_time`` mean the same thing in every view, which is
what lets a provider be written once against a view instead of once against
every source schema.

Target and feature columns keep the names the caller gave them: a provider has
to hand results back labeled with the target the user asked about.

``ViewKind`` is defined in :mod:`openforecast.protocol.vocabulary` and
re-exported here. A model's training contract has to name the view it consumes,
and ``models/`` sits above ``views/`` in the layering, so the enum lives where
both can reach it rather than being spelled twice.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from hashlib import blake2b
from typing import Any, ClassVar, Self

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from openforecast.data._arrow import canonicalize, column_values, require_table, summarize
from openforecast.data.features import FeatureSpec
from openforecast.data.frequency import Frequency
from openforecast.data.schema import reject_duplicate_names
from openforecast.errors import DataError, SchemaError
from openforecast.protocol.vocabulary import ViewKind

__all__ = [
    "CONTEXT_END",
    "CONTEXT_START",
    "EVENT_TIME",
    "FORECAST_END",
    "FORECAST_START",
    "HORIZON_STEP",
    "ORIGIN_TIME",
    "ROW_ID",
    "SAMPLE_ID",
    "SERIES_ID",
    "FeatureGroups",
    "ViewKind",
    "opaque_id",
]

#: The identity of one training unit, per view.
SERIES_ID = "series_id"
SAMPLE_ID = "sample_id"
ROW_ID = "row_id"

#: The two time axes, named the same way wherever they appear.
EVENT_TIME = "event_time"
ORIGIN_TIME = "origin_time"

#: The bounds a sequence sample covers, and where a tabular row sits in it.
CONTEXT_START = "context_start"
CONTEXT_END = "context_end"
FORECAST_START = "forecast_start"
FORECAST_END = "forecast_end"
HORIZON_STEP = "horizon_step"

# Wide enough that a collision is not a practical concern, short enough to read
# in an error message.
_ID_BYTES = 8


def opaque_id(*parts: object) -> str:
    """A deterministic identifier for one training unit.

    Deterministic so that the same source and the same plan produce the same
    view twice, and opaque so that nothing downstream can recover the instance
    or the origin from it — the mapping lives in the view's key table, where a
    provider is not looking.
    """
    payload = "\x1f".join(_render(part) for part in parts)
    return blake2b(payload.encode("utf-8"), digest_size=_ID_BYTES).hexdigest()


def _render(part: object) -> str:
    if isinstance(part, datetime):
        return part.isoformat()
    if isinstance(part, tuple):
        return "\x1e".join(_render(item) for item in part)  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    return repr(part)


class FeatureGroups(BaseModel):
    """The feature-role accessors every view schema needs.

    Providers map the three roles onto their own vocabulary — past covariates,
    future covariates, static covariates — so every view has to expose them the
    same way.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    features: tuple[FeatureSpec, ...] = ()

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
    def static_features(self) -> tuple[FeatureSpec, ...]:
        return tuple(feature for feature in self.features if feature.is_static)

    @property
    def has_observed_features(self) -> bool:
        return bool(self.observed_features)

    @property
    def has_known_features(self) -> bool:
        return bool(self.known_features)

    @property
    def has_static_features(self) -> bool:
        return bool(self.static_features)

    @property
    def temporal_feature_names(self) -> tuple[str, ...]:
        """Observed then known — the canonical order of the temporal columns."""
        return (
            *(feature.name for feature in self.observed_features),
            *(feature.name for feature in self.known_features),
        )

    @property
    def static_feature_names(self) -> tuple[str, ...]:
        return tuple(feature.name for feature in self.static_features)


class ViewSchema(FeatureGroups):
    """What every view schema declares, however it lays its tables out.

    ``kind`` is a field rather than a class attribute so that a serialized
    schema says which view it describes; ``expected_kind`` is what keeps the two
    from disagreeing.
    """

    expected_kind: ClassVar[ViewKind]
    #: Column names the view uses for itself, which a caller's column may not reuse.
    reserved: ClassVar[tuple[str, ...]] = ()

    kind: ViewKind
    frequency: Frequency
    targets: tuple[str, ...]
    instance_keys: tuple[str, ...] = ()

    @field_validator("frequency", mode="before")
    @classmethod
    def _parse_frequency(cls, value: object) -> object:
        """Accept ``frequency="1h"`` and store the native representation."""
        return Frequency.parse(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def _check_declaration(self) -> Self:
        if self.kind is not type(self).expected_kind:
            raise SchemaError(
                f"a {type(self).expected_kind} view schema cannot declare kind {self.kind}"
            )
        if not self.targets:
            raise SchemaError(f"a {self.kind} view must declare at least one target")
        reject_duplicate_names(self.targets, "target")
        reject_duplicate_names(self.feature_names, "feature")
        reject_duplicate_names(self.instance_keys, "instance key")
        overlap = set(self.targets) & set(self.feature_names)
        if overlap:
            raise SchemaError(f"a column cannot be both a target and a feature: {sorted(overlap)}")
        reject_reserved_names(self.targets, type(self).reserved, "target")
        reject_reserved_names(self.feature_names, type(self).reserved, "feature")
        reject_reserved_names(self.instance_keys, type(self).reserved, "instance key")
        return self


# -- validation shared by the view types ------------------------------------


def reject_reserved_names(names: Iterable[str], reserved: Iterable[str], label: str) -> None:
    """A caller's column may not be spelled like one of the view's own columns.

    Views own ``series_id``, ``sample_id``, ``event_time`` and friends. A target
    called ``event_time`` would collide with the axis rather than travel beside
    it, so it is rejected here instead of silently overwriting a column.
    """
    taken = set(reserved)
    offenders = sorted(name for name in names if name in taken)
    if offenders:
        raise SchemaError(
            f"{offenders} cannot be {label} names in a view: "
            f"a view reserves {sorted(taken)} for its own columns"
        )


def prepare(table: pa.Table, columns: Sequence[str], label: str) -> pa.Table:
    """Require a real Arrow table and put its declared columns in canonical order."""
    return canonicalize(require_table(table, label), columns, label)


def require_matching_ids(
    table: pa.Table, key_table: pa.Table, column: str, label: str, key_label: str
) -> tuple[str, ...]:
    """The two tables must describe exactly the same set of training units.

    An id in the data with no key row cannot be mapped back to an instance, and
    a key row with no data is a training unit that was announced and never
    materialized. Both mean the view was built wrong.
    """
    ids: list[str] = column_values(table, column)
    keys: list[str] = column_values(key_table, column)
    if len(set(keys)) != len(keys):
        raise DataError(f"{key_label} has duplicate {column} values")
    orphans = sorted(set(ids) - set(keys))
    if orphans:
        raise DataError(
            f"{label} holds {column} values absent from {key_label}: {summarize(orphans)}"
        )
    absent = sorted(set(keys) - set(ids))
    if absent:
        raise DataError(f"{key_label} announces {column} values absent from {label}: {absent}")
    return tuple(dict.fromkeys(ids))


def require_rows(table: pa.Table, label: str) -> pa.Table:
    if not table.num_rows:
        raise DataError(f"{label} is empty; a view must hold at least one row")
    return table


def group_positions(values: Iterable[Any]) -> dict[Any, list[int]]:
    """``value -> the row positions holding it``, in first-seen order."""
    grouped: dict[Any, list[int]] = {}
    for position, value in enumerate(values):
        grouped.setdefault(value, []).append(position)
    return grouped
