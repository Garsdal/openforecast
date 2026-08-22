"""The event-time schema: ``instance x event_time x variable``.

This describes ordinary time-series data. It deliberately cannot express
forecast vintages — a value here belongs to an event time and nothing else.
Point-in-time semantics arrive as their own schema in Step 3 rather than as
optional fields bolted onto this one.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from openforecast.data.features import FeatureSpec
from openforecast.data.frequency import Frequency
from openforecast.errors import SchemaError

__all__ = ["TimeSeriesSchema"]


def reject_duplicate_names(names: Sequence[str], role: str) -> None:
    """Shared by both semantic schemas: a repeated column name has no one meaning."""
    counts = Counter(names)
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    if duplicates:
        raise SchemaError(f"duplicate {role} names: {duplicates}")


class TimeSeriesSchema(BaseModel):
    """What the columns of a :class:`~openforecast.data.frame.TimeSeriesFrame` mean.

    The semantic axes are orthogonal and the interesting categories are derived
    from them: a panel of several targets is ``is_panel and is_multivariate``,
    not a ``PANEL_MULTIVARIATE`` enum member.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    time: str
    frequency: Frequency

    instance_keys: tuple[str, ...] = ()
    targets: tuple[str, ...]
    features: tuple[FeatureSpec, ...] = ()

    @field_validator("frequency", mode="before")
    @classmethod
    def _parse_frequency(cls, value: object) -> object:
        """Accept ``frequency="1h"`` and store the native representation."""
        return Frequency.parse(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def _check_names(self) -> Self:
        if not self.targets:
            raise SchemaError("a time series schema must declare at least one target")

        reject_duplicate_names(self.instance_keys, "instance key")
        reject_duplicate_names(self.targets, "target")
        reject_duplicate_names(self.feature_names, "feature")

        for name in (self.time, *self.instance_keys, *self.targets, *self.feature_names):
            if not name.strip():
                raise SchemaError("column names must not be empty")

        overlap = set(self.targets) & set(self.feature_names)
        if overlap:
            raise SchemaError(
                f"a column cannot be both a target and a feature: {sorted(overlap)}; "
                f"declare it once with the role it actually has"
            )
        for role, names in (
            ("target", self.targets),
            ("feature", self.feature_names),
            ("instance key", self.instance_keys),
        ):
            if self.time in names:
                raise SchemaError(f"the time column {self.time!r} cannot also be a {role}")
        keyed = set(self.instance_keys) & (set(self.targets) | set(self.feature_names))
        if keyed:
            raise SchemaError(
                f"a column cannot be both an instance key and a target or feature: {sorted(keyed)}"
            )
        return self

    # -- feature groups ----------------------------------------------------

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(feature.name for feature in self.features)

    @property
    def temporal_features(self) -> tuple[FeatureSpec, ...]:
        return tuple(feature for feature in self.features if feature.is_temporal)

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

    # -- derived shape -----------------------------------------------------

    @property
    def is_panel(self) -> bool:
        """Several instances share this schema, identified by ``instance_keys``."""
        return bool(self.instance_keys)

    @property
    def target_count(self) -> int:
        return len(self.targets)

    @property
    def is_univariate(self) -> bool:
        return self.target_count == 1

    @property
    def is_multivariate(self) -> bool:
        return self.target_count > 1

    # -- canonical table layouts -------------------------------------------

    @property
    def history_columns(self) -> tuple[str, ...]:
        """``instance keys, event time, targets, observed features, known features``."""
        return (
            *self.instance_keys,
            self.time,
            *self.targets,
            *(feature.name for feature in self.observed_features),
            *(feature.name for feature in self.known_features),
        )

    @property
    def future_columns(self) -> tuple[str, ...]:
        """``instance keys, event time, known temporal features``."""
        return (
            *self.instance_keys,
            self.time,
            *(feature.name for feature in self.known_features),
        )

    @property
    def static_columns(self) -> tuple[str, ...]:
        """``instance keys, static features``."""
        return (*self.instance_keys, *(feature.name for feature in self.static_features))
