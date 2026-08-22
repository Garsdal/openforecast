"""Feature semantics: what a column is, and when its values are knowable.

The two axes are orthogonal on purpose. ``kind`` says whether a column varies
over the time axis; ``availability`` says whether its values for future event
times are knowable at forecast time. Combining them into one enum would make
every consumer pattern-match on combinations that are really independent.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from openforecast.errors import SchemaError

__all__ = ["FeatureAvailability", "FeatureKind", "FeatureSpec"]


class FeatureAvailability(StrEnum):
    #: Known only up to the forecast origin, e.g. a measured temperature.
    OBSERVED = "observed"
    #: Known for future event times too, e.g. a calendar flag or a weather forecast.
    KNOWN = "known"


class FeatureKind(StrEnum):
    #: Varies along the time axis.
    TEMPORAL = "temporal"
    #: Constant per instance, e.g. installed capacity.
    STATIC = "static"


class FeatureSpec(BaseModel):
    """One non-target column, with its semantics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    kind: FeatureKind = FeatureKind.TEMPORAL
    availability: FeatureAvailability | None = None

    @model_validator(mode="after")
    def _check_availability_matches_kind(self) -> Self:
        if not self.name.strip():
            raise SchemaError("feature name must not be empty")
        if self.kind is FeatureKind.TEMPORAL and self.availability is None:
            raise SchemaError(
                f"temporal feature {self.name!r} must declare an availability "
                f"of 'observed' or 'known'"
            )
        if self.kind is FeatureKind.STATIC and self.availability is not None:
            raise SchemaError(
                f"static feature {self.name!r} must not declare an availability; "
                f"a value that never varies over time is knowable at every origin"
            )
        return self

    @classmethod
    def observed(cls, name: str) -> FeatureSpec:
        """A temporal feature known only up to the forecast origin."""
        return cls(name=name, kind=FeatureKind.TEMPORAL, availability=FeatureAvailability.OBSERVED)

    @classmethod
    def known(cls, name: str) -> FeatureSpec:
        """A temporal feature known for future event times as well."""
        return cls(name=name, kind=FeatureKind.TEMPORAL, availability=FeatureAvailability.KNOWN)

    @classmethod
    def static(cls, name: str) -> FeatureSpec:
        """A feature that is constant within an instance."""
        return cls(name=name, kind=FeatureKind.STATIC, availability=None)

    @property
    def is_temporal(self) -> bool:
        return self.kind is FeatureKind.TEMPORAL

    @property
    def is_static(self) -> bool:
        return self.kind is FeatureKind.STATIC

    @property
    def is_observed(self) -> bool:
        return self.availability is FeatureAvailability.OBSERVED

    @property
    def is_known(self) -> bool:
        return self.availability is FeatureAvailability.KNOWN
