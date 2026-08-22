"""What a model can be asked to do with the data it is given.

Structured rather than a flat bag of booleans, because the questions are
independent: how many instances, how many targets, which feature roles, what
kind of output, and what to do about missing values. The engine checks a
materialized view against these before a provider is started, so an unsupported
request fails as a declaration mismatch rather than as a provider stack trace.

Every default is the conservative answer. A descriptor that declares nothing
describes a single-series, univariate, point-forecast model with no feature
support that cannot see a missing value — so a capability a provider has is
something it states, never something it is assumed to have.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from openforecast.data.features import FeatureSpec
from openforecast.errors import SchemaError

__all__ = [
    "FeatureCapabilities",
    "InstanceCapabilities",
    "MissingValueSupport",
    "ModelCapabilities",
    "OutputCapabilities",
    "TargetCapabilities",
]


class InstanceCapabilities(BaseModel):
    """How many series at once."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    single: bool = True
    panel: bool = False

    @model_validator(mode="after")
    def _check_something_is_supported(self) -> Self:
        if not (self.single or self.panel):
            raise SchemaError("a model must support single series, a panel, or both")
        return self

    def supports(self, *, is_panel: bool) -> bool:
        return self.panel if is_panel else self.single


class TargetCapabilities(BaseModel):
    """How many targets at once."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    univariate: bool = True
    multivariate: bool = False

    @model_validator(mode="after")
    def _check_something_is_supported(self) -> Self:
        if not (self.univariate or self.multivariate):
            raise SchemaError("a model must support univariate targets, multivariate, or both")
        return self

    def supports(self, target_count: int) -> bool:
        if target_count < 1:
            raise SchemaError("a forecast needs at least one target")
        return self.univariate if target_count == 1 else self.multivariate


class FeatureCapabilities(BaseModel):
    """Which feature roles the model can consume.

    Named after the roles OpenForecast declares, not after any provider's
    covariate vocabulary; ``hist_exog_list`` and friends are a translation that
    happens inside an integration.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    observed: bool = False
    known: bool = False
    static: bool = False

    def supports(self, feature: FeatureSpec) -> bool:
        if feature.is_static:
            return self.static
        return self.observed if feature.is_observed else self.known

    def unsupported(self, features: Iterable[FeatureSpec]) -> tuple[str, ...]:
        """The names of the features this model cannot be given.

        Returned rather than raised: the caller reporting the mismatch knows
        which model and which dataset are involved, and can say so.
        """
        return tuple(feature.name for feature in features if not self.supports(feature))


class OutputCapabilities(BaseModel):
    """What the model can produce."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    point: bool = True
    quantiles: bool = False
    samples: bool = False

    @model_validator(mode="after")
    def _check_something_is_produced(self) -> Self:
        if not (self.point or self.quantiles or self.samples):
            raise SchemaError("a model must produce point forecasts, quantiles or samples")
        return self


class MissingValueSupport(StrEnum):
    #: The model consumes gaps and NaNs as they are.
    NATIVE = "native"
    #: The recipe must carry an explicit, recorded transform — an imputation
    #: step the user asked for, never one OpenForecast inserted.
    REQUIRES_TRANSFORM = "requires_transform"
    #: The model cannot be given data with missing values at all.
    UNSUPPORTED = "unsupported"


class ModelCapabilities(BaseModel):
    """The full capability declaration of one model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    instances: InstanceCapabilities = InstanceCapabilities()
    targets: TargetCapabilities = TargetCapabilities()
    features: FeatureCapabilities = FeatureCapabilities()
    outputs: OutputCapabilities = OutputCapabilities()
    #: Point-in-time data is full of real missing values — a feature that had not
    #: been published at an origin. What a model does about them is part of its
    #: contract, because the alternative is imputing them silently.
    missing_values: MissingValueSupport = MissingValueSupport.UNSUPPORTED

    @property
    def tolerates_missing_values(self) -> bool:
        return self.missing_values is MissingValueSupport.NATIVE

    @property
    def requires_missing_value_transform(self) -> bool:
        return self.missing_values is MissingValueSupport.REQUIRES_TRANSFORM
