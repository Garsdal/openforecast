"""``ModelLifecycle``: whether a model has to be fitted, and whether it can be.

The two questions are separate. A statistical forecaster must be fitted and can
be. A pretrained foundation model may be usable zero-shot and fine-tunable, or
usable zero-shot and frozen. Collapsing them into one flag would make the third
combination inexpressible.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from openforecast.errors import SchemaError

__all__ = ["ModelLifecycle"]


class ModelLifecycle(BaseModel):
    """What has to happen to a model before it can forecast."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Forecasting without a fitted artifact is an error rather than zero-shot use.
    requires_fit: bool
    #: ``of.fit`` accepts this model at all.
    supports_fit: bool
    #: A fitted artifact can be advanced with new data instead of refitted.
    supports_update: bool = False

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.requires_fit and not self.supports_fit:
            raise SchemaError(
                "a model that requires fitting must support fitting; "
                "as declared it could never be used"
            )
        if self.supports_update and not self.supports_fit:
            raise SchemaError(
                "a model that cannot be fitted has no artifact to update; "
                "declare supports_fit to allow updates"
            )
        return self

    @classmethod
    def trainable(cls, *, supports_update: bool = False) -> ModelLifecycle:
        """Must be fitted before it forecasts — the ordinary case."""
        return cls(requires_fit=True, supports_fit=True, supports_update=supports_update)

    @classmethod
    def pretrained(cls, *, supports_fit: bool = False) -> ModelLifecycle:
        """Usable zero-shot; ``supports_fit`` says whether it can also be tuned."""
        return cls(requires_fit=False, supports_fit=supports_fit)

    @property
    def is_zero_shot(self) -> bool:
        return not self.requires_fit
