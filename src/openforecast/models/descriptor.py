"""``ModelDescriptor``: everything OpenForecast knows about a model it can name.

A descriptor is what a provider advertises and what the engine plans against. It
has to be complete enough that ``fit()`` can materialize data without asking the
provider anything first — which view, at how many origins, with which features,
and what to do about missing values are all answered here.

It describes a model, not a fitted one. Its reference is therefore never pinned
to a revision: ``nixtla/nhits`` is a model, ``local/de-price@01K...`` is an
artifact, and Step 7's registry is what turns the second into something you can
forecast with.

Since Step 23 the training contract is optional, because the second model
lifecycle has no training unit to describe. A pretrained foundation model
forecasts from its reference and is never handed a fit view, so ``training`` is
``None`` rather than a contract nothing will read — inventing one would put a
view kind and an origin scope in the catalog that no code path can honour.
Which of the two a descriptor is is not a second declaration: it follows from
``lifecycle.supports_fit``, and the validator below refuses the two ways of
disagreeing with it.
"""

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, model_validator

from openforecast.errors import SchemaError
from openforecast.models.capabilities import ModelCapabilities
from openforecast.models.contract import TrainingContract
from openforecast.models.lifecycle import ModelLifecycle
from openforecast.models.ref import ModelRef

__all__ = ["ModelDescriptor"]


class ModelDescriptor(BaseModel):
    """One model, as the catalog and the engine see it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Accepts a plain ``"nixtla/nhits"``; stored parsed.
    ref: ModelRef
    #: The provider that executes it, which is also the reference's namespace.
    provider: str
    display_name: str

    lifecycle: ModelLifecycle
    #: What the model learns from. ``None`` for a pretrained model that cannot
    #: be fitted, which has no training unit to describe.
    training: TrainingContract | None = None
    capabilities: ModelCapabilities = ModelCapabilities()

    #: JSON Schema for the provider-specific ``params`` of ``of.Model(...)``.
    #: Opaque to OpenForecast: anything OpenForecast owns — context length,
    #: horizon, seeds — is expressed in the recipe and compiled by the provider,
    #: not passed through here.
    parameters_schema: dict[str, Any] = {}

    @model_validator(mode="after")
    def _check_identity(self) -> Self:
        if self.ref.is_pinned:
            raise SchemaError(
                f"{self.ref} pins a revision; a descriptor describes a model, and a "
                f"revision names one fitted artifact of it"
            )
        if self.ref.namespace != self.provider:
            raise SchemaError(
                f"{self.ref} is namespaced {self.ref.namespace!r} but is advertised by "
                f"provider {self.provider!r}; a provider may only advertise its own models"
            )
        if not self.display_name.strip():
            raise SchemaError(f"{self.ref} must have a display name")
        if self.parameters_schema and self.parameters_schema.get("type") != "object":
            raise SchemaError(
                f"the parameter schema of {self.ref} must be a JSON Schema object of "
                f"type 'object'; model parameters are always a mapping"
            )
        self._check_training()
        return self

    def _check_training(self) -> None:
        """A training contract exists exactly when the model can be fitted.

        Both directions are refused. A fittable model without one cannot be
        planned against at all — ``fit()`` reads the contract to decide what to
        materialize. A contract on a model that cannot be fitted is the invented
        one Step 23 forbids: nothing would ever read it, so nothing would ever
        find out it was wrong.
        """
        if self.lifecycle.supports_fit and self.training is None:
            raise SchemaError(
                f"{self.ref} supports fitting and declares no training contract; a fit is "
                f"planned from the contract, so there would be nothing to materialize"
            )
        if not self.lifecycle.supports_fit and self.training is not None:
            raise SchemaError(
                f"{self.ref} cannot be fitted and declares a {self.training.view} training "
                f"contract; nothing would ever read it, so declare training=None"
            )

    @property
    def is_fittable(self) -> bool:
        """Whether ``of.fit`` accepts this model — and whether it has a contract."""
        return self.lifecycle.supports_fit

    @property
    def required_training(self) -> TrainingContract:
        """The training contract, for the code paths that only run for a fit.

        Raised on rather than returned as ``None``: reaching here for a model
        that cannot be fitted means a fit was planned for one, which is a bug
        in the caller rather than a missing declaration.
        """
        if self.training is None:
            raise SchemaError(
                f"{self.ref} is used zero-shot and has no training contract; forecast with "
                f"the reference directly"
            )
        return self.training

    def __str__(self) -> str:
        return str(self.ref)

    def __repr__(self) -> str:
        """``of.models.list()`` is a discovery call, so it has to be readable.

        The generated pydantic repr spells out every capability of every model,
        which turns listing a catalog into several screens of text. What a
        listing is for is the references; the fields are what ``of.models.get``
        is for.
        """
        return f"ModelDescriptor({self.ref})"
