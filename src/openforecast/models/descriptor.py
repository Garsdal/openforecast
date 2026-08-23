"""``ModelDescriptor``: everything OpenForecast knows about a model it can name.

A descriptor is what a provider advertises and what the engine plans against. It
has to be complete enough that ``fit()`` can materialize data without asking the
provider anything first — which view, at how many origins, with which features,
and what to do about missing values are all answered here.

It describes a model, not a fitted one. Its reference is therefore never pinned
to a revision: ``nixtla/nhits`` is a model, ``local/de-price@01K...`` is an
artifact, and Step 7's registry is what turns the second into something you can
forecast with.
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
    training: TrainingContract
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
        return self

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
