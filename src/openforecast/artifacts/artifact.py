"""``ModelArtifact``: the whole of what one fit produced.

```text
<artifact-id>/
    manifest.json     what this is
    recipe.json       what was fitted
    schema.json       the training view's schema
    provider/         opaque
```

Three files rather than one, for two different reasons. The recipe is the thing
a user recognizes — it is what they wrote — and a recipe that has to be dug out
of a manifest is a recipe nobody reads. The view schema is the input to the
provider's own conversion, so keeping it whole means a provider can be handed
the exact schema it was fitted against instead of a summary of it. Both are
hashed into the manifest, so the split costs nothing in integrity.

Constructing one describes a fit; it does not perform it. The provider directory
is filled by the provider, between the moment the artifact is described and the
moment it is published.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, model_validator

from openforecast.artifacts.identity import new_artifact_id
from openforecast.artifacts.manifest import (
    COMPOSITE_PROVIDER,
    ModelManifest,
    TrainedSchema,
    TrainingRecord,
    content_hash,
    missing_value_transforms,
    view_schema_payload,
)
from openforecast.errors import ArtifactError
from openforecast.models.ref import ModelRef
from openforecast.recipes.nodes import Recipe
from openforecast.tasks.plan import FitPlan
from openforecast.views.planner import FitView

__all__ = ["ModelArtifact"]


class ModelArtifact(BaseModel):
    """A manifest, the recipe it describes, and the schema it was fitted against."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest: ModelManifest
    recipe: Recipe
    #: The training view's own schema, kept verbatim so that a provider can be
    #: handed exactly what it was fitted against rather than a projection of it.
    training_schema: dict[str, Any]

    @model_validator(mode="after")
    def _check_hashes(self) -> Self:
        """The manifest has to describe the two files it travels with.

        Checked on construction as well as on read, so an artifact assembled in
        memory cannot be published with hashes that were never true.
        """
        expected = content_hash(self.recipe.model_dump(mode="json"))
        if self.manifest.recipe_hash != expected:
            raise ArtifactError(
                f"the recipe of {self.manifest.artifact_id} does not hash to what its "
                f"manifest records; the artifact was modified after it was written"
            )
        if self.manifest.training_schema_hash != content_hash(self.training_schema):
            raise ArtifactError(
                f"the training schema of {self.manifest.artifact_id} does not hash to what "
                f"its manifest records; the artifact was modified after it was written"
            )
        return self

    @classmethod
    def of_fit(
        cls,
        *,
        name: str,
        source_model: ModelRef | str,
        recipe: Recipe,
        view: FitView,
        provider: str,
        provider_version: str,
        openforecast_version: str,
        plan: FitPlan | None = None,
        artifact_id: str | None = None,
    ) -> ModelArtifact:
        """Describe the artifact a fit of ``recipe`` on ``view`` produces.

        The view is the source of every training fact — sample count, context,
        horizon, origin fidelity — so a manifest cannot claim a fit that was not
        the one materialized. The plan contributes only what it alone knows: the
        origin selection that was asked for, and the seed.
        """
        plan = FitPlan() if plan is None else plan
        training_schema = view_schema_payload(view)
        manifest = ModelManifest(
            artifact_id=new_artifact_id() if artifact_id is None else artifact_id,
            name=name,
            source_model=ModelRef.parse(source_model),
            provider=provider,
            provider_version=provider_version,
            openforecast_version=openforecast_version,
            recipe_hash=content_hash(recipe.model_dump(mode="json")),
            training_schema_hash=content_hash(training_schema),
            training=TrainingRecord.of_view(view, origins=plan.origins, seed=plan.seed),
            data_schema=TrainedSchema.of_view(view.schema),
            missing_value_transforms=missing_value_transforms(recipe),
        )
        return cls(manifest=manifest, recipe=recipe, training_schema=training_schema)

    @classmethod
    def of_composite(
        cls,
        *,
        name: str,
        recipe: Recipe,
        views: Sequence[FitView],
        data_schema: TrainedSchema,
        openforecast_version: str,
        plan: FitPlan | None = None,
        artifact_id: str | None = None,
    ) -> ModelArtifact:
        """Describe the artifact a fit of a pipeline or an ensemble produces.

        A composite is executed by OpenForecast rather than by a provider, so it
        names none — and its leaves may consume different views, so it records
        one training record per leaf instead of one for itself. ``schema.json``
        holds the data schema every leaf was materialized from, which is the only
        schema the artifact as a whole was fitted against.
        """
        plan = FitPlan() if plan is None else plan
        if not views:
            raise ArtifactError("a composite artifact holds at least one fitted model")
        training_schema = data_schema.model_dump(mode="json")
        manifest = ModelManifest(
            artifact_id=new_artifact_id() if artifact_id is None else artifact_id,
            name=name,
            provider=COMPOSITE_PROVIDER,
            provider_version=openforecast_version,
            openforecast_version=openforecast_version,
            recipe_hash=content_hash(recipe.model_dump(mode="json")),
            training_schema_hash=content_hash(training_schema),
            members=tuple(
                TrainingRecord.of_view(view, origins=plan.origins, seed=plan.seed) for view in views
            ),
            data_schema=data_schema,
            missing_value_transforms=missing_value_transforms(recipe),
        )
        return cls(manifest=manifest, recipe=recipe, training_schema=training_schema)

    @property
    def artifact_id(self) -> str:
        return self.manifest.artifact_id

    @property
    def ref(self) -> ModelRef:
        """``local/de-price@01K...``."""
        return self.manifest.ref

    def __str__(self) -> str:
        return str(self.ref)
