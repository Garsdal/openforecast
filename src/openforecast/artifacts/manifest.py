"""``ModelManifest``: what a fitted artifact is, in terms nobody has to guess.

The provider directory inside an artifact is opaque — a pickled StatsForecast
object, a Lightning checkpoint, whatever the library persists. Everything
OpenForecast needs in order to decide whether that directory can answer a
forecast request therefore has to be stated outside it, in the manifest:

```json
{
  "training": {
    "view": "sequences",
    "origin_fidelity": "observed",
    "context": 168,
    "horizon": 72,
    "samples": 8832
  }
}
```

Two fields carry most of the weight. ``origin_fidelity`` says whether the model
learned from real vintages or from windows cut out of one freshest series, which
is the difference between a model that saw the past as it was and one that was
told the past was cleaner than it was — a difference no metric recovers later.
``horizon`` says how far each training sample reached, and ``horizon_bound``
whether the model can be asked about any other — so a request a model cannot
serve is refused instead of silently truncated, and one it can serve is not
refused for a horizon that was only ever the shape of its training data. The two
are separate because they are separate facts: ``darts/tide`` bakes 72 into its
architecture, while ``sktime/pooled-trees`` learns one step from the same samples
and rolls, and the sample shape does not say which of those a model is.

A recipe that is not one model — a pipeline, an ensemble — is executed by
OpenForecast rather than by a provider, and its members may consume different
views. Such an artifact records one training record per fitted leaf in
``members`` and no single ``training``, because there is no one materialization
it could honestly describe.

The recipe and the full training-view schema are their own files beside the
manifest, and the manifest carries their hashes. An artifact is immutable, so a
hash that no longer matches means the directory was edited underneath us, and
reading on would produce forecasts from a model the manifest no longer
describes.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from hashlib import blake2b
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from openforecast.data.features import FeatureSpec
from openforecast.data.frequency import Frequency
from openforecast.errors import ArtifactError, SchemaError
from openforecast.models.contract import TrainingContract
from openforecast.models.ref import ModelRef
from openforecast.protocol.version import PROTOCOL_VERSION
from openforecast.protocol.vocabulary import ViewKind
from openforecast.recipes.nodes import Recipe, declared_transforms
from openforecast.recipes.transforms import Impute, MissingIndicator
from openforecast.tasks.origins import AllOrigins, OriginSelection
from openforecast.views.base import FeatureGroups, ViewSchema
from openforecast.views.planner import FitView
from openforecast.views.provenance import MATERIALIZER_VERSION, OriginFidelity, SourceKind
from openforecast.views.sequences import SequenceView
from openforecast.views.series import SeriesView

__all__ = [
    "COMPOSITE_PROVIDER",
    "LOCAL_NAMESPACE",
    "MissingValueTransform",
    "ModelManifest",
    "TrainedSchema",
    "TrainingRecord",
    "content_hash",
    "missing_value_transforms",
    "view_schema_payload",
]

#: The namespace every fitted artifact lives in. ``nixtla/nhits`` is a model;
#: ``local/de-price@01K...`` is something that was fitted here.
LOCAL_NAMESPACE = "local"

#: The provider recorded for an artifact OpenForecast executes itself. A
#: pipeline's scaling and an ensemble's combination are OpenForecast's own
#: semantics, so no library is named as having produced them.
COMPOSITE_PROVIDER = "openforecast"

#: The transforms that change what missing data means, and therefore have to be
#: legible in the manifest rather than only inside the recipe tree.
MissingValueTransform = MissingIndicator | Impute


def content_hash(payload: object) -> str:
    """A stable hash of anything JSON-serializable.

    Canonical JSON — sorted keys, no incidental whitespace — so that the same
    recipe hashes the same however it was constructed, and a reordered field is
    not mistaken for an edit.
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return blake2b(encoded, digest_size=16).hexdigest()


class TrainedSchema(FeatureGroups):
    """The data shape a fitted artifact expects to be given again.

    A projection of the training view's schema, without the view's own layout:
    what the forecast data has to declare for the artifact to be applicable at
    all. Kept in the manifest so that answering "can this model forecast this
    dataset" needs no provider and no Arrow file.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    frequency: Frequency
    targets: tuple[str, ...]
    instance_keys: tuple[str, ...] = ()

    @classmethod
    def of_view(cls, schema: ViewSchema) -> TrainedSchema:
        return cls(
            frequency=schema.frequency,
            targets=schema.targets,
            instance_keys=schema.instance_keys,
            features=schema.features,
        )

    @classmethod
    def merge(cls, schemas: Sequence[TrainedSchema]) -> TrainedSchema:
        """What a composite artifact needs to be given: the union of its members'.

        Every member was materialized from the same data, so the frequency, the
        targets and the instance keys agree by construction and disagreement is
        a bug rather than a case to reconcile. The features do not have to: a
        tabular member consumes no observed feature, so the artifact requires one
        exactly when some member of it does.
        """
        if not schemas:
            raise SchemaError("a composite artifact is fitted from at least one view")
        first = schemas[0]
        for other in schemas[1:]:
            if (first.frequency, first.targets, first.instance_keys) != (
                other.frequency,
                other.targets,
                other.instance_keys,
            ):
                raise SchemaError(
                    f"the members of this artifact were fitted on different data: "
                    f"{first.frequency}/{list(first.targets)}/{list(first.instance_keys)} "
                    f"and {other.frequency}/{list(other.targets)}/{list(other.instance_keys)}"
                )
        features: dict[str, FeatureSpec] = {}
        for schema in schemas:
            for feature in schema.features:
                features.setdefault(feature.name, feature)
        return cls(
            frequency=first.frequency,
            targets=first.targets,
            instance_keys=first.instance_keys,
            features=tuple(features.values()),
        )

    @property
    def is_panel(self) -> bool:
        return bool(self.instance_keys)


class TrainingRecord(BaseModel):
    """How the model was trained, as opposed to what it was trained from.

    Everything here is decided by the ``ViewPlanner`` rather than by the
    provider, which is why it can be recorded without asking one: the view, how
    much context and horizon each sample spanned, how many samples there were,
    from which origins, and whether those origins were real.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    view: ViewKind
    source: SourceKind
    origin_fidelity: OriginFidelity
    origins: OriginSelection = AllOrigins()

    #: Sized by the fit plan's window, and only for a view that has samples.
    context: int | None = Field(default=None, ge=1)
    #: How far each training sample reached past its origin. A property of the
    #: materialization, which is why every view that has samples records one.
    horizon: int | None = Field(default=None, ge=1)
    #: Whether that horizon is also the only one the model can answer. A property
    #: of the model's own contract rather than of the data, and conservative by
    #: default: a record that does not say is read as bound.
    horizon_bound: bool = True
    #: Training units: series, sequences or rows, depending on the view.
    samples: int = Field(ge=1)

    materializer_version: int = MATERIALIZER_VERSION
    #: ``None`` is an unseeded fit saying so, rather than claiming reproducibility.
    seed: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check_view_fields(self) -> Self:
        if self.view is ViewKind.FORECAST:
            raise SchemaError("an artifact is trained from a fit view, not from a forecast view")
        if self.view is ViewKind.SERIES and (self.context is not None or self.horizon is not None):
            raise SchemaError(
                "a series view binds neither a context length nor a horizon; one complete "
                "time series is the training unit"
            )
        if self.view is not ViewKind.SERIES and self.horizon is None:
            raise SchemaError(f"a {self.view} view bounds its samples by a horizon")
        if self.view is ViewKind.SEQUENCES and self.context is None:
            raise SchemaError(
                "a sequence view sizes its samples by a context length, so a model "
                "trained on one has to record it"
            )
        if self.view is ViewKind.TABULAR and self.context is not None:
            raise SchemaError(
                "a tabular view binds no context length; lagged features are declared on the recipe"
            )
        return self

    @classmethod
    def of_view(
        cls,
        view: FitView,
        *,
        contract: TrainingContract | None = None,
        origins: OriginSelection | None = None,
        seed: int | None = None,
    ) -> TrainingRecord:
        """Read the record straight off the view that was handed to the provider.

        Nothing is passed in that the view already knows, so a manifest cannot
        describe a fit that did not happen: the sample count, the context and
        horizon, and the fidelity of the origins all come from the materialized
        view rather than from what was requested.

        ``contract`` is the one thing the view cannot answer — whether the model
        is bound to the horizon its samples happen to span. Left out, the record
        says bound, which is the answer that refuses rather than the one that
        forecasts something nobody promised.
        """
        return cls(
            view=view.kind,
            source=view.provenance.source,
            origin_fidelity=view.provenance.origin_fidelity,
            origins=AllOrigins() if origins is None else origins,
            context=view.schema.context if isinstance(view, SequenceView) else None,
            horizon=None if isinstance(view, SeriesView) else view.schema.horizon,
            horizon_bound=contract is None or contract.horizon_bound_at_fit,
            samples=_sample_count(view),
            materializer_version=view.provenance.materializer_version,
            seed=seed,
        )

    @property
    def is_observed(self) -> bool:
        """Whether the model learned from real vintages rather than simulated ones."""
        return self.origin_fidelity is OriginFidelity.OBSERVED

    def serves_horizon(self, horizon: int) -> bool:
        """Whether a forecast at ``horizon`` is one this artifact can produce.

        A model that bound no horizon at fit time serves any; one that bound a
        horizon serves exactly that one. Which of the two a model is, is its
        contract's business — this only knows what the contract said when the
        record was written.
        """
        return not self.horizon_bound or self.horizon is None or self.horizon == horizon


class ModelManifest(BaseModel):
    """The complete, immutable description of one fitted artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_id: str
    #: The alias name this revision was fitted under: ``local/de-price``.
    name: str
    #: The model that was fitted — never local, never pinned. ``None`` for a
    #: composite recipe, which names several: the recipe is the answer there.
    source_model: ModelRef | None = None
    provider: str
    provider_version: str
    openforecast_version: str
    protocol_version: int = PROTOCOL_VERSION

    #: Of ``recipe.json`` and ``schema.json``, which hold the recipe that was
    #: fitted and the training view's own schema.
    recipe_hash: str
    training_schema_hash: str

    #: How the one model this artifact holds was trained. ``None`` for a
    #: composite recipe, whose leaves are recorded in :attr:`members` instead.
    training: TrainingRecord | None = None
    #: One record per fitted leaf of a composite recipe, in recipe order. An
    #: ensemble's members may consume different views, so there is no single
    #: training record a composite artifact could honestly carry.
    members: tuple[TrainingRecord, ...] = ()
    data_schema: TrainedSchema
    #: Lifted out of the recipe: whether, and how, missingness was altered.
    missing_value_transforms: tuple[MissingValueTransform, ...] = ()

    @model_validator(mode="after")
    def _check_identity(self) -> Self:
        # Imported here: identity/ has no business importing the manifest back.
        from openforecast.artifacts.identity import is_artifact_id

        if not is_artifact_id(self.artifact_id):
            raise ArtifactError(f"{self.artifact_id!r} is not an artifact id")
        if self.source_model is not None:
            self._check_source_model(self.source_model)
        if (self.training is None) == (not self.members):
            raise ArtifactError(
                f"{self.artifact_id} must record either how its one model was trained or "
                f"one record per leaf of a composite recipe, and not both"
            )
        if (self.source_model is None) != (self.training is None):
            raise ArtifactError(
                f"{self.artifact_id} names a source model if and only if it holds one "
                f"model; a composite recipe names several, and the recipe records them"
            )
        # Rejects a name that is not one path segment, among other things.
        ModelRef(namespace=LOCAL_NAMESPACE, name=self.name)
        return self

    def _check_source_model(self, source_model: ModelRef) -> None:
        if source_model.is_pinned:
            raise ArtifactError(
                f"{source_model} pins a revision; the source model is the model that "
                f"was fitted, and the revision it produced is {self.artifact_id}"
            )
        if source_model.namespace == LOCAL_NAMESPACE:
            raise ArtifactError(
                f"{source_model} is itself an artifact reference; the source model "
                f"names what a provider executes, as in 'nixtla/nhits'"
            )

    @property
    def training_records(self) -> tuple[TrainingRecord, ...]:
        """Every fitted leaf, however many this artifact holds.

        One for a leaf model, one per member for a composite. What the engine
        walks when it has to ask something of everything that was fitted.
        """
        return self.members if self.training is None else (self.training,)

    @property
    def is_composite(self) -> bool:
        """Whether OpenForecast, rather than one provider, executes this artifact."""
        return self.training is None

    def serves_horizon(self, horizon: int) -> bool:
        """Whether every model in this artifact can produce ``horizon`` steps."""
        return all(record.serves_horizon(horizon) for record in self.training_records)

    @property
    def ref(self) -> ModelRef:
        """``local/de-price@01K...`` — this exact immutable revision."""
        return ModelRef(namespace=LOCAL_NAMESPACE, name=self.name, revision=self.artifact_id)

    @property
    def alias(self) -> ModelRef:
        """``local/de-price`` — the mutable reference that may point here."""
        return ModelRef(namespace=LOCAL_NAMESPACE, name=self.name)

    @property
    def imputes_missing_values(self) -> bool:
        """Whether anything in the recipe filled a missing value."""
        return any(isinstance(transform, Impute) for transform in self.missing_value_transforms)

    def __str__(self) -> str:
        return str(self.ref)


def missing_value_transforms(recipe: Recipe) -> tuple[MissingValueTransform, ...]:
    """The transforms in ``recipe`` that change what a missing value means."""
    return tuple(
        transform
        for transform in declared_transforms(recipe)
        if isinstance(transform, MissingIndicator | Impute)
    )


def view_schema_payload(view: FitView) -> dict[str, Any]:
    """The training view's own schema, as it is written to ``schema.json``."""
    return view.schema.model_dump(mode="json")


def _sample_count(view: FitView) -> int:
    """The training units the view holds, whatever the view calls them."""
    if isinstance(view, SeriesView):
        return len(view.series_ids)
    if isinstance(view, SequenceView):
        return len(view.sample_ids)
    return view.num_rows
