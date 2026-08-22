"""What a manifest has to be able to say about a fit that already happened.

The manifest is read instead of the provider directory, so every question the
engine asks of a fitted model has to be answerable here: which view it consumed,
what horizon it was bound to, whether its origins were real vintages, and
whether anything touched the missing values on the way in.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import openforecast as of
from openforecast import ArtifactError, ModelRefError, SchemaError
from openforecast.artifacts import (
    ModelArtifact,
    ModelManifest,
    TrainedSchema,
    TrainingRecord,
    content_hash,
)
from openforecast.protocol import PROTOCOL_VERSION
from openforecast.recipes import ColumnSet, ImputeMethod, declared_transforms
from openforecast.views import OriginFidelity, SourceKind, ViewKind
from tests import artifacts

# -- the training record ----------------------------------------------------


def test_the_record_is_read_off_the_view_that_was_materialized() -> None:
    view = artifacts.sequence_view()
    record = TrainingRecord.of_view(view, origins=of.AllOrigins(), seed=42)

    assert record.view is ViewKind.SEQUENCES
    assert record.context == artifacts.CONTEXT
    assert record.horizon == artifacts.HORIZON
    assert record.samples == len(view.sample_ids)
    assert record.source is SourceKind.TIME_SERIES
    assert record.seed == 42


def test_event_time_data_records_simulated_origins() -> None:
    """Windows cut out of one freshest series are not the past as it was."""
    record = TrainingRecord.of_view(artifacts.sequence_view())
    assert record.origin_fidelity is OriginFidelity.SIMULATED
    assert not record.is_observed


def test_point_in_time_data_records_observed_origins() -> None:
    record = TrainingRecord.of_view(artifacts.sequence_view(artifacts.dataset()))
    assert record.origin_fidelity is OriginFidelity.OBSERVED
    assert record.is_observed


def test_a_series_fit_records_neither_context_nor_horizon() -> None:
    view = artifacts.series_view()
    record = TrainingRecord.of_view(view)

    assert record.view is ViewKind.SERIES
    assert record.context is None
    assert record.horizon is None
    assert record.samples == len(view.series_ids)


def test_a_series_fit_serves_any_horizon() -> None:
    """A local forecaster is asked for a horizon at inference, so any is fine."""
    record = TrainingRecord.of_view(artifacts.series_view())
    assert record.serves_horizon(1)
    assert record.serves_horizon(1000)


def test_a_bound_horizon_serves_only_itself() -> None:
    record = TrainingRecord.of_view(artifacts.sequence_view())
    assert record.serves_horizon(artifacts.HORIZON)
    assert not record.serves_horizon(artifacts.HORIZON + 1)


def test_a_tabular_fit_records_a_horizon_and_no_context() -> None:
    view = artifacts.tabular_view()
    record = TrainingRecord.of_view(view)

    assert record.view is ViewKind.TABULAR
    assert record.context is None
    assert record.horizon == artifacts.HORIZON
    assert record.samples == view.num_rows


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"view": ViewKind.SERIES, "horizon": 24}, "binds neither"),
        ({"view": ViewKind.SEQUENCES, "horizon": 24}, "context length"),
        ({"view": ViewKind.SEQUENCES, "context": 168}, "bounds its samples"),
        ({"view": ViewKind.TABULAR, "horizon": 24, "context": 168}, "no context length"),
        ({"view": ViewKind.FORECAST, "horizon": 24}, "not from a forecast view"),
    ],
)
def test_a_record_cannot_describe_a_view_that_binds_otherwise(
    fields: dict[str, object], message: str
) -> None:
    with pytest.raises(SchemaError, match=message):
        TrainingRecord(
            source=SourceKind.TIME_SERIES,
            origin_fidelity=OriginFidelity.SIMULATED,
            samples=10,
            **fields,  # pyright: ignore[reportArgumentType]
        )


# -- the trained schema -----------------------------------------------------


def test_the_trained_schema_is_the_view_schema_without_the_layout() -> None:
    view = artifacts.sequence_view()
    schema = TrainedSchema.of_view(view.schema)

    assert schema.targets == view.schema.targets
    assert schema.instance_keys == view.schema.instance_keys
    assert schema.frequency == view.schema.frequency
    assert schema.feature_names == view.schema.feature_names
    assert schema.has_known_features
    assert schema.is_panel


# -- the manifest -----------------------------------------------------------


def test_a_fit_produces_a_pinned_local_reference() -> None:
    artifact = artifacts.artifact(name="de-price")

    assert artifact.ref.namespace == "local"
    assert artifact.ref.name == "de-price"
    assert artifact.ref.revision == artifact.artifact_id
    assert str(artifact.manifest.alias) == "local/de-price"


def test_the_manifest_records_what_this_build_can_read_back() -> None:
    manifest = artifacts.artifact().manifest

    assert manifest.protocol_version == PROTOCOL_VERSION
    assert manifest.openforecast_version == of.__version__
    assert manifest.provider == "nixtla"
    assert manifest.provider_version == "0.1.0"
    assert str(manifest.source_model) == "nixtla/nhits"


def test_the_manifest_records_the_origin_selection_that_was_asked_for() -> None:
    """The selection is the request; the fidelity is what the data could offer."""
    plan = of.FitPlan(
        origins=of.AllOrigins(stride=2), window=of.WindowPlan(context=artifacts.CONTEXT)
    )
    manifest = artifacts.artifact(plan=plan).manifest
    training = manifest.training
    assert training is not None

    assert training.origins == of.AllOrigins(stride=2)
    assert training.seed is None


@pytest.mark.parametrize("source", ["local/de-price", "nixtla/nhits@01K5Z6QK3M9TQK1W2E3R4T5Y6U"])
def test_the_source_model_is_a_model_not_an_artifact(source: str) -> None:
    with pytest.raises(ArtifactError):
        artifacts.artifact(source_model=source)


def test_an_artifact_id_that_was_not_generated_is_refused() -> None:
    with pytest.raises(ArtifactError, match="not an artifact id"):
        artifacts.artifact(artifact_id="de-price-v2")


def test_a_name_that_is_not_a_reference_segment_is_refused() -> None:
    """A name is half of a reference and becomes an alias filename.

    Validated through ``ModelRef``, so the answer to "what may an artifact be
    called" is the same as the answer to "what may a model be called" — and a
    name that could climb out of the aliases directory is not one of them.
    """
    with pytest.raises(ModelRefError):
        artifacts.artifact(name="../escape")


# -- missing values ---------------------------------------------------------


def test_missing_value_transforms_are_lifted_out_of_the_recipe() -> None:
    """Reading the manifest answers "was missingness altered", without a tree walk."""
    recipe = of.Pipeline(
        steps=(
            of.MissingIndicator(columns=ColumnSet.FEATURES),
            of.Impute(columns=ColumnSet.FEATURES, method=ImputeMethod.MEDIAN),
            of.StandardScaler(columns=ColumnSet.TARGETS),
            of.Model("nixtla/nhits"),
        )
    )
    manifest = artifacts.artifact(recipe=recipe).manifest

    assert [type(transform) for transform in manifest.missing_value_transforms] == [
        of.MissingIndicator,
        of.Impute,
    ]
    assert manifest.imputes_missing_values


def test_a_recipe_that_touches_nothing_records_nothing() -> None:
    manifest = artifacts.artifact(recipe=of.Model("nixtla/nhits")).manifest
    assert manifest.missing_value_transforms == ()
    assert not manifest.imputes_missing_values


def test_transforms_are_collected_through_ensembles_of_pipelines() -> None:
    recipe = of.Ensemble(
        models=(
            of.Pipeline(
                steps=(
                    of.Impute(columns=ColumnSet.FEATURES, method=ImputeMethod.ZERO),
                    of.Model("a/one"),
                )
            ),
            of.Pipeline(steps=(of.StandardScaler(columns=ColumnSet.TARGETS), of.Model("a/two"))),
        )
    )
    assert [type(transform) for transform in declared_transforms(recipe)] == [
        of.Impute,
        of.StandardScaler,
    ]


# -- integrity --------------------------------------------------------------


def test_an_artifact_round_trips_through_json() -> None:
    artifact = artifacts.artifact()
    assert ModelArtifact.model_validate(artifact.model_dump(mode="json")) == artifact


def test_the_recipe_has_to_hash_to_what_the_manifest_records() -> None:
    artifact = artifacts.artifact()
    with pytest.raises(ArtifactError, match="does not hash"):
        ModelArtifact(
            manifest=artifact.manifest,
            recipe=of.Model("nixtla/autoarima"),
            training_schema=artifact.training_schema,
        )


def test_the_training_schema_has_to_hash_to_what_the_manifest_records() -> None:
    artifact = artifacts.artifact()
    edited = {**artifact.training_schema, "horizon": 999}
    with pytest.raises(ArtifactError, match="does not hash"):
        ModelArtifact(manifest=artifact.manifest, recipe=artifact.recipe, training_schema=edited)


def test_the_hash_does_not_depend_on_how_the_payload_was_built() -> None:
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})
    assert content_hash({"a": 1}) != content_hash({"a": 2})


def test_the_manifest_is_frozen() -> None:
    manifest: ModelManifest = artifacts.artifact().manifest
    with pytest.raises(ValidationError):
        manifest.name = "something-else"  # pyright: ignore[reportAttributeAccessIssue]
