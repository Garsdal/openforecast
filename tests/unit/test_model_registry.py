"""What one reference means right now, across the catalog and the store.

Step 5's rule is that a model reference is a name, not a state. This is where the
state comes from, and the case worth being strict about is forecasting with a
string that names a model nobody fitted: it has to fail, because the alternative
is a number that looks like a forecast from a model the caller never trained.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openforecast import ModelError, ModelRequiresFit, UnknownModelError
from openforecast.artifacts import ArtifactStore, ModelHandle
from openforecast.models import (
    ModelCatalog,
    ModelDescriptor,
    ModelLifecycle,
    ModelRef,
    TrainingContract,
)
from openforecast.registry import ModelRegistry
from tests import artifacts

NHITS = ModelDescriptor(
    ref=ModelRef.parse("nixtla/nhits"),
    provider="nixtla",
    display_name="NHiTS",
    lifecycle=ModelLifecycle.trainable(),
    training=TrainingContract.sequences(),
)

FOUNDATION = ModelDescriptor(
    ref=ModelRef.parse("nixtla/timegpt"),
    provider="nixtla",
    display_name="TimeGPT",
    lifecycle=ModelLifecycle.pretrained(),
)

TUNABLE = ModelDescriptor(
    ref=ModelRef.parse("nixtla/timegpt-tunable"),
    provider="nixtla",
    display_name="TimeGPT (tunable)",
    lifecycle=ModelLifecycle.pretrained(supports_fit=True),
    training=TrainingContract.sequences(),
)


@pytest.fixture
def registry(tmp_path: Path) -> ModelRegistry:
    return ModelRegistry(
        catalog=ModelCatalog([NHITS, FOUNDATION, TUNABLE]),
        store=ArtifactStore(tmp_path / "openforecast"),
    )


def fit(registry: ModelRegistry, **options: object) -> str:
    artifact = artifacts.artifact(**options)  # pyright: ignore[reportArgumentType]
    with registry.store.stage(artifact):
        pass
    return artifact.artifact_id


# -- forecasting ------------------------------------------------------------


def test_a_local_reference_resolves_to_the_artifact_it_names(registry: ModelRegistry) -> None:
    revision = fit(registry)
    resolved = registry.resolve("local/de-price")

    assert isinstance(resolved, ModelHandle)
    assert resolved.artifact_id == revision
    assert resolved.manifest.source_model == NHITS.ref


def test_forecasting_with_an_unfitted_model_reference_is_refused(
    registry: ModelRegistry,
) -> None:
    with pytest.raises(ModelRequiresFit, match="has to be fitted"):
        registry.resolve("nixtla/nhits")


def test_a_zero_shot_model_resolves_to_its_descriptor(registry: ModelRegistry) -> None:
    """Zero-shot use is something a model declares, not something assumed."""
    assert registry.resolve("nixtla/timegpt") == FOUNDATION


def test_an_unknown_reference_is_unknown(registry: ModelRegistry) -> None:
    with pytest.raises(UnknownModelError):
        registry.resolve("nixtla/nothing")
    with pytest.raises(UnknownModelError):
        registry.resolve("local/nothing")


def test_membership_is_the_question_resolve_answers(registry: ModelRegistry) -> None:
    fit(registry)
    assert "local/de-price" in registry
    assert "nixtla/timegpt" in registry
    # Known, but not something that can forecast as it stands.
    assert "nixtla/nhits" not in registry


# -- fitting ----------------------------------------------------------------


def test_a_model_reference_resolves_to_the_descriptor_to_plan_against(
    registry: ModelRegistry,
) -> None:
    descriptor = registry.for_fit("nixtla/nhits")
    assert descriptor == NHITS
    assert descriptor.required_training.view == NHITS.required_training.view


def test_an_artifact_is_not_something_to_fit(registry: ModelRegistry) -> None:
    """Refitting means fitting the model it came from, and the manifest names it."""
    fit(registry)
    with pytest.raises(ModelError, match="fitted from nixtla/nhits"):
        registry.for_fit("local/de-price")


def test_a_model_that_cannot_be_fitted_says_so(registry: ModelRegistry) -> None:
    with pytest.raises(ModelError, match="cannot be fitted"):
        registry.for_fit("nixtla/timegpt")


def test_a_pretrained_model_that_can_be_tuned_may_be_fitted(registry: ModelRegistry) -> None:
    assert registry.for_fit("nixtla/timegpt-tunable") == TUNABLE


# -- wiring -----------------------------------------------------------------


def test_the_artifact_accessor_is_about_artifacts_specifically(
    registry: ModelRegistry,
) -> None:
    revision = fit(registry)
    assert registry.artifact(f"local/de-price@{revision}").artifact_id == revision
    with pytest.raises(UnknownModelError):
        registry.artifact("nixtla/nhits")


def test_a_registry_touches_no_filesystem_until_it_is_used(tmp_path: Path) -> None:
    root = tmp_path / "unused"
    registry = ModelRegistry(catalog=ModelCatalog(), store=ArtifactStore(root))
    assert registry.catalog.refs() == ()
    assert not root.exists()
