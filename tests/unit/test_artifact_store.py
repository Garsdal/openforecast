"""The artifact lifecycle: created, resolved, aliased, reloaded, deleted.

Two properties are what the rest of the design leans on. A published revision is
immutable, so resolving the same pinned reference twice gives the same model
forever. And nothing is published until the fit succeeded, so an artifact that is
resolvable is an artifact whose provider state is complete.

No provider appears anywhere in here, which is the point: the store is what makes
a fitted model addressable, and it never opens the directory the provider wrote.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import openforecast as of
from openforecast import ArtifactError, UnknownModelError
from openforecast.artifacts import ArtifactStore, default_root
from openforecast.recipes import ColumnSet, ImputeMethod
from tests import artifacts

PROVIDER_STATE = "native model bytes"


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "openforecast")


def fit(store: ArtifactStore, **options: object) -> str:
    """Publish an artifact the way an engine would, provider payload included."""
    artifact = artifacts.artifact(**options)  # pyright: ignore[reportArgumentType]
    with store.stage(artifact) as staging:
        (staging.provider_path / "state.bin").write_text(PROVIDER_STATE, encoding="utf-8")
    return artifact.artifact_id


# -- creation ---------------------------------------------------------------


def test_a_fit_publishes_an_artifact_and_an_alias(store: ArtifactStore) -> None:
    artifact = artifacts.artifact(name="de-price")
    with store.stage(artifact) as staging:
        (staging.provider_path / "state.bin").write_text(PROVIDER_STATE, encoding="utf-8")

    handle = staging.handle
    assert str(handle.ref) == f"local/de-price@{artifact.artifact_id}"
    assert store.get("local/de-price") == handle
    assert store.get(handle.ref) == handle


def test_the_provider_directory_survives_publication_untouched(store: ArtifactStore) -> None:
    """Opaque means opaque: the store moves the bytes and never reads them."""
    revision = fit(store)
    handle = store.get(f"local/de-price@{revision}")
    assert (handle.provider_path / "state.bin").read_text(encoding="utf-8") == PROVIDER_STATE


def test_an_artifact_directory_holds_the_three_metadata_files(store: ArtifactStore) -> None:
    handle = store.get(f"local/de-price@{fit(store)}")
    assert sorted(path.name for path in handle.path.iterdir()) == [
        "manifest.json",
        "provider",
        "recipe.json",
        "schema.json",
    ]


def test_a_failed_fit_publishes_nothing(store: ArtifactStore) -> None:
    """A resolvable artifact with half-written provider state would forecast."""
    artifact = artifacts.artifact()
    with (
        pytest.raises(RuntimeError, match="the provider crashed"),
        store.stage(artifact) as staging,
    ):
        (staging.provider_path / "state.bin").write_text("partial", encoding="utf-8")
        raise RuntimeError("the provider crashed")

    assert store.list() == ()
    assert list(store.staging_root.iterdir()) == []
    with pytest.raises(UnknownModelError):
        store.get(artifact.ref)


def test_an_unpublished_staging_has_no_handle(store: ArtifactStore) -> None:
    artifact = artifacts.artifact()
    with (
        store.stage(artifact) as staging,
        pytest.raises(ArtifactError, match="not been published"),
    ):
        _ = staging.handle


def test_a_fit_can_be_published_without_moving_the_alias(store: ArtifactStore) -> None:
    """A candidate model is addressable before it is the one anything uses."""
    artifact = artifacts.artifact()
    with store.stage(artifact, alias=False):
        pass

    assert store.get(artifact.ref).artifact_id == artifact.artifact_id
    assert store.aliases() == ()
    with pytest.raises(UnknownModelError):
        store.get("local/de-price")


def test_the_same_revision_cannot_be_published_twice(store: ArtifactStore) -> None:
    revision = fit(store)
    with pytest.raises(ArtifactError, match="already published"):
        fit(store, artifact_id=revision)


def test_publishing_twice_is_refused(store: ArtifactStore) -> None:
    artifact = artifacts.artifact()
    with store.stage(artifact) as staging:
        staging.commit()

    assert staging.is_published
    with pytest.raises(ArtifactError, match="already been published"):
        staging.commit()


def test_a_revision_cannot_be_staged_twice_at_once(store: ArtifactStore) -> None:
    artifact = artifacts.artifact()
    with store.stage(artifact) as staging:
        assert staging.artifact == artifact
        assert staging.path.parent == store.staging_root
        with (
            pytest.raises(ArtifactError, match="already being written"),
            store.stage(artifact),
        ):
            pass  # pragma: no cover - staging the same id twice never opens


# -- the handle -------------------------------------------------------------


def test_a_handle_answers_from_the_manifest_alone(store: ArtifactStore) -> None:
    """The questions asked of a fitted model, none of which load a model."""
    revision = fit(store)
    handle = store.get("local/de-price")

    assert handle.name == "de-price"
    assert handle.artifact_id == revision
    assert handle.provider == "nixtla"
    assert handle.training == handle.manifest.training
    assert handle.data_schema == handle.manifest.data_schema
    assert handle.serves_horizon(artifacts.HORIZON)
    assert not handle.serves_horizon(artifacts.HORIZON + 1)
    assert str(handle) == f"local/de-price@{revision}"
    assert repr(handle).startswith("ModelHandle(local/de-price@")


# -- resolution -------------------------------------------------------------


def test_the_alias_follows_the_newest_fit_and_older_revisions_stay(
    store: ArtifactStore,
) -> None:
    first = fit(store, artifact_id=artifacts.artifact_id(1))
    second = fit(store, artifact_id=artifacts.artifact_id(2))

    assert store.get("local/de-price").artifact_id == second
    assert store.get(f"local/de-price@{first}").artifact_id == first
    assert store.revisions("de-price") == (first, second)


def test_revisions_are_listed_oldest_first(store: ArtifactStore) -> None:
    """Chronological because an artifact id sorts that way, not because of a field."""
    published = [fit(store, artifact_id=artifacts.artifact_id(step)) for step in (3, 1, 2)]
    assert [handle.artifact_id for handle in store.list()] == sorted(published)


def test_listing_can_be_narrowed_to_one_name(store: ArtifactStore) -> None:
    fit(store, name="de-price")
    fit(store, name="fr-load")
    assert [handle.name for handle in store.list(name="fr-load")] == ["fr-load"]
    assert len(store.list()) == 2


def test_a_reloaded_artifact_is_the_artifact_that_was_fitted(store: ArtifactStore) -> None:
    recipe = of.Pipeline(
        steps=(
            of.Impute(columns=ColumnSet.FEATURES, method=ImputeMethod.MEDIAN),
            of.Model("nixtla/nhits"),
        )
    )
    artifact = artifacts.artifact(recipe=recipe)
    with store.stage(artifact):
        pass

    assert store.read(artifact.ref) == artifact


def test_a_reference_in_another_namespace_is_not_an_artifact(store: ArtifactStore) -> None:
    with pytest.raises(UnknownModelError, match="fitted artifact"):
        store.get("nixtla/nhits")


def test_a_pinned_reference_has_to_pin_an_artifact_id(store: ArtifactStore) -> None:
    with pytest.raises(UnknownModelError, match="artifact id"):
        store.get("local/de-price@v2")


def test_a_revision_cannot_be_read_under_another_name(store: ArtifactStore) -> None:
    """``name@revision`` has to agree, or one reference would mean two models."""
    revision = fit(store, name="de-price")
    with pytest.raises(UnknownModelError, match="was fitted as"):
        store.get(f"local/fr-load@{revision}")


def test_an_unfitted_name_says_what_is_known(store: ArtifactStore) -> None:
    with pytest.raises(UnknownModelError, match="nothing has been fitted yet"):
        store.get("local/de-price")

    fit(store, name="fr-load")
    with pytest.raises(UnknownModelError, match="local/fr-load"):
        store.get("local/de-price")


def test_membership_does_not_raise(store: ArtifactStore) -> None:
    fit(store)
    assert "local/de-price" in store
    assert "local/nothing" not in store
    assert "nixtla/nhits" not in store


# -- aliases ----------------------------------------------------------------


def test_an_alias_can_be_pointed_back_at_an_older_revision(store: ArtifactStore) -> None:
    """A rollback moves the pointer; the revisions themselves never change."""
    first = fit(store, artifact_id=artifacts.artifact_id(1))
    fit(store, artifact_id=artifacts.artifact_id(2))

    store.set_alias("de-price", first)
    assert store.get("local/de-price").artifact_id == first


def test_an_alias_cannot_point_at_another_lineage(store: ArtifactStore) -> None:
    revision = fit(store, name="de-price")
    fit(store, name="fr-load")
    with pytest.raises(ArtifactError, match="names a lineage"):
        store.set_alias("fr-load", revision)


def test_an_alias_cannot_point_at_a_revision_that_does_not_exist(store: ArtifactStore) -> None:
    with pytest.raises(UnknownModelError, match="does not exist"):
        store.set_alias("de-price", artifacts.artifact_id(1))


def test_aliases_are_listed_as_references(store: ArtifactStore) -> None:
    fit(store, name="de-price")
    fit(store, name="fr-load")
    assert [str(ref) for ref in store.aliases()] == ["local/de-price", "local/fr-load"]


def test_an_alias_pointing_at_a_missing_revision_is_an_error_not_a_fallback(
    store: ArtifactStore,
) -> None:
    """Silently resolving to some other revision would forecast from the wrong model."""
    fit(store)
    (store.aliases_root / "de-price.json").write_text(
        json.dumps({"name": "de-price", "revision": artifacts.artifact_id(9)}), encoding="utf-8"
    )
    with pytest.raises(ArtifactError, match="no longer in the store"):
        store.get("local/de-price")


# -- deletion ---------------------------------------------------------------


def test_deleting_a_revision_moves_the_alias_to_the_newest_remaining(
    store: ArtifactStore,
) -> None:
    first = fit(store, artifact_id=artifacts.artifact_id(1))
    second = fit(store, artifact_id=artifacts.artifact_id(2))

    assert store.delete(f"local/de-price@{second}") == (second,)
    assert store.get("local/de-price").artifact_id == first


def test_deleting_the_only_revision_removes_the_alias_too(store: ArtifactStore) -> None:
    """An alias that resolves to nothing is worse than an alias that is gone."""
    revision = fit(store)
    assert store.delete(f"local/de-price@{revision}") == (revision,)
    assert store.aliases() == ()
    with pytest.raises(UnknownModelError):
        store.get("local/de-price")


def test_deleting_an_alias_deletes_every_revision_of_it(store: ArtifactStore) -> None:
    published = {fit(store, artifact_id=artifacts.artifact_id(step)) for step in (1, 2)}
    fit(store, name="fr-load")

    assert set(store.delete("local/de-price")) == published
    assert [handle.name for handle in store.list()] == ["fr-load"]


def test_deleting_what_was_never_fitted_says_so(store: ArtifactStore) -> None:
    with pytest.raises(UnknownModelError):
        store.delete("local/de-price")


# -- integrity --------------------------------------------------------------


def test_an_edited_recipe_fails_to_load(store: ArtifactStore) -> None:
    """An artifact is immutable, so an edit means it is no longer what it says."""
    revision = fit(store)
    handle = store.get(f"local/de-price@{revision}")
    (handle.path / "recipe.json").write_text(
        json.dumps(of.Model("nixtla/autoarima").model_dump(mode="json")), encoding="utf-8"
    )
    with pytest.raises(ArtifactError, match="does not hash"):
        store.read(handle.ref)


def test_a_renamed_artifact_directory_fails_to_load(store: ArtifactStore) -> None:
    revision = fit(store)
    renamed = artifacts.artifact_id(5)
    (store.models_root / revision).rename(store.models_root / renamed)
    with pytest.raises(ArtifactError, match="calls itself"):
        store.get(f"local/de-price@{renamed}")


def test_an_artifact_from_another_protocol_version_is_refused(store: ArtifactStore) -> None:
    """The provider directory's layout is not something to guess at."""
    revision = fit(store)
    path = store.models_root / revision / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**manifest, "protocol_version": 99}), encoding="utf-8")

    with pytest.raises(ArtifactError, match="protocol version"):
        store.get(f"local/de-price@{revision}")


def test_a_manifest_that_is_not_json_is_refused(store: ArtifactStore) -> None:
    revision = fit(store)
    (store.models_root / revision / "manifest.json").write_text("{", encoding="utf-8")
    with pytest.raises(ArtifactError, match="not valid JSON"):
        store.get(f"local/de-price@{revision}")


def test_something_that_is_not_an_artifact_directory_is_skipped(store: ArtifactStore) -> None:
    fit(store)
    (store.models_root / "notes.txt").write_text("scratch", encoding="utf-8")
    (store.models_root / "leftovers").mkdir()
    assert len(store.list()) == 1


# -- the default location ---------------------------------------------------


def test_the_default_root_is_the_platform_data_directory() -> None:
    root = default_root()
    assert root.name == "openforecast"
    assert root.is_absolute()


def test_a_store_touches_no_filesystem_until_it_is_used(tmp_path: Path) -> None:
    """Constructing one is cheap, so the engine can hold one without deciding to."""
    root = tmp_path / "unused"
    ArtifactStore(root)
    assert not root.exists()
