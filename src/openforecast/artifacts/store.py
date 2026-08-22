"""``ArtifactStore``: immutable revisions on disk, and the aliases that move.

```text
~/.local/share/openforecast/
    models/
        01K5Z6QK3M9TQK1W2E3R4T5Y6U/
            manifest.json
            recipe.json
            schema.json
            provider/
    aliases/
        de-price.json
    .tmp/
```

Two kinds of thing live here, and the difference is the whole design. A
*revision* is immutable: once published it is never written to again, so a
forecast made from it today is the forecast it would have made a month ago.
An *alias* is a pointer that moves: ``local/de-price`` means whichever revision
was last selected, which is what lets a scheduled forecast job name a model once
and pick up retrainings without being edited.

Nothing is published until the fit has succeeded. A provider trains into
``.tmp/<artifact-id>`` and the directory is renamed into ``models/`` only after,
because a crashed fit that leaves a half-written artifact behind is worse than a
crashed fit: the artifact would be resolvable, and only the forecasts would be
wrong.

No provider is involved in any of this. The store reads and writes manifests,
recipes and schemas; the ``provider/`` subdirectory is created, handed over and
never looked inside.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from platformdirs import user_data_path
from pydantic import BaseModel, ConfigDict

from openforecast.artifacts.artifact import ModelArtifact
from openforecast.artifacts.handle import ModelHandle
from openforecast.artifacts.identity import is_artifact_id
from openforecast.artifacts.manifest import LOCAL_NAMESPACE, ModelManifest
from openforecast.errors import ArtifactError, UnknownModelError
from openforecast.models.ref import ModelRef
from openforecast.protocol.version import PROTOCOL_VERSION
from openforecast.recipes.nodes import parse_recipe

__all__ = ["ALIASES_DIRNAME", "ArtifactStaging", "ArtifactStore", "MODELS_DIRNAME"]

MODELS_DIRNAME = "models"
ALIASES_DIRNAME = "aliases"
STAGING_DIRNAME = ".tmp"
PROVIDER_DIRNAME = "provider"

MANIFEST_FILENAME = "manifest.json"
RECIPE_FILENAME = "recipe.json"
SCHEMA_FILENAME = "schema.json"


class ArtifactAlias(BaseModel):
    """``aliases/de-price.json``: which revision this name currently means."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    revision: str


class ArtifactStaging:
    """One artifact being written, not yet visible under its reference.

    The provider writes into :attr:`provider_path` and knows nothing else about
    the store. Publication is a single rename, so a reader either sees a complete
    artifact or sees nothing.
    """

    def __init__(self, store: ArtifactStore, artifact: ModelArtifact) -> None:
        self._store = store
        self._artifact = artifact
        self._path = store.staging_root / artifact.artifact_id
        self._handle: ModelHandle | None = None
        if self._path.exists():
            raise ArtifactError(f"{self._path} is already being written")
        if store.contains_revision(artifact.artifact_id):
            raise ArtifactError(f"{artifact.artifact_id} is already published")
        (self._path / PROVIDER_DIRNAME).mkdir(parents=True)

    @property
    def artifact(self) -> ModelArtifact:
        return self._artifact

    @property
    def artifact_id(self) -> str:
        return self._artifact.artifact_id

    @property
    def path(self) -> Path:
        return self._path

    @property
    def provider_path(self) -> Path:
        """Where the provider persists its native state. Opaque to OpenForecast."""
        return self._path / PROVIDER_DIRNAME

    @property
    def is_published(self) -> bool:
        return self._handle is not None

    @property
    def handle(self) -> ModelHandle:
        """The published artifact — only after :meth:`commit`."""
        if self._handle is None:
            raise ArtifactError(
                f"{self.artifact_id} has not been published yet; a staged artifact becomes "
                f"a handle when the fit that produced it succeeds"
            )
        return self._handle

    def commit(self, *, alias: bool = True) -> ModelHandle:
        """Write the metadata and move the artifact into place, atomically."""
        if self._handle is not None:
            raise ArtifactError(f"{self.artifact_id} has already been published")
        _write_json(self._path / MANIFEST_FILENAME, self._artifact.manifest.model_dump(mode="json"))
        _write_json(self._path / RECIPE_FILENAME, self._artifact.recipe.model_dump(mode="json"))
        _write_json(self._path / SCHEMA_FILENAME, self._artifact.training_schema)
        self._handle = self._store.publish(self._path, self._artifact, alias=alias)
        return self._handle

    def abort(self) -> None:
        """Discard everything written so far. A failed fit produces no artifact."""
        shutil.rmtree(self._path, ignore_errors=True)


class ArtifactStore:
    """The local artifact registry: create, resolve, alias, reload, delete."""

    def __init__(self, root: str | Path | None = None) -> None:
        self._root = Path(root) if root is not None else default_root()

    @property
    def root(self) -> Path:
        return self._root

    @property
    def models_root(self) -> Path:
        return self._root / MODELS_DIRNAME

    @property
    def aliases_root(self) -> Path:
        return self._root / ALIASES_DIRNAME

    @property
    def staging_root(self) -> Path:
        return self._root / STAGING_DIRNAME

    # -- creation ----------------------------------------------------------

    @contextmanager
    def stage(self, artifact: ModelArtifact, *, alias: bool = True) -> Generator[ArtifactStaging]:
        """Materialize ``artifact`` under ``.tmp`` and publish it on a clean exit.

        ```python
        with store.stage(artifact) as staging:
            provider.fit(view, into=staging.provider_path)
        handle = staging.handle
        ```

        An exception anywhere inside removes the staged directory and propagates:
        a fit that raised produced no model, and a resolvable artifact whose
        provider state is half-written would be indistinguishable from one that
        worked.
        """
        self.staging_root.mkdir(parents=True, exist_ok=True)
        staging = ArtifactStaging(self, artifact)
        try:
            yield staging
        except BaseException:
            staging.abort()
            raise
        if not staging.is_published:  # a caller is free to commit inside the block
            staging.commit(alias=alias)

    def publish(self, staged: Path, artifact: ModelArtifact, *, alias: bool = True) -> ModelHandle:
        """Move a fully written staging directory into place, atomically.

        What :meth:`ArtifactStaging.commit` calls once the metadata is on disk.
        The rename is the moment the artifact starts existing, which is why it is
        the last thing that happens.
        """
        destination = self.models_root / artifact.artifact_id
        self.models_root.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise ArtifactError(f"{artifact.artifact_id} is already published")
        # Atomic within a filesystem, which .tmp and models/ share by construction.
        os.rename(staged, destination)
        if alias:
            self.set_alias(artifact.manifest.name, artifact.artifact_id)
        return ModelHandle(manifest=artifact.manifest, path=destination)

    # -- resolution --------------------------------------------------------

    def get(self, ref: ModelRef | str) -> ModelHandle:
        """The handle ``ref`` names, following the alias when it is unpinned."""
        parsed = self._local(ref)
        revision = parsed.revision if parsed.revision is not None else self.alias(parsed.name)
        path = self.models_root / revision
        manifest = self._read_manifest(path, revision)
        if parsed.is_pinned and manifest.name != parsed.name:
            raise UnknownModelError(
                f"{parsed} does not exist; revision {revision} was fitted as {manifest.alias}"
            )
        return ModelHandle(manifest=manifest, path=path)

    def read(self, ref: ModelRef | str) -> ModelArtifact:
        """The whole artifact — manifest, recipe and training schema — re-validated.

        What a fit reads back when it has to know exactly what was fitted: the
        engine reloading an artifact to forecast, or a benchmark comparing what
        two artifacts did. The hashes in the manifest are checked here, so an
        artifact that was edited on disk fails to load rather than forecasting.
        """
        handle = self.get(ref)
        recipe_payload = _read_json(handle.path / RECIPE_FILENAME, handle.artifact_id)
        schema_payload = _read_json(handle.path / SCHEMA_FILENAME, handle.artifact_id)
        if not isinstance(schema_payload, dict):
            raise ArtifactError(f"the schema of {handle.artifact_id} is not a schema object")
        return ModelArtifact(
            manifest=handle.manifest,
            recipe=parse_recipe(recipe_payload),
            training_schema=schema_payload,  # pyright: ignore[reportUnknownArgumentType]
        )

    def list(self, *, name: str | None = None) -> tuple[ModelHandle, ...]:
        """Every published artifact, oldest first; optionally one alias's history.

        Chronological because an artifact id sorts that way, so listing needs no
        manifest field anybody could have written differently.
        """
        handles = [
            ModelHandle(manifest=self._read_manifest(path, path.name), path=path)
            for path in sorted(self._revision_paths())
        ]
        return tuple(handle for handle in handles if name is None or handle.name == name)

    def revisions(self, name: str) -> tuple[str, ...]:
        """Every revision fitted under ``name``, oldest first."""
        return tuple(handle.artifact_id for handle in self.list(name=name))

    def contains_revision(self, revision: str) -> bool:
        return (self.models_root / revision / MANIFEST_FILENAME).is_file()

    def __contains__(self, ref: ModelRef | str) -> bool:
        try:
            self.get(ref)
        except (UnknownModelError, ArtifactError):
            return False
        return True

    # -- aliases -----------------------------------------------------------

    def alias(self, name: str) -> str:
        """The revision ``local/<name>`` currently points at."""
        path = self._alias_path(name)
        if not path.is_file():
            known = [str(ref) for ref in self.aliases()]
            raise UnknownModelError(
                f"no fitted model named {LOCAL_NAMESPACE}/{name}"
                + (f"; known: {known}" if known else "; nothing has been fitted yet")
            )
        alias = ArtifactAlias.model_validate(_read_json(path, name))
        if not self.contains_revision(alias.revision):
            raise ArtifactError(
                f"{LOCAL_NAMESPACE}/{name} points at revision {alias.revision}, which is "
                f"no longer in the store; point it at an existing revision or delete it"
            )
        return alias.revision

    def set_alias(self, name: str, revision: str) -> ModelRef:
        """Point ``local/<name>`` at ``revision``.

        How a retrained model becomes the one a scheduled job picks up, and how a
        rollback happens: the alias moves, the revisions do not.
        """
        reference = ModelRef(namespace=LOCAL_NAMESPACE, name=name, revision=revision)
        if not self.contains_revision(revision):
            raise UnknownModelError(f"{reference} does not exist, so no alias can point at it")
        manifest = self._read_manifest(self.models_root / revision, revision)
        if manifest.name != name:
            raise ArtifactError(
                f"revision {revision} was fitted as {manifest.alias}; an alias names a "
                f"lineage, so pointing {LOCAL_NAMESPACE}/{name} at it would make one "
                f"reference mean two different models"
            )
        self.aliases_root.mkdir(parents=True, exist_ok=True)
        _write_json(
            self._alias_path(name), ArtifactAlias(name=name, revision=revision).model_dump()
        )
        return reference

    def aliases(self) -> tuple[ModelRef, ...]:
        """Every alias in the store, as the references a user would write."""
        if not self.aliases_root.is_dir():
            return ()
        return tuple(
            ModelRef(namespace=LOCAL_NAMESPACE, name=path.stem)
            for path in sorted(self.aliases_root.glob("*.json"))
        )

    # -- deletion ----------------------------------------------------------

    def delete(self, ref: ModelRef | str) -> tuple[str, ...]:
        """Delete a revision, or every revision of an alias, and return the ids.

        A pinned reference deletes that revision only. If its alias pointed
        there, the alias follows to the newest remaining revision of the same
        name, or is removed when none is left — an alias resolving to nothing is
        worse than an alias that is gone, because only one of the two says so at
        the point of use.
        """
        parsed = self._local(ref)
        if not parsed.is_pinned:
            deleted = self.revisions(parsed.name)
            if not deleted:
                raise UnknownModelError(f"no fitted model named {parsed}")
            for revision in deleted:
                shutil.rmtree(self.models_root / revision)
            self._alias_path(parsed.name).unlink(missing_ok=True)
            return deleted

        handle = self.get(parsed)
        shutil.rmtree(handle.path)
        remaining = self.revisions(handle.name)
        if remaining:
            self.set_alias(handle.name, remaining[-1])
        else:
            self._alias_path(handle.name).unlink(missing_ok=True)
        return (handle.artifact_id,)

    # -- internals ---------------------------------------------------------

    def _local(self, ref: ModelRef | str) -> ModelRef:
        """Parse ``ref`` and require it to name something this store could hold."""
        parsed = ModelRef.parse(ref)
        if parsed.namespace != LOCAL_NAMESPACE:
            raise UnknownModelError(
                f"{parsed} names a model provided by {parsed.namespace!r}, not a fitted "
                f"artifact; fitted models live in the {LOCAL_NAMESPACE!r} namespace"
            )
        if parsed.revision is not None and not is_artifact_id(parsed.revision):
            raise UnknownModelError(f"{parsed} does not pin an artifact id")
        return parsed

    def _alias_path(self, name: str) -> Path:
        # Through ModelRef, so a name that is not one path segment cannot become
        # a path at all.
        reference = ModelRef(namespace=LOCAL_NAMESPACE, name=name)
        return self.aliases_root / f"{reference.name}.json"

    def _revision_paths(self) -> Iterator[Path]:
        if not self.models_root.is_dir():
            return
        for path in self.models_root.iterdir():
            if is_artifact_id(path.name) and (path / MANIFEST_FILENAME).is_file():
                yield path

    def _read_manifest(self, path: Path, revision: str) -> ModelManifest:
        if not (path / MANIFEST_FILENAME).is_file():
            raise UnknownModelError(f"no artifact {revision} in {self.models_root}")
        manifest = ModelManifest.model_validate(_read_json(path / MANIFEST_FILENAME, revision))
        if manifest.artifact_id != revision:
            raise ArtifactError(
                f"the artifact in {path} calls itself {manifest.artifact_id}; an artifact "
                f"id is its directory name, so one of the two was renamed"
            )
        if manifest.protocol_version != PROTOCOL_VERSION:
            raise ArtifactError(
                f"{manifest.ref} was written for protocol version "
                f"{manifest.protocol_version} and this build speaks {PROTOCOL_VERSION}; "
                f"refit it rather than reading a layout that may have changed"
            )
        return manifest

    def __repr__(self) -> str:
        return f"ArtifactStore({self._root})"


def default_root() -> Path:
    """``~/.local/share/openforecast`` and its equivalent on each platform."""
    return user_data_path("openforecast", appauthor=False)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path, subject: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise UnknownModelError(f"{subject} cannot be read: {error}") from error
    except json.JSONDecodeError as error:
        raise ArtifactError(f"{path} is not valid JSON: {error}") from error
