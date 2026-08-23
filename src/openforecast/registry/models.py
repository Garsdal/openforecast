"""``ModelRegistry``: what one model reference means right now.

```python
registry.for_fit("nixtla/nhits")        # a descriptor: fit this
registry.resolve("local/de-price")      # a handle: forecast with this
registry.resolve("nixtla/nhits")        # ModelRequiresFit
```

A reference is a name, not a state — that is Step 5's rule — and this is the one
place that answers what state it currently has. The catalog knows what a provider
advertises; the artifact store knows what was fitted here. Neither can answer on
its own, and the engine should not be asking two things and combining them.

```python
registry.for_fit("amazon/chronos-2")    # ModelDoesNotSupportFit
```

The lifecycle distinction is the interesting part. Forecasting from a bare
``nixtla/nhits`` raises :class:`~openforecast.errors.ModelRequiresFit`, because a
model that has to be fitted is not a model that can forecast, and quietly fitting
it on whatever data the forecast call was handed would return a number that looks
like a forecast from a model the caller never trained. A pretrained model that
declares ``requires_fit=False`` resolves to its descriptor instead — zero-shot use
is a property the model declares, not one OpenForecast assumes.
"""

from __future__ import annotations

from openforecast.artifacts.handle import ModelHandle
from openforecast.artifacts.manifest import LOCAL_NAMESPACE
from openforecast.artifacts.store import ArtifactStore
from openforecast.errors import ModelDoesNotSupportFit, ModelError, ModelRequiresFit
from openforecast.models.catalog import DEFAULT_CATALOG, ModelCatalog
from openforecast.models.descriptor import ModelDescriptor
from openforecast.models.ref import ModelRef

__all__ = ["ModelRegistry", "Resolution"]

#: What a reference resolves to for forecasting: a fitted artifact, or a model
#: that needs no fitting.
Resolution = ModelHandle | ModelDescriptor


class ModelRegistry:
    """The catalog and the artifact store, behind one reference lookup."""

    def __init__(
        self, catalog: ModelCatalog | None = None, store: ArtifactStore | None = None
    ) -> None:
        self._catalog = catalog if catalog is not None else DEFAULT_CATALOG
        # Built lazily: constructing a registry should not touch the filesystem.
        self._store = store

    @property
    def catalog(self) -> ModelCatalog:
        return self._catalog

    @property
    def store(self) -> ArtifactStore:
        if self._store is None:
            self._store = ArtifactStore()
        return self._store

    def resolve(self, ref: ModelRef | str) -> Resolution:
        """What ``ref`` means for a forecast, right now.

        A local reference resolves to the artifact it names, following the alias
        when it is unpinned. A provider reference resolves to its descriptor only
        if the model can forecast without being fitted.
        """
        parsed = ModelRef.parse(ref)
        if parsed.namespace == LOCAL_NAMESPACE:
            return self.store.get(parsed)
        descriptor = self._catalog.get(parsed)
        if descriptor.lifecycle.requires_fit:
            raise ModelRequiresFit(
                f"{parsed} has to be fitted before it can forecast; fit it with of.fit and "
                f"forecast with the {LOCAL_NAMESPACE}/... reference that comes back"
            )
        return descriptor

    def artifact(self, ref: ModelRef | str) -> ModelHandle:
        """The fitted artifact ``ref`` names.

        For callers that need an artifact specifically — reading a manifest,
        deleting a revision — rather than whatever the reference resolves to.
        """
        return self.store.get(ref)

    def for_fit(self, ref: ModelRef | str) -> ModelDescriptor:
        """The descriptor to plan a fit against.

        A local reference is rejected: an artifact is the *result* of a fit, and
        refitting one means fitting the model it came from, on data of the
        caller's choosing. The manifest says which model that was.
        """
        parsed = ModelRef.parse(ref)
        if parsed.namespace == LOCAL_NAMESPACE:
            handle = self.store.get(parsed)
            source = handle.manifest.source_model or "the recipe recorded in it"
            raise ModelError(
                f"{parsed} is a fitted artifact, not a model to fit; it was fitted from "
                f"{source}, so fit that instead"
            )
        descriptor = self._catalog.get(parsed)
        if not descriptor.is_fittable:
            raise ModelDoesNotSupportFit(
                f"{parsed} cannot be fitted; it is used zero-shot, so forecast with the "
                f"reference directly: of.forecast(model='{parsed}', data=..., horizon=...)"
            )
        return descriptor

    def __contains__(self, ref: ModelRef | str) -> bool:
        try:
            self.resolve(ref)
        except (ModelError, ModelRequiresFit):
            return False
        return True

    def __repr__(self) -> str:
        return f"ModelRegistry(models={len(self._catalog)}, store={self.store.root})"
