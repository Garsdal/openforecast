"""``ModelCatalog``: the models OpenForecast can currently name.

```python
of.models.list()
descriptor = of.models.get("nixtla/nhits")
```

The catalog holds descriptors only. Resolving a reference to a *fitted* model —
following the ``local/de-price`` alias to the revision it points at — belongs to
the artifact store, and :class:`~openforecast.registry.ModelRegistry` is where
the two halves of "what does this string mean" meet. It reads descriptors from
here rather than duplicating them.

Descriptors are pushed in rather than discovered here: the built-in reference
provider registers itself in Step 8 and external providers advertise their
models over the handshake in Step 9. Both of those live in outer layers, which
is why the catalog is something they fill rather than something that imports
them. Until then it is legitimately empty.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from openforecast.errors import DuplicateModelError, UnknownModelError
from openforecast.models.descriptor import ModelDescriptor
from openforecast.models.ref import ModelRef

__all__ = ["DEFAULT_CATALOG", "ModelCatalog"]


class ModelCatalog:
    """A set of model descriptors, keyed by reference."""

    def __init__(self, descriptors: Iterable[ModelDescriptor] = ()) -> None:
        self._descriptors: dict[ModelRef, ModelDescriptor] = {}
        for descriptor in descriptors:
            self.register(descriptor)

    def register(self, descriptor: ModelDescriptor) -> ModelDescriptor:
        """Add ``descriptor``, refusing to shadow one already registered.

        Registering the same reference twice is a wiring bug — two providers
        claiming one name, or one provider registered twice — and letting the
        later call win would make which model you get depend on load order.
        """
        existing = self._descriptors.get(descriptor.ref)
        if existing is not None:
            raise DuplicateModelError(
                f"{descriptor.ref} is already registered by provider "
                f"{existing.provider!r}; a reference names one model"
            )
        self._descriptors[descriptor.ref] = descriptor
        return descriptor

    def get(self, ref: ModelRef | str) -> ModelDescriptor:
        """The descriptor for ``ref``, which may be a plain string."""
        parsed = ModelRef.parse(ref)
        descriptor = self._descriptors.get(parsed)
        if descriptor is not None:
            return descriptor
        if parsed.is_pinned:
            raise UnknownModelError(
                f"{parsed} names a fitted revision; the model catalog holds model "
                f"descriptors, so ask for {parsed.unpinned} instead"
            )
        raise UnknownModelError(f"no model named {parsed}{self._known()}", model=str(parsed))

    def list(self, *, provider: str | None = None) -> tuple[ModelDescriptor, ...]:
        """Every registered descriptor, in reference order."""
        return tuple(
            descriptor
            for descriptor in sorted(self._descriptors.values(), key=lambda item: str(item.ref))
            if provider is None or descriptor.provider == provider
        )

    def refs(self) -> tuple[ModelRef, ...]:
        return tuple(descriptor.ref for descriptor in self.list())

    def providers(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(descriptor.provider for descriptor in self.list()))

    def __contains__(self, ref: ModelRef | str) -> bool:
        return ModelRef.parse(ref) in self._descriptors

    def __iter__(self) -> Iterator[ModelDescriptor]:
        return iter(self.list())

    def __len__(self) -> int:
        return len(self._descriptors)

    def _known(self) -> str:
        if not self._descriptors:
            return "; no models are registered yet"
        return f"; known models: {[str(ref) for ref in self.refs()]}"


#: The catalog behind ``of.models.list()`` and ``of.models.get(...)``.
DEFAULT_CATALOG = ModelCatalog()
