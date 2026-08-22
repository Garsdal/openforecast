"""The providers this process can execute models with, by name.

```python
providers = ProviderRegistry([BUILTIN_PROVIDER, subprocess_provider])

providers.get("nixtla").fit(...)
```

A ``ProviderRegistry`` maps a provider's name to its client. It is what stops
the engine from ever asking which provider it is talking to: a descriptor names
one, the registry hands it over, and the engine calls the same two methods
whatever came back — an in-process provider and a subprocess in its own
environment are the same thing from here.

:class:`~openforecast.providers.client.ProviderClient`, the shape of what the
registry holds, is defined in ``providers/`` and re-exported here. Both sides of
the boundary have to name it, and ``runtime/`` is not on a provider's import
surface.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from openforecast.errors import ProviderError
from openforecast.providers.client import ProviderClient

__all__ = ["ProviderClient", "ProviderRegistry"]


class ProviderRegistry:
    """The providers this process can execute models with, by name."""

    def __init__(self, providers: Iterable[ProviderClient] = ()) -> None:
        self._providers: dict[str, ProviderClient] = {}
        for provider in providers:
            self.register(provider)

    def register(self, provider: ProviderClient) -> ProviderClient:
        """Add ``provider``, refusing to shadow one already registered."""
        existing = self._providers.get(provider.name)
        if existing is not None:
            raise ProviderError(
                f"a provider named {provider.name!r} is already registered; a provider "
                f"name is the namespace of the models it advertises, so it names one"
            )
        self._providers[provider.name] = provider
        return provider

    def get(self, name: str) -> ProviderClient:
        """The provider called ``name``.

        The engine reaches this through a model descriptor, so a miss means a
        model was advertised by something that has since gone away.
        """
        provider = self._providers.get(name)
        if provider is None:
            raise ProviderError(
                f"no provider named {name!r} is installed; available: {sorted(self._providers)}"
            )
        return provider

    def __contains__(self, name: str) -> bool:
        return name in self._providers

    def __iter__(self) -> Iterator[ProviderClient]:
        return iter(self._providers.values())

    def __len__(self) -> int:
        return len(self._providers)

    def __repr__(self) -> str:
        return f"ProviderRegistry({sorted(self._providers)})"
