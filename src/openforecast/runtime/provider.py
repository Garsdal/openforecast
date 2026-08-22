"""What the engine requires of anything that executes a model.

```python
provider.descriptors()
provider.fit(model=..., params=..., view=..., seed=..., into=...)
provider.forecast(model=..., params=..., view=..., output=..., state=...)
```

A :class:`typing.Protocol` rather than a base class, deliberately. Step 9's
provider runs in another process and another environment; what it shares with
the in-process one is the shape of these three calls, not an inheritance chain
it would have to import across a subprocess boundary. Structural typing says
exactly that and nothing more.

Everything crossing the call is either bulk data in a provider-neutral view or a
plain mapping — ``params`` as the user wrote them, ``output`` as it serializes.
That is not a coincidence: these are the arguments that become a JSON control
message and an Arrow bundle in Step 9, so the in-process provider is exercising
the same contract the subprocess one will.

A ``ProviderRegistry`` maps a provider's name to its client. It is what stops
the engine from ever asking which provider it is talking to: a descriptor names
one, the registry hands it over, and the engine calls the same two methods
whatever came back.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pyarrow as pa

from openforecast.errors import ProviderError
from openforecast.models.descriptor import ModelDescriptor
from openforecast.models.ref import ModelRef
from openforecast.views.forecast import ForecastView
from openforecast.views.planner import FitView

__all__ = ["ProviderClient", "ProviderRegistry"]


@runtime_checkable
class ProviderClient(Protocol):
    """One provider, in this process or behind a subprocess transport."""

    @property
    def name(self) -> str:
        """The namespace of every model it advertises: ``nixtla``, ``builtin``."""
        ...

    @property
    def version(self) -> str:
        """Recorded in every artifact this provider fits."""
        ...

    def descriptors(self) -> tuple[ModelDescriptor, ...]:
        """Every model it can execute."""
        ...

    def fit(
        self,
        *,
        model: ModelRef | str,
        params: Mapping[str, Any],
        view: FitView,
        seed: int | None,
        into: Path,
    ) -> None:
        """Fit ``model`` on ``view``, persisting its native state into ``into``.

        ``into`` is the provider's own directory and nothing else reads it. A
        provider that raises leaves no artifact behind: the engine stages a fit
        and publishes it only on success.
        """
        ...

    def forecast(
        self,
        *,
        model: ModelRef | str,
        params: Mapping[str, Any],
        view: ForecastView,
        output: Mapping[str, Any],
        state: Path,
    ) -> pa.Table:
        """Answer ``view`` from the state a previous fit wrote into ``state``.

        The answer is one long table in the canonical forecast columns — the
        instance keys, ``event_time``, ``target``, ``kind``, ``quantile``,
        ``sample`` and ``value``. The engine validates it against what it asked
        for, so a provider cannot quietly answer a different question.
        """
        ...


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
