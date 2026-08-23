"""What ``amazon`` advertises, and which adapter executes each of it.

```python
descriptors("amazon")                    # every model, as the handshake reports it
adapter_for("amazon/chronos-2", "amazon")  # the one that executes it
```

A catalog rather than a chain of ``if`` statements in the provider: a model is
added by naming its adapter here, and both the handshake and the dispatch read
the same table.

One entry, and the plan says so outright: the point of the step is the second
model lifecycle, not a shelf of checkpoints. ``amazon/chronos-bolt-base`` and
the rest are a row here each once there is a reason to prefer one, and if adding
the second needs more than a row then the zero-shot path is doing less than it
claims.

Asking for the descriptors imports neither ``chronos`` nor ``torch`` — the
adapter imports them inside the calls that need them — so a handshake stays
cheap, which is what makes ``openforecast providers list`` and installation-time
discovery cheap.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import pyarrow as pa

from openforecast.errors import UnknownModelError
from openforecast.models import ModelDescriptor, ModelRef
from openforecast.views import ForecastView
from openforecast_chronos.adapter import CHRONOS_2

__all__ = ["Adapter", "adapter_for", "descriptors", "model_names"]


class Adapter(Protocol):
    """One pretrained checkpoint, behind the operations it supports.

    Two rather than three: there is no ``fit``, and the absence is the
    declaration. A provider whose descriptors say ``training=None`` is never
    asked to fit — the registry refuses it before a process is started — so an
    adapter that implemented one would be implementing something unreachable.
    """

    @property
    def name(self) -> str:
        """The model half of the reference: ``chronos-2`` in ``amazon/chronos-2``."""
        ...

    def descriptor(self, provider: str) -> ModelDescriptor: ...

    def forecast(self, view: ForecastView, output: Mapping[str, Any], state: Path) -> pa.Table: ...


#: Every model this integration provides, by the name in its reference.
ADAPTERS: Mapping[str, Adapter] = {
    CHRONOS_2.name: CHRONOS_2,
}


def model_names() -> tuple[str, ...]:
    return tuple(ADAPTERS)


def descriptors(provider: str) -> tuple[ModelDescriptor, ...]:
    """Every model this integration advertises, namespaced to ``provider``."""
    return tuple(adapter.descriptor(provider) for adapter in ADAPTERS.values())


def adapter_for(model: ModelRef | str, provider: str) -> Adapter:
    """The adapter that executes ``model``, or a refusal naming what there is."""
    ref = ModelRef.parse(model)
    adapter = ADAPTERS.get(ref.name) if ref.namespace == provider else None
    if adapter is None:
        raise UnknownModelError(
            f"{ref} is not a model of the {provider!r} provider; it provides "
            f"{[f'{provider}/{name}' for name in ADAPTERS]}"
        )
    return adapter
