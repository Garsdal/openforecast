"""What ``sktime`` advertises, and which adapter executes each of it.

```python
descriptors("sktime")                          # every model, as the handshake reports it
adapter_for("sktime/pooled-trees", "sktime")    # the one that executes it
```

A catalog rather than a chain of ``if`` statements in the provider: a model is
added by naming its adapter here, and both the handshake and the dispatch read
the same table. Nothing else in the integration knows how many models there are.

Asking for the descriptors imports no forecasting library — the adapters import
theirs inside the calls that need them — so a handshake stays cheap, which is
what makes ``openforecast providers list`` and installation-time discovery cheap.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import pyarrow as pa

from openforecast.errors import UnknownModelError
from openforecast.models import ModelDescriptor, ModelRef
from openforecast.views import FitView, ForecastView
from openforecast_sktime.adapters import local_models, panel_models

__all__ = ["Adapter", "adapter_for", "descriptors", "model_names"]


class Adapter(Protocol):
    """One sktime forecaster, behind the three provider operations."""

    @property
    def name(self) -> str:
        """The model half of the reference: ``theta`` in ``sktime/theta``."""
        ...

    def descriptor(self, provider: str) -> ModelDescriptor: ...

    def fit(
        self, view: FitView, params: Mapping[str, Any], into: Path, *, seed: int | None
    ) -> None: ...

    def forecast(self, view: ForecastView, output: Mapping[str, Any], state: Path) -> pa.Table: ...


#: Every model this integration provides, by the name in its reference.
ADAPTERS: Mapping[str, Adapter] = {
    local_models.THETA.name: local_models.THETA,
    panel_models.POOLED_TREES.name: panel_models.POOLED_TREES,
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
