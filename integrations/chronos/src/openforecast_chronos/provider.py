"""``ChronosProvider``: the provider contract, for a model that is never fitted.

```python
provider.descriptors()
provider.forecast(model=..., params=..., view=..., output=..., state=...)
provider.fit(...)                       # refused: there is nothing to fit
```

The same :class:`~openforecast.providers.ProviderClient` contract the built-in
provider and the four trainable integrations implement, which is what lets this
run in a subprocess in its own environment without the engine knowing.

``fit`` is present and refuses. The protocol has three operations and a provider
answers all three or it is not one — but what it answers with here is the same
refusal the registry already made on the caller's side, because the models this
provider advertises declare ``training=None`` and the engine never plans a fit
for one. So this method is unreachable through ``of.fit``, and it exists for the
case where something reaches the wire directly: a provider that quietly did
nothing and reported success would leave an artifact recording a fit that never
happened.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa

from openforecast.errors import ModelDoesNotSupportFit
from openforecast.models import ModelDescriptor, ModelRef
from openforecast.views import FitView, ForecastView
from openforecast_chronos import catalog
from openforecast_chronos._version import __version__

__all__ = ["PROVIDER_NAME", "PROVIDER_VERSION", "ChronosProvider"]

#: Also the namespace of every model advertised: ``amazon/chronos-2``. Named
#: after the publisher rather than after this distribution, because that is the
#: reference the model is known by.
PROVIDER_NAME = "amazon"

#: The version of this distribution, reported at the handshake so that
#: ``openforecast providers list`` says which implementation is installed.
PROVIDER_VERSION = __version__


class ChronosProvider:
    """Executes pretrained Chronos checkpoints, in whatever process it was started in."""

    name = PROVIDER_NAME
    version = PROVIDER_VERSION

    def descriptors(self) -> tuple[ModelDescriptor, ...]:
        """Every model this integration can execute."""
        return catalog.descriptors(self.name)

    def fit(
        self,
        *,
        model: ModelRef | str,
        params: Mapping[str, Any],
        view: FitView,
        seed: int | None,
        into: Path,
    ) -> None:
        """Refuse: every model this provider advertises is used as it was published."""
        del params, view, seed, into
        raise ModelDoesNotSupportFit(
            f"{model} is a pretrained model and cannot be fitted; forecast with the "
            f"reference directly"
        )

    def forecast(
        self,
        *,
        model: ModelRef | str,
        params: Mapping[str, Any],
        view: ForecastView,
        output: Mapping[str, Any],
        state: Path,
    ) -> pa.Table:
        """Answer ``view`` from the pretrained checkpoint ``model`` names.

        ``state`` is the empty directory a zero-shot forecast is given, and
        ``params`` is empty for the same reason: there was no fit to compile
        parameters at, so the model advertises none and nothing is dropped.
        """
        del params
        return catalog.adapter_for(model, self.name).forecast(view, output, state)

    def __repr__(self) -> str:
        return f"ChronosProvider(version={self.version}, models={len(catalog.model_names())})"
