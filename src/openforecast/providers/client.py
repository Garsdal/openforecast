"""What anything that executes a model has to look like.

```python
provider.descriptors()
provider.fit(model=..., params=..., view=..., seed=..., into=...)
provider.forecast(model=..., params=..., view=..., output=..., state=...)
```

A :class:`typing.Protocol` rather than a base class, deliberately. A provider
may run in this process or in a subprocess in its own environment; what the two
share is the shape of these three calls, not an inheritance chain one of them
would have to import across a process boundary. Structural typing says exactly
that and nothing more.

It lives in ``providers/`` rather than in ``runtime/`` because both sides of the
boundary have to name it: the engine calls it, and the serving harness an
integration's ``__main__`` runs implements it. ``runtime/`` is not on a
provider's import surface, so a contract defined there could only be duplicated.

Everything crossing the call is either bulk data in a provider-neutral view or a
plain mapping — ``params`` as the user wrote them, ``output`` as it serializes.
That is not a coincidence: those are exactly the arguments that become a JSON
control message and an Arrow bundle when the provider is a subprocess, so an
in-process provider exercises the same contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pyarrow as pa

from openforecast.models.descriptor import ModelDescriptor
from openforecast.models.ref import ModelRef
from openforecast.views.forecast import ForecastView
from openforecast.views.planner import FitView

__all__ = ["ProviderClient"]


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
