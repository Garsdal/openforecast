"""Model identity, contracts and the model catalog.

A model reference is a string with a shape:

```text
<namespace>/<name>[@revision]
```

and a descriptor is what that string resolves to. The descriptor is complete
enough to plan against on its own — it says which execution view the model
trains on, at how many origins, with which feature roles, and what it does about
missing values — so the engine can materialize data without consulting the
provider first.

```python
import openforecast as of

of.models.list()

descriptor = of.models.get("nixtla/nhits")
descriptor.lifecycle.requires_fit   # True
descriptor.training.view            # ViewKind.SEQUENCES
```

``list`` and ``get`` read the default catalog. It is empty until the built-in
reference provider registers itself in Step 8 and external providers advertise
their models in Step 9 — the catalog is filled by outer layers rather than
importing them.
"""

from openforecast.models.capabilities import (
    FeatureCapabilities,
    InstanceCapabilities,
    MissingValueSupport,
    ModelCapabilities,
    OutputCapabilities,
    TargetCapabilities,
)
from openforecast.models.catalog import DEFAULT_CATALOG, ModelCatalog
from openforecast.models.contract import OriginScope, TrainingContract
from openforecast.models.descriptor import ModelDescriptor
from openforecast.models.lifecycle import ModelLifecycle
from openforecast.models.ref import ModelRef
from openforecast.protocol.vocabulary import ViewKind

__all__ = [
    "DEFAULT_CATALOG",
    "FeatureCapabilities",
    "InstanceCapabilities",
    "MissingValueSupport",
    "ModelCapabilities",
    "ModelCatalog",
    "ModelDescriptor",
    "ModelLifecycle",
    "ModelRef",
    "OriginScope",
    "OutputCapabilities",
    "TargetCapabilities",
    "TrainingContract",
    "ViewKind",
    "get",
    "list",
    "register",
]


def get(ref: ModelRef | str) -> ModelDescriptor:
    """The descriptor named by ``ref``, from the default catalog."""
    return DEFAULT_CATALOG.get(ref)


def list(*, provider: str | None = None) -> tuple[ModelDescriptor, ...]:  # noqa: A001
    """Every model the default catalog can name, in reference order."""
    return DEFAULT_CATALOG.list(provider=provider)


def register(descriptor: ModelDescriptor) -> ModelDescriptor:
    """Add ``descriptor`` to the default catalog.

    How a provider makes itself discoverable. Kept out of the read path on
    purpose: a caller listing models should not be able to change what is
    listed by accident.
    """
    return DEFAULT_CATALOG.register(descriptor)
