"""Which providers exist, and how their models become discoverable.

```python
providers = install_default_providers()      # registers into the default catalog

of.models.list()                             # (builtin/seasonal-naive,)
```

A provider is discovered and its models are registered; nothing imports a
provider in order to know what it offers. Today that is one in-process provider
whose descriptors are read directly. In Step 9 the same two lines happen over a
handshake with a subprocess in its own environment, and the catalog cannot tell
the difference — which is the point of registering descriptors rather than
importing model classes.

Registration is idempotent so that installing twice is not an error. Registering
a *different* descriptor under a name already taken still is: which model a
reference means may not depend on load order.
"""

from __future__ import annotations

from collections.abc import Iterable

from openforecast.errors import DuplicateModelError
from openforecast.models.catalog import DEFAULT_CATALOG, ModelCatalog
from openforecast.providers.builtin import BUILTIN_PROVIDER
from openforecast.runtime.provider import ProviderClient, ProviderRegistry

__all__ = ["default_providers", "install_default_providers", "register_descriptors"]


def default_providers() -> ProviderRegistry:
    """The providers this build ships with."""
    return ProviderRegistry([BUILTIN_PROVIDER])


def install_default_providers(catalog: ModelCatalog | None = None) -> ProviderRegistry:
    """Register the shipped providers' models and return the registry.

    What makes ``of.models.list()`` answer anything at all: the catalog holds
    what providers advertise, and this is the advertising.
    """
    providers = default_providers()
    register_descriptors(providers, catalog)
    return providers


def register_descriptors(
    providers: Iterable[ProviderClient], catalog: ModelCatalog | None = None
) -> ModelCatalog:
    """Add every model these providers advertise to ``catalog``."""
    target = DEFAULT_CATALOG if catalog is None else catalog
    for provider in providers:
        for descriptor in provider.descriptors():
            if descriptor.ref in target:
                # Installing the same provider twice is not a conflict; two
                # providers claiming one reference is, and the catalog says so.
                if target.get(descriptor.ref) == descriptor:
                    continue
                raise DuplicateModelError(
                    f"{descriptor.ref} is already registered as a different model"
                )
            target.register(descriptor)
    return target
