"""Which providers exist, and how their models become discoverable.

```python
providers = install_default_providers()      # registers into the default catalog

of.models.list()                             # (builtin/seasonal-naive, nixtla/nhits)
```

A provider is discovered and its models are registered; nothing imports a
provider in order to know what it offers. Two kinds are discovered here and the
catalog cannot tell them apart, which is the point of registering descriptors
rather than importing model classes:

```text
shipped     the built-in provider, in this process
installed   an integration in its own uv environment, over the subprocess protocol
```

Discovery of the second kind reads the handshake each environment recorded when
it was installed. It starts no process and imports nothing from the integration
— a provider process starts when a model is actually fitted or forecast with,
and the handshake is repeated then to check that the environment still is what
it said it was.

Registration is idempotent so that installing twice is not an error. Registering
a *different* descriptor under a name already taken still is: which model a
reference means may not depend on load order.
"""

from __future__ import annotations

from collections.abc import Iterable

from openforecast.errors import DuplicateModelError
from openforecast.models.catalog import DEFAULT_CATALOG, ModelCatalog
from openforecast.providers.builtin import BUILTIN_PROVIDER
from openforecast.runtime.environments import ProviderEnvironments
from openforecast.runtime.provider import ProviderClient, ProviderRegistry

__all__ = [
    "default_providers",
    "install_default_providers",
    "installed_providers",
    "register_descriptors",
]


def default_providers(environments: ProviderEnvironments | None = None) -> ProviderRegistry:
    """The provider this build ships with, plus every environment installed."""
    registry = ProviderRegistry([BUILTIN_PROVIDER])
    for client in installed_providers(environments):
        registry.register(client)
    return registry


def installed_providers(
    environments: ProviderEnvironments | None = None,
) -> tuple[ProviderClient, ...]:
    """A client per installed provider environment. Starts nothing."""
    store = environments if environments is not None else ProviderEnvironments()
    return tuple(store.clients())


def install_default_providers(
    catalog: ModelCatalog | None = None, environments: ProviderEnvironments | None = None
) -> ProviderRegistry:
    """Register the discoverable providers' models and return the registry.

    What makes ``of.models.list()`` answer anything at all: the catalog holds
    what providers advertise, and this is the advertising.
    """
    providers = default_providers(environments)
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
