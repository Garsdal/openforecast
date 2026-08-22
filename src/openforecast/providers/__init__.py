"""Providers that execute models, and the SDK an external one is written with.

A provider is the only thing in OpenForecast that touches a forecasting
implementation, and it sees exactly one kind of data: execution views. That is
the boundary rule the whole architecture rests on, so the built-in provider is
held to it as strictly as an external integration would be — its import surface
is :mod:`openforecast.views`, :mod:`openforecast.errors`,
:mod:`openforecast.protocol`, :mod:`openforecast.models` and this package.

Three things live here:

```text
ProviderClient    the three calls anything that executes a model has to answer
serve             the harness an integration's __main__ runs on stdin/stdout
builtin           the reference provider, in-process and over the wire
```

``ProviderClient`` and ``serve`` are the provider half of the boundary, which is
why they are here rather than in ``runtime/``: the engine may import a provider,
but a provider may not import the engine, so a contract defined there could only
be duplicated on this side.

``builtin/seasonal-naive`` exists so that the engine can be proved end to end
before any external library is involved. It is a real model with a real
contract, not a stub: fit it, persist it, reload it and forecast with it, and
whatever a conformance suite later asks of Nixtla it can ask of this.
"""

from openforecast.providers.builtin import BUILTIN_PROVIDER, BuiltinProvider
from openforecast.providers.client import ProviderClient
from openforecast.providers.serve import ProviderServer, serve

__all__ = [
    "BUILTIN_PROVIDER",
    "BuiltinProvider",
    "ProviderClient",
    "ProviderServer",
    "serve",
]
