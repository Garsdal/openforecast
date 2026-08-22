"""An installable provider: the stub, with the ``__main__`` an integration has.

```bash
python -m tests.stub_provider
```

The environment tests need a provider that is *not* the one OpenForecast ships,
because installing a second copy of ``builtin`` is refused — a provider name is
the namespace of the models it advertises, and one namespace is one provider. So
the stub of :mod:`tests.providers` grows the two lines an integration's
``__main__`` will have, and becomes something that can be installed, discovered
and executed over the wire.
"""

from __future__ import annotations

from openforecast.providers import serve
from tests.providers import StubProvider, descriptor

PROVIDER_NAME = "example"
PROVIDER_VERSION = "0.4.2"

PROVIDER = StubProvider(
    name=PROVIDER_NAME,
    version=PROVIDER_VERSION,
    models=(descriptor("echo", provider=PROVIDER_NAME),),
    value=42.0,
)

if __name__ == "__main__":
    raise SystemExit(serve(PROVIDER))
