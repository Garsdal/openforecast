"""``builtin`` — the models OpenForecast ships with.

One model today, ``builtin/seasonal-naive``, and it earns its place twice over:
it is a genuinely useful baseline, and it is the provider the engine is proved
against before any external library exists. Everything an integration has to do
it does — advertise a descriptor, consume an execution view, persist state,
answer a ``ForecastView`` — under the same import restrictions.
"""

from openforecast.providers.builtin.provider import (
    BUILTIN_PROVIDER,
    PROVIDER_NAME,
    PROVIDER_VERSION,
    BuiltinProvider,
)

__all__ = ["BUILTIN_PROVIDER", "PROVIDER_NAME", "PROVIDER_VERSION", "BuiltinProvider"]
