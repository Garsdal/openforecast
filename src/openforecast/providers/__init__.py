"""Providers that execute models, starting with the built-in reference one.

A provider is the only thing in OpenForecast that touches a forecasting
implementation, and it sees exactly one kind of data: execution views. That is
the boundary rule the whole architecture rests on, so the built-in provider is
held to it as strictly as an external integration would be — its import surface
is :mod:`openforecast.views`, :mod:`openforecast.errors`,
:mod:`openforecast.protocol` and :mod:`openforecast.models`, the last of which
is how it says what it provides.

``builtin/seasonal-naive`` exists so that the engine can be proved end to end
before any external library is involved. It is a real model with a real
contract, not a stub: fit it, persist it, reload it and forecast with it, and
whatever a conformance suite later asks of Nixtla it can ask of this.
"""

from openforecast.providers.builtin import BUILTIN_PROVIDER, BuiltinProvider

__all__ = ["BUILTIN_PROVIDER", "BuiltinProvider"]
