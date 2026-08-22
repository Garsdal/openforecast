"""``python -m openforecast.providers.builtin`` — the reference provider, served.

The built-in provider is in-process by default, and it does not need to be
anything else. This entry point exists so that the *transport* has something
real to be proved against: the same provider, the same models and the same
views, reached over the subprocess protocol an external integration will use.

An integration's ``__main__`` is this file with a different import.
"""

from __future__ import annotations

from openforecast.providers.builtin.provider import BUILTIN_PROVIDER
from openforecast.providers.serve import serve

if __name__ == "__main__":
    raise SystemExit(serve(BUILTIN_PROVIDER))
