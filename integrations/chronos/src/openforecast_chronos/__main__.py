"""``python -m openforecast_chronos``: the provider, on stdin and stdout.

What the engine starts inside this integration's environment. Everything the
transport needs — parsing requests, reading view bundles, writing Arrow answers,
turning a failure into an error envelope and keeping stdout free of anything a
library prints — is the serving harness's job, so this file is two lines.
"""

from __future__ import annotations

from openforecast.providers import serve
from openforecast_chronos.provider import ChronosProvider

if __name__ == "__main__":
    raise SystemExit(serve(ChronosProvider()))
