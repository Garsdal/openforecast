"""The one place the version of this distribution is written down.

``hatchling`` reads it from here, the provider reports it at the handshake, and
every artifact fitted by this integration records it — so a bump is visible in
the environment record and in the artifacts, and there is nothing to keep in
sync by hand.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
