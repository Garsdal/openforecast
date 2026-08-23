"""The one place the version of this distribution is written down.

``hatchling`` reads it from here and the provider reports it at the handshake,
so a bump is visible in the environment record and there is nothing to keep in
sync by hand.

Unlike the trainable integrations, no artifact records it: a zero-shot forecast
publishes nothing. What identifies the model that produced a number is the
reference — ``amazon/chronos-2`` — plus this version and the checkpoint it
pins, which is what the handshake reports.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
