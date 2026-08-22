"""The protocol version, and what a change to it means.

One integer, bumped when a message or a bundle a provider exchanges with
OpenForecast changes shape in a way an older reader cannot handle. It is stamped
into every artifact manifest, so an artifact written by a future OpenForecast is
refused with an explanation rather than misread — the provider directory inside
it is opaque, and guessing at its layout is exactly the mistake worth making
impossible.

It lives here rather than in Step 9's transport because the artifact lifecycle
needs it first, and both have to mean the same number.
"""

from __future__ import annotations

__all__ = ["PROTOCOL_VERSION"]

#: The wire and on-disk protocol this build speaks.
PROTOCOL_VERSION = 1
