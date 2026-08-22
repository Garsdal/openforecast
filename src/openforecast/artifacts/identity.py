"""Artifact identifiers: sortable, opaque, generated once.

```text
01K5Z6QK3M9TQK1W2E3R4T5Y6U
└──────────┬─────────┘└─┬─┘
   48-bit millisecond   80 bits of randomness
```

A ULID, in Crockford base32. Three properties are why:

*Time-ordered.* Lexicographic order is chronological order, so listing a
directory of artifacts lists them oldest first without reading a manifest, and
"the newest revision of this name" is a string comparison rather than a
timestamp field somebody could disagree with.

*Not a content hash.* Two fits of the same recipe on the same data are two
artifacts, because a fit is an event. What was fitted is recorded — and hashed —
in the manifest; the identity says *which fit*.

*Opaque and filename-safe.* It is a directory name, so it may not carry a slash,
a case-collapsing pair on a case-insensitive filesystem, or a character a shell
would eat. Crockford base32 is uppercase-only and excludes ``I``, ``L``, ``O``
and ``U``, which is also what keeps a transcribed id from turning into a
different valid one.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime

from openforecast.errors import ArtifactError

__all__ = ["ARTIFACT_ID_LENGTH", "artifact_time", "is_artifact_id", "new_artifact_id"]

#: Crockford base32: no ``I``, ``L``, ``O`` or ``U``, so no two ids differ only
#: by a character a human would read the same way.
_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_DECODE = {character: value for value, character in enumerate(_ALPHABET)}

_TIME_BITS = 48
_RANDOM_BITS = 80
_RANDOM_BYTES = _RANDOM_BITS // 8

#: 128 bits in base32, and the length every id has.
ARTIFACT_ID_LENGTH = (_TIME_BITS + _RANDOM_BITS) // 5 + 1


def new_artifact_id(*, moment: datetime | None = None, entropy: bytes | None = None) -> str:
    """A fresh identifier for one fit.

    Both inputs are injectable so that a test can pin an id without patching the
    clock — nothing else should pass them.
    """
    moment = datetime.now(tz=UTC) if moment is None else moment
    entropy = secrets.token_bytes(_RANDOM_BYTES) if entropy is None else entropy
    if len(entropy) != _RANDOM_BYTES:
        raise ArtifactError(f"an artifact id takes {_RANDOM_BYTES} bytes of entropy")
    milliseconds = int(moment.timestamp() * 1000)
    if not 0 <= milliseconds < 1 << _TIME_BITS:
        raise ArtifactError(f"{moment.isoformat()} is outside the range an artifact id can encode")
    return _encode((milliseconds << _RANDOM_BITS) | int.from_bytes(entropy, "big"))


def is_artifact_id(text: str) -> bool:
    """Whether ``text`` could have been produced by :func:`new_artifact_id`.

    What the store uses to tell an artifact directory from anything else that
    ended up beside it, so a stray file is skipped rather than half-read.
    """
    return (
        len(text) == ARTIFACT_ID_LENGTH
        and all(character in _DECODE for character in text)
        # 26 base32 characters hold 130 bits; the top three must be zero.
        and _DECODE[text[0]] < 8
    )


def artifact_time(artifact_id: str) -> datetime:
    """When the artifact was created, read back out of its identifier."""
    if not is_artifact_id(artifact_id):
        raise ArtifactError(f"{artifact_id!r} is not an artifact id")
    value = 0
    for character in artifact_id:
        value = value * 32 + _DECODE[character]
    return datetime.fromtimestamp((value >> _RANDOM_BITS) / 1000, tz=UTC)


def _encode(value: int) -> str:
    characters = [""] * ARTIFACT_ID_LENGTH
    for position in reversed(range(ARTIFACT_ID_LENGTH)):
        characters[position] = _ALPHABET[value & 0x1F]
        value >>= 5
    return "".join(characters)
