"""``ModelRef``: what a model identifier means.

```text
<namespace>/<name>[@revision]

nixtla/nhits
nixtla/autoarima
darts/nhits
local/de-price
local/de-price@01K5Z6QK3M9TQK1W2E3R4T5Y6U
```

A reference is a name, not a state. ``nixtla/nhits`` says nothing about whether
anything has been fitted — it is the registry that answers that, by resolving
the reference to either a provider-advertised descriptor or a locally fitted
artifact. Keeping the two apart is what lets the same string appear in
``of.fit(model=...)`` and ``of.forecast(model=...)`` and mean the right thing in
each: the first names what to train, the second names what to run.

The revision suffix pins one immutable artifact. An unpinned ``local/de-price``
is an alias that follows the latest selected revision, which is a different
thing to ask for and therefore a different reference.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from openforecast.errors import ModelRefError

__all__ = ["ModelRef"]

#: Lowercase words joined by a single ``.``, ``-`` or ``_``. Deliberately narrow:
#: two references differing only in case would be two names for one model.
_SEGMENT = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")

#: An artifact id is generated, so this only has to reject what cannot be one.
_REVISION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

_SYNTAX = "<namespace>/<name>[@revision]"


class ModelRef(BaseModel):
    """A parsed model reference.

    Frozen and hashable, so it can key a catalog: two references that print the
    same are the same reference.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Who provides the model — a provider name, or ``local`` for a fitted artifact.
    namespace: str
    name: str
    #: The immutable revision this reference is pinned to, if any.
    revision: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _accept_a_string(cls, value: object) -> object:
        """Let a plain ``"nixtla/nhits"`` stand in for a reference anywhere.

        Applies to every field typed ``ModelRef`` as well as to
        :meth:`parse`, so a descriptor, a recipe or a request can be written
        with the string the user would have typed.
        """
        return _split(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def _check_syntax(self) -> ModelRef:
        for part, label in ((self.namespace, "namespace"), (self.name, "name")):
            if not _SEGMENT.match(part):
                raise ModelRefError(
                    f"{part!r} is not a valid model {label}: expected lowercase letters, "
                    f"digits and single '.', '-' or '_' separators, as in 'nixtla/nhits'"
                )
        if self.revision is not None and not _REVISION.match(self.revision):
            raise ModelRefError(f"{self.revision!r} is not a valid model revision")
        return self

    @classmethod
    def parse(cls, text: str | ModelRef) -> ModelRef:
        """``ModelRef.parse("nixtla/nhits@01K...")``, or a reference unchanged."""
        return text if isinstance(text, ModelRef) else cls.model_validate(text)

    def __str__(self) -> str:
        pinned = f"@{self.revision}" if self.revision is not None else ""
        return f"{self.namespace}/{self.name}{pinned}"

    @property
    def is_pinned(self) -> bool:
        """Whether this names one immutable revision rather than an alias."""
        return self.revision is not None

    @property
    def unpinned(self) -> ModelRef:
        """The same model without its revision — the alias form."""
        return self if self.revision is None else ModelRef(namespace=self.namespace, name=self.name)

    def at_revision(self, revision: str) -> ModelRef:
        """This model pinned to ``revision``."""
        return ModelRef(namespace=self.namespace, name=self.name, revision=revision)


def _split(text: str) -> dict[str, Any]:
    """``"nixtla/nhits@01K"`` -> the three fields, or a reference error."""
    remainder, separator, revision = text.partition("@")
    if separator and "@" in revision:
        raise ModelRefError(f"{text!r} pins more than one revision; expected {_SYNTAX}")
    if separator and not revision:
        raise ModelRefError(
            f"{text!r} ends in '@' without naming a revision; drop it to mean the "
            f"latest selected revision"
        )
    namespace, separator, name = remainder.partition("/")
    if not separator:
        raise ModelRefError(
            f"{text!r} names no provider; a model reference is {_SYNTAX}, as in 'nixtla/nhits'"
        )
    if "/" in name:
        raise ModelRefError(f"{text!r} has more than one '/'; a model reference is {_SYNTAX}")
    return {"namespace": namespace, "name": name, "revision": revision or None}
