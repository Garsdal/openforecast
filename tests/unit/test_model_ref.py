"""``ModelRef`` parsing: what a model identifier may and may not be.

The reference is the whole user-facing model UX, so the rules are worth pinning
down exactly. Nothing here touches a registry: a reference is a name, and
whether anything answers to it is a different question.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from openforecast import ModelRefError
from openforecast.models import ModelRef

REVISION = "01K5Z6QK3M9TQK1W2E3R4T5Y6U"


@pytest.mark.parametrize(
    ("text", "namespace", "name", "revision"),
    [
        ("nixtla/nhits", "nixtla", "nhits", None),
        ("nixtla/autoarima", "nixtla", "autoarima", None),
        ("darts/nhits", "darts", "nhits", None),
        ("local/de-price", "local", "de-price", None),
        (f"local/de-price@{REVISION}", "local", "de-price", REVISION),
        ("builtin/seasonal-naive", "builtin", "seasonal-naive", None),
        ("openforecast/auto", "openforecast", "auto", None),
        ("provider.x/model_2.b", "provider.x", "model_2.b", None),
    ],
)
def test_a_reference_parses_into_its_parts(
    text: str, namespace: str, name: str, revision: str | None
) -> None:
    ref = ModelRef.parse(text)

    assert (ref.namespace, ref.name, ref.revision) == (namespace, name, revision)
    assert str(ref) == text


@pytest.mark.parametrize(
    "text",
    [
        "nhits",  # no provider
        "nixtla/nhits/extra",  # two slashes
        "/nhits",  # empty namespace
        "nixtla/",  # empty name
        "nixtla/nhits@",  # pinned to nothing
        f"nixtla/nhits@{REVISION}@2",  # pinned twice
        "Nixtla/nhits",  # uppercase
        "nixtla/NHiTS",
        "nixtla /nhits",  # whitespace
        "nixtla/-nhits",  # leading separator
        "nixtla/nhits-",
        "nixtla/n--hits",  # doubled separator
        "",
    ],
)
def test_a_malformed_reference_is_rejected(text: str) -> None:
    with pytest.raises(ModelRefError):
        ModelRef.parse(text)


def test_a_reference_round_trips_through_its_string() -> None:
    for text in ("nixtla/nhits", f"local/de-price@{REVISION}"):
        assert ModelRef.parse(str(ModelRef.parse(text))) == ModelRef.parse(text)


def test_a_reference_is_hashable_so_it_can_key_a_catalog() -> None:
    assert ModelRef.parse("nixtla/nhits") == ModelRef.parse("nixtla/nhits")
    assert ModelRef.parse("nixtla/nhits") != ModelRef.parse("darts/nhits")

    keyed: dict[ModelRef, str] = {}
    keyed[ModelRef.parse("nixtla/nhits")] = "first"
    keyed[ModelRef.parse("nixtla/nhits")] = "second"
    keyed[ModelRef.parse("darts/nhits")] = "other"

    assert len(keyed) == 2
    assert keyed[ModelRef.parse("nixtla/nhits")] == "second"


def test_pinning_a_revision_makes_a_different_reference() -> None:
    """The alias and the revision it currently points at are not interchangeable.

    Asking for ``local/de-price`` means "whatever is selected now"; asking for
    the revision means "this one, forever". A catalog that treated them as one
    key could not tell those apart.
    """
    alias = ModelRef.parse("local/de-price")
    pinned = alias.at_revision(REVISION)

    assert not alias.is_pinned
    assert pinned.is_pinned
    assert pinned != alias
    assert pinned.unpinned == alias
    assert alias.unpinned is alias


def test_a_reference_is_frozen() -> None:
    ref = ModelRef.parse("nixtla/nhits")

    with pytest.raises(Exception, match="frozen|immutable"):
        ref.name = "autoarima"  # pyright: ignore[reportAttributeAccessIssue]


def test_a_string_stands_in_for_a_reference_wherever_one_is_expected() -> None:
    """So that a descriptor or a request can be written the way a user types it."""

    class Request(BaseModel):
        model: ModelRef

    assert Request(model="nixtla/nhits").model == ModelRef.parse("nixtla/nhits")  # pyright: ignore[reportArgumentType]

    with pytest.raises(ModelRefError):
        Request(model="nhits")  # pyright: ignore[reportArgumentType]


def test_parsing_a_reference_returns_it_unchanged() -> None:
    ref = ModelRef.parse("nixtla/nhits")

    assert ModelRef.parse(ref) is ref
