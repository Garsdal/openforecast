"""Artifact identifiers: time-ordered, opaque, and impossible to confuse.

The two properties the store depends on are that ids sort chronologically —
listing revisions needs no manifest field — and that anything else in the models
directory is recognizably not one.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from openforecast import ArtifactError
from openforecast.artifacts import (
    ARTIFACT_ID_LENGTH,
    artifact_time,
    is_artifact_id,
    new_artifact_id,
)

MOMENT = datetime(2026, 8, 22, 11, 0, 0, tzinfo=UTC)


def test_an_id_is_26_uppercase_base32_characters() -> None:
    generated = new_artifact_id(moment=MOMENT)
    assert len(generated) == ARTIFACT_ID_LENGTH == 26
    assert generated == generated.upper()
    assert is_artifact_id(generated)


def test_ids_sort_chronologically() -> None:
    """The ordering the store lists revisions by, without reading a manifest."""
    ids = [new_artifact_id(moment=MOMENT + timedelta(seconds=step)) for step in range(5)]
    assert ids == sorted(ids)


def test_the_same_moment_still_yields_distinct_ids() -> None:
    """Two fits in the same millisecond are two artifacts, not one."""
    generated = {new_artifact_id(moment=MOMENT) for _ in range(50)}
    assert len(generated) == 50


def test_an_id_carries_the_moment_it_was_created() -> None:
    recovered = artifact_time(new_artifact_id(moment=MOMENT))
    assert recovered == MOMENT


def test_generation_is_reproducible_when_both_inputs_are_given() -> None:
    entropy = bytes(range(10))
    assert new_artifact_id(moment=MOMENT, entropy=entropy) == new_artifact_id(
        moment=MOMENT, entropy=entropy
    )


@pytest.mark.parametrize(
    "text",
    [
        "",
        "not-an-id",
        # Lowercase: two spellings of one id would be two directories on a
        # case-sensitive filesystem and one on a case-insensitive one.
        new_artifact_id(moment=MOMENT).lower(),
        new_artifact_id(moment=MOMENT)[:-1],
        new_artifact_id(moment=MOMENT) + "0",
        # The letters Crockford base32 leaves out precisely so that a
        # transcribed id cannot turn into a different valid one.
        "01ILOU" + new_artifact_id(moment=MOMENT)[6:],
        # 26 base32 characters hold 130 bits; an id has 128.
        "Z" + new_artifact_id(moment=MOMENT)[1:],
    ],
)
def test_what_is_not_an_id(text: str) -> None:
    assert not is_artifact_id(text)
    with pytest.raises(ArtifactError):
        artifact_time(text)


def test_entropy_of_the_wrong_size_is_refused() -> None:
    with pytest.raises(ArtifactError, match="bytes of entropy"):
        new_artifact_id(moment=MOMENT, entropy=b"\x00")


def test_a_moment_outside_the_encodable_range_is_refused() -> None:
    with pytest.raises(ArtifactError, match="outside the range"):
        new_artifact_id(moment=datetime(1969, 1, 1, tzinfo=UTC))
