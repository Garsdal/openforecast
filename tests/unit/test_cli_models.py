"""``openforecast models`` — the catalog, read from a shell.

The assertions are about the projection: which facts reach stdout, that the JSON
and the table are the same facts, and that a reference nobody advertises is a
failure with a non-zero exit code rather than an empty listing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import openforecast as of
from openforecast.commands import build_parser
from tests.cli import run

MODEL = "builtin/seasonal-naive"


@pytest.fixture
def store(tmp_path: Path) -> str:
    """A store of this test's own, so nothing here reads the real one."""
    return str(tmp_path / "openforecast")


def test_listing_names_every_model_this_build_can_fit(store: str) -> None:
    result = run("models", "--store", store, "list")

    assert result.code == 0
    assert result.err == ""
    assert result.out.splitlines()[0].split() == ["MODEL", "NAME", "FIT", "VIEW", "OUTPUTS"]
    assert MODEL in result.out


def test_the_json_listing_is_the_descriptors_themselves(store: str) -> None:
    """The same document ``GET /v1/models`` answers with, so there is one schema."""
    result = run("models", "--store", store, "list", "--json")

    listed = result.json["models"]
    assert listed == [descriptor.model_dump(mode="json") for descriptor in of.models.list()]


def test_getting_one_model_is_the_same_facts_as_the_listing(store: str) -> None:
    """One descriptor, whichever verb asked for it.

    ``ref`` is the parsed reference rather than the string that was typed, which
    is what the HTTP projection answers with too: the CLI prints the model
    exactly as ``GET /v1/models/{ref}`` does, so an agent reading one has read
    both.
    """
    listed = run("models", "--store", store, "list", "--json").json["models"]
    got = run("models", "--store", store, "get", MODEL, "--json").json

    assert got in listed
    assert got["ref"] == {"namespace": "builtin", "name": "seasonal-naive", "revision": None}


def test_a_human_summary_says_what_the_model_trains_on(store: str) -> None:
    result = run("models", "--store", store, "get", MODEL)

    assert result.code == 0
    assert MODEL in result.out
    assert "provider     builtin" in result.out
    assert "view         series" in result.out


def test_filtering_by_provider_keeps_only_that_provider(store: str) -> None:
    listed = run("models", "--store", store, "list", "--provider", "builtin", "--json").json

    assert listed["models"]
    assert {item["provider"] for item in listed["models"]} == {"builtin"}


def test_filtering_by_a_provider_nobody_advertises_lists_nothing(store: str) -> None:
    """An empty *filter* is empty. An unknown *reference* is an error — see below."""
    listed = run("models", "--store", store, "list", "--provider", "nixtla", "--json").json

    assert listed["models"] == []


def test_a_reference_nobody_advertises_fails_loudly(store: str) -> None:
    result = run("models", "--store", store, "get", "nixtla/nhits")

    assert result.code == 1
    assert result.out == ""
    assert "nixtla/nhits" in result.err
    assert result.err.startswith("error: ")


def test_the_verbs_are_parsed_rather_than_guessed() -> None:
    assert build_parser().parse_args(["models", "list"]).verb == "list"
    with pytest.raises(SystemExit):
        run("models", "levitate")
