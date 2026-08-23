"""``openforecast schema`` — the discovery half of Step 27.

> inspect schema -> construct request -> execute

So the assertions follow that loop: the command answers with the committed
document, the document describes the request the command next to it validates,
and a request built from it is accepted. Plus the CLI's own contract, which is the
one every command here keeps: stdout is the answer, and ``--json`` is parseable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from openforecast.docs.schemas import SCHEMAS, document, find
from tests.cli import run, write_config


def test_every_schema_the_build_has_is_reachable_by_name() -> None:
    """The names are the discovery surface, so all of them answer."""
    for schema in SCHEMAS:
        result = run("schema", schema.name, "--json")

        assert result.code == 0
        assert result.json["$id"] == schema.filename


def test_the_json_is_the_committed_document() -> None:
    """One schema, whether it is read from the checkout or asked of the build."""
    result = run("schema", "fit", "--json")

    assert result.json == json.loads(json.dumps(document(find("fit"))))


def test_the_human_rendering_lists_the_fields_and_says_which_are_required() -> None:
    result = run("schema", "forecast")

    assert result.code == 0
    assert "forecast  Forecast request" in result.out
    assert "spec/schemas/forecast-request.json" in result.out
    lines = [line.split() for line in result.out.splitlines() if line.startswith(("model", "hor"))]
    assert ["model", "yes", "string"] in lines
    assert ["horizon", "yes", "integer"] in lines


def test_a_union_says_what_it_is_one_of() -> None:
    """A recipe has no fields of its own; the alternatives are the answer."""
    result = run("schema", "recipe")

    assert result.code == 0
    assert "one of: Model, Pipeline, Ensemble, Reduction" in result.out


def test_an_unknown_schema_is_refused_by_the_parser() -> None:
    """A closed set of names, so a typo is a usage error listing the real ones."""
    with pytest.raises(SystemExit):
        run("schema", "fitt")


def test_a_request_built_from_the_schema_is_accepted(tmp_path: Path) -> None:
    """The loop, closed: read the schema, write a request, run the command.

    The request below names only fields the schema declared, and the fit is
    refused for the one thing the schema cannot promise — a dataset that is not
    on this disk — rather than for a field the command did not recognize.
    """
    fields = run("schema", "fit", "--json").json["properties"]
    payload: dict[str, Any] = {
        "model": "builtin/seasonal-naive",
        "data": str(tmp_path / "dataset"),
        "horizon": 24,
    }
    assert set(payload) <= set(fields)

    result = run("fit", "--config", write_config(tmp_path / "fit.json", payload), "--json")

    assert result.code == 1
    assert json.loads(result.err)["error"]["code"] == "INVALID_DATA"
