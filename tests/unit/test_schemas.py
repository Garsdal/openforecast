"""Step 27.1: every executable request has a schema, generated and committed.

```bash
uv run generate-schemas
git diff --exit-code spec/schemas
```

CI runs those two lines. This module is the same check as a test, plus what the
diff alone would not catch: that the documents are a pure function of the types —
so two machines with different providers installed produce the same bytes — that
a schema really describes the request the command validates rather than something
adjacent to it, and that an agent following the schema can construct a request
that is accepted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from openforecast.commands import config as configs
from openforecast.docs.schemas import SCHEMA_ROOT, SCHEMAS, document, find, render, write

REPO_ROOT = Path(__file__).resolve().parents[2]

#: What Step 27.1 asks to be generated, as the files it asks for.
EXPECTED_FILES = {
    "fit-request.json",
    "forecast-request.json",
    "backtest-request.json",
    "model-recipe.json",
    "fit-plan.json",
    "output-spec.json",
    "time-series-schema.json",
    "model-descriptor.json",
    "error.json",
}


def test_the_committed_documents_are_the_generated_ones() -> None:
    """A stale schema fails here first, before an agent constructs a request from it."""
    directory = REPO_ROOT / SCHEMA_ROOT
    committed = {
        path.name: path.read_text(encoding="utf-8") for path in sorted(directory.glob("*.json"))
    }
    generated = {schema.filename: render(schema) for schema in SCHEMAS}

    assert committed == generated, "spec/schemas is out of date; run: uv run generate-schemas"


def test_the_documents_are_the_ones_the_step_names() -> None:
    assert {schema.filename for schema in SCHEMAS} == EXPECTED_FILES


def test_names_and_slugs_are_both_unique() -> None:
    """One is typed at a command line and one is a filename; neither may collide."""
    assert len({schema.name for schema in SCHEMAS}) == len(SCHEMAS)
    assert len({schema.slug for schema in SCHEMAS}) == len(SCHEMAS)


def test_every_document_declares_its_dialect_and_identity() -> None:
    for schema in SCHEMAS:
        body = document(schema)

        assert body["$schema"].startswith("https://json-schema.org/")
        assert body["$id"] == schema.filename
        assert body["title"] == schema.title
        assert body["description"] == schema.summary


def test_a_fit_request_is_the_config_the_command_validates() -> None:
    """Not a document beside it: the schema is generated from that same model.

    So an agent that reads the schema, writes a request and runs
    ``openforecast fit --config`` cannot be refused by a field the schema
    promised — which is the loop 27.2 exists to close.
    """
    body = document(find("fit"))
    payload = {
        "model": "builtin/seasonal-naive",
        "data": "./dataset",
        "horizon": 24,
        "plan": {"window": {"context": 168}},
    }

    assert set(body["properties"]) == set(configs.FitConfig.model_fields)
    assert set(body["required"]) == {"model", "data"}
    assert configs.validate(payload, configs.FitConfig, source="the schema").horizon == 24


def test_a_backtest_request_is_described_where_http_cannot_describe_it() -> None:
    """The one operation with no endpoint, so the OpenAPI document has no body for it."""
    body = document(find("backtest"))

    assert set(body["required"]) == {"models", "data", "validation", "metrics"}
    assert "Validation" in str(body["$defs"].keys())


def test_the_error_envelope_is_the_three_fields() -> None:
    """One envelope, and this is the document that says so to a machine."""
    body = document(find("error"))
    info = body["$defs"]["ErrorInfo"]["properties"]

    assert set(body["properties"]) == {"error"}
    assert {"code", "message", "details"} <= set(info)


def test_a_union_is_described_as_its_alternatives() -> None:
    """A recipe is a discriminated union, and the discriminator survives generation."""
    body = document(find("recipe"))

    assert set(body["$defs"]) >= {"Model", "Pipeline", "Ensemble", "Reduction"}
    assert body["discriminator"]["propertyName"] == "kind"


def test_generating_a_document_consults_no_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    """A schema is a function of the types, so what is installed cannot change it."""
    from openforecast.models.catalog import ModelCatalog

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("generating a schema must not list a model catalog")

    monkeypatch.setattr(ModelCatalog, "list", refuse)

    assert document(find("model"))["title"] == "Model descriptor"


def test_writing_the_documents_is_idempotent(tmp_path: Path) -> None:
    """Same script, same bytes — which is what ``--exit-code`` depends on."""
    first = {path.name: path.read_text(encoding="utf-8") for path in write(tmp_path)}
    second = {path.name: path.read_text(encoding="utf-8") for path in write(tmp_path)}

    assert first == second
    assert set(first) == EXPECTED_FILES
    assert json.loads(first["fit-request.json"])["title"] == "Fit request"


def test_a_document_that_is_no_longer_generated_is_removed(tmp_path: Path) -> None:
    """A stale schema is worse than a missing one: it looks current."""
    directory = tmp_path / SCHEMA_ROOT
    directory.mkdir(parents=True)
    stale = directory / "fit-request-v0.json"
    stale.write_text("{}", encoding="utf-8")

    write(tmp_path)

    assert not stale.exists()


def test_an_unknown_name_is_a_miss_rather_than_something_close() -> None:
    with pytest.raises(KeyError):
        find("fitt")
