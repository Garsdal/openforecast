"""Rule 7: OpenAPI is a projection of OpenForecast semantics, not their source.

```bash
uv run generate-openapi
git diff --exit-code spec/openapi/openapi.json
```

CI runs those two lines. This module is the same check as a test, plus the
things the diff alone would not catch: that the document is a pure function of
the models (no engine consulted, no provider started), that the five endpoints
the plan names are the five that exist, and that the semantic vocabulary really
did reach the schemas rather than being flattened into anonymous objects on the
way out.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from openforecast.server.openapi import SPEC_PATH, document, render, write

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_PATHS = {
    "/v1/models": {"get"},
    "/v1/models/{ref}": {"get"},
    "/v1/fit": {"post"},
    "/v1/forecast": {"post"},
    "/v1/artifacts/{ref}": {"get"},
}


@pytest.fixture(scope="module")
def spec() -> dict[str, Any]:
    return document()


def test_the_committed_document_is_the_generated_one(spec: dict[str, Any]) -> None:
    """The check CI runs, so that a stale spec fails here first.

    A document written by hand beside the code is one that drifts; a generated
    one that nobody regenerates is the same thing more slowly. Committing it and
    diffing it is what makes the dependency direction enforceable.
    """
    committed = (REPO_ROOT / SPEC_PATH).read_text(encoding="utf-8")

    assert committed == render(spec), (
        "spec/openapi/openapi.json is out of date; run: uv run generate-openapi"
    )


def test_the_endpoints_are_exactly_the_ones_step_sixteen_names(spec: dict[str, Any]) -> None:
    """Distributed asynchronous training is deliberately not among them."""
    found = {path: set(operations) for path, operations in spec["paths"].items()}

    assert found == EXPECTED_PATHS


def test_generating_the_document_executes_nothing(spec: dict[str, Any]) -> None:
    """It is a function of the route signatures and the models, and nothing else.

    ``document()`` builds the application over a transport that raises if any
    route is called, so this passing at all is the assertion; the version below
    is what would differ between two builds if it were reading the environment.
    """
    from openforecast import __version__

    assert spec["info"]["version"] == __version__
    assert spec["openapi"].startswith("3.")


def test_a_fit_request_carries_a_recipe_and_a_plan_as_themselves(spec: dict[str, Any]) -> None:
    """Control is JSON, so the OpenAPI document describes it in full."""
    schemas = spec["components"]["schemas"]
    body = schemas["FitBody"]["properties"]

    assert set(body) == {"model", "data", "horizon", "plan", "name", "params"}
    assert "FitPlan" in schemas
    assert "WindowPlan" in schemas
    assert {"Model", "Pipeline", "Ensemble", "Reduction"} <= set(schemas)


def test_bulk_data_is_one_opaque_field_rather_than_rows_of_json(spec: dict[str, Any]) -> None:
    """A hundred thousand training rows do not belong in nested JSON.

    The tables are strings here — base64 Arrow IPC — which is what lets the
    encoding of the bulk channel change to multipart later without any control
    model in the document changing.
    """
    payload = spec["components"]["schemas"]["TimeSeriesPayload"]["properties"]

    assert payload["history"]["type"] == "string"
    assert payload["data_schema"]["$ref"].endswith("/TimeSeriesSchema")


def test_every_endpoint_documents_the_failures_it_can_produce(spec: dict[str, Any]) -> None:
    """A generated SDK should report the same failures the Python one does."""
    for path, operations in spec["paths"].items():
        for method, operation in operations.items():
            declared = set(operation["responses"])
            assert {"404", "422", "502"} <= declared, f"{method.upper()} {path}"


def test_writing_the_document_is_idempotent(tmp_path: Path) -> None:
    """Same script, same bytes — which is what ``--exit-code`` depends on."""
    first = write(tmp_path).read_text(encoding="utf-8")
    second = write(tmp_path).read_text(encoding="utf-8")

    assert first == second
    assert json.loads(first)["info"]["title"] == "OpenForecast"
