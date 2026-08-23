"""The generated reference: current, complete, and a function of the code alone.

```bash
uv run generate-reference
git diff --exit-code docs/reference/generated
```

CI runs those two lines. This module is the same check as a test, plus the things
the diff alone would not catch: that the pages cover the whole public surface,
that generating them consults no provider catalog — so two machines with different
providers installed produce the same bytes — and that writing them twice is
idempotent, which is what ``--exit-code`` depends on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import openforecast as of
from openforecast.docs.reference import PAGES, REFERENCE_ROOT, pages, write
from openforecast.models.catalog import ModelCatalog

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def rendered() -> dict[str, str]:
    return pages()


def test_the_committed_pages_are_the_generated_ones(rendered: dict[str, str]) -> None:
    """A stale reference page fails here first, before anyone reads it."""
    directory = REPO_ROOT / REFERENCE_ROOT
    committed = {
        path.name: path.read_text(encoding="utf-8") for path in sorted(directory.glob("*.md"))
    }

    assert committed == rendered, (
        "docs/reference/generated is out of date; run: uv run generate-reference"
    )


def test_every_public_name_is_documented(rendered: dict[str, str]) -> None:
    """``__all__`` is the surface, so the reference covers exactly it."""
    documented = {
        name
        for page in PAGES
        for line in rendered[f"{page.slug}.md"].splitlines()
        if line.startswith("## `")
        for name in [line.removeprefix("## `").removesuffix("`")]
    }
    expected = set(of.__all__) - {"models", "__version__"} | set(of.models.__all__)

    assert documented == expected


def test_the_index_names_every_page_and_every_name(rendered: dict[str, str]) -> None:
    index = rendered["index.md"]

    assert of.__version__ in index
    for page in PAGES:
        assert f"({page.slug}.md)" in index
    for name in of.__all__:
        if name in {"models", "__version__"}:
            continue
        assert f"| `{name}` |" in index


def test_generating_the_pages_lists_no_model_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pages are a function of the types, never of what is installed.

    A page built from ``of.models.list()`` would differ between two machines and
    could not be diffed, so the generator must not consult it — asserted by making
    the call fail rather than by reading the output and hoping.
    """

    def refuse(self: object, *_: object, **__: object) -> None:
        """Every model, sorted by reference."""
        raise AssertionError("generating the reference must not list the model catalog")

    monkeypatch.setattr(ModelCatalog, "list", refuse)

    assert pages()


def test_the_pages_describe_descriptor_types_rather_than_a_catalog(
    rendered: dict[str, str],
) -> None:
    """What a machine has installed is not documentation; the vocabulary is."""
    models = rendered["models.md"]

    for vocabulary in ("ModelDescriptor", "TrainingContract", "ModelCapabilities", "ModelRef"):
        assert f"## `{vocabulary}`" in models


def test_writing_is_idempotent_and_removes_stale_pages(tmp_path: Path) -> None:
    """Same generator, same bytes — and a page that is no longer generated goes."""
    first = {path.name: path.read_text(encoding="utf-8") for path in write(tmp_path)}
    stale = tmp_path / REFERENCE_ROOT / "gone.md"
    stale.write_text("# a page that used to be generated\n", encoding="utf-8")

    second = {path.name: path.read_text(encoding="utf-8") for path in write(tmp_path)}

    assert first == second
    assert not stale.exists()


def test_a_pydantic_model_is_documented_by_its_fields(rendered: dict[str, str]) -> None:
    """Not by a generated ``__init__``, which would name every field twice."""
    tasks = rendered["tasks.md"]

    assert "## `WindowPlan`" in tasks
    assert "| `context` | `int` | *required* |" in tasks
    assert "How much history one training sample conditions on." in tasks


def test_an_error_documents_the_code_a_caller_branches_on(rendered: dict[str, str]) -> None:
    errors = rendered["errors.md"]

    assert "`error.code` is `MODEL_DOES_NOT_SUPPORT_FIT`." in errors
    assert "`error.code` is `ORIGIN_SCOPE_ERROR`." in errors


def test_a_class_documents_the_methods_it_defines(rendered: dict[str, str]) -> None:
    """Including the four operations, which is what a client is for."""
    client = rendered["client.md"]

    for operation in ("fit", "forecast", "backtest", "eligible_models"):
        assert f"| `{operation}(self, " in client


def test_a_discriminated_union_documents_its_members(rendered: dict[str, str]) -> None:
    """``of.Recipe`` is one of four nodes, and the page says which four."""
    recipes = rendered["recipes.md"]

    assert "One of: `Model`, `Pipeline`, `Ensemble`, `Reduction`." in recipes
