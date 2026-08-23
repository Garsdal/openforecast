"""The agent-facing interface over the documentation: `llms.txt`, and `.md` sources.

Step 29. `docs/llms.txt` is generated and committed, so this suite is the same
arrangement CI uses for the reference pages and the JSON Schemas: the committed
bytes have to be the generated ones, and a page that is added, moved or retitled
in the nav is a diff here rather than a map that quietly points at nothing.

What the map *says* is checked too. A generated index whose every line said
"page" would regenerate cleanly and be useless, so every nav page has to appear,
every link has to resolve to a file in this repository, and every summary has to
be a sentence off the page it describes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openforecast.docs import llms
from tests.docs.blocks import DOCS_ROOT, REPO_ROOT

CONFIG = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
SITE = llms.site_from(CONFIG)
ENTRIES = llms.entries(CONFIG)


def pages() -> tuple[tuple[llms.Entry, str], ...]:
    site, pages = llms.read_site(REPO_ROOT)
    assert site == SITE
    return pages


def test_the_committed_map_is_the_generated_one() -> None:
    """`uv run generate-llms` in CI, diffed — asserted here without running git."""
    committed = (REPO_ROOT / llms.INDEX_PATH).read_text(encoding="utf-8")

    assert committed == llms.index(SITE, pages()), (
        "docs/llms.txt is stale; run `uv run generate-llms`"
    )


def test_writing_the_map_is_idempotent(tmp_path: Path) -> None:
    """Otherwise the CI diff would fail on a machine rather than on a change."""
    root = tmp_path / "repo"
    (root / DOCS_ROOT.name).mkdir(parents=True)
    (root / "mkdocs.yml").write_text(CONFIG, encoding="utf-8")
    for entry, page in pages():
        target = entry.source(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(page, encoding="utf-8")

    first = llms.write(root)
    text = first.read_text(encoding="utf-8")

    assert llms.write(root).read_text(encoding="utf-8") == text
    assert text == (REPO_ROOT / llms.INDEX_PATH).read_text(encoding="utf-8")


def test_the_nav_parser_reads_sections_and_pages() -> None:
    """A parser that found nothing would look like a suite that passed."""
    parsed = llms.entries(
        "site_name: Example\n"
        "nav:\n"
        "  - Home: index.md\n"
        "  - Guides:\n"
        "      - Fitting: guides/fitting.md\n"
        "      - Forecasting: guides/forecasting.md\n"
        "\n"
        "extra:\n"
        "  - Not: a-page.md\n"
    )

    assert [(entry.section, entry.title, entry.slug) for entry in parsed] == [
        (llms.UNSECTIONED, "Home", "index.md"),
        ("Guides", "Fitting", "guides/fitting.md"),
        ("Guides", "Forecasting", "guides/forecasting.md"),
    ]


def test_a_nav_with_no_pages_is_an_error() -> None:
    """A map generated from an unreadable nav would be an empty file, silently."""
    with pytest.raises(ValueError, match="lists no pages"):
        llms.entries("site_name: Example\nnav:\n")


def test_a_config_missing_a_site_field_is_an_error() -> None:
    with pytest.raises(ValueError, match="site_url"):
        llms.site_from("site_name: Example\nsite_description: A site\n")


def test_every_page_in_the_nav_appears_in_the_map() -> None:
    text = (REPO_ROOT / llms.INDEX_PATH).read_text(encoding="utf-8")

    missing = [entry.slug for entry in ENTRIES if SITE.link(entry.slug) not in text]

    assert not missing, f"pages in the nav that the map does not name: {missing}"


def test_every_section_of_the_nav_is_a_heading_in_the_map() -> None:
    text = (REPO_ROOT / llms.INDEX_PATH).read_text(encoding="utf-8")

    for section in dict.fromkeys(entry.section for entry in ENTRIES):
        assert f"\n## {section}\n" in text


def test_every_link_in_the_map_resolves_to_a_file_in_this_repository() -> None:
    """The map is only useful if its addresses are the site's real ones."""
    text = (REPO_ROOT / llms.INDEX_PATH).read_text(encoding="utf-8")
    prefix = f"{SITE.url.rstrip('/')}/"
    broken: list[str] = []
    for line in text.splitlines():
        if not line.startswith("- ["):
            continue
        target = line.split("](", 1)[1].split(")", 1)[0]
        if not target.startswith(prefix):
            continue
        page = target[len(prefix) :]
        if page == llms.FULL_NAME:  # assembled at build time, not committed
            continue
        if not (REPO_ROOT / DOCS_ROOT / page).is_file():
            broken.append(target)

    assert not broken, f"the map points at pages that do not exist: {broken}"


@pytest.mark.parametrize(("entry", "page"), pages(), ids=[entry.slug for entry in ENTRIES])
def test_every_summary_is_a_sentence_from_its_page(entry: llms.Entry, page: str) -> None:
    """Not a heading, not a fence, not the generated-file notice, and not empty."""
    found = llms.summary(page)

    assert len(found) > 20, f"{entry.slug}: {found!r} is not a description"
    assert found in " ".join(page.split()), f"{entry.slug}: {found!r} is not on the page"
    assert not found.startswith(("#", "|", "```", "*Generated"))
    assert found.endswith((".", "!", "?"))


def test_a_paragraph_that_introduces_a_block_is_not_the_summary() -> None:
    """A colon says "look below", which a reader deciding what to fetch cannot."""
    found = llms.summary(
        "# Title\n\nThe loop this exists for:\n\n```text\na -> b\n```\n\nIt is a loop.\n"
    )

    assert found == "It is a loop."


def test_the_corpus_holds_every_page_whole() -> None:
    """`llms-full.txt` is one fetch instead of thirty, so nothing may be summarized."""
    corpus = llms.full(SITE, pages())

    for entry, page in pages():
        assert (llms.DOCS_ROOT / entry.path).as_posix() in corpus
        assert page.strip() in corpus


def test_the_built_site_gets_the_corpus_and_the_markdown_sources(tmp_path: Path) -> None:
    """Step 29.1: the source of a page is served beside the page.

    This is what `docs_hooks.py` calls, so a reader fetching
    `.../guides/fitting.md` gets Markdown rather than a themed application.
    """
    site_dir = tmp_path / "site"
    site_dir.mkdir()

    written = llms.write_site(site_dir, REPO_ROOT)

    assert (site_dir / llms.FULL_NAME).read_text(encoding="utf-8").startswith("# OpenForecast")
    for entry, page in pages():
        assert (site_dir / entry.slug).read_text(encoding="utf-8") == page
    assert len(written) == len(ENTRIES) + 1


def test_the_hook_the_site_build_runs_is_the_one_the_config_names() -> None:
    """A hook the config does not name is a hook that never runs."""
    assert "docs_hooks.py" in CONFIG
    assert (REPO_ROOT / "docs_hooks.py").is_file()
