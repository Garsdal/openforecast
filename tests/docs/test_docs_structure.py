"""The site's shape: every page in the nav, every link resolving.

`mkdocs build --strict` in CI says the same thing about the built site; this says
it without a static-site generator installed, so a broken link fails the ordinary
test run too. The nav is read as text rather than as YAML on purpose — the docs
group is not part of `uv sync`, and a structural check that needed a parser from
it would only run where the site is built.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.docs.blocks import DOCS_ROOT, GENERATED_ROOT, REPO_ROOT, blocks_in, pages_under

MKDOCS = REPO_ROOT / "mkdocs.yml"

_NAV_ENTRY = re.compile(r":\s*(?P<path>[\w./-]+\.md)\s*$")
#: A Markdown link to somewhere in this repository: `[text](path.md#anchor)`.
_LINK = re.compile(r"\[[^\]]*\]\((?P<target>[^)\s]+)\)")


def nav_pages() -> tuple[Path, ...]:
    lines = MKDOCS.read_text(encoding="utf-8").splitlines()
    return tuple(
        DOCS_ROOT / match.group("path")
        for line in lines
        if (match := _NAV_ENTRY.search(line)) is not None
    )


def all_pages() -> tuple[Path, ...]:
    return tuple(sorted(DOCS_ROOT.rglob("*.md")))


def test_the_nav_and_the_pages_are_the_same_set() -> None:
    """A page not in the nav is a page nobody finds; a nav entry with no page is a 404."""
    listed = set(nav_pages())
    present = set(all_pages())

    def relative(paths: set[Path]) -> list[str]:
        return sorted(str(path.relative_to(REPO_ROOT)) for path in paths)

    assert listed == present, (
        "docs/ and the mkdocs nav disagree:\n"
        f"  in the nav, not on disk: {relative(listed - present)}\n"
        f"  on disk, not in the nav: {relative(present - listed)}"
    )


def test_the_nav_lists_every_page_once() -> None:
    entries = nav_pages()

    assert len(entries) == len(set(entries))


def test_the_four_purposes_each_have_pages() -> None:
    """Step 25.1: tutorials, guides, concepts and reference are separate sections.

    Separate because they answer different questions — "run this", "how do I do
    X?", "why does it work this way?", "what exactly is the signature?" — and a
    page that answers two of them answers neither well.
    """
    for section in ("getting-started", "guides", "concepts", "integrations"):
        assert list((DOCS_ROOT / section).glob("*.md")), f"docs/{section} is empty"

    assert (GENERATED_ROOT / "index.md").is_file()


def test_the_agent_section_answers_its_four_questions() -> None:
    """Step 29.3: discover, choose, construct, recover — and nothing narrative.

    Listed as pages rather than checked as prose, because the section's value is
    that a reader with a task lands on one page rather than reading five.
    """
    section = DOCS_ROOT / "agents"
    expected = {
        "overview.md",
        "choosing-a-model.md",
        "point-in-time.md",
        "structured-cli.md",
        "errors.md",
    }

    assert {path.name for path in section.glob("*.md")} == expected

    overview = (section / "overview.md").read_text(encoding="utf-8")
    for question in ("discover", "choose", "construct", "recover"):
        assert question in overview, f"the overview does not say how to {question}"


@pytest.mark.parametrize("page", all_pages(), ids=lambda path: str(path.relative_to(REPO_ROOT)))
def test_every_relative_link_resolves(page: Path) -> None:
    """Link checking, without the network: only links into the repository."""
    broken: list[str] = []
    for match in _LINK.finditer(page.read_text(encoding="utf-8")):
        target = match.group("target")
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        path, _, _anchor = target.partition("#")
        if not path:
            continue
        if not (page.parent / path).resolve().exists():
            broken.append(target)

    assert not broken, f"{page.relative_to(REPO_ROOT)} links to nothing: {broken}"


def test_the_hand_written_pages_do_not_retype_a_signature() -> None:
    """Step 25.2: reference material is generated, so prose never duplicates it.

    A hand-written page may of course *call* the API. What it may not do is
    restate a definition — a ``def`` or a ``class`` from the public surface — since
    that is the thing the generated reference is the single source of.
    """
    definitions = [
        f"{block.location}: {line.strip()}"
        for page in pages_under()
        for block in blocks_in(page)
        if block.is_python
        for line in block.source.splitlines()
        if line.startswith(("def ", "class "))
    ]
    assert not definitions, "signatures belong in the generated reference:\n" + "\n".join(
        definitions
    )
