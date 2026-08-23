"""``/llms.txt``, and the Markdown behind every page, generated from the nav.

```bash
uv run generate-llms
git diff --exit-code docs/llms.txt
```

Step 29. The documentation of Step 25 is the documentation; this is a second
*interface* over it, for the reader that arrives with a fetch tool rather than a
browser. Three things make that reader's job possible:

```text
/llms.txt          the map: every page, its section, and one sentence about it
/llms-full.txt     the whole corpus in one response, for a single fetch
/<page>.md         the Markdown source of any page, served beside its HTML
```

None of it is written by hand. The map is read off the `nav` in `mkdocs.yml` and
the first sentence of each page, so a page that is added, moved, retitled or
rewritten moves in `llms.txt` too — the alternative is a curated index that is
accurate on the day it is written and misleading afterwards, which is worse than
no index at all for a reader that cannot see the site to check.

`llms.txt` is committed and diffed in CI, like the generated reference and the
JSON Schemas: it is small, and what it says about the shape of the documentation
is worth reading in a pull request. `llms-full.txt` is not committed, because it
is a byte-for-byte concatenation of files that are already in this repository —
committing it would double the diff of every documentation change to say nothing
new. It is assembled into the built site by ``docs_hooks.py``, together with the
Markdown sources, which is also where they are *served* from.

The `nav` is read as text rather than as YAML on purpose, for the reason
``tests/docs/test_docs_structure.py`` gives: the docs dependency group is not
part of ``uv sync``, so anything that needed a parser from it could only run
where the site is built, and this has to run wherever the tests do.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "DOCS_ROOT",
    "INDEX_PATH",
    "Entry",
    "Site",
    "entries",
    "index",
    "full",
    "main",
    "read_site",
    "site_from",
    "summary",
    "write",
    "write_site",
]

#: The documentation sources, relative to the repository root.
DOCS_ROOT = Path("docs")
#: The committed map, which mkdocs copies to the site root verbatim.
INDEX_PATH = DOCS_ROOT / "llms.txt"
#: The single-fetch corpus, assembled into the built site rather than committed.
FULL_NAME = "llms-full.txt"

#: A page with no section above it in the nav still needs a heading to live under.
UNSECTIONED = "Overview"
#: Where the things a site cannot serve — the code, the specs, the examples — live.
_REPO = "https://github.com/Garsdal/openforecast"

_SITE_FIELD = re.compile(r"^(?P<key>site_name|site_description|site_url):\s*(?P<value>.+?)\s*$")
_NAV_START = re.compile(r"^nav:\s*$")
_NAV_ENTRY = re.compile(r"^(?P<indent> +)- (?P<title>[^:]+):\s*(?P<value>.*?)\s*$")
_FENCE = re.compile(r"^\s*```")
#: A whole paragraph in emphasis — the "generated, do not edit" notice — is a
#: statement about the page rather than the sentence the page opens with.
_ASIDE = re.compile(r"^\*.*\*$")


@dataclass(frozen=True)
class Site:
    """What ``mkdocs.yml`` says the site is, for the header of the map."""

    name: str
    description: str
    url: str

    def link(self, target: str) -> str:
        """The published address of one file in the site."""
        return f"{self.url.rstrip('/')}/{target}"


@dataclass(frozen=True)
class Entry:
    """One page of the nav: where it sits, what it is called, where it lives."""

    section: str
    title: str
    #: The page's path relative to :data:`DOCS_ROOT`, which is also its address
    #: within the built site: mkdocs copies ``docs/`` to the site root.
    path: Path

    @property
    def slug(self) -> str:
        return self.path.as_posix()

    def source(self, root: Path) -> Path:
        """Where the page is read from, under a repository root."""
        return root / DOCS_ROOT / self.path


def site_from(config: str) -> Site:
    """The site fields of a ``mkdocs.yml`` document."""
    found = {
        match.group("key"): match.group("value")
        for line in config.splitlines()
        if (match := _SITE_FIELD.match(line)) is not None
    }
    missing = {"site_name", "site_description", "site_url"} - found.keys()
    if missing:
        raise ValueError(f"mkdocs.yml declares no {', '.join(sorted(missing))}")
    return Site(
        name=found["site_name"],
        description=found["site_description"],
        url=found["site_url"],
    )


def entries(config: str) -> tuple[Entry, ...]:
    """Every page of the nav, in the order a reader meets it.

    A line with a title and no value opens a section; the pages indented under it
    belong to it. The order is the nav's own, so the map is the site's table of
    contents rather than a second opinion about how to read it.
    """
    found: list[Entry] = []
    section = UNSECTIONED
    section_indent = 0
    in_nav = False
    for line in config.splitlines():
        if _NAV_START.match(line):
            in_nav = True
            continue
        if not in_nav:
            continue
        match = _NAV_ENTRY.match(line)
        if match is None:
            # A blank line inside the nav is nothing; anything else at column
            # zero is the next top-level key, and the nav has ended.
            if line.strip() and not line.startswith(" "):
                break
            continue
        indent = len(match.group("indent"))
        title = match.group("title").strip()
        value = match.group("value")
        if not value:
            section, section_indent = title, indent
            continue
        if indent <= section_indent:
            section, section_indent = UNSECTIONED, indent
        found.append(Entry(section=section, title=title, path=Path(value)))
    if not found:
        raise ValueError("mkdocs.yml lists no pages; the nav could not be read")
    return tuple(found)


def summary(page: str) -> str:
    """The first sentence of the first paragraph of prose on a page.

    Headings, fenced code, tables, lists, the generated-file notice and the HTML
    comments that mark an example as unrunnable are all skipped: what is wanted is
    the sentence the author wrote to say what the page is about, which is the one
    a reader deciding whether to fetch it needs.

    A paragraph ending in a colon is skipped too. It introduces the block after
    it rather than saying anything on its own, and "The loop the CLI exists for:"
    is not a description of a page.
    """
    lines = page.splitlines()
    start = next((index + 1 for index, line in enumerate(lines) if line.startswith("# ")), 0)
    paragraph: list[str] = []
    index = start
    while index < len(lines):
        line = lines[index].strip()
        if _FENCE.match(line):
            index += 1
            while index < len(lines) and not _FENCE.match(lines[index]):
                index += 1
            index += 1
            continue
        skipped = (
            not line
            or line.startswith(("#", "|", "<!--", "- ", "* ", "> ", "!!!"))
            or _ASIDE.match(line) is not None
        )
        if skipped:
            if paragraph and not paragraph[-1].endswith(":"):
                break
            paragraph.clear()
            index += 1
            continue
        paragraph.append(line)
        index += 1
    return _first_sentence(" ".join(paragraph))


#: A "sentence" this short is a version number or an abbreviation — "OpenForecast
#: 0.1.0." — so the next one is taken as well rather than published as a summary.
_TOO_SHORT = 24


def _first_sentence(text: str) -> str:
    """The opening sentence of a paragraph, or as many as it takes to say something."""
    found = ""
    for match in re.finditer(r"(?s).+?(?:[.!?](?=\s|$)|$)", text):
        found = text[: match.end()].strip()
        if len(found) > _TOO_SHORT:
            break
    return found


def index(site: Site, pages: tuple[tuple[Entry, str], ...]) -> str:
    """The map: the site, then one line per page under its section heading."""
    lines = [
        f"# {site.name}",
        "",
        f"> {site.description}. Point-in-time forecasting is first-class: real",
        "> historical forecast vintages are trained on directly, and an artifact",
        "> records whether its origins were observed or simulated.",
        "",
        "One name per intent — `fit`, `forecast`, `backtest`, `eligible_models` — over",
        "any provider, any transport and both semantic data sources. Every page below",
        "is served as Markdown at the address shown, and every Python example on one is",
        "executed by the test suite.",
    ]
    current: str | None = None
    for entry, page in pages:
        if entry.section != current:
            current = entry.section
            lines += ["", f"## {current}", ""]
        lines.append(f"- [{entry.title}]({site.link(entry.slug)}): {summary(page)}")
    lines += [
        "",
        "## Optional",
        "",
        f"- [Everything above, in one file]({site.link(FULL_NAME)}): the whole "
        "corpus concatenated, for a single fetch.",
        f"- [Repository]({_REPO}): the library, the integrations, the executable "
        "examples under `examples/`, and the seven architecture rules.",
        f"- [OpenAPI document]({_REPO}/blob/main/spec/openapi/openapi.json): the HTTP "
        "projection, generated from the request models.",
        f"- [JSON Schemas]({_REPO}/tree/main/spec/schemas): what each request has to "
        "look like, generated from the models that validate it.",
        "",
    ]
    return "\n".join(lines)


def full(site: Site, pages: tuple[tuple[Entry, str], ...]) -> str:
    """Every page, concatenated, each under the path it came from."""
    lines = [
        f"# {site.name} — the complete documentation",
        "",
        f"> {site.description}",
        "",
        f"{len(pages)} pages, in the order of the site's navigation. Each is preceded by "
        "the path it is generated from and served at.",
        "",
    ]
    for entry, page in pages:
        lines += [
            "---",
            "",
            f"<!-- {entry.section} / {entry.title} — {(DOCS_ROOT / entry.path).as_posix()} -->",
            "",
            page.strip(),
            "",
        ]
    return "\n".join(lines)


def read_site(root: Path | None = None) -> tuple[Site, tuple[tuple[Entry, str], ...]]:
    """``mkdocs.yml`` and every page it lists, read once for both documents."""
    base = Path.cwd() if root is None else root
    config = (base / "mkdocs.yml").read_text(encoding="utf-8")
    site = site_from(config)
    pages = tuple(
        (entry, entry.source(base).read_text(encoding="utf-8")) for entry in entries(config)
    )
    return site, pages


def write(root: Path | None = None) -> Path:
    """Write ``docs/llms.txt``, and return where it went."""
    base = Path.cwd() if root is None else root
    site, pages = read_site(base)
    path = base / INDEX_PATH
    path.write_text(index(site, pages), encoding="utf-8")
    return path


def write_site(site_dir: Path, root: Path | None = None) -> tuple[Path, ...]:
    """Add the agent-facing files to a built site, and say what was added.

    ``llms.txt`` is already there — mkdocs copies it out of ``docs/`` like any
    other static file. What is missing from a built site is the corpus in one
    file and the Markdown *source* of each page: a rendered page is a document
    with a theme, a search index and a navigation drawer around it, and asking a
    reader to recover the prose from that is asking it to parse an application.
    """
    base = Path.cwd() if root is None else root
    site, pages = read_site(base)
    written = [site_dir / FULL_NAME]
    written[0].write_text(full(site, pages), encoding="utf-8")
    for entry, _ in pages:
        target = site_dir / entry.slug
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(entry.source(base), target)
        written.append(target)
    return tuple(written)


def main() -> int:
    """The ``generate-llms`` console script."""
    print(write())
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a console script
    raise SystemExit(main())
