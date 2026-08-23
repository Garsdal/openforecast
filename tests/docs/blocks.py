"""Reading fenced code blocks, and the markers beside them, out of a Markdown page.

An example that cannot be executed here says so *in the page*, above the fence:

```text
<!-- docs-exec: skip — needs `openforecast providers install nixtla` -->
```

The reason is required, so "this cannot run" is a statement somebody had to write
rather than a default. The comment is invisible in the rendered page and visible
in the source, which is where the person changing the example is looking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

__all__ = ["SKIP_MARKER", "Block", "blocks_in", "pages_under"]

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPO_ROOT / "docs"
#: Written by ``uv run generate-reference``, so its blocks are signatures rather
#: than examples: nothing in them is meant to be executed.
GENERATED_ROOT = DOCS_ROOT / "reference" / "generated"

SKIP_MARKER = re.compile(r"<!--\s*docs-exec:\s*skip\s*(?:—|-|:)\s*(?P<reason>.+?)\s*-->")
_FENCE = re.compile(r"^```(?P<info>[^\s`]*)")


@dataclass(frozen=True)
class Block:
    """One fenced block, with where it came from and whether it is executed."""

    path: Path
    line: int
    language: str
    source: str
    skip_reason: str | None

    @property
    def location(self) -> str:
        return f"{self.path.relative_to(REPO_ROOT)}:{self.line}"

    @property
    def is_python(self) -> bool:
        return self.language == "python"


def pages_under(root: Path = DOCS_ROOT) -> tuple[Path, ...]:
    """Every hand-written page, in a stable order."""
    return tuple(path for path in sorted(root.rglob("*.md")) if GENERATED_ROOT not in path.parents)


def blocks_in(path: Path) -> tuple[Block, ...]:
    """Every fenced block on one page.

    A skip marker applies to the next fence that opens after it, and to nothing
    else: a marker followed by prose and then a block still names that block,
    which is how a sentence can explain the skip.
    """
    found: list[Block] = []
    pending: str | None = None
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        marker = SKIP_MARKER.search(lines[index])
        if marker is not None:
            pending = marker.group("reason")
            index += 1
            continue
        fence = _FENCE.match(lines[index])
        if fence is None:
            index += 1
            continue
        start = index + 1
        end = start
        while end < len(lines) and not lines[end].startswith("```"):
            end += 1
        found.append(
            Block(
                path=path,
                line=index + 1,
                language=fence.group("info"),
                source="\n".join(lines[start:end]) + "\n",
                skip_reason=pending,
            )
        )
        pending = None
        index = end + 1
    return tuple(found)
