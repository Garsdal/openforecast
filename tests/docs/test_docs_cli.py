"""Every ``openforecast ...`` command line in the documentation, parsed.

The Python examples on these pages are executed, so a stale call is a failing
test. A shell example is prose as far as an interpreter is concerned, which is
exactly how a documented command outlives the flag it uses — so the parser is
asked about each one instead. That catches everything a parse can catch: a verb
that no longer exists, a flag that was renamed, a group that moved.

Running them is deliberately not what happens here. ``openforecast providers
install nixtla`` builds an environment from a network, and ``openforecast serve``
binds a socket and does not return; what is checkable without either is that the
command line is one this build accepts.
"""

from __future__ import annotations

import argparse
import shlex
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

import pytest

from openforecast.commands import build_parser
from tests.docs.blocks import DOCS_ROOT, REPO_ROOT, blocks_in, pages_under

#: The hand-written pages, plus the two documents at the repository root: a
#: command in the README is as public as one in the guide.
PAGES = (*pages_under(DOCS_ROOT), REPO_ROOT / "README.md", REPO_ROOT / "ARCHITECTURE.md")

#: Lines that ask the parser to print and exit rather than to parse a command.
_EXITING = frozenset({"--version", "-h", "--help"})


def command_lines() -> list[tuple[Path, int, str]]:
    """Every line of a fenced block that invokes the CLI."""
    found: list[tuple[Path, int, str]] = []
    for page in PAGES:
        for block in blocks_in(page):
            if block.is_python:
                continue
            for offset, line in enumerate(block.source.splitlines()):
                # A documented pipeline is documented as one; only the command
                # OpenForecast owns is this parser's business.
                text = line.split("|")[0].split("#")[0].strip()
                if text.startswith("openforecast "):
                    found.append((page, block.line + offset, text))
    return found


LINES = command_lines()


def test_the_documentation_shows_commands_at_all() -> None:
    """Otherwise a scanner that found nothing would look like a suite that passed."""
    assert len(LINES) >= 10


@pytest.mark.parametrize(
    ("page", "line", "command"),
    LINES,
    ids=[f"{page.relative_to(REPO_ROOT)}:{line}" for page, line, _ in LINES],
)
def test_every_documented_command_is_one_this_build_accepts(
    page: Path, line: int, command: str
) -> None:
    argv = shlex.split(command)[1:]
    if set(argv) & _EXITING:
        return
    parser = build_parser()
    # argparse reports a bad command line by printing usage and exiting, so the
    # failure has to be caught to be reported as the documentation problem it is.
    with redirect_stderr(StringIO()) as usage:
        try:
            parsed = parser.parse_args(argv)
        except SystemExit:
            pytest.fail(
                f"{page.relative_to(REPO_ROOT)}:{line} documents a command this build "
                f"does not accept:\n  {command}\n{usage.getvalue()}"
            )
    assert isinstance(parsed, argparse.Namespace)
    assert hasattr(parsed, "handler")
