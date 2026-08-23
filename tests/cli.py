"""Running the CLI in-process, and everything one invocation produced.

The CLI's contract is three things a script depends on — what went to stdout,
what went to stderr, and what the exit code was — so a test has to be able to
see all three separately. ``main`` takes both streams for exactly that reason,
which is also how a test asserts the contract without a subprocess: the entry
point returns the code rather than exiting, so a failing command is an assertion
rather than a ``CalledProcessError``.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from openforecast.commands import main

__all__ = ["Run", "run", "write_config"]


class Run:
    """One CLI invocation, and everything it produced."""

    def __init__(self, code: int, out: str, err: str) -> None:
        self.code = code
        self.out = out
        self.err = err

    @property
    def json(self) -> Any:
        """The document on stdout.

        Parsed here rather than in each test, so that a command which printed a
        log line to stdout fails as a broken stream contract — the JSON would no
        longer parse — rather than as a puzzling assertion further down.
        """
        return json.loads(self.out)

    def __repr__(self) -> str:
        return f"Run(code={self.code}, out={self.out!r}, err={self.err!r})"


def run(*argv: str) -> Run:
    """One command, with both streams captured."""
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, out=out, err=err)
    return Run(code, out.getvalue(), err.getvalue())


def write_config(path: Path, payload: dict[str, Any]) -> str:
    """A config file, as the string a ``--config`` argument takes."""
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)
