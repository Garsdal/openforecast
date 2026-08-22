"""Fake providers on the far side of a real process boundary.

The transport is only worth testing against a real subprocess: a mock would
agree with whatever the client believes about pipes, buffering and exit codes,
which is exactly where the interesting failures are. So each test writes a small
script and runs it with the interpreter the suite is running under.

Two things live here. :func:`script` builds a badly behaved provider — one that
crashes, or answers nonsense, or never answers — from a few lines of source.
:data:`REFERENCE_PROVIDER` is the real thing: the built-in provider, reached
through the entry point it ships, over the protocol an integration will speak.
"""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

__all__ = ["ANSWERS_ANYTHING", "REFERENCE_PROVIDER", "script"]


def script(tmp_path: Path, *sources: str, name: str = "fake_provider.py") -> list[str]:
    """A command running ``sources`` as one script, with the test's interpreter.

    Each part is dedented on its own, so a caller can concatenate an indented
    triple-quoted block with a shared one without the two having to agree about
    indentation.
    """
    path = tmp_path / name
    path.write_text("".join(dedent(source) for source in sources), encoding="utf-8")
    return [sys.executable, str(path)]


#: Reads request lines and replies with whatever ``respond(request)`` returns:
#: a string is written verbatim — which is how a test writes something that is
#: not JSON at all — and anything else is encoded. ``None`` means "say nothing",
#: which is how a hang is written.
ANSWERS_ANYTHING = """
    import json
    import sys

    for line in sys.stdin:
        answer = respond(json.loads(line))
        if answer is not None:
            sys.stdout.write(answer if isinstance(answer, str) else json.dumps(answer))
            sys.stdout.write("\\n")
            sys.stdout.flush()
"""


#: The built-in provider over the wire instead of in this process, through the
#: entry point it ships: ``python -m openforecast.providers.builtin``. An
#: integration's is ``python -m openforecast_nixtla``, and it is the same file.
REFERENCE_PROVIDER = [sys.executable, "-m", "openforecast.providers.builtin"]
