"""The ``openforecast`` entry point.

```bash
openforecast models list
openforecast fit --config fit.json
openforecast serve
```

A thin projection over the same objects the Python API uses — never a second
API architecture. Each command group parses arguments and calls into
``openforecast``; nothing is computed here that ``import openforecast`` could
not do.

The command tree is deliberately two levels deep at most: a group and a verb,
or a bare command. ``openforecast fit`` is not under ``openforecast run``, and
there is no ``openforecast artifacts models revisions list``. What a nested tree
buys is tidiness in the help output, and what it costs is that nobody can guess
the command they want.

The stream contract is the one the provider protocol uses, for the same reason:
**stdout is the answer, stderr is everything else.** A command's output can be
piped into ``jq`` with ``--json``, and a failure appears on stderr with a
non-zero exit code rather than as a traceback. Step 27.3 made that failure
structured too: with ``--json`` it is the same ``{"error": {"code", "message",
"details"}}`` document the HTTP projection answers with, still on stderr, so an
agent branches on a code rather than on prose and a script reading stdout still
cannot mistake a failure for an answer.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from typing import IO

from openforecast import __version__
from openforecast.commands import doctor, models, operations, output, providers, schema, serve
from openforecast.commands.exit_codes import EXIT_ERROR
from openforecast.errors import OpenForecastError

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openforecast",
        description="The unified interface for forecasting.",
    )
    parser.add_argument("--version", action="version", version=f"openforecast {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    models.add_parser(commands)
    providers.add_parser(commands)
    operations.add_parsers(commands)
    schema.add_parser(commands)
    doctor.add_parser(commands)
    serve.add_parser(commands)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    out: IO[str] | None = None,
    err: IO[str] | None = None,
) -> int:
    """Run one command. Returns the process exit code rather than exiting."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    stdout = out if out is not None else sys.stdout
    stderr = err if err is not None else sys.stderr
    try:
        # Each command group registers its own handler, so adding one is adding
        # a parser rather than editing a chain of comparisons here.
        handler: Callable[[argparse.Namespace, IO[str]], int] = args.handler
        return handler(args, stdout)
    except OpenForecastError as error:
        # Every failure a caller can act on is one of these, so this is the only
        # place a message is written to stderr — and nothing partial has been
        # written to stdout, because a command prints its answer once it has one.
        _report(error, stderr, as_json=bool(getattr(args, "json", False)))
        return EXIT_ERROR


def _report(error: OpenForecastError, err: IO[str], *, as_json: bool) -> None:
    """One failure, written the way the caller asked to be spoken to.

    Step 27.3 in the CLI: a caller who asked for ``--json`` gets the error as the
    same ``{"error": {"code", "message", "details"}}`` document the HTTP
    projection answers with, so recovery is a branch on ``code`` rather than a
    match against prose that is free to be rewritten.

    It goes to **stderr**, and stdout stays empty. That is not a hedge between
    the two rules — 26.4 says stdout carries the requested output and 26.5 says
    failure is the exit code, so a document on stdout would be indistinguishable
    from an answer to a script reading the stream. A failed command therefore
    writes nothing to stdout, exits non-zero, and says why on stderr in whichever
    of the two shapes was asked for.
    """
    if as_json:
        output.dump({"error": error.as_json()}, err)
    else:
        print(f"error: {error}", file=err)


if __name__ == "__main__":  # pragma: no cover - exercised as a console script
    raise SystemExit(main())
