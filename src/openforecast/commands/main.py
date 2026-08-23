"""The ``openforecast`` entry point.

```bash
openforecast providers list
openforecast serve
```

A thin projection over the same objects the Python API uses — never a second
API architecture. Each command group parses arguments and calls into
``openforecast``; nothing is computed here that ``import openforecast`` could
not do.

The stream contract is the one the provider protocol uses, for the same reason:
**stdout is the answer, stderr is everything else.** A command's output can be
piped into ``jq`` with ``--json``, and a failure appears on stderr with a
non-zero exit code rather than as a traceback.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from typing import IO

from openforecast import __version__
from openforecast.commands import providers, serve
from openforecast.errors import OpenForecastError

__all__ = ["build_parser", "main"]

#: Something the user can fix — an uninstalled provider, a failed build. A
#: traceback would say the same thing less clearly and imply a bug.
EXIT_ERROR = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openforecast",
        description="The unified interface for forecasting.",
    )
    parser.add_argument("--version", action="version", version=f"openforecast {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    providers.add_parser(commands)
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
        print(f"error: {error}", file=stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover - exercised as a console script
    raise SystemExit(main())
