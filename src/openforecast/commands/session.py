"""The one client every command runs against.

```bash
openforecast fit --config fit.json
openforecast fit --config fit.json --store ./artifacts
```

Step 26.1: the CLI calls the OpenForecast Python SDK rather than implementing a
second execution path. This is where that becomes mechanical — a command builds
an :class:`~openforecast.client.OpenForecast` here and then only calls its public
methods, so there is nothing a command could compute that ``import openforecast``
could not.

``--store`` is the only thing there is to configure, and it is the argument the
client already takes: where fitted artifacts live. Absent, it is the user data
directory, which is where ``of.fit`` puts them — so a fit from the shell and a
fit from Python land in the same place and either can forecast with the other's
artifact.
"""

from __future__ import annotations

import argparse

from openforecast.client import OpenForecast

__all__ = ["add_store_argument", "client_for"]


def add_store_argument(parser: argparse.ArgumentParser) -> None:
    """Register ``--store`` on a command that touches artifacts."""
    parser.add_argument(
        "--store",
        default=None,
        help="the artifact store to use (default: the user data directory)",
    )


def client_for(args: argparse.Namespace) -> OpenForecast:
    """The client the command runs against.

    Built per invocation rather than shared: a process runs one command, and a
    module-level default would be a store chosen before ``--store`` was read.
    """
    return OpenForecast(store=args.store)
