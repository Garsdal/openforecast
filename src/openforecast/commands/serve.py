"""``openforecast serve`` — the same engine, over HTTP.

```bash
openforecast serve
openforecast serve --host 0.0.0.0 --port 8321
openforecast serve --store /var/lib/openforecast
```

A projection over a projection: the command builds the FastAPI application of
:mod:`openforecast.server.app` over a
:class:`~openforecast.server.transport.LocalTransport` and runs it. Nothing is
computed here that ``import openforecast`` could not do, which is the same rule
``openforecast providers`` follows.

The server half needs the ``openforecast[server]`` extra, so a missing framework
is reported as the one thing the user can do about it rather than as an
``ImportError`` traceback. Everything else about OpenForecast — including
*calling* a remote service with
:class:`~openforecast.server.transport.HttpTransport` — works without it.

It binds to loopback by default. A forecasting service has no authentication
yet, so the default has to be the one that does not publish an unauthenticated
service to a network by accident; ``--host 0.0.0.0`` is a decision the operator
makes out loud.
"""

from __future__ import annotations

import argparse
from typing import IO, Any

from openforecast.commands.exit_codes import EXIT_OK
from openforecast.errors import OpenForecastError
from openforecast.server.transport import DEFAULT_PORT, LocalTransport

__all__ = ["add_parser", "run"]

DEFAULT_HOST = "127.0.0.1"


def add_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Register ``openforecast serve``."""
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "serve",
        help="serve this engine over HTTP",
        description="Run the OpenForecast HTTP API over the models this build can execute.",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"the address to bind (default: {DEFAULT_HOST}, loopback only)",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"the port to bind (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--store",
        default=None,
        help="the artifact store to fit into (default: the user data directory)",
    )
    parser.set_defaults(handler=run)
    return parser


def run(args: argparse.Namespace, out: IO[str]) -> int:
    """Serve until interrupted. Returns 0 on a clean shutdown."""
    app = _application(LocalTransport(store=args.store))
    print(f"openforecast serving on http://{args.host}:{args.port}/v1", file=out)
    _uvicorn().run(app, host=args.host, port=args.port, log_level="info")
    return EXIT_OK


def _application(transport: LocalTransport) -> Any:
    from openforecast.server.app import create_app

    return create_app(transport)


def _uvicorn() -> Any:
    try:
        import uvicorn
    except ModuleNotFoundError as error:  # pragma: no cover - depends on the install
        raise OpenForecastError(
            "serving needs the HTTP extra, which is not installed: "
            "pip install 'openforecast[server]'"
        ) from error
    return uvicorn
