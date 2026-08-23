"""The ``openforecast`` CLI.

```bash
openforecast providers install nixtla
openforecast providers list
openforecast serve
```

A thin projection over the same client the Python API uses — never a second API
architecture. Two things belong at a command line rather than in the library.
Managing provider environments is one: installing an integration builds an
isolated interpreter, and that is a machine-level action rather than a
forecasting one. Serving is the other: ``openforecast serve`` runs the HTTP
projection of Step 16 over this build's own engine, and needs the
``openforecast[server]`` extra.

It is built on ``argparse`` deliberately. A CLI framework would be a fourth
runtime dependency for a projection, and rule 1 makes a dependency an
architectural decision rather than a convenience.
"""

from openforecast.commands.main import build_parser, main

__all__ = ["build_parser", "main"]
