"""The ``openforecast`` CLI.

```bash
openforecast providers install nixtla
openforecast providers list
```

A thin projection over the same client the Python API uses — never a second API
architecture. Today it manages provider environments, which is the one thing
that genuinely belongs at a command line: installing an integration builds an
isolated interpreter, and that is a machine-level action rather than a
forecasting one.

It is built on ``argparse`` deliberately. A CLI framework would be a fourth
runtime dependency for a projection, and rule 1 makes a dependency an
architectural decision rather than a convenience.
"""

from openforecast.commands.main import build_parser, main

__all__ = ["build_parser", "main"]
