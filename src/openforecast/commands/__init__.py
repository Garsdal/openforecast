"""The ``openforecast`` CLI.

```bash
openforecast models list
openforecast models get builtin/seasonal-naive

openforecast providers list
openforecast providers install nixtla

openforecast fit --config fit.json
openforecast forecast --config forecast.json --json
openforecast backtest --config backtest.json --json

openforecast schema fit --json

openforecast doctor
openforecast serve
```

A thin projection over the same client the Python API uses — never a second API
architecture. Step 26 is what made that the whole surface rather than only the
machine-level part of it: ``fit``, ``forecast`` and ``backtest`` here are
``client.fit``, ``client.forecast`` and ``client.backtest`` with a config file
deserialized into the Pydantic types the SDK already has, so the shell can do
everything Python can and neither can do something the other cannot.

The CLI is intentionally uncreative. The tree is a group and a verb at most.
Every information-producing command takes ``--json``. stdout is the requested
output and stderr is everything else, so ``| jq`` is reliable. A failure is a
non-zero exit code and a sentence on stderr, never prose on stdout that a script
would have to read — and with ``--json``, that sentence is the structured error
envelope of Step 27.3, so recovery is a branch on a code rather than a match
against prose. ``openforecast schema`` is the other half of that step: what a
request has to look like, answered by the build that would execute it.

It is built on ``argparse`` deliberately. A CLI framework would be a fourth
runtime dependency for a projection, and rule 1 makes a dependency an
architectural decision rather than a convenience.
"""

from openforecast.commands.main import build_parser, main

# The CLI's surface is the two functions that run it. The codes it exits with
# are a contract too, and they live in :mod:`openforecast.commands.exit_codes`
# rather than here: a caller who needs them is writing a command, not running
# one.
__all__ = ["build_parser", "main"]
