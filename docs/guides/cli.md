# The command line

Everything OpenForecast does is reachable from a shell, and the CLI is
deliberately uncreative: a group and a verb at most, `--json` on everything that
produces information, stdout for the answer and stderr for everything else.

```bash
openforecast models list
openforecast models get builtin/seasonal-naive

openforecast providers list
openforecast providers install nixtla

openforecast fit --model builtin/seasonal-naive --data ./dataset --name de-load
openforecast forecast --model local/de-load --data ./dataset --horizon 24
openforecast backtest --config backtest.json

openforecast doctor
openforecast serve
```

Every one of those calls the same Python SDK. There is no second execution path,
which is why a fit from the shell and a fit from Python land in the same artifact
store and either can forecast with the other's model.

## Getting data to disk

A shell cannot hand over a `TimeSeriesFrame`, so the CLI reads one that was
written. That is the library's own `write`, not a CLI format:

```python
import pandas as pd

import openforecast as of

hours = pd.date_range("2026-01-01", periods=24 * 14, freq="1h")
data = of.TimeSeriesFrame.from_pandas(
    history=pd.DataFrame(
        {
            "timestamp": hours,
            "load": [50.0 + step % 24 for step in range(len(hours))],
        }
    ),
    time="timestamp",
    frequency="1h",
    targets=["load"],
)

data.write("./dataset")
```

`--data ./dataset` then points at that directory. Which of the three written
datasets it holds — a `TimeSeriesFrame`, a `PointInTimeFrame` or a
`ForecastDataset` — is read off what is in it rather than declared, because the
directory already says. Every one of them is loaded through the ordinary
constructor, so a truncated table fails to load exactly as it would in Python.

## Flags for the simple case, a config file for the rest

The flags are the top-level scalars:

```bash
openforecast fit --model builtin/seasonal-naive --data ./dataset --horizon 24
```

Anything with a recipe, a plan, a validation strategy or a metric in it is a
JSON file, because a flag syntax for a nested pipeline would be a second way to
write something the library already has one way to write:

```bash
openforecast fit --config fit.json
```

```json
{
  "model": {
    "kind": "pipeline",
    "steps": [
      {"kind": "standard-scaler", "columns": "targets"},
      {"kind": "model", "ref": "builtin/seasonal-naive", "params": {"season_length": 24}}
    ]
  },
  "data": "./dataset",
  "horizon": 24,
  "name": "de-load",
  "plan": {"window": {"context": 168}, "seed": 7}
}
```

The file is the arguments of `of.fit` as JSON: every nested field is the same
Pydantic type the SDK uses, so `plan` is a `FitPlan`, `model` is a `Recipe` or a
model reference, and a key nobody recognizes is refused by name rather than
ignored. Paths inside it resolve against the working directory, like every other
path on a command line — `--config` does not change what `./dataset` means.

The two forms are not mixed. Passing `--config` *and* a flag it configures is an
error, so there is no precedence rule to remember.

A backtest has no flag-only form at all, since its validation strategy and its
metrics are nested objects:

```bash
openforecast backtest --config backtest.json --json
```

```json
{
  "models": [
    "builtin/seasonal-naive",
    {"model": "builtin/seasonal-naive", "name": "with-a-longer-context",
     "plan": {"window": {"context": 336}}}
  ],
  "data": "./dataset",
  "validation": {"mode": "rolling", "horizon": 24, "windows": 5},
  "metrics": [{"metric": "mae"}, {"metric": "bias"}]
}
```

## `--json` everywhere

Every information-producing command prints either an aligned table for a person
or one JSON document for something that parses:

```bash
openforecast models list --json | jq -r '.models[].display_name'
openforecast forecast --config forecast.json --json | jq '.rows | length'
openforecast backtest --config backtest.json --json | jq -r '.leaderboards.mae[0].model'
```

The JSON is the complete answer — a forecast's `rows` holds every row, where the
table rendering shows a preview and says how many it is not showing. A model
prints as the `ModelDescriptor` the HTTP projection returns for
`GET /v1/models`, so an agent reading a model over the CLI and one reading it
over the network are reading one schema rather than two spellings of it.

## Streams and exit codes

```text
stdout    the requested output, and nothing else
stderr    logs, progress, warnings, and the message when something fails
0         the command did what it was asked
non-zero  it did not
```

Failure is never prose on stdout. A command that cannot do what it was asked
writes one sentence to stderr and exits non-zero:

```bash
openforecast models get nixtla/nhits
# error: nixtla/nhits is not a model this build can execute ...
```

That is what makes a pipeline reliable: `openforecast models list --json | jq`
either parses or the exit code says why not.

## Checking an installation

```bash
openforecast doctor
openforecast doctor --json
```

One command for the questions a broken setup raises before the breakage does:
which Python this is, where artifacts are written and whether that is writable,
whether `uv` is present so a provider can be installed, which provider
environments exist and whether their interpreters are still there, how many
models the catalog holds, and whether the HTTP extra is installed.

Something broken is a `fail` and the command exits non-zero — a container health
check is an exit status, not a grep. Something merely absent is a `warn` and does
not fail: an installation with no providers is a working installation of the
built-in models.

## Serving

```bash
openforecast serve
openforecast serve --host 0.0.0.0 --port 8321
```

The HTTP projection over this build's own engine, behind the
`openforecast[server]` extra. It binds to loopback by default, because a
forecasting service has no authentication yet and `--host 0.0.0.0` should be a
decision somebody makes out loud.
