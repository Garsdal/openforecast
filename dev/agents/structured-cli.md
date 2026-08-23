# The structured CLI

The loop the CLI exists for:

```text
inspect schema  ->  construct request  ->  execute  ->  read JSON
```

No Python signature to reverse-engineer, no doc page to trust, and no schema
pinned to a version other than the one installed. [The command
line](../guides/cli.md) is the full guide; this is the short version.

## Inspect

```bash
openforecast schema fit --json
openforecast schema forecast --json
openforecast schema backtest --json
```

That is the JSON Schema of the request, generated from the Pydantic model the
command validates against — byte-for-byte the document committed under
`spec/schemas`. An unrecognized key is refused *by name*, so a request that
validates is a request that runs.

```bash
openforecast doctor --json
openforecast models list --json
openforecast providers list --json
```

`doctor` answers "can this installation forecast" as an exit code, which makes it
a reasonable first call.

## Construct

Flags are the top-level scalars; anything with structure in it is a JSON file,
because a flag syntax for a nested pipeline would be a second way to write
something the library has one way to write.

```bash
openforecast fit --model builtin/seasonal-naive --data ./dataset --horizon 24 --name de-load
openforecast fit --config fit.json
```

```json
{
  "model": "builtin/seasonal-naive",
  "data": "./dataset",
  "horizon": 24,
  "name": "de-load",
  "plan": {"window": {"context": 168}, "seed": 7}
}
```

The config file *is* the arguments of `of.fit`, written down: each nested field
deserializes into the same type the library uses.

## Data crosses as a directory

A shell cannot hand over a frame, so `--data` points at what `write` produced:

```python
import pandas as pd

import openforecast as of

hours = pd.date_range("2026-01-01", periods=24 * 14, freq="1h")
data = of.TimeSeriesFrame.from_pandas(
    history=pd.DataFrame({"timestamp": hours, "load": [50.0 + step % 24 for step in range(len(hours))]}),
    time="timestamp",
    frequency="1h",
    targets=["load"],
)

data.write("./dataset")
```

Which of the three datasets that directory holds — a `TimeSeriesFrame`, a
`PointInTimeFrame` or a `ForecastDataset` — is read off what is in it rather than
declared.

## Execute, and read the streams

```bash
openforecast forecast --model local/de-load --data ./dataset --horizon 24 --json
```

```text
stdout    the requested output, and nothing else
stderr    logs, progress, warnings, and the message when something fails
0         the command did what it was asked
non-zero  it did not
```

So `openforecast models list --json | jq` is reliable, and a failure never
appears on stdout as prose a script would have to interpret. Failures carry the
same structured shape everywhere — see [Recovering from
errors](errors.md).

## It is the same execution path

`fit`, `forecast` and `backtest` are `client.fit`, `client.forecast` and
`client.backtest`: the same SDK, the same artifact store. A fit from a shell can
be forecast with from Python and the other way round. There is no CLI-only
behaviour to discover.

## Remotely, if that is where the models are

```bash
openforecast serve
```

`of.OpenForecast(transport=of.HttpTransport("http://localhost:8321"))` exposes the
same calls, and `spec/openapi/openapi.json` describes the endpoints. Serving needs
`pip install 'openforecast[server]'`; calling a service needs nothing extra.
