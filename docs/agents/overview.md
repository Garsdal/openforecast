# Overview

This section is for a reader that arrives with a fetch tool and a task, not with
an afternoon. It answers four questions and nothing else: how to **discover**
what a build can do, how to **choose** a model, how to **construct** a request,
and how to **recover** from a refusal.

## The four answers

| Question | Ask this | Page |
| --- | --- | --- |
| What can this build run? | `openforecast models list --json` | below |
| Which model fits my data? | `of.eligible_models(data, horizon=...)` | [Choosing a model](choosing-a-model.md) |
| What does a request look like? | `openforecast schema fit --json` | [The structured CLI](structured-cli.md) |
| Why was this refused? | `error.code` | [Recovering from errors](errors.md) |

Point-in-time data has its own page, because it is the one part of the library
where a plausible-looking request can be silently wrong: [Point-in-time
rules](point-in-time.md).

## One name per intent

There is exactly one call per intent, and it is the same name on the client, on
the package, on the CLI and over HTTP. `fit`, never `train`; `forecast`, never
`predict` or `infer`; `backtest`, never `evaluate` or `historical_forecasts`.

```python
import openforecast as of

client = of.OpenForecast()

[str(ref) for ref in client.models.refs()]   # what this build can run
```

`of.fit(...)` and `client.fit(...)` are the same operation — the module-level
functions are methods on a default client, so the signatures differ only by
`client=`. A test asserts that, and asserts that no alias exists. So there is no
second way to do any of this to go looking for.

## Everything is machine-readable

| Surface | What it is |
| --- | --- |
| [`/llms.txt`](https://garsdal.github.io/openforecast/llms.txt) | this site's map: every page, its section, one sentence about it |
| [`/llms-full.txt`](https://garsdal.github.io/openforecast/llms-full.txt) | the whole corpus in one fetch |
| `<page>.md` | the Markdown source of any page, served beside its HTML |
| `openforecast schema <request> --json` | the JSON Schema of a request, from the installed build |
| `--json` on any command | the answer on stdout, and nothing else on stdout |
| `error.code` | a stable, frozen identifier for every failure |

The first three are generated from the navigation by `uv run generate-llms`, so
they cannot describe a page that no longer exists. The schemas are generated from
the Pydantic models the commands validate against, so a request built from one is
a request this build accepts.

## Two things worth knowing before starting

**Nothing is repaired.** Construction validates and refuses: a duplicate
timestamp, a target in the future table, a static feature that varies within an
instance, a missing value fed to a model that declared it cannot see one. A
refusal names the field and what to do instead. Nothing is imputed, deduplicated
or forward-filled on the way in.

**A capability is declared, never assumed.** A model's descriptor states which
execution view it trains on, which data shapes and feature roles it takes,
whether it can learn from several origins jointly, what it does about missing
values and which output forms it can produce. Every request is checked against
that declaration *before* a provider process is started, which is why an
impossible request fails in the first second rather than after an hour.
