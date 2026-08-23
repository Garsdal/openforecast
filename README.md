# OpenForecast

The unified interface for forecasting.

You describe your data and the model you want in OpenForecast's own vocabulary,
and the library compiles that into whatever the underlying forecasting library
expects — statistical, neural, tree-based or a zero-shot foundation model —
behind one stable surface. Point-in-time data is first-class rather than bolted
on: if you have real historical forecast vintages, you train on them directly.

**[Documentation](https://garsdal.github.io/openforecast/)** ·
[Quickstart](https://garsdal.github.io/openforecast/getting-started/quickstart/) ·
[For agents](https://garsdal.github.io/openforecast/agents/overview/) ·
[llms.txt](https://garsdal.github.io/openforecast/llms.txt)

## Getting started

```bash
uv add openforecast && uv run openforecast doctor
```

or `pip install openforecast && openforecast doctor`. That is the whole install:
the core dependencies are `pydantic`, `pyarrow` and `platformdirs`, and no
forecasting framework is among them. `doctor` answers whether this installation
can forecast, as an exit code, which makes it a reasonable first call. Then read
the
[quickstart](https://garsdal.github.io/openforecast/getting-started/quickstart/),
or run any of the seven complete scripts in [`examples/`](examples/):

```bash
uv run examples/01_quickstart.py
```

## One ensemble, three libraries, one leaderboard

<!-- docs-exec: skip — needs the nixtla, sklearn and amazon provider environments -->

```python
import openforecast as of

# Real forecast vintages: what was actually known at each origin, kept apart.
# Nothing is deduplicated on the event time, forward-filled or imputed — a
# missing value means the feature had not been published yet.
data = of.ForecastDataset.from_pandas(
    df,
    origin_time="ref_time",
    event_time="target_time",
    instance_keys=["zone"],
    targets=["price"],
    known_features=["wind_fc", "load_fc"],
    event_frequency="1h",
    origin_frequency="1h",
)

# One ensemble across two libraries with two different training units. The
# neural model declares a sequence contract and is handed context -> horizon
# windows; the gradient booster declares a tabular one and is handed one
# supervised row per origin and lead of the same data. Neither provider learns
# that an ensemble exists, and neither has to agree with the other about `torch`:
# each integration lives in its own environment, reached over a subprocess
# protocol.
model = of.fit(
    of.Ensemble(
        models=[
            of.Model("nixtla/nhits", params={"max_steps": 500}),
            of.Model("sklearn/hist-gradient-boosting"),
        ],
        weights=[0.6, 0.4],
    ),
    data=data,
    horizon=24,
    name="de-price",
)

# The ensemble, its two members, and a zero-shot foundation model that is never
# fitted at all — on one leaderboard, over real historical origins. At each
# origin the features come from that vintage; later ones are absent from the
# object the model is handed rather than merely unused.
result = of.backtest(
    models=[
        model.ref,                          # the pinned fit above: scored, not refitted
        "nixtla/nhits",
        "sklearn/hist-gradient-boosting",
        "amazon/chronos-2",                 # pretrained: forecasts as it stands
    ],
    data=data,
    validation=of.ForecastOriginValidation(origins=of.AllOrigins(stride=24), horizon=24),
    metrics=[of.MAE(), of.Bias()],
)

result.leaderboard("mae")
result.metrics_by(["horizon_step", "zone"])   # does it degrade after 12, and where?
```

`origin_fidelity` in that result says `observed` rather than `simulated`, because
the origins were real. Switching any candidate to `darts/tide` or
`sktime/pooled-trees` changes the string and nothing else.

## What is different about it

- **One name per intent.** `fit`, `forecast`, `backtest`, `eligible_models` — the
  same four on the client, the package, the CLI and over HTTP. A test fails if a
  second way to do any of them appears.
- **Point-in-time is a representation, not a convention.** Vintages have their
  own type, leakage rules are enforced in one place, and an artifact records
  whether its origins were observed or simulated.
- **Providers are quarantined.** The core never imports a forecasting framework;
  each integration has its own environment and only ever sees a provider-neutral
  execution view.
- **Nothing is assumed.** Every capability is declared by the model and checked
  before a provider starts, so an impossible request fails in the first second
  with a stable `error.code`.
- **Machine-readable throughout.** JSON Schemas per request, an OpenAPI document,
  `--json` on every command, and `llms.txt` over the docs — all generated and
  diffed in CI.

## Documentation

[garsdal.github.io/openforecast](https://garsdal.github.io/openforecast/) — one
published site per release.

| Section | Answers |
| --- | --- |
| [Getting started](https://garsdal.github.io/openforecast/getting-started/installation/) | Install it, run the workflow once, learn the five ideas. |
| [Guides](https://garsdal.github.io/openforecast/guides/event-time/) | "How do I do X?" — one task per page, with code that runs. |
| [For agents](https://garsdal.github.io/openforecast/agents/overview/) | Discover, choose, construct, recover — and the point-in-time rules. |
| [Concepts](https://garsdal.github.io/openforecast/concepts/data-model/) | "Why does OpenForecast work this way?" |
| [Reference](https://garsdal.github.io/openforecast/reference/generated/) | Exact signatures and types, generated from the code. |
| [ARCHITECTURE.md](ARCHITECTURE.md) | The seven rules, and how each one is enforced. |

Every page is also served as Markdown at the same address with `.md` appended,
and [`/llms-full.txt`](https://garsdal.github.io/openforecast/llms-full.txt) is
the whole corpus in one fetch. Every Python example in the documentation is
executed by the test suite, and the generated reference is diffed in CI, so a
signature cannot drift from the library.

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
uv sync                 # create .venv, install core + dev dependencies
uv run pytest           # unit, contract, conformance, docs, examples, e2e
uv run ruff check .     # lint
uv run ruff format .    # format
uv run pyright          # type check
```

Four things are generated from the code, committed, and diffed in CI — a change
to what a request means is a diff rather than a document that quietly disagrees:

```bash
uv run generate-openapi     # spec/openapi/openapi.json
uv run generate-schemas     # spec/schemas
uv run generate-reference   # docs/reference/generated
uv run generate-llms        # docs/llms.txt
```

```bash
uv sync --group docs && uv run mkdocs build --strict
```

## Repository layout

```text
src/openforecast/   the library: data/, views/, models/, recipes/, tasks/,
                    artifacts/, registry/, runtime/, providers/, protocol/,
                    commands/, server/, evaluation/, docs/, client.py
integrations/       provider distributions, each independently versioned:
                    nixtla, darts, sktime, sklearn, chronos
examples/           seven complete scripts, executed in CI
tests/              unit, contract, conformance, docs, examples and e2e suites
spec/               protocol, Arrow, OpenAPI and JSON Schema specifications
docs/               the documentation site
```

Imports flow one way — `protocol` → data and models → `views` → runtime →
client, CLI, server, evaluation — and that is a test rather than a diagram.
[ARCHITECTURE.md](ARCHITECTURE.md) states the rules and names the test enforcing
each.

> **Status: early development.** The public V1 surface is frozen and the
> semantics are implemented end to end over five integrations. See
> [PLAN.md](PLAN.md) for the roadmap that got here and
> [PLAN_2.md](PLAN_2.md) for what comes next.
