# OpenForecast

The unified interface for forecasting.

OpenForecast is a framework-agnostic forecasting API. You describe your data and
the model you want in OpenForecast's own vocabulary, and the library compiles
that into whatever the underlying forecasting library expects — statistical,
neural, or tree-based — behind one stable surface:

```python
import openforecast as of

model = of.fit(model="nixtla/nhits", data=train, horizon=24)
forecast = of.forecast(model=model, data=context, horizon=24)
```

Point-in-time forecasting is first-class rather than bolted on. If you have real
historical forecast vintages — what was actually known at each origin — you can
train on them directly, and the model records that its origins were *observed*
rather than *simulated* by cutting windows out of a single freshest series.

> **Status: early development.** This repository currently contains the
> foundation from Step 1 — packaging, layering rules, tooling and tests — and
> the event-time semantic model from Step 2: `TimeSeriesFrame` and the
> vocabulary that describes one. Models, views, recipes and the engine shown
> above land in later steps. See [PLAN.md](PLAN.md) for the full 17-step
> roadmap.

## The event-time semantic model

`TimeSeriesFrame` represents ordinary `instance × event_time × variable` data as
three Arrow tables — history, future and static — against one schema:

```python
import openforecast as of

frame = of.TimeSeriesFrame.from_pandas(
    history=df,
    time="timestamp",
    frequency="1h",
    instance_keys=["country"],
    targets=["load"],
    observed_features=["temperature_actual"],
    known_features=["temperature_forecast"],
    static_features=["capacity"],
)

frame.schema.is_panel          # True
frame.schema.is_univariate     # True
frame.write("de-load")
frame = of.TimeSeriesFrame.read("de-load")
```

Features carry two orthogonal axes: `kind` (temporal or static) and
`availability` (observed only up to the origin, or known into the future).
Interesting categories are derived from those axes rather than enumerated, so
there is no `PANEL_MULTIVARIATE`.

Construction validates and never repairs. Duplicate instance/time rows,
timestamps off the declared frequency grid, targets or observed features in the
future table, and static features that vary within an instance are all errors —
each of them silently changes what the data means. Gaps and missing values are
preserved as they are: a missing observation is information.

Forecast vintages are deliberately *not* expressible here. Point-in-time data
gets its own first-class representation in Step 3 rather than optional fields on
this one.

## The architectural invariant

> OpenForecast owns forecasting semantics. Providers only consume
> provider-neutral **execution views**. Point-in-time and ordinary event-time
> data are materialized into those views before crossing the provider boundary.

Two things follow from this, and they shape the whole repository:

**The core never depends on a forecasting framework.** No Nixtla, Darts,
sktime, PyTorch, JAX or LightGBM — the core install is `pydantic`, `pyarrow`
and `platformdirs`. Integrations depend on OpenForecast, never the reverse, and
each lives in its own distribution under `integrations/` with its own lockfile
and virtual environment, so providers with incompatible dependency graphs
(Torch vs. JAX, say) can coexist without ever meeting.

**No provider branches on where the data came from.** A provider is handed a
`SeriesView`, `WindowView`, `TabularView` or `ForecastView` — never a
`TimeSeriesFrame` or a `ForecastDataset`. Point-in-time handling lives in the
`ViewPlanner`, once, instead of being re-derived in every integration.

[ARCHITECTURE.md](ARCHITECTURE.md) states all seven rules and how each is
enforced.

## Layering

Imports flow in one direction only:

```text
                    protocol/
                        ↓
      data/  models/  recipes/  tasks/
                        ↓
                     views/
                        ↓
      runtime/  registry/  artifacts/
                        ↓
        client.py  commands/  server/
```

These rules are tests, not documentation: `tests/unit/test_architecture.py`
AST-scans the package and fails on any forbidden import, any forecasting
framework in the declared dependencies, and any import pointing down the stack.
CI additionally greps `uv tree --no-dev`, so a framework cannot slip in as
somebody else's transitive dependency.

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
uv sync                 # create .venv and install core + dev dependencies
uv run ruff check .     # lint
uv run ruff format .    # format
uv run pyright          # type check
uv run pytest           # test
```

Arrow is the canonical data-plane representation, so anything crossing a
process or language boundary is Arrow IPC rather than JSON.

## Repository layout

```text
src/openforecast/
    data/        semantic source datasets — event-time and point-in-time
    views/       provider-neutral execution views and the ViewPlanner
    models/      model refs, descriptors, capabilities, training contracts
    recipes/     the model-construction IR: models, pipelines, ensembles
    tasks/       fit plans, origin selection, forecast tasks, output specs
    artifacts/   artifact lifecycle, manifests, atomic writes
    registry/    model and provider resolution
    runtime/     the execution engine and provider clients
    protocol/    the provider wire protocol
    commands/    the CLI
    server/      the HTTP projection
    client.py    the user-facing client

integrations/    provider distributions, each independently versioned
tests/           unit, contract, conformance and e2e suites
spec/            protocol, Arrow and OpenAPI specifications
```

Every package exists with a docstring naming the step that fills it. Nothing is
a stub API — if a name is not implemented yet, it is absent rather than raising
`NotImplementedError`.
