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
> the semantic data layer from Steps 2 and 3: `TimeSeriesFrame` for ordinary
> event-time data, and `PointInTimeFrame`, `ForecastDataset` and
> `ForecastContext` for real forecast vintages. Models, views, recipes and the
> engine shown above land in later steps. See [PLAN.md](PLAN.md) for the full
> 17-step roadmap.

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

Forecast vintages are deliberately *not* expressible here — they get their own
representation rather than optional fields on this one.

## The point-in-time semantic model

`PointInTimeFrame` represents `instance × origin_time × event_time × variable`:
what was knowable, vintage by vintage.

```text
zone origin_time event_time wind_fc load_fc
DE   08:00       12:00      10.1    54.2
DE   09:00       12:00      11.7    54.8
DE   10:00       12:00      12.4    55.1
```

Three rows, not one. The same event time appears once per origin and the values
differ between them, which is the whole point: nothing collapses, deduplicates
or forward-fills a vintage. Lead time is derived rather than stored — ask for it
with `pit.with_lead_time(unit="hour")`.

`ForecastDataset` pairs that information with the outcome it was trying to
predict:

```text
information   PointInTimeFrame   every vintage, exactly as it was issued
truth         TimeSeriesFrame    the realized outcome, once per event time
```

The `(ref_time, target_time)` tables production pipelines already emit carry
both at once — the label is repeated on every vintage of the same event time —
so there is a constructor that splits them apart:

```python
dataset = of.ForecastDataset.from_pandas(
    df,
    origin_time="ref_time",
    event_time="target_time",
    instance_keys=["zone"],
    targets=["price"],
    known_features=["wind_fc", "solar_fc", "load_fc"],
    event_frequency="1h",
    origin_frequency="1h",
)
```

If the repeated labels disagree, that is a contradiction in the source data and
raises `InconsistentTruthError` — OpenForecast does not pick one. A label that
is merely missing in an earlier vintage is not a disagreement: it is a label
that was not published yet.

`ForecastContext` is exactly one inference origin, the shape production
inference always has:

```python
context = dataset.at_origin("2026-08-22T11:00:00Z")
```

Only that vintage contributes. A feature value revised at 12:00 cannot appear in
the context of the 11:00 origin, and an observed feature is rejected outright if
it holds a value for an event time after the origin that supposedly produced it.
Contexts can also be built directly from live data with
`of.ForecastContext.from_pandas(...)`.

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
