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
> foundation from Step 1 — packaging, layering rules, tooling and tests — the
> semantic data layer from Steps 2 and 3: `TimeSeriesFrame` for ordinary
> event-time data, and `PointInTimeFrame`, `ForecastDataset` and
> `ForecastContext` for real forecast vintages — the execution views and
> `ViewPlanner` from Step 4, the model references, descriptors and execution
> contracts from Step 5, and the recipes, fit plans and forecast tasks from
> Step 6. `of.fit` and `of.forecast` themselves land with the engine in Step 8.
> See [PLAN.md](PLAN.md) for the full 17-step roadmap.

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

## Execution views

A provider never sees a semantic dataset. It is handed an **execution view**,
named after the training unit it holds rather than after a model family:

| View           | Training unit                     | Typical models              |
| -------------- | --------------------------------- | --------------------------- |
| `SeriesView`   | one complete time series          | ARIMA, ETS, Theta           |
| `SequenceView` | many context → horizon sequences  | NHiTS, TFT, PatchTST        |
| `TabularView`  | individual supervised target rows | LightGBM, XGBoost, CatBoost |

`ForecastView` is the inference counterpart of all three: one origin, one
horizon.

The `ViewPlanner` is the only place in OpenForecast that knows which semantic
source it is materializing from:

```python
from openforecast.views import ViewKind, ViewPlanner, ViewRequest

planner = ViewPlanner()
request = ViewRequest(kind=ViewKind.SEQUENCES, context=168, horizon=72)

from_event_time = planner.fit_view(timeseries, request)
from_vintages = planner.fit_view(forecast_dataset, request)
```

Both calls return a `SequenceView` with the same schema and the same sample
layout. What differs is provenance: windows cut out of one freshest series
record `OriginFidelity.SIMULATED`, and windows built from real vintages record
`OriginFidelity.OBSERVED`. A model trained on the first was told the past was
cleaner than it was, and the artifact has to be able to say so.

Each sample is exactly one forecast origin — a context window ending at the
origin and a forecast window after it — and the view validates that rather than
trusting it, so no integration can accidentally learn across two origins.
Samples are keyed by an opaque, deterministic `sample_id`, with the instance
keys and origins in a separate `samples` table; a provider cannot condition on
what it cannot see. A window the data does not fully cover is dropped rather
than padded, and a value the source did not have stays missing rather than
being imputed.

`ForecastView` materializes one inference origin, trimmed to the context the
model was trained on:

```python
context = dataset.at_origin("2026-08-22T11:00:00Z")

view = planner.forecast_view(
    context,
    ViewRequest(kind=ViewKind.FORECAST, horizon=72, context=168),
)
```

Its `future` table names exactly the event times being asked about, so a
provider never derives them from a horizon count and a frequency.

## Models

A model is named by a string:

```text
<namespace>/<name>[@revision]

nixtla/nhits
nixtla/autoarima
darts/nhits
local/de-price
local/de-price@01K5Z6QK3M9TQK1W2E3R4T5Y6U
```

The string is a name, not a state — whether anything has been fitted is a
question for the registry. That is what lets a provider model and your own
fitted artifact appear in the same argument position. An unpinned
`local/de-price` is an alias that follows the latest selected revision; pinning
one names it forever.

```python
import openforecast as of

of.models.list()

descriptor = of.models.get("nixtla/nhits")

descriptor.lifecycle.requires_fit          # True
descriptor.training.view                   # ViewKind.SEQUENCES
descriptor.training.context_required       # True
descriptor.capabilities.missing_values     # MissingValueSupport.REQUIRES_TRANSFORM
```

A descriptor is complete enough to plan against on its own: it says which
execution view to materialize, whether several forecast origins may be learned
from jointly, which feature roles the model accepts, and what it does about
missing values. No provider is started to answer any of that, which is why the
engine has no reason to know which provider it is talking to.

Contracts are checked where they are declared. A `SeriesView` is one complete
time series, so a series model cannot claim to learn across origins, to bind a
horizon at fit time, or to forecast an instance it never saw — it was fitted per
series and has nothing to generalize with. Capability defaults are the
conservative ones throughout: a descriptor that declares nothing describes a
single-series, univariate, point-forecast model that cannot see a missing value.
A capability is something a provider states, never something it is assumed to
have.

`of.models.list()` is empty until Step 8 registers the built-in reference
provider and Step 9 lets external providers advertise their models.

## Recipes, plans and tasks

Three things stay separate: what to fit, how to fit it, and what to predict.

```python
recipe = of.Pipeline(
    steps=[
        of.MissingIndicator(columns="features"),
        of.Impute(columns="features", method="median"),
        of.StandardScaler(columns="targets"),
        of.Model("nixtla/nhits", params={"max_steps": 500}),
    ]
)

plan = of.FitPlan(
    origins=of.AllOrigins(),
    window=of.WindowPlan(context=168),
    seed=42,
)

task = of.ForecastTask(horizon=72)
output = of.OutputSpec.quantiles([0.1, 0.5, 0.9])
```

That separation is what lets one recipe be fitted at a single origin and across
every origin, or asked for a different horizon, without being rewritten.
Recipes compose: `of.Ensemble` and `of.Pipeline` hold recipes rather than
models, so an ensemble of pipelines needs no new vocabulary, and
`of.Reduction(estimator="lightgbm/regressor", strategy="direct", lags=[1, 24, 168])`
expresses the tabular reduction that point-in-time LightGBM setups use.

**OpenForecast owns what it can own.** A context length is stated once, as
`WindowPlan(context=168)`, and compiled into `input_size` for Nixtla or
`input_chunk_length` for Darts. Passing one of those as a provider parameter is
an error that names the field to use instead — the same for a horizon, a seed,
a frequency or a covariate list. Two copies of one number, free to disagree,
with the provider's spelling winning silently, is not a convenience.

**Nothing is imputed silently.** A missing value in point-in-time data is
information: the feature had not been published at that origin. A model that
cannot consume one declares `MissingValueSupport.REQUIRES_TRANSFORM`, and the
caller writes `MissingIndicator` and `Impute` down as steps — recorded in the
artifact, visible to whoever reads the forecast later. Putting the indicator
*after* the imputation is refused, because it would come out constant.

**The origin selections mean the same thing for both sources.**

```python
of.AllOrigins(stride=1)
of.LatestOrigin()
of.AtOrigin(timestamp)
of.OriginsBetween(start, end, stride=12)
```

On a `TimeSeriesFrame` they pick among the origins that can be simulated; on a
`ForecastDataset`, among the vintages that exist. The same `FitPlan` therefore
works on both, and only the recorded `OriginFidelity` differs.

Recipes and plans are a serializable AST, tagged by `kind`, and
`of.parse_recipe` reads one back. The same JSON is what reaches an artifact
manifest in Step 7, a provider subprocess in Step 9 and an HTTP body in
Step 16 — no part of it is provider-specific.

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
`SeriesView`, `SequenceView`, `TabularView` or `ForecastView` — never a
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
