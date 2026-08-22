# Phase Two — From forecasting protocol to agent-ready forecasting platform

Phase One, Steps 1–17, proves the core architecture:

```text
semantic data
    │
    ├── TimeSeriesFrame
    └── ForecastDataset
            │
            ▼
        ViewPlanner
            │
      ┌─────┼────────┐
      ▼     ▼        ▼
 SeriesView SequenceView TabularView
      │     │        │
      ▼     ▼        ▼
   providers / models
            │
            ▼
        Forecast
```

Phase Two should **not redesign this foundation**.

Its purpose is to:

1. make `TabularView` genuinely useful;
2. make evaluation trustworthy;
3. finish the common forecast output protocol;
4. prove cross-provider composition;
5. prove the pretrained/no-fit lifecycle;
6. then harden OpenForecast into an extremely simple, documented, machine-readable interface for agents.

The most important Phase Two architectural rule remains:

> **New functionality should reuse existing OpenForecast primitives wherever possible. Avoid adding new public types unless there is no clean way to express the functionality with what already exists.**

And for agent-readiness:

> **An agent should be able to discover → understand → execute → inspect → recover without reading implementation code or guessing provider behavior.**

---

# Step 18 — Native `TabularView` execution with scikit-learn

## Goal

Make `TabularView` a complete execution path independent of sktime, Darts, or any other forecasting framework.

Add scikit-learn as the first direct consumer of:

```text
TabularView
```

This proves that OpenForecast itself owns the transformation:

```text
forecasting problem
       ↓
supervised rows
       ↓
X / y
```

rather than delegating this responsibility to sktime's reduction APIs.

## Why

Previously, sktime was an attractive place to execute `Reduction`, because sktime already knows how to turn forecasting into regression.

But that would make the path:

```text
ForecastDataset
      ↓
OpenForecast TabularView
      ↓
sktime
      ↓
sktime reduction
      ↓
sklearn estimator
```

That is the wrong ownership boundary.

OpenForecast already knows:

```text
forecast origin
target time
lead
information vintage
training row
truth alignment
```

Once `TabularView` exists, another forecasting framework does not need to reinterpret those semantics.

The correct path is:

```text
ForecastDataset
      ↓
ViewPlanner
      ↓
TabularView
      ↓
scikit-learn
      ↓
fit(X, y)
```

sktime remains an independent provider for sktime-native forecasting models.

---

## 18.1 Add sklearn integration

Create:

```text
integrations/
    sklearn/
        pyproject.toml
        uv.lock

        src/
            openforecast_sklearn/
                __main__.py
                provider.py
                catalog.py
                adapter.py
```

Use the same isolated-provider architecture as:

```text
nixtla
darts
sktime
```

even though sklearn itself has relatively simple dependencies.

This keeps provider execution consistent.

---

## 18.2 Start with one estimator

Expose:

```text
sklearn/hist-gradient-boosting
```

backed by:

```python
from sklearn.ensemble import HistGradientBoostingRegressor
```

This is a particularly useful first estimator because it can handle missing feature values natively, which is important for PIT datasets.

Do not start by exposing the whole sklearn catalog.

Later additions should be trivial:

```text
sklearn/random-forest
sklearn/extra-trees
sklearn/ridge
```

If adding these later requires architectural changes, `TabularView` is insufficient.

---

## 18.3 Descriptor

Approximately:

```yaml
ref: sklearn/hist-gradient-boosting
provider: sklearn

lifecycle:
  requires_fit: true
  supports_fit: true

training:
  view: tabular
  origin_scope: multiple

capabilities:
  instances:
    single: true
    panel: true

  targets:
    univariate: true
    multivariate: false

  features:
    observed: true
    known: true
    static: true

  outputs:
    point: true
    quantiles: false
    samples: false

  missing_features: native
```

Start with one target only.

Do not build multi-target wrappers yet.

---

## 18.4 Finalize `TabularView`

Use:

```python
class TabularView:
    X: pa.Table
    y: pa.Array | None
    keys: pa.Table
    schema: TabularViewSchema
```

During training:

```text
X = model inputs
y = truth
```

During inference:

```text
X = model inputs
y = None
```

`keys` carries structural information:

```text
row_id
instance keys...
origin_time
event_time
horizon_step
```

Example:

```text
row_id  zone  origin_time  event_time  horizon_step
001     DE    08:00        12:00       4
002     DE    08:00        13:00       5
003     DE    09:00        12:00       3
```

Those fields are not automatically model features.

`X` might contain:

```text
wind_fc
solar_fc
load_fc
exchange_fc
```

If the user wants lead time as a feature, they explicitly request it through the existing derived-feature mechanism.

---

## 18.5 PIT materialization

Given:

```text
ref_time target_time wind_fc load_fc price

08:00    12:00       NaN     54      80
08:00    13:00       NaN     53      76

09:00    12:00       11      55      80
09:00    13:00       12      54      76
```

produce:

```text
X
────────────────
wind_fc load_fc
NaN     54
NaN     53
11      55
12      54
```

and:

```text
y
──
80
76
80
76
```

The duplicated targets are intentional.

These are four distinct forecast examples because their information vintages differ.

No deduplication occurs based on `target_time`.

---

## 18.6 Public API

This should work directly:

```python
model = client.fit(
    model="sklearn/hist-gradient-boosting",
    data=forecast_dataset,
    horizon=72,
    params={
        "learning_rate": 0.05,
        "max_iter": 500,
        "max_leaf_nodes": 31,
    },
)
```

Forecast:

```python
context = forecast_dataset.at_origin(current_ref_time)

forecast = client.forecast(
    model=model,
    data=context,
    horizon=72,
)
```

The provider effectively only needs to do:

```python
estimator.fit(X, y)
```

and later:

```python
prediction = estimator.predict(X)
```

---

## 18.7 Keep `Reduction`, but narrow its purpose

Do **not** require:

```python
of.Reduction(...)
```

for PIT data that already contains production forecasting features.

This works directly:

```python
client.fit(
    model="sklearn/hist-gradient-boosting",
    data=forecast_dataset,
)
```

`Reduction` is for cases where OpenForecast must create a tabular problem from an ordinary event-time series.

Example:

```python
recipe = of.Reduction(
    estimator="sklearn/hist-gradient-boosting",
    strategy="recursive",
    lags=[1, 24, 168],
)
```

Then:

```text
TimeSeriesFrame
      ↓
Reduction
      ↓
OpenForecast generates lagged supervised rows
      ↓
TabularView
      ↓
sklearn
```

Whereas:

```text
ForecastDataset
      ↓
TabularView directly
      ↓
sklearn
```

---

## 18.8 Be conservative with PIT lag generation

Do not automatically generate target lags from `ForecastDataset.truth`.

For PIT data, there is a subtle distinction between:

```text
event happened before origin
```

and:

```text
the realized value was actually available at origin
```

So initially:

```text
ForecastDataset
    → use the supplied PIT features

TimeSeriesFrame + Reduction
    → OpenForecast may generate lags
```

Avoid inventing an availability model before it is needed.

---

## 18.9 Persistence

Inside provider artifact:

```text
provider/
    estimator.pkl
    metadata.json
```

OpenForecast's artifact manifest remains responsible for:

```text
model ref
recipe
provider version
training view
origin fidelity
schema
horizon
```

---

## 18.10 Required tests

Test:

* multiple origins forecasting the same target remain separate rows;
* NaN patterns reach sklearn unchanged;
* target values join correctly;
* prediction rows map back to the correct instance/origin/event time;
* provider never imports `ForecastDataset` or `PointInTimeFrame`;
* no sktime code executes in this path.

### Done when

This works:

```python
model = client.fit(
    "sklearn/hist-gradient-boosting",
    data=pit_dataset,
    horizon=72,
)

forecast = client.forecast(
    model=model,
    data=pit_dataset.at_origin(current_ref),
    horizon=72,
)
```

---

# Step 19 — Leakage-safe point-in-time backtesting

## Goal

Make historical evaluation a first-class OpenForecast capability with genuine point-in-time correctness.

Add only one major public operation:

```python
client.backtest(...)
```

Reuse the existing:

```text
origin selectors
FitPlan
ViewPlanner
ForecastContext
Forecast
```

rather than building another validation framework.

---

## 19.1 Public API

Example:

```python
result = client.backtest(
    model="sklearn/hist-gradient-boosting",
    data=pit_dataset,
    origins=of.OriginsBetween(
        start="2026-01-01",
        end="2026-07-01",
        stride=24,
    ),
    horizon=72,
    metrics=[
        "mae",
        "rmse",
        "bias",
    ],
)
```

Sequence model:

```python
result = client.backtest(
    model="nixtla/nhits",
    data=pit_dataset,
    origins=of.AllOrigins(stride=24),
    horizon=72,
    fit_plan=of.FitPlan(
        sequences=of.SequencePlan(
            context=168,
        ),
    ),
    metrics=[
        "mae",
        "bias",
    ],
)
```

Use metric names as strings.

Do not introduce:

```text
MAE()
RMSE()
MetricRegistry
SlicePlan
ValidationStrategy
```

unless later requirements prove they are needed.

---

## 19.2 Historical PIT forecast semantics

For validation origin:

```text
2026-05-15 11:00
```

the backtester must create:

```python
context = dataset.at_origin(
    "2026-05-15T11:00"
)
```

That exact object passes through the normal forecast path.

Therefore:

```text
Backtester
    ↓
ForecastContext
    ↓
ViewPlanner
    ↓
SeriesView / SequenceView / TabularView
    ↓
provider
```

There is no alternate "historical prediction" implementation.

This is important because backtesting and production forecasting should exercise exactly the same materialization logic.

---

## 19.3 Prevent training leakage

For validation origin `T`, training should conservatively use only rows satisfying:

```text
origin_time < T
```

and labels satisfying:

```text
event_time < T
```

for V1.

This establishes a safe baseline.

Later, OpenForecast may support explicit target publication/availability semantics, but do not add that yet.

---

## 19.4 Event-time datasets

`TimeSeriesFrame` works through the same API.

OpenForecast simulates historical origins.

So:

```text
ForecastDataset
    → observed historical information sets

TimeSeriesFrame
    → simulated historical information sets
```

Results record:

```text
origin_fidelity = observed | simulated
```

---

## 19.5 `BacktestResult`

Add only one public result primitive:

```python
BacktestResult
```

Predictions are Arrow-backed:

```text
model
fold
instance keys...
origin_time
event_time
horizon_step
target
prediction
actual
```

Metrics:

```text
model
fold
metric
value
```

Convenience:

```python
result.predictions
result.metrics
result.to_pandas()
```

For grouped analysis:

```python
result.metrics_by("horizon_step")
```

or:

```python
result.metrics_by(
    ["horizon_step", "zone"]
)
```

Avoid creating a slicing DSL.

---

## 19.6 Critical leakage tests

Use poisoned vintages.

Example:

```text
08:00 → wind = 10
09:00 → wind = 20
10:00 → wind = 999999
```

Evaluate at:

```text
09:00
```

Assert provider input contains:

```text
20
```

and can never contain:

```text
999999
```

This should become a permanent high-level conformance test.

---

## 19.7 Frozen artifacts

If:

```python
model="local/de-price@..."
```

is passed to `backtest()`, evaluate the frozen artifact.

Do not refit it.

If a trainable model definition/recipe is passed, fit per fold.

Infer this from model lifecycle rather than introducing a separate mode argument.

---

## Done when

All model families can be evaluated using the same historical-origin semantics, with a hard PIT leakage guarantee.

---

# Step 20 — Complete probabilistic forecast normalization

## Goal

Finish `Forecast` so models producing:

```text
point predictions
quantiles
samples
```

are interoperable.

Do not create separate result classes for each form.

Reuse:

```text
Forecast
OutputSpec
```

---

## 20.1 Public requests

Support:

```python
of.OutputSpec.point()
```

```python
of.OutputSpec.quantiles(
    [0.1, 0.5, 0.9]
)
```

```python
of.OutputSpec.samples(100)
```

Do not add parametric distribution objects yet.

---

## 20.2 Canonical Arrow representation

Use one long-format representation:

```text
instance keys...
origin_time
event_time
target
kind
quantile
sample_id
value
```

Point:

```text
DE 11:00 12:00 price point null null 80
```

Quantiles:

```text
DE 11:00 12:00 price quantile 0.1 null 65
DE 11:00 12:00 price quantile 0.5 null 79
DE 11:00 12:00 price quantile 0.9 null 96
```

Samples:

```text
DE 11:00 12:00 price sample null 0 72
DE 11:00 12:00 price sample null 1 84
DE 11:00 12:00 price sample null 2 79
```

---

## 20.3 Capability validation

Existing model capability metadata determines what can be requested:

```yaml
outputs:
  point: true
  quantiles: true
  samples: false
```

If a user asks for samples from a model without sample support:

```python
output=of.OutputSpec.samples(100)
```

raise before provider execution.

---

## 20.4 Safe generic conversion

Allow:

```text
samples
  ↓
quantiles
```

because quantiles can be calculated from samples.

Do not do:

```text
quantiles
  ↓
invented samples
```

and do not manufacture probabilistic output from deterministic models.

---

## 20.5 Backtesting integration

Add metrics:

```text
pinball
coverage
interval_width
```

Example:

```python
result = client.backtest(
    model="nixtla/nhits",
    data=dataset,
    horizon=72,
    output=of.OutputSpec.quantiles(
        [0.1, 0.5, 0.9]
    ),
    metrics=[
        "mae",
        "pinball",
        "coverage",
    ],
)
```

Still use string metric identifiers.

---

## 20.6 sklearn

Do not complicate the sklearn provider just to make it probabilistic.

The purpose of Step 20 is primarily to normalize providers that already have meaningful native probabilistic outputs.

A proper deterministic→probabilistic calibration layer can come later.

---

## Done when

Downstream application code can consume exactly the same `Forecast` regardless of which probabilistic provider produced it.

---

# Step 21 — Simple cross-provider ensembles

## Goal

Prove that OpenForecast can compose models consuming different execution views because they converge on a common `Forecast`.

Keep this stage deliberately simple.

No stacking.

No dynamic weights.

No mixture-of-experts.

---

## 21.1 Simplify `Ensemble`

Use:

```python
recipe = of.Ensemble(
    models=[
        of.Model("nixtla/nhits"),
        of.Model(
            "sklearn/hist-gradient-boosting"
        ),
    ],
)
```

Default behavior:

```text
equal averaging
```

Optional fixed weights:

```python
recipe = of.Ensemble(
    models=[
        of.Model("nixtla/nhits"),
        of.Model(
            "sklearn/hist-gradient-boosting"
        ),
    ],
    weights=[0.7, 0.3],
)
```

Avoid requiring:

```python
of.Mean()
of.WeightedMean()
```

as extra primitives.

---

## 21.2 Each child plans independently

For:

```text
nixtla/nhits
```

the planner generates:

```text
SequenceView
```

For:

```text
sklearn/hist-gradient-boosting
```

it generates:

```text
TabularView
```

So:

```text
                         ForecastDataset
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
             nixtla/nhits        sklearn/hgb
                    │                     │
             SequenceView           TabularView
                    │                     │
                    ▼                     ▼
                Forecast              Forecast
                    └──────────┬──────────┘
                               ▼
                             mean
```

This is an important architecture milestone.

---

## 21.3 Parent artifact

Store child model references rather than binaries:

```json
{
  "type": "ensemble",
  "children": [
    "local/model-a@...",
    "local/model-b@..."
  ],
  "weights": [
    0.5,
    0.5
  ]
}
```

---

## 21.4 Compatibility checking

Validate all children before starting training.

For example, if a child only supports:

```text
origin_scope=single
```

but the requested `FitPlan` contains all PIT origins, reject the entire ensemble rather than silently changing how that child is trained.

---

## 21.5 Quantile averaging

For quantiles, require children to produce the same requested levels.

Then:

```text
ensemble P10 = weighted mean(child P10)
ensemble P50 = weighted mean(child P50)
ensemble P90 = weighted mean(child P90)
```

Document that this is **quantile averaging**, not an exact mixture distribution.

Do not combine samples yet.

---

## 21.6 Backtesting

The existing API simply works:

```python
result = client.backtest(
    model=recipe,
    data=pit_dataset,
    origins=...,
    horizon=72,
    metrics=["mae", "bias"],
)
```

The backtester should not need ensemble-specific evaluation logic beyond calling the normal model lifecycle.

---

## Done when

One persisted model can combine a `SequenceView` child and a `TabularView` child without either provider knowing that an ensemble exists.

---

# Step 22 — Intentionally deferred

Do not fill this number merely because it exists.

The functionality previously considered for this stage—hyperparameter search—adds considerable new API and execution complexity.

Explicitly defer:

```text
hyperparameter optimization
search spaces
trial orchestration
learned ensemble weights
stacking
```

until there is evidence they belong in core OpenForecast.

---

# Step 23 — First zero-shot foundation model

## Goal

Prove the second major model lifecycle:

```text
trainable model
    requires fit

pretrained model
    forecast immediately
```

Start with **one** foundation-model integration.

Do not add multiple providers at once.

---

## 23.1 Add Chronos integration

Create:

```text
integrations/
    chronos/
        pyproject.toml
        uv.lock

        src/
            openforecast_chronos/
                __main__.py
                provider.py
                catalog.py
                adapter.py
```

Expose a clean model ref such as:

```text
amazon/chronos-2
```

for the selected supported checkpoint/version.

---

## 23.2 Lifecycle

Descriptor:

```yaml
lifecycle:
  requires_fit: false
  supports_fit: false
```

Even if underlying Chronos supports fine-tuning, don't expose it yet.

The integration initially supports:

```text
zero-shot inference
```

only.

This keeps lifecycle semantics clean.

---

## 23.3 Allow model descriptors without training contracts

Change:

```python
training: TrainingContract
```

to:

```python
training: TrainingContract | None
```

For pretrained-only models:

```python
training = None
```

Do not invent a fake training contract.

---

## 23.4 Public UX

This should work:

```python
forecast = client.forecast(
    model="amazon/chronos-2",
    data=context,
    horizon=72,
)
```

without calling `fit()`.

This should fail:

```python
client.fit(
    model="amazon/chronos-2",
    data=data,
)
```

with a structured:

```text
MODEL_DOES_NOT_SUPPORT_FIT
```

error.

---

## 23.5 No foundation-model-specific data primitive

Do **not** introduce:

```text
FoundationView
PretrainedView
ChronosFrame
```

Use the existing forecasting-context path:

```text
ForecastContext
      ↓
ViewPlanner / normalized forecast input
      ↓
Chronos adapter
```

The adapter handles whatever native structure Chronos requires.

---

## 23.6 PIT should work automatically

```python
context = pit_dataset.at_origin(
    historical_ref_time
)

forecast = client.forecast(
    model="amazon/chronos-2",
    data=context,
    horizon=72,
)
```

OpenForecast owns the information-vintage semantics.

Chronos receives the correct current information set.

---

## 23.7 PIT backtesting

Because Step 19 already exists:

```python
result = client.backtest(
    model="amazon/chronos-2",
    data=pit_dataset,
    origins=of.AllOrigins(stride=24),
    horizon=72,
    metrics=[
        "mae",
        "bias",
    ],
)
```

No training folds are necessary.

For every origin:

```text
ForecastContext
      ↓
zero-shot forecast
      ↓
truth
```

This lets users compare zero-shot foundation models fairly against fitted classical/tabular/sequence models.

---

## 23.8 Probabilistic output

Map native Chronos output into the Step 20 protocol.

Never expose Chronos-specific forecast objects publicly.

---

## Done when

The same PIT dataset can evaluate:

```text
sklearn/hist-gradient-boosting
nixtla/nhits
darts/...
amazon/chronos-2
```

behind one forecast/backtest interface.

---

# Step 24 — Freeze and simplify the public Python SDK

At this point, the forecasting architecture is quite broad.

Now stop adding modeling capabilities temporarily and make the developer interface extremely predictable.

## Goal

Establish one canonical Python API for every common intent.

I would use:

```python
import openforecast as of

client = of.OpenForecast()
```

Then exactly:

```python
client.models.list()
client.models.get(...)

client.fit(...)
client.forecast(...)
client.backtest(...)
```

Data construction remains:

```python
of.TimeSeriesFrame.from_pandas(...)
of.ForecastDataset.from_pandas(...)
```

---

## 24.1 One method per intent

Avoid aliases.

Don't have:

```text
fit()
train()

forecast()
predict()
infer()

backtest()
evaluate()
historical_forecasts()
```

Choose one term.

Agents benefit enormously from there being one obvious action.

---

## 24.2 Audit public types

Aim for a small public surface such as:

```text
OpenForecast

TimeSeriesFrame
ForecastDataset
ForecastContext

Model
Pipeline
Ensemble
Reduction

FitPlan
SequencePlan

OutputSpec

Forecast
BacktestResult

origin selectors
a small set of transforms
```

Internal implementation classes should stay internal:

```text
ViewPlanner
SequenceView
SeriesView
TabularView
SubprocessProviderClient
```

These may be documented conceptually without being public SDK objects.

---

## 24.3 Freeze exports

Maintain:

```python
openforecast.__all__
```

and test it.

That prevents accidental public API growth.

---

## 24.4 Ergonomic signatures

Common simple cases should remain short:

```python
model = client.fit(
    "sklearn/hist-gradient-boosting",
    data=data,
    horizon=24,
)
```

not require users to construct five request objects.

More explicit forms remain available when needed.

---

## Done when

The entire normal workflow can be taught on one README page.

---

# Step 25 — Version-controlled documentation as code

## Goal

Documentation changes with the implementation, is tested in CI, and can be accessed for every released OpenForecast version.

## Structure

```text
docs/
├── index.md
│
├── getting-started/
│   ├── installation.md
│   ├── quickstart.md
│   └── concepts.md
│
├── guides/
│   ├── event-time.md
│   ├── point-in-time.md
│   ├── fitting.md
│   ├── forecasting.md
│   ├── backtesting.md
│   ├── probabilistic.md
│   └── ensembles.md
│
├── concepts/
│   ├── data-model.md
│   ├── point-in-time.md
│   ├── execution-views.md
│   ├── model-lifecycle.md
│   └── providers.md
│
├── integrations/
│   ├── nixtla.md
│   ├── darts.md
│   ├── sktime.md
│   ├── sklearn.md
│   └── chronos.md
│
└── reference/
    └── generated/
```

---

## 25.1 Separate docs by purpose

### Tutorials

Complete runnable workflows.

### Guides

“How do I do X?”

### Concepts

“Why does OpenForecast work this way?”

### Reference

Exact signatures/types generated from code.

This keeps docs useful to both humans and agents.

---

## 25.2 Generate reference documentation

Generate from:

```text
Python type annotations
docstrings
Pydantic models
model descriptors
```

Do not manually duplicate signatures.

---

## 25.3 Version documentation

Published docs should retain:

```text
latest
v0.1
v0.2
...
```

So an agent using:

```text
openforecast==0.3
```

can retrieve the matching documentation.

---

## 25.4 CI

Every docs build should:

```text
build successfully
check links
execute code examples
verify generated reference is current
```

If an API example no longer executes, CI should fail.

---

## Done when

Documentation drift becomes difficult to introduce accidentally.

---

# Step 26 — Build a boring, deterministic CLI

## Goal

Make OpenForecast extremely easy to use from:

```text
terminal
scripts
CI
coding agents
```

The CLI should be intentionally uncreative.

## Initial commands

```bash
openforecast models list
openforecast models get nixtla/nhits

openforecast providers list
openforecast providers install nixtla

openforecast fit
openforecast forecast
openforecast backtest

openforecast doctor
```

Avoid deep trees of nested commands.

---

## 26.1 Reuse SDK semantics

The CLI must call:

```text
OpenForecast Python SDK
```

rather than implementing another execution path.

Likewise config files deserialize into the same Pydantic types already used by the SDK.

---

## 26.2 Config for complex commands

Simple:

```bash
openforecast models get nixtla/nhits
```

Complex:

```bash
openforecast fit --config fit.json
```

Example:

```json
{
  "model": "nixtla/nhits",
  "horizon": 24,
  "data": "./dataset",
  "plan": {
    "sequences": {
      "context": 168
    }
  }
}
```

Don't create dozens of CLI flags for nested recipes.

---

## 26.3 `--json` everywhere

Human:

```bash
openforecast models get nixtla/nhits
```

Agent:

```bash
openforecast models get nixtla/nhits --json
```

Structured result:

```json
{
  "ref": "nixtla/nhits",
  "lifecycle": {
    "requires_fit": true
  },
  "training": {
    "view": "sequences",
    "origin_scope": "multiple"
  }
}
```

Every information-producing command should support machine-readable output.

---

## 26.4 stdout/stderr contract

Enforce:

```text
stdout = requested output
stderr = logs/progress/warnings
```

This allows:

```bash
openforecast models list --json | jq ...
```

to remain reliable.

---

## 26.5 Exit codes

Define stable exit behavior:

```text
0 success
non-zero failure
```

Do not use stdout prose to communicate failure state.

---

## Done when

An agent can perform the normal OpenForecast workflow entirely through shell commands and structured JSON.

---

# Step 27 — Machine-readable schemas and structured errors

## Goal

Make OpenForecast discoverable without requiring an agent to reverse-engineer Python signatures or prose.

This is one of the most important agent-readiness steps.

---

## 27.1 Generate JSON Schema

Automatically generate schemas for existing protocol objects:

```text
Fit request
Forecast request
Backtest request
ModelRecipe
FitPlan
OutputSpec
TimeSeriesSchema
ModelDescriptor
```

Store:

```text
spec/
    schemas/
        fit-request.json
        forecast-request.json
        backtest-request.json
        model-recipe.json
        model-descriptor.json
```

Generated from Pydantic.

Never maintain these manually.

---

## 27.2 CLI schema discovery

Add:

```bash
openforecast schema fit --json
openforecast schema forecast --json
openforecast schema backtest --json
```

An agent can now:

```text
inspect schema
    ↓
construct request
    ↓
execute
```

without guessing.

---

## 27.3 One error protocol

Every OpenForecast error should expose:

```text
code
message
details
```

Example:

```json
{
  "code": "MODEL_REQUIRES_FIT",
  "message": "nixtla/nhits must be fitted before forecasting.",
  "details": {
    "model": "nixtla/nhits"
  }
}
```

Python:

```python
try:
    ...
except OpenForecastError as exc:
    print(exc.code)
    print(exc.message)
    print(exc.details)
```

CLI JSON error:

```bash
openforecast ... --json
```

returns the same logical envelope.

HTTP later uses the same structure.

---

## 27.4 Stable error codes

Examples:

```text
MODEL_NOT_FOUND
MODEL_REQUIRES_FIT
MODEL_DOES_NOT_SUPPORT_FIT

UNSUPPORTED_DATA_SHAPE
UNSUPPORTED_FEATURE
UNSUPPORTED_OUTPUT

INVALID_MODEL_PARAMETERS
INVALID_DATA

ORIGIN_SCOPE_ERROR
INCOMPATIBLE_FORECAST_TASK

PROVIDER_NOT_INSTALLED
PROVIDER_EXECUTION_FAILED
```

Agents should be able to recover based on `code`, not string-match prose.

---

## Done when

Every executable API has a discoverable schema and every expected failure has structured machine-readable semantics.

---

# Step 28 — Executable canonical examples

## Goal

Provide a small corpus of short, complete examples that agents can copy, run, adapt, and trust.

Create:

```text
examples/
    01_quickstart.py
    02_panel.py
    03_point_in_time.py
    04_backtest.py
    05_probabilistic.py
    06_ensemble.py
    07_zero_shot.py
```

Don't create dozens.

Seven excellent examples are more useful than fifty stale ones.

---

## 28.1 PIT example

For example:

```python
import openforecast as of

client = of.OpenForecast()

dataset = of.ForecastDataset.from_pandas(
    df,
    origin_time="ref_time",
    event_time="target_time",
    targets=["price"],
    known_features=[
        "wind_fc",
        "load_fc",
    ],
    event_frequency="1h",
)

model = client.fit(
    "sklearn/hist-gradient-boosting",
    data=dataset,
    horizon=24,
)

forecast = client.forecast(
    model=model,
    data=dataset.at_origin(latest_origin),
    horizon=24,
)
```

This should be complete enough to run.

---

## 28.2 Tiny deterministic data

Examples should generate or bundle small datasets.

An agent cloning the repository should be able to run:

```bash
uv run examples/03_point_in_time.py
```

without finding external datasets.

---

## 28.3 Execute examples in CI

Every example is tested.

An API change that breaks an example breaks CI.

---

## 28.4 Avoid example duplication

Where practical, documentation should incorporate tested examples rather than copying slightly different versions manually.

---

## Done when

An agent can learn most of OpenForecast by reading a handful of executable examples.

---

# Step 29 — Agent-readable documentation and discovery

## Goal

Make the documentation itself easy for LLM-based agents to retrieve and consume.

This is an additional interface over the docs from Step 25, not a separate documentation system.

---

## 29.1 Markdown-first accessibility

Ensure documentation pages have clean Markdown source and can be accessed without needing to parse large client-rendered pages.

---

## 29.2 Publish `llms.txt`

Generate:

```text
/llms.txt
```

and optionally:

```text
/llms-full.txt
```

from the actual docs navigation.

Example structure:

```text
# OpenForecast

> Unified forecasting protocol with first-class point-in-time data.

## Getting started
- Quickstart
- Installation
- Point-in-time forecasting

## API
- Models
- Fit
- Forecast
- Backtest

## Concepts
- ForecastDataset
- SequenceView
- TabularView
- Model lifecycle
```

Generate this file.

Do not maintain it manually.

---

## 29.3 Agent docs section

Add concise documentation focused on agent usage:

```text
docs/agents/
    overview.md
    choosing-a-model.md
    point-in-time.md
    structured-cli.md
    errors.md
```

These should emphasize:

```text
how to discover
how to choose
how to construct
how to recover
```

rather than long narrative explanations.

---

## Done when

An agent encountering OpenForecast for the first time can quickly locate the exact canonical docs relevant to its task.

---

# Step 30 — Capability-driven model discovery

## Goal

Make the existing model registry rich enough that an agent can decide which models are applicable without trying them.

Do not add a model-search DSL yet.

Simply ensure:

```python
client.models.list()
```

and:

```bash
openforecast models list --json
```

return sufficient structured metadata.

---

## 30.1 Model listing

Example:

```json
[
  {
    "ref": "nixtla/nhits",
    "lifecycle": {
      "requires_fit": true
    },
    "training": {
      "view": "sequences",
      "origin_scope": "multiple"
    },
    "capabilities": {
      "instances": {
        "single": true,
        "panel": true
      },
      "features": {
        "observed": true,
        "known": true,
        "static": true
      },
      "outputs": {
        "point": true,
        "quantiles": true,
        "samples": false
      }
    }
  }
]
```

An agent can filter this itself.

---

## 30.2 Model inspection

```python
descriptor = client.models.get(
    "nixtla/nhits"
)
```

should also expose the model's parameter schema.

Then an agent can discover:

```text
model exists
model requires fit
model handles PIT multiple origins
model accepts known covariates
model supports quantiles
model accepts these parameters
```

without reading integration source code.

---

## 30.3 Installed vs available

Model discovery should clearly distinguish:

```text
available and installed
known but provider not installed
```

so agents can determine whether:

```bash
openforecast providers install nixtla
```

is required.

---

## 30.4 Don't add query methods yet

Avoid:

```python
client.models.search(
    point_in_time=True,
    panel=True,
    quantiles=True,
)
```

for now.

Structured lists are already easy for agents to filter and keep the Python API much smaller.

---

## Done when

An agent can programmatically answer:

> Which installed models are capable of fitting this dataset and producing the requested forecast output?

without running any model.

---

# Step 31 — Thin MCP server over the existing SDK

## Goal

Expose OpenForecast directly to MCP-compatible agents without introducing any MCP-specific forecasting semantics.

Only do this **after** the Python API, CLI, schemas, model descriptors, and structured errors are stable.

Architecture:

```text
Agent
  │
  ▼
MCP adapter
  │
  ▼
OpenForecast SDK
  │
  ▼
Engine
```

Not:

```text
Agent
  ↓
new MCP forecasting implementation
```

---

## 31.1 Initial tools

Keep the tool set tiny:

```text
models_list
model_get

fit
forecast
backtest
```

Possibly:

```text
providers_list
```

That's enough.

---

## 31.2 Reuse schemas

Input/output schemas should be generated from the same underlying Pydantic types already used by:

```text
Python
CLI
HTTP
```

Do not manually define separate MCP schemas.

---

## 31.3 Structured errors

MCP tool failures should preserve the same:

```text
code
message
details
```

semantics from Step 27.

---

## 31.4 Optional documentation resources

Later, expose resources such as:

```text
openforecast://docs/quickstart
openforecast://models
openforecast://models/nixtla/nhits
```

but treat this as secondary.

The first milestone is reliable tool execution.

---

## Done when

An MCP agent can discover available forecasting models, inspect them, fit models, issue forecasts, and run PIT-safe backtests using exactly the same OpenForecast engine as Python users.

---

# Phase Two dependency order

The final sequence is:

```text
PHASE ONE
Steps 1–17
Core protocol + providers + PIT architecture
        │
        ▼

18
sklearn / native TabularView
        │
        │ proves direct supervised forecasting
        ▼

19
PIT-safe backtesting
        │
        │ proves historical correctness
        ▼

20
Probabilistic normalization
        │
        │ completes Forecast protocol
        ▼

21
Cross-provider ensembles
        │
        │ proves composability
        ▼

22
DEFERRED
        │
        ▼

23
Zero-shot foundation model
        │
        │ proves pretrained/no-fit lifecycle
        ▼

────────────────────────────────────
       INTERFACE HARDENING
────────────────────────────────────

24
Freeze simple Python SDK
        │
        ▼

25
Versioned docs-as-code
        │
        ▼

26
Deterministic CLI + --json
        │
        ▼

27
JSON Schema + structured errors
        │
        ▼

28
Executable canonical examples
        │
        ▼

29
Agent-readable docs + llms.txt
        │
        ▼

30
Capability-driven model discovery
        │
        ▼

31
Thin MCP adapter
```

## What I would deliberately **not** add during Phase Two

This matters as much as the roadmap itself.

I would defer:

```text
hyperparameter optimization
generic search spaces

openforecast/auto
automatic model routing

model update / partial_fit lifecycle
retraining schedules

cloud execution
distributed jobs

generic provider authoring SDK

stacking
learned ensemble weights
mixture-of-experts

natural-language-to-recipe layer

LangChain/CrewAI/AutoGen-specific integrations

custom agent framework
```

None of those are necessary to prove the central idea.

After Step 31, OpenForecast would already have a very strong shape:

```text
                         OpenForecast

              ┌─────────────────────────┐
              │  Forecasting semantics  │
              │                         │
              │ event-time + PIT        │
              └────────────┬────────────┘
                           ▼

                     ViewPlanner

          ┌────────────────┼─────────────────┐
          ▼                ▼                 ▼

     SeriesView       SequenceView       TabularView

          │                │                 │
          ▼                ▼                 ▼

      classical        global/NN         supervised ML
       models            models              models

          │                │                 │
          └────────────────┼─────────────────┘
                           ▼

                       Forecast
               point / quantiles / samples
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
             backtesting         ensembles

                           +

                   pretrained models
                       zero-shot

                           │
                           ▼

                  ONE PUBLIC INTERFACE

                   Python SDK
                       │
                   CLI + JSON
                       │
                   JSON Schema
                       │
                versioned docs
                       │
               executable examples
                       │
                  MCP adapter
```

The central product idea at that point becomes quite crisp: **OpenForecast is not just a wrapper around forecasting libraries; it is a stable forecasting protocol that humans and agents can use to discover, train, evaluate, compose, and execute fundamentally different forecasting approaches while preserving production-grade point-in-time semantics.**
