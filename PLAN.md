Absolutely. Below is the **full revised 1–17 implementation sequence**, with point-in-time forecasting built in as a first-class concept from the start rather than bolted on later.

The architectural invariant for the entire implementation is:

> **OpenForecast owns forecasting semantics. Providers only consume provider-neutral execution views. Point-in-time and ordinary event-time data are materialized into those views before crossing the provider boundary.**

And:

> **Providers must never branch on whether source data came from a `TimeSeriesFrame` or a `ForecastDataset`.**

---

# Step 1 — Repository foundation and architecture boundaries

## Goal

Create the OpenForecast repository, package structure, dependency boundaries, CI, typing, linting, testing, and architectural rules.

At the end of this stage, no forecasting functionality exists yet, but the repository structure makes it difficult to violate the intended architecture later.

## Repository structure

Create:

```text
openforecast/
│
├── pyproject.toml
├── uv.lock
├── README.md
├── ARCHITECTURE.md
│
├── src/
│   └── openforecast/
│       ├── __init__.py
│       ├── client.py
│       │
│       ├── data/
│       ├── views/
│       ├── models/
│       ├── recipes/
│       ├── tasks/
│       ├── artifacts/
│       ├── registry/
│       ├── runtime/
│       ├── protocol/
│       ├── commands/
│       └── server/
│
├── integrations/
│   ├── nixtla/
│   ├── darts/
│   └── sktime/
│
├── tests/
│   ├── unit/
│   ├── contract/
│   ├── conformance/
│   └── e2e/
│
└── spec/
    ├── protocol/
    ├── arrow/
    └── openapi/
```

The main Python distribution is simply:

```text
openforecast
```

Do not create separate internal packages named:

```text
openforecast_core
openforecast_api
openforecast_cli
```

The CLI lives under:

```text
openforecast.commands
```

The HTTP server lives under:

```text
openforecast.server
```

## Dependencies

Root package should stay lightweight:

```toml
dependencies = [
    "pydantic>=2",
    "pyarrow",
    "platformdirs",
]
```

Development tooling:

```text
pytest
pytest-cov
hypothesis
ruff
pyright
```

Do not install:

```text
statsforecast
neuralforecast
darts
sktime
torch
jax
lightgbm
```

into the root OpenForecast environment.

Each integration gets its own `pyproject.toml`, `uv.lock`, and isolated runtime.

## Architectural rules

Write these explicitly into `ARCHITECTURE.md`.

1. OpenForecast semantic types must never import provider libraries.
2. Providers consume execution views, not source semantic datasets.
3. Providers must not branch on `TimeSeriesFrame` versus `ForecastDataset`.
4. Point-in-time vintages must never be silently replaced by newer information.
5. Missing values must never be silently imputed.
6. Provider-specific terminology must not leak into the public OpenForecast protocol.
7. OpenAPI is a projection of OpenForecast semantics, not their source.

## Architecture tests

Add tests that fail if root imports:

```text
neuralforecast
statsforecast
darts
sktime
torch
jax
```

Later add equivalent tests ensuring provider packages cannot import source PIT semantic classes directly.

## CI

CI must run:

```bash
uv sync
uv run ruff check .
uv run pyright
uv run pytest
```

## Done when

The repository installs cleanly and all architecture/quality checks pass.

---

# Step 2 — Event-time semantic model

## Goal

Implement OpenForecast's basic event-time time-series primitive:

```text
instance × event_time × variable
```

This represents ordinary time-series data.

It deliberately does **not** represent forecast vintages.

## Implement frequency

```python
class FrequencyUnit(StrEnum):
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class Frequency(BaseModel):
    unit: FrequencyUnit
    step: int = 1
```

Support parsing convenience strings such as:

```python
Frequency.parse("15m")
Frequency.parse("1h")
```

but store OpenForecast-native semantics internally.

## Feature semantics

```python
class FeatureAvailability(StrEnum):
    OBSERVED = "observed"
    KNOWN = "known"


class FeatureKind(StrEnum):
    TEMPORAL = "temporal"
    STATIC = "static"


class FeatureSpec(BaseModel):
    name: str
    kind: FeatureKind = FeatureKind.TEMPORAL
    availability: FeatureAvailability | None = None
```

Rules:

```text
temporal -> availability required
static   -> availability must be None
```

## `TimeSeriesSchema`

```python
class TimeSeriesSchema(BaseModel):
    time: str
    frequency: Frequency

    instance_keys: tuple[str, ...] = ()
    targets: tuple[str, ...]
    features: tuple[FeatureSpec, ...] = ()
```

Derived properties:

```python
schema.is_panel
schema.is_univariate
schema.is_multivariate
schema.target_count

schema.has_observed_features
schema.has_known_features
schema.has_static_features
```

Do not create separate semantic enums like:

```text
PANEL_MULTIVARIATE
```

These properties are derived from orthogonal axes.

## `TimeSeriesFrame`

```python
class TimeSeriesFrame:
    history: pa.Table
    future: pa.Table | None
    static: pa.Table | None
    schema: TimeSeriesSchema
```

Canonical history layout:

```text
instance_keys...
event_time
target columns...
observed features...
known features...
```

Future table:

```text
instance_keys...
event_time
known temporal features...
```

Static table:

```text
instance_keys...
static features...
```

## Public API

```python
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
```

## Validation

Implement:

```text
required columns exist
targets unique
features unique
target cannot also be feature
no duplicate instance/time rows
timestamps satisfy declared frequency
future contains no target columns
future contains no observed-only features
static contains one row per instance
```

Do not silently repair malformed data.

## Serialization

```python
frame.write(path)
frame = TimeSeriesFrame.read(path)
```

using:

```text
schema.json
history.arrow
future.arrow
static.arrow
```

## Tests

Test:

```text
single univariate
single multivariate
panel univariate
panel multivariate
static/known/observed features
Arrow round-trip
```

Use Hypothesis for shape/property tests.

## Done when

`TimeSeriesFrame` completely represents ordinary event-time time series without any provider dependency.

---

# Step 3 — Point-in-time semantic model

## Goal

Make point-in-time forecasting a first-class OpenForecast concept.

Represent:

```text
instance × origin_time × event_time × variable
```

where:

```text
origin_time = when information was available
event_time  = what time the information refers to
```

## Implement `PointInTimeSchema`

```python
class PointInTimeSchema(BaseModel):
    origin_time: str
    event_time: str

    event_frequency: Frequency
    origin_frequency: Frequency | None = None

    instance_keys: tuple[str, ...] = ()
    features: tuple[FeatureSpec, ...]
```

## Implement `PointInTimeFrame`

```python
class PointInTimeFrame:
    table: pa.Table
    schema: PointInTimeSchema
```

Canonical Arrow key:

```text
(instance_keys..., origin_time, event_time)
```

Example:

```text
zone origin_time event_time wind_fc load_fc
DE   08:00       12:00      10.1    54.2
DE   09:00       12:00      11.7    54.8
DE   10:00       12:00      12.4    55.1
```

Preserve NaNs exactly.

## Lead time

Do not store `lead_time` as a required structural column.

Derive:

```python
lead = event_time - origin_time
```

Expose:

```python
pit.with_lead_time(unit="hour")
```

## Implement `ForecastDataset`

```python
class ForecastDataset:
    information: PointInTimeFrame
    truth: TimeSeriesFrame
```

This deliberately separates:

```text
what was knowable
```

from:

```text
what actually happened
```

## Existing `(ref_time, target_time)` convenience constructor

Support:

```python
dataset = of.ForecastDataset.from_pandas(
    df,
    origin_time="ref_time",
    event_time="target_time",
    instance_keys=["zone"],
    targets=["price"],
    known_features=[
        "wind_fc",
        "solar_fc",
        "load_fc",
    ],
    observed_features=[],
    event_frequency="1h",
    origin_frequency="1h",
)
```

If:

```text
08:00 -> 12:00 -> price 80
09:00 -> 12:00 -> price 80
```

extract one truth row:

```text
12:00 -> price 80
```

If labels disagree:

```text
08:00 -> 12:00 -> 80
09:00 -> 12:00 -> 81
```

raise:

```text
InconsistentTruthError
```

Do not arbitrarily choose one.

## Implement `ForecastContext`

Represent exactly one inference origin.

```python
context = dataset.at_origin(
    "2026-08-22T11:00:00Z"
)
```

returns:

```python
ForecastContext
```

Also support live construction:

```python
context = of.ForecastContext.from_pandas(
    history=history_df,
    future=future_df,
    origin_time=ref_time,
    event_time="target_time",
    instance_keys=["zone"],
    targets=["price"],
    observed_features=[...],
    known_features=[...],
    frequency="1h",
)
```

## Validation

Test:

```text
duplicate origin/event rejected
NaNs preserved
different vintages preserved
truth unique by instance/event
at_origin(t) selects only vintage t
future vintage never leaks backward
```

Use poisoned future values such as `999999` to detect leakage.

## Done when

OpenForecast can faithfully represent production PIT training data without any provider-specific representation.

---

# Step 4 — Execution View intermediate representation and ViewPlanner

## Goal

Create the abstraction that prevents PIT branching from appearing in every provider.

Semantic source data:

```text
TimeSeriesFrame
ForecastDataset
ForecastContext
```

must be converted into provider-neutral **execution views**.

Providers consume only execution views.

## Implement package

```text
openforecast/views/
    base.py
    series.py
    windows.py
    tabular.py
    forecast.py
    planner.py
    provenance.py
```

## `SeriesView`

For classical single-time-axis forecasters.

```python
class SeriesView:
    temporal: pa.Table
    static: pa.Table | None
    schema: SeriesViewSchema
```

Shape:

```text
series_id
event_time
target columns
feature columns
```

Used by:

```text
AutoARIMA
ETS
Theta
local forecasters
```

## `WindowView`

For global/window-learning models.

```python
class WindowView:
    temporal: pa.Table
    static: pa.Table | None
    samples: pa.Table
    schema: WindowViewSchema
```

`temporal`:

```text
sample_id
event_time
targets...
features...
```

`samples`:

```text
sample_id
instance keys...
origin_time
context_start
context_end
forecast_start
forecast_end
```

One:

```text
instance × origin
```

equals one training sample.

`sample_id` should be opaque and deterministic.

## `TabularView`

For reduction/regression models.

```python
class TabularView:
    X: pa.Table
    y: pa.Table
    keys: pa.Table
    schema: TabularViewSchema
```

Keys include:

```text
row_id
instance keys
origin_time
event_time
horizon_step
```

This view preserves exactly the type of training data currently used for PIT LightGBM.

## `ForecastView`

Standardized inference representation:

```python
class ForecastView:
    origin_time: datetime
    history: pa.Table
    future: pa.Table
    static: pa.Table | None
    metadata: ForecastViewMetadata
```

## `ViewPlanner`

```python
class ViewPlanner:

    def fit_view(
        self,
        data,
        contract,
        fit_plan,
        task,
    ) -> FitView:
        ...

    def forecast_view(
        self,
        context,
        contract,
        task,
    ) -> ForecastView:
        ...
```

## Required mapping

```text
                     Series      Windows      Tabular

TimeSeriesFrame       yes         yes          yes
ForecastDataset       selected    yes          yes
ForecastContext       forecast    forecast     forecast
```

For ordinary `TimeSeriesFrame` → windows:

```text
historical forecast origins are simulated
```

For `ForecastDataset` → windows:

```text
actual historical forecast vintages are used
```

## Provenance

Define:

```python
class OriginFidelity(StrEnum):
    SIMULATED = "simulated"
    OBSERVED = "observed"
```

A model trained using real PIT data must record:

```text
OBSERVED
```

A model trained by cutting windows from one freshest historical time series records:

```text
SIMULATED
```

## Provider boundary enforcement

Add an architecture test prohibiting integrations from importing:

```text
ForecastDataset
PointInTimeFrame
```

Providers can import only:

```text
SeriesView
WindowView
TabularView
ForecastView
```

## Done when

The same `WindowView` type can be generated from both ordinary event-time and PIT data.

---

# Step 5 — Model references, descriptors, capabilities and execution contracts

## Goal

Define what a model identifier means and what execution view a model consumes.

Keep the clean OpenRouter-style string UX.

## `ModelRef`

Syntax:

```text
<namespace>/<name>[@revision]
```

Examples:

```text
nixtla/nhits
nixtla/autoarima
darts/nhits
local/de-price
local/de-price@01K...
```

A string does not itself imply whether it is fitted.

The registry resolves it.

## Lifecycle

```python
class ModelLifecycle(BaseModel):
    requires_fit: bool
    supports_fit: bool
    supports_update: bool = False
```

## Execution contract

```python
class FitViewKind(StrEnum):
    SERIES = "series"
    WINDOWS = "windows"
    TABULAR = "tabular"


class OriginScope(StrEnum):
    SINGLE = "single"
    MULTIPLE = "multiple"


class TrainingContract(BaseModel):
    view: FitViewKind
    origin_scope: OriginScope

    context_required: bool = False
    horizon_bound_at_fit: bool = False
    supports_unseen_instances: bool = False
```

Examples:

### AutoARIMA

```yaml
view: series
origin_scope: single
horizon_bound_at_fit: false
```

### NHiTS

```yaml
view: windows
origin_scope: multiple
context_required: true
horizon_bound_at_fit: true
supports_unseen_instances: true
```

### LightGBM reduction

```yaml
view: tabular
origin_scope: multiple
```

## Data capabilities

Implement structured:

```python
class InstanceCapabilities(BaseModel):
    single: bool
    panel: bool


class TargetCapabilities(BaseModel):
    univariate: bool
    multivariate: bool


class FeatureCapabilities(BaseModel):
    observed: bool
    known: bool
    static: bool


class OutputCapabilities(BaseModel):
    point: bool
    quantiles: bool
    samples: bool
```

## Missing values

```python
class MissingValueSupport(StrEnum):
    NATIVE = "native"
    REQUIRES_TRANSFORM = "requires_transform"
    UNSUPPORTED = "unsupported"
```

Never silently impute PIT NaNs.

## `ModelDescriptor`

```python
class ModelDescriptor(BaseModel):
    ref: ModelRef
    provider: str
    display_name: str

    lifecycle: ModelLifecycle
    training: TrainingContract
    capabilities: ModelCapabilities

    parameters_schema: dict
```

## Registry API

```python
of.models.list()

descriptor = of.models.get("nixtla/nhits")
```

## Done when

A model descriptor fully explains how OpenForecast must materialize data before provider execution.

---

# Step 6 — Recipes, FitPlan, origin selection and forecast tasks

## Goal

Define OpenForecast's provider-independent model construction and training language.

## Leaf model

```python
of.Model(
    "nixtla/nhits",
    params={
        "max_steps": 500,
    },
)
```

Do not expose semantic concepts such as context length through provider params when OpenForecast can own them.

## Pipeline

```python
of.Pipeline(
    steps=[
        of.StandardScaler(columns="targets"),
        of.Model("nixtla/nhits"),
    ]
)
```

## Ensemble

```python
of.Ensemble(
    models=[
        of.Model("nixtla/nhits"),
        of.Model("nixtla/autoarima"),
    ],
    combine=of.Mean(),
)
```

Also:

```python
of.WeightedMean(weights=[0.7, 0.3])
```

## Reduction

```python
of.Reduction(
    estimator="lightgbm/regressor",
    strategy="direct",
    lags=[1, 24, 168],
)
```

Strategies:

```text
recursive
direct
multioutput
```

Execution can be unsupported initially, but the protocol is defined now.

## Origin selection

Implement:

```python
of.AllOrigins(stride=1)
of.LatestOrigin()
of.AtOrigin(timestamp)
of.OriginsBetween(start, end, stride=12)
```

These work identically for:

```text
TimeSeriesFrame -> simulated origins
ForecastDataset -> observed origins
```

## Window semantics

```python
of.WindowPlan(
    context=168,
)
```

This is OpenForecast-native.

Do not require callers to additionally specify:

```text
Nixtla input_size=168
Darts input_chunk_length=168
```

Those compile from `WindowPlan`.

## `FitPlan`

```python
plan = of.FitPlan(
    origins=of.AllOrigins(),
    window=of.WindowPlan(context=168),
    seed=42,
    resources=of.Resources(
        accelerator="auto",
    ),
)
```

Reserve HPO/search fields but reject unsupported configurations explicitly.

## PIT transforms

Implement:

```python
of.LeadTimeFeature(
    name="lead_hours",
    unit="hour",
)
```

and optionally:

```python
of.OriginCalendarFeatures(
    hour=True,
    weekday=True,
)
```

## Explicit missing handling

```python
of.MissingIndicator(columns="features")
```

and:

```python
of.Impute(
    columns="features",
    method="median",
)
```

A neural model requiring imputation can therefore be:

```python
of.Pipeline(
    steps=[
        of.MissingIndicator(columns="features"),
        of.Impute(columns="features", method="median"),
        of.Model("nixtla/nhits"),
    ]
)
```

## Forecast task

```python
of.ForecastTask(
    horizon=24,
)
```

V1 horizon = steps.

## Output

```python
of.OutputSpec.point()
of.OutputSpec.quantiles([0.1, 0.5, 0.9])
of.OutputSpec.samples(100)
```

## Done when

All recipes/tasks serialize and deserialize independently of providers.

---

# Step 7 — Model artifact lifecycle and local model registry

## Goal

Make fitted models first-class immutable resources while preserving string model identifiers.

## Flow

```text
ModelDefinition
      ↓
fit
      ↓
ModelArtifact
      ↓
forecast
```

## Artifact refs

Fit:

```python
model = of.fit(
    model="...",
    data=data,
    name="de-price",
)
```

returns:

```text
local/de-price@01K...
```

Alias:

```text
local/de-price
```

points to the latest selected immutable revision.

## Artifact structure

```text
~/.local/share/openforecast/
    models/
        <artifact-id>/
            manifest.json
            recipe.json
            schema.json
            provider/
                ...
    aliases/
        de-price.json
```

Provider directory is opaque.

## Manifest

Include:

```text
artifact ID
source model
recipe
provider
provider version
OpenForecast version
protocol version
training schema hash

training view:
    series/windows/tabular

origin fidelity:
    observed/simulated

origin selection
context
horizon
number of samples
materializer version
feature schema
missing-value transforms
```

Example:

```json
{
  "training": {
    "view": "windows",
    "origin_fidelity": "observed",
    "context": 168,
    "horizon": 72,
    "samples": 8832
  }
}
```

## Atomic writes

Train into:

```text
.tmp/<artifact-id>
```

and rename atomically only after success.

## `ModelHandle`

```python
class ModelHandle:
    ref: ModelRef
    manifest: ModelManifest
```

`ModelHandle` must not keep a heavyweight native model loaded.

## Done when

Artifacts can be:

```text
created
resolved
aliased
reloaded
deleted
```

without providers being involved in registry semantics.

---

# Step 8 — Core execution engine and built-in reference provider

## Goal

Make `fit()` and `forecast()` work end-to-end before adding external providers.

## Reference model

Implement:

```text
builtin/seasonal-naive
```

Use it to test:

```text
single/panel
univariate/multivariate
fit/artifact/forecast
```

## Engine fit flow

```python
def fit(model, data, horizon, plan):

    recipe = normalize_recipe(model)

    descriptor = registry.resolve(recipe)

    view = view_planner.fit_view(
        data=data,
        contract=descriptor.training,
        fit_plan=plan,
        task=ForecastTask(horizon=horizon),
    )

    validate_view(
        view,
        descriptor.capabilities,
    )

    artifact = provider.fit(
        recipe=recipe,
        view=view,
        plan=plan,
    )

    return persist(artifact)
```

The engine should never contain:

```python
if provider == "nixtla":
```

## Forecast flow

```python
def forecast(model, data, horizon, output):

    artifact = resolve_model(model)

    context = normalize_forecast_context(data)

    view = view_planner.forecast_view(
        context=context,
        contract=artifact.forecast_contract,
        task=ForecastTask(horizon=horizon),
    )

    return provider.forecast(
        artifact=artifact,
        view=view,
        output=output,
    )
```

## PIT handling

The only source-type branching belongs in:

```text
ViewPlanner
```

Never provider code.

## Series model + PIT

If a `SeriesView` model gets PIT data with:

```text
AllOrigins()
```

raise:

```text
OriginScopeError
```

But this is valid:

```python
of.fit(
    "builtin/some-series-model",
    data=forecast_dataset,
    plan=of.FitPlan(
        origins=of.AtOrigin(ref_time)
    ),
)
```

because the planner materializes one `SeriesView`.

## Public API

```python
model = of.fit(
    model="builtin/seasonal-naive",
    data=train,
    params={"season_length": 24},
)

forecast = of.forecast(
    model=model,
    data=context,
    horizon=48,
)
```

## Pipeline/ensemble

Implement OpenForecast-owned execution for:

```text
StandardScaler -> model
ensemble of child artifacts
```

## Done when

A completely local built-in model can fit/persist/reload/forecast from the public API.

---

# Step 9 — Provider subprocess protocol and isolated uv environments

## Goal

Run integrations in separate uv-managed environments while leaving the engine unchanged.

## Environment structure

```text
~/.cache/openforecast/providers/
    nixtla/
        0.1.0/
            .venv/
```

Create with uv.

## CLI

```bash
openforecast providers list
openforecast providers install nixtla
openforecast providers inspect nixtla
openforecast providers remove nixtla
```

## Provider handshake

Request:

```json
{
  "protocol_version": 1,
  "operation": "handshake"
}
```

Response:

```json
{
  "protocol_version": 1,
  "provider": "nixtla",
  "provider_version": "0.1.0",
  "models": []
}
```

## RPC

Control:

```text
JSON Lines over stdin/stdout
```

Bulk data:

```text
Arrow IPC bundles
```

## Critical change

Providers receive views, not source datasets.

Example:

```json
{
  "operation": "fit",
  "view": {
    "kind": "windows",
    "path": "/tmp/openforecast/view"
  }
}
```

Window bundle:

```text
schema.json
temporal.arrow
static.arrow
samples.arrow
provenance.json
```

Tabular bundle:

```text
schema.json
x.arrow
y.arrow
keys.arrow
```

## Logging

```text
stdout = protocol only
stderr = logs
```

## Errors

Standardized:

```json
{
  "status": "error",
  "error": {
    "code": "INVALID_MODEL_PARAMETERS",
    "message": "...",
    "details": {}
  }
}
```

## Tests

Test:

```text
handshake
fit
forecast
provider crash
malformed response
protocol mismatch
timeout
stderr pollution
```

## Done when

`Engine` can swap in a subprocess provider without knowing it is a subprocess.

---

# Step 10 — Full conformance suite including point-in-time behavior

## Goal

Create the contract all future providers/models must satisfy.

## Golden semantic datasets

Create:

```text
single_univariate
single_multivariate
panel_univariate
panel_multivariate

pit_panel_univariate
pit_panel_multivariate
pit_missingness
pit_varying_vintages
pit_known_future
pit_observed_features
```

## View tests

Test:

```text
TimeSeriesFrame -> SeriesView
TimeSeriesFrame -> WindowView
TimeSeriesFrame -> TabularView

ForecastDataset -> SeriesView at one origin
ForecastDataset -> WindowView
ForecastDataset -> TabularView
```

## Leakage sentinel

Example:

```text
origin 08 -> target 12 -> wind=10
origin 09 -> target 12 -> wind=20
origin 10 -> target 12 -> wind=999999
```

Materialize origin 09.

Assert:

```text
20 exists
999999 does not
```

## Window sample count

For:

```text
100 origins
3 instances
```

expect:

```text
300 samples
```

for `AllOrigins()`.

## Missingness

If availability evolves:

```text
08 -> NaN
09 -> NaN
10 -> 42
```

preserve exactly that.

## Event-time equivalence

Construct PIT data where every vintage contains identical values.

Then compare:

```text
TimeSeriesFrame -> WindowView
ForecastDataset -> WindowView
```

The numerical windows should match.

Only:

```text
OriginFidelity
```

differs.

## Provider conformance

A model declaring:

```text
view=windows
```

automatically gets tests against both:

```text
event-time source
PIT source
```

The provider itself only receives `WindowView`.

## Done when

The built-in reference provider passes every capability it declares.

---

# Step 11 — Nixtla integration: StatsForecast / AutoARIMA

## Goal

Add the first external provider with a `SeriesView` consumer.

## Integration

```text
integrations/nixtla/
    pyproject.toml
    uv.lock

    src/openforecast_nixtla/
        __main__.py
        provider.py
        catalog.py
        conversion.py

        adapters/
            statsforecast.py
            neuralforecast.py
```

## Model

Advertise:

```text
nixtla/autoarima
```

Contract approximately:

```yaml
training:
  view: series
  origin_scope: single
  horizon_bound_at_fit: false
```

## Conversion

`SeriesView` becomes StatsForecast's expected representation internally.

Provider is allowed to construct:

```text
unique_id
ds
y
```

but these concepts must never escape the integration.

## Fit

Compile:

```python
of.Model(
    "nixtla/autoarima",
    params={...}
)
```

into the StatsForecast implementation.

Persist native state under the provider artifact directory.

## Event-time API

```python
model = of.fit(
    model="nixtla/autoarima",
    data=timeseries,
)
```

## PIT API

Valid:

```python
model = of.fit(
    model="nixtla/autoarima",
    data=forecast_dataset,
    plan=of.FitPlan(
        origins=of.AtOrigin(ref_time)
    ),
)
```

because both become `SeriesView`.

This must fail:

```python
of.fit(
    model="nixtla/autoarima",
    data=forecast_dataset,
    plan=of.FitPlan(
        origins=of.AllOrigins()
    ),
)
```

because AutoARIMA does not learn jointly across historical forecast origins.

## String lifecycle behavior

This:

```python
of.forecast(
    model="nixtla/autoarima",
    ...
)
```

must raise:

```text
ModelRequiresFit
```

## Done when

StatsForecast works through the isolated provider without understanding PIT semantics.

---

# Step 12 — Nixtla integration: NeuralForecast / NHiTS with true PIT learning

## Goal

Prove that a global neural forecasting model can train from true point-in-time vintages using `WindowView`.

This is the most important architecture-validation stage.

## Model contract

```yaml
nixtla/nhits:

  training:
    view: windows
    origin_scope: multiple
    context_required: true
    horizon_bound_at_fit: true
    supports_unseen_instances: true
```

Only declare `supports_unseen_instances=true` after verifying it in integration tests.

## Public API

```python
dataset = of.ForecastDataset.from_pandas(
    df,
    origin_time="ref_time",
    event_time="target_time",
    instance_keys=["zone"],
    targets=["price"],
    known_features=[
        "wind_fc",
        "solar_fc",
        "load_fc",
    ],
    event_frequency="1h",
    origin_frequency="1h",
)
```

Fit:

```python
model = of.fit(
    model=of.Model(
        "nixtla/nhits",
        params={
            "max_steps": 500,
        },
    ),
    data=dataset,
    horizon=72,
    plan=of.FitPlan(
        origins=of.AllOrigins(),
        window=of.WindowPlan(
            context=168,
        ),
    ),
    name="de-price",
)
```

## Internal materialization

```text
ForecastDataset
      ↓
ViewPlanner
      ↓
WindowView

(instance, origin) -> sample_id
```

Provider maps:

```text
sample_id  -> unique_id
event_time -> ds
target     -> y
```

Features:

```text
observed -> hist_exog_list
known    -> futr_exog_list
static   -> stat_exog_list
```

These mappings exist only inside the Nixtla provider.

## Semantic compilation

OpenForecast:

```text
WindowPlan(context=168)
```

becomes:

```text
NHiTS input_size=168
```

OpenForecast:

```text
horizon=72
```

becomes:

```text
NHiTS h=72
```

The user must not specify these twice.

## One-window invariant

Each synthetic sample must represent exactly:

```text
168 context steps
72 forecast steps
```

and exactly one forecast origin.

Test explicitly that Nixtla is not accidentally constructing cross-origin training samples.

## Current-origin inference

```python
context = dataset.at_origin(current_ref_time)

forecast = of.forecast(
    model="local/de-price",
    data=context,
    horizon=72,
)
```

The provider receives a `ForecastView`, not a `ForecastDataset`.

## Missing values

If NHiTS cannot consume native PIT NaNs, require an explicit pipeline:

```python
recipe = of.Pipeline(
    steps=[
        of.MissingIndicator(columns="features"),
        of.Impute(
            columns="features",
            method="median",
        ),
        of.Model(
            "nixtla/nhits",
            params={"max_steps": 500},
        ),
    ]
)
```

Never silently destroy missingness.

## Horizon validation

If artifact was trained with:

```text
horizon=72
```

and user requests:

```text
horizon=48
```

or another incompatible horizon, raise:

```text
IncompatibleForecastTask
```

if the native model binds horizon during training.

## Done when

NHiTS trains from real PIT vintages while the provider contains zero `ForecastDataset`-specific logic.

---

# Step 13 — Darts integration and library-neutral WindowView validation

## Goal

Prove the OpenForecast abstraction is not secretly Nixtla-shaped.

Add Darts as the second global-model implementation.

## Structure

```text
integrations/darts/
    pyproject.toml
    uv.lock
    src/openforecast_darts/
```

## Models

Start with:

```text
one global model
one local model
```

For example:

```text
darts/nhits
```

plus an appropriate local statistical model.

## Global model contract

Global model consumes:

```text
WindowView
```

Darts adapter converts each `sample_id` into one Darts `TimeSeries`.

Conceptually:

```python
target_series = []
past_covariates = []
future_covariates = []

for sample in view.samples:
    target_series.append(...)
    past_covariates.append(...)
    future_covariates.append(...)
```

The provider sees no origin semantics.

## Identical UX

These should differ only in model ref:

```python
of.fit(
    "nixtla/nhits",
    data=dataset,
    horizon=72,
    plan=plan,
)
```

and:

```python
of.fit(
    "darts/nhits",
    data=dataset,
    horizon=72,
    plan=plan,
)
```

## Semantic compilation

Map OpenForecast:

```text
context
```

to Darts:

```text
input_chunk_length
```

and:

```text
horizon
```

to appropriate Darts output horizon semantics.

Again, users should not have to manually specify equivalent concepts twice.

## Local Darts model

A local Darts model consumes:

```text
SeriesView
```

and therefore supports PIT only when selecting a single origin.

Same behavior as AutoARIMA.

## Conformance

Run exactly the same PIT `WindowView` tests used for NHiTS.

## Done when

Switching Nixtla ↔ Darts does not change OpenForecast's public PIT API.

---

# Step 14 — sktime integration and reduction/panel validation

## Goal

Use sktime to validate:

```text
SeriesView
WindowView
TabularView
```

against a third ecosystem with explicit panel/global semantics.

## Structure

```text
integrations/sktime/
    pyproject.toml
    uv.lock
    src/openforecast_sktime/
```

## Panel mapping

For `WindowView`:

```text
sample_id
event_time
```

becomes a sktime panel MultiIndex.

Example:

```text
sample_id event_time y    wind_fc
001       00:00      ...  ...
001       01:00      ...  ...
002       00:00      ...  ...
002       01:00      ...  ...
```

## Series models

Local sktime forecasters consume `SeriesView`.

## Global/panel models

Eligible global models consume `WindowView`.

## Reduction support

Implement the first execution path for:

```python
of.Reduction(
    estimator="lightgbm/regressor",
    strategy="recursive",
    lags=[1, 24, 168],
)
```

through sktime if appropriate.

Alternatively add a direct OpenForecast reduction provider if that proves cleaner, but the public recipe remains unchanged.

## PIT reduction

For:

```python
of.Reduction(...)
```

with `ForecastDataset`, the ViewPlanner creates:

```text
TabularView
```

containing all:

```text
instance × origin × target
```

rows.

This is the architecture needed for the user's existing LightGBM style.

## Acceptance

The PIT data must preserve:

```text
ref-specific feature values
NaN distribution
lead time if explicitly requested
target duplication across multiple origins
```

## Done when

The same `ForecastDataset` can power:

```text
Nixtla global neural model
Darts global model
sktime/global model
tabular reduction
```

without changing its source representation.

---

# Step 15 — Public V1 API stabilization and full E2E suite

## Goal

Freeze the initial developer experience.

Users should never need to know about:

```text
ViewPlanner
WindowView
unique_id
ds
Nixtla
Darts internal TimeSeries
sktime MultiIndex conversion
IPC files
provider subprocesses
```

unless debugging.

## Discovery

```python
import openforecast as of

of.models.list()
```

Example:

```text
builtin/seasonal-naive
nixtla/autoarima
nixtla/nhits
darts/nhits
...
```

Inspect:

```python
model = of.models.get("nixtla/nhits")

model.lifecycle.requires_fit
model.training.view
model.capabilities
```

## Event-time fit

```python
model = of.fit(
    "nixtla/nhits",
    data=timeseries,
    horizon=24,
    plan=of.FitPlan(
        window=of.WindowPlan(context=168),
    ),
)
```

## Point-in-time fit

```python
model = of.fit(
    "nixtla/nhits",
    data=forecast_dataset,
    horizon=24,
    plan=of.FitPlan(
        origins=of.AllOrigins(),
        window=of.WindowPlan(context=168),
    ),
)
```

## String artifact forecast

```python
forecast = of.forecast(
    model="local/de-price",
    data=current_context,
    horizon=24,
)
```

## Explicit model recipe

```python
model = of.fit(
    model=of.Model(
        "nixtla/nhits",
        params={
            "max_steps": 500,
        },
    ),
    data=dataset,
    horizon=24,
    plan=plan,
)
```

## Pipeline

```python
model = of.fit(
    model=of.Pipeline(
        steps=[
            of.MissingIndicator(columns="features"),
            of.Impute(
                columns="features",
                method="median",
            ),
            of.StandardScaler(columns="targets"),
            of.Model("nixtla/nhits"),
        ]
    ),
    data=dataset,
    horizon=24,
    plan=plan,
)
```

## Ensemble

```python
model = of.fit(
    model=of.Ensemble(
        models=[
            of.Model("nixtla/nhits"),
            of.Model("nixtla/autoarima"),
        ],
        combine=of.Mean(),
    ),
    data=data,
    horizon=24,
)
```

Only permit combinations whose child training contracts can be satisfied by the source data.

## Output schema

Canonical Arrow forecast should use a stable long representation:

```text
instance keys...
event_time
target
kind
quantile
sample
value
```

Example:

```text
DE 12:00 price point    null null 80
DE 12:00 price quantile 0.1  null 65
DE 12:00 price quantile 0.5  null 78
DE 12:00 price quantile 0.9  null 95
```

Convenience:

```python
forecast.to_pandas()
forecast.to_wide()
forecast.point()
forecast.quantile(0.5)
```

## E2E test

One full test should:

```text
install Nixtla provider
construct PIT dataset
discover AutoARIMA
discover NHiTS

fit AutoARIMA at one origin
reload
forecast

fit PIT NHiTS across many origins
reload
forecast current origin

fit Darts model using same PIT dataset

fit PIT reduction

verify artifact aliases

verify no provider-specific field names
escape into public objects
```

## Forbidden terminology test

Public protocol objects should not expose:

```text
unique_id
ds
hist_exog_list
futr_exog_list
stat_exog_list
input_chunk_length
```

These belong only inside provider adapters.

## Done when

The local OpenForecast V1 experience is coherent and provider-independent.

---

# Step 16 — HTTP/OpenAPI projection and remote transport

## Goal

Expose the exact same OpenForecast semantics remotely without making HTTP the architecture.

Dependency direction:

```text
OpenForecast semantics
        ↓
Engine
        ↓
HTTP projection
        ↓
OpenAPI
        ↓
generated remote SDKs
```

## Local server

```bash
openforecast serve
```

Expose initial endpoints:

```text
GET  /v1/models
GET  /v1/models/{ref}

POST /v1/fit
POST /v1/forecast

GET  /v1/artifacts/{ref}
```

Do not solve distributed asynchronous training yet.

## Transport abstraction

```python
client = OpenForecast(
    transport=LocalTransport()
)
```

and:

```python
client = OpenForecast(
    transport=HttpTransport(
        "http://localhost:8321"
    )
)
```

Both expose:

```python
client.models.list()
client.models.get(...)
client.fit(...)
client.forecast(...)
```

## Bulk data

Do not put huge datasets into nested JSON if avoidable.

Use:

```text
JSON/Pydantic for control
Arrow IPC for bulk data
```

The HTTP API can later support multipart or uploaded Arrow objects.

## OpenAPI

Generate from Pydantic request/response models.

Commit:

```text
spec/openapi/openapi.json
```

CI:

```bash
uv run generate-openapi
git diff --exit-code spec/openapi/openapi.json
```

## Cross-language

Generated TypeScript/Go/Java SDKs initially support remote transport.

Python remains hand-written because it also supports local execution/provider management.

## Done when

This:

```python
OpenForecast(LocalTransport())
```

and:

```python
OpenForecast(HttpTransport(...))
```

provide the same user-facing forecasting semantics.

---

# Step 17 — Benchmarking, PIT evaluation and foundation for `openforecast/auto`

## Goal

Use the universal abstraction to benchmark models across providers and eventually build automatic routing/model selection.

This is where OpenForecast begins becoming more than a wrapper.

## Benchmark API

```python
result = of.benchmark(
    models=[
        "builtin/seasonal-naive",
        "nixtla/autoarima",
        "nixtla/nhits",
        "darts/nhits",
    ],
    data=data,
    validation=of.RollingOrigin(
        horizon=24,
        windows=5,
    ),
    metrics=[
        of.MAE(),
        of.Bias(),
    ],
)
```

## PIT-aware validation

For `ForecastDataset`, benchmarking must use **actual historical origins** rather than reconstructing historical features from latest values.

Example:

```python
result = of.benchmark(
    models=[
        "nixtla/nhits",
        "darts/nhits",
        of.Reduction(
            estimator="lightgbm/regressor",
            strategy="direct",
        ),
    ],
    data=pit_dataset,
    validation=of.ForecastOriginValidation(
        origins=of.OriginsBetween(
            start=...,
            end=...,
            stride=24,
        ),
        horizon=72,
    ),
)
```

For each validation origin:

```text
features must come from that exact historical origin
truth comes from the truth TimeSeriesFrame
later feature vintages are inaccessible
```

This should become a fundamental OpenForecast guarantee.

## Benchmark result

Arrow-backed:

```text
model
fold
origin
metric
value
fit_seconds
forecast_seconds
```

Also track:

```text
origin fidelity
provider
artifact
```

## Event-time vs PIT comparison

OpenForecast should eventually make it possible to explicitly benchmark:

```text
simulated historical availability
vs
true PIT historical availability
```

This could be highly valuable.

## `openforecast/auto`

Lay the foundation for:

```python
model = of.fit(
    model="openforecast/auto",
    data=dataset,
    horizon=24,
)
```

Internally it can:

```text
inspect data semantics
determine eligible model contracts
benchmark models
rank results
fit winner or ensemble
persist selected recipe
```

Eligibility can automatically rule out:

```text
AutoARIMA for multi-origin learning
models without missing-value support
models without required feature capabilities
models that cannot handle the target dimensionality
```

## Done when

Benchmarking is entirely built on:

```text
ModelRecipe
ForecastDataset / TimeSeriesFrame
ViewPlanner
FitPlan
ForecastTask
Forecast
```

with no Nixtla/Darts/sktime-specific benchmarking implementation.

---

# Final architecture after Step 17

```text
                              OPENFORECAST

                         SEMANTIC DATA LAYER

              ┌────────────────────────────────┐
              │                                │
              ▼                                ▼

       TimeSeriesFrame                  ForecastDataset
                                               │
                                    ┌──────────┴─────────┐
                                    ▼                    ▼
                              PointInTimeFrame      TimeSeriesFrame
                               information              truth

              │                                │
              └───────────────┬────────────────┘
                              ▼

                         ViewPlanner

                ┌─────────────┼─────────────┐
                ▼             ▼             ▼

           SeriesView     WindowView    TabularView
                │             │             │
                │             │             │
                ▼             ▼             ▼

          local models    global models    reductions

          AutoARIMA       NHiTS            LightGBM
          ETS             TFT              XGBoost
                          PatchTST          CatBoost

                │             │             │
                └─────────────┼─────────────┘
                              ▼

                       PROVIDER LAYER

             ┌────────────────┼────────────────┐
             ▼                ▼                ▼

           Nixtla            Darts           sktime

             │                │                │
             └────────────────┼────────────────┘
                              ▼

                       ModelArtifact
                              │
                              ▼

                       ForecastContext
                        one origin only
                              │
                              ▼
                          Forecast
```

The biggest principle I would preserve throughout all 17 steps is that **`ForecastDataset → WindowView` is an OpenForecast operation, not a Nixtla trick**. Nixtla might represent each window using `unique_id`; Darts might represent it as a `Sequence[TimeSeries]`; sktime might represent it as a panel MultiIndex. Those are compilation targets. The semantic meaning of the training sample belongs to OpenForecast.

That is what makes point-in-time forecasting a genuine first-class capability rather than a special branch that will become painful as you add providers.
