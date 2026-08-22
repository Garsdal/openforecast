Yes. I would freeze the architecture now and implement it in **vertical stages**, where every stage ends with green tests and a usable public surface. The most important rule is that no stage is allowed to leak Nixtla semantics into OpenForecast.

A few implementation decisions are worth locking first. Arrow should be the canonical in-memory/data-plane representation; Arrow supports typed schemas, metadata, and IPC serialization, which makes it a good cross-language boundary. ([Apache Arrow][1]) External providers should remain isolated Python projects with their own uv environments rather than workspace members sharing one resolution. uv explicitly supports isolated execution environments, while workspaces share dependency resolution. ([Astral Docs][2])

For Nixtla specifically, the adapter boundary is important: NeuralForecast still fundamentally operates on `unique_id`, `ds`, `y` plus its historic/future/static exogenous conventions, while StatsForecast exposes fit/predict and persisted fitted models. OpenForecast should translate into those representations only inside the Nixtla provider. ([Nixtla][3])

---

# Target repository after these plans

This is the structure I would build toward:

```text
openforecast/
│
├── pyproject.toml
├── uv.lock
├── README.md
│
├── src/
│   └── openforecast/
│       ├── __init__.py
│       ├── client.py
│       │
│       ├── data/
│       │   ├── frame.py
│       │   ├── schema.py
│       │   ├── features.py
│       │   ├── frequency.py
│       │   └── validation.py
│       │
│       ├── models/
│       │   ├── refs.py
│       │   ├── descriptors.py
│       │   ├── capabilities.py
│       │   └── handles.py
│       │
│       ├── recipes/
│       │   ├── base.py
│       │   ├── model.py
│       │   ├── pipeline.py
│       │   ├── ensemble.py
│       │   ├── reduction.py
│       │   └── transforms.py
│       │
│       ├── tasks/
│       │   ├── fit.py
│       │   ├── forecast.py
│       │   ├── output.py
│       │   └── resources.py
│       │
│       ├── artifacts/
│       │   ├── manifest.py
│       │   ├── store.py
│       │   └── aliases.py
│       │
│       ├── registry/
│       │   ├── models.py
│       │   └── providers.py
│       │
│       ├── runtime/
│       │   ├── engine.py
│       │   ├── providers.py
│       │   ├── environments.py
│       │   ├── process.py
│       │   └── ipc.py
│       │
│       ├── protocol/
│       │   ├── version.py
│       │   ├── messages.py
│       │   └── errors.py
│       │
│       ├── commands/
│       │   └── ...
│       │
│       └── server/
│           └── ...
│
├── integrations/
│   └── nixtla/
│       ├── pyproject.toml
│       ├── uv.lock
│       ├── src/
│       │   └── openforecast_nixtla/
│       │       ├── __main__.py
│       │       ├── provider.py
│       │       ├── catalog.py
│       │       ├── conversion.py
│       │       └── adapters/
│       │           ├── statsforecast.py
│       │           └── neuralforecast.py
│       └── tests/
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

`openforecast_nixtla` is acceptable here because it is an independently distributed integration package. I would avoid `openforecast_core`, `openforecast_cli`, `openforecast_api`, etc. inside the main project.

---

# Plan 1 — Repository foundation and architectural boundaries

## Goal

Create the repository, packaging, dependency rules, CI, formatting, typing and test structure without implementing forecasting functionality yet.

This stage should establish an architectural rule:

> `openforecast` has no dependency on Nixtla, Darts, sktime, PyTorch or any forecasting framework.

External integrations depend on OpenForecast, never the reverse.

## Implement

Create one main Python distribution:

```toml
[project]
name = "openforecast"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2",
    "pyarrow",
    "platformdirs",
]
```

Development dependencies should include at least:

```text
pytest
pytest-cov
hypothesis
ruff
mypy or pyright
```

CLI dependencies can be introduced when the CLI is implemented rather than forcing them into the first commit.

Do **not** make `integrations/nixtla` part of the root uv workspace. It needs its own:

```text
pyproject.toml
uv.lock
.venv
```

because future providers may need incompatible Torch/JAX/etc. dependency graphs.

Define import boundaries:

```text
data/
models/
recipes/
tasks/

         ↓

runtime/
registry/
artifacts/

         ↓

client.py
```

`protocol/` is allowed underneath provider/runtime boundaries but should not know anything about Nixtla.

Set up CI commands:

```bash
uv sync
uv run ruff check .
uv run pytest
uv run pyright
```

Add an architecture test that ensures the main package cannot accidentally import:

```text
neuralforecast
statsforecast
darts
sktime
torch
jax
```

A simple AST/import scan is enough.

## Public API target

`src/openforecast/__init__.py` should initially export nothing except:

```python
__version__
```

Do not prematurely create stub APIs.

## Done when

```bash
uv run pytest
uv run ruff check .
```

passes and:

```bash
uv tree
```

contains no forecasting framework.

---

# Plan 2 — Define OpenForecast's time-series semantic model

This is the most important implementation stage.

## Goal

Create an OpenForecast-native representation for:

```text
single-instance univariate
single-instance multivariate
panel univariate
panel multivariate
```

without creating four different container classes.

The shape must derive from orthogonal axes.

## Core primitives

Implement:

```python
class FeatureAvailability(StrEnum):
    OBSERVED = "observed"
    KNOWN = "known"


class FeatureKind(StrEnum):
    TEMPORAL = "temporal"
    STATIC = "static"
```

Then:

```python
class FeatureSpec(BaseModel):
    name: str
    kind: FeatureKind = FeatureKind.TEMPORAL
    availability: FeatureAvailability | None = None
```

Validation:

```text
temporal feature
    -> availability required

static feature
    -> availability must be None
```

Define OpenForecast's own frequency representation rather than exposing Pandas aliases as protocol semantics:

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

For example:

```python
Frequency(unit="minute", step=15)
Frequency(unit="hour", step=1)
```

Adapters later compile this into `"15min"`, `"H"`, etc.

## `TimeSeriesSchema`

Implement:

```python
class TimeSeriesSchema(BaseModel):
    time: str
    frequency: Frequency

    instance_keys: tuple[str, ...] = ()
    targets: tuple[str, ...]

    features: tuple[FeatureSpec, ...] = ()
```

Then expose derived properties:

```python
schema.is_panel
schema.is_univariate
schema.is_multivariate
schema.target_count
schema.has_known_features
schema.has_observed_features
schema.has_static_features
```

Examples:

```python
TimeSeriesSchema(
    time="timestamp",
    frequency=Frequency(unit="hour"),
    targets=("load",),
)
```

is:

```text
single-instance
univariate
```

This:

```python
TimeSeriesSchema(
    time="timestamp",
    frequency=Frequency(unit="hour"),
    instance_keys=("country",),
    targets=("load",),
)
```

is:

```text
panel
univariate
```

And:

```python
TimeSeriesSchema(
    time="timestamp",
    frequency=Frequency(unit="hour"),
    instance_keys=("station",),
    targets=("wind_u", "wind_v"),
)
```

is:

```text
panel
multivariate
```

Do not create enum values called `PANEL_MULTIVARIATE`.

The axes derive from schema.

---

## `TimeSeriesFrame`

Implement the physical container:

```python
class TimeSeriesFrame:
    history: pa.Table
    future: pa.Table | None
    static: pa.Table | None
    schema: TimeSeriesSchema
```

This is important.

### History table

One row:

```text
instance × timestamp
```

Columns include:

```text
instance keys
timestamp
targets
observed features
known features where historically available
```

Example:

```text
country timestamp             load  temperature  wind_fc
DE      2026-01-01 00:00      51.1     4.2       12.1
DE      2026-01-01 01:00      49.8     4.0       12.8
FR      2026-01-01 00:00      42.4     7.1        8.4
```

### Future table

Contains:

```text
instance keys
timestamp
known temporal features
```

It must **not** contain:

```text
targets
observed-only features
```

Example:

```text
country timestamp             wind_fc holiday
DE      2026-01-03 00:00       15.2    false
DE      2026-01-03 01:00       15.8    false
```

### Static table

One row per logical instance.

```text
country capacity region
DE      81.2     central
FR      63.1     west
```

For a single-instance dataset, permit a single-row static table without instance keys.

---

## Convenience constructors

The core should accept Pandas, Polars and Arrow eventually, but canonicalize immediately to Arrow.

Start with:

```python
frame = of.TimeSeriesFrame.from_pandas(
    history=df,
    time="timestamp",
    frequency="1h",
    targets="load",
)
```

Also:

```python
frame = of.TimeSeriesFrame.from_pandas(
    history=train,
    future=future,
    static=static,
    time="timestamp",
    frequency="1h",
    instance_keys=["country"],
    targets=["load"],
    observed_features=["temperature"],
    known_features=["wind_fc", "holiday"],
    static_features=["capacity"],
)
```

The convenience layer may parse `"1h"`.

The underlying `Frequency` object remains canonical.

## Validation rules

Implement all of these now:

```text
time column exists
all instance keys exist
all targets exist in history
target names are unique
feature names are unique
target cannot also be feature
history contains no duplicate instance/time keys
future contains no duplicate instance/time keys
static contains exactly one row per instance
future contains only known temporal features
observed features cannot extend into future
timestamps are compatible with declared frequency
future begins after history for each instance
all required future timestamps exist for declared horizon when validated against a ForecastTask
```

Do not silently sort duplicates or silently drop invalid fields.

## Arrow serialization

Implement:

```python
frame.write(path)
TimeSeriesFrame.read(path)
```

using an OpenForecast bundle:

```text
frame/
    schema.json
    history.arrow
    future.arrow
    static.arrow
```

Only files that exist are written.

Use Arrow IPC rather than JSON for data. Arrow's IPC formats are explicitly intended for record-batch serialization/interprocess exchange. ([Apache Arrow][4])

## Tests

Use Hypothesis/property tests to generate combinations of:

```text
single/panel
uni/multivariate
0/1/multiple features
observed/known/static
```

Test Arrow round-trip equality.

## Done when

These all work:

```python
frame.schema.is_panel
frame.schema.is_multivariate
frame.write(...)
TimeSeriesFrame.read(...)
```

with zero dependency on forecasting libraries.

---

# Plan 3 — Model references, descriptors and capabilities

## Goal

Solve the `"nixtla/nhits"` problem before fitting anything.

A string identifies a **model resource**, not necessarily a fitted model.

## `ModelRef`

Implement parsing for:

```text
nixtla/nhits
nixtla/autoarima

local/electricity-price
local/electricity-price@01K3ABC...
```

Model reference grammar:

```text
<namespace>/<name>[@<revision>]
```

Do not encode resource type into the string.

Resolve it through the registry.

Implement:

```python
ModelRef.parse("nixtla/nhits")
```

but all public APIs accept plain strings.

---

## `ModelDescriptor`

Implement:

```python
class ModelLifecycle(BaseModel):
    requires_fit: bool
    supports_fit: bool
    supports_update: bool = False


class ModelDescriptor(BaseModel):
    ref: ModelRef
    provider: str
    display_name: str
    lifecycle: ModelLifecycle
    capabilities: ModelCapabilities
    parameters_schema: dict
```

A future foundation model might say:

```yaml
ref: amazon/chronos-2

lifecycle:
  requires_fit: false
  supports_fit: true
```

NHiTS says:

```yaml
ref: nixtla/nhits

lifecycle:
  requires_fit: true
  supports_fit: true
```

---

# Model capability axes

Make this structured.

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


class ModelCapabilities(BaseModel):
    instances: InstanceCapabilities
    targets: TargetCapabilities
    features: FeatureCapabilities
    outputs: OutputCapabilities

    regular_time: bool = True
    irregular_time: bool = False
```

Add:

```python
capabilities.validate_frame(frame)
capabilities.validate_output(output)
```

This must produce OpenForecast-native exceptions:

```text
UnsupportedDataShape
UnsupportedFeature
UnsupportedOutput
ModelRequiresFit
```

Never allow native Nixtla errors to be the first validation layer.

---

# Model registry

Implement:

```python
of.models.list()
of.models.get("builtin/seasonal-naive")
```

The registry initially only knows built-in models.

Later installed providers contribute descriptors.

## Done when

You can run:

```python
model = of.models.get("builtin/seasonal-naive")

print(model.capabilities.targets.multivariate)
```

without executing anything.

---

# Plan 4 — Define recipes, fit tasks and forecast tasks

## Goal

Define the shared model-construction language **before** integrating Nixtla.

This is your forecasting intermediate representation.

---

## Leaf model recipe

Implement:

```python
recipe = of.Model(
    "nixtla/nhits",
    params={
        "input_size": 168,
        "max_steps": 1000,
    },
)
```

String shorthand:

```python
of.fit(
    model="nixtla/nhits",
    ...
)
```

must normalize internally to:

```python
Model(ref="nixtla/nhits")
```

Provider-specific hyperparameters are allowed in `params`.

They are validated against:

```text
ModelDescriptor.parameters_schema
```

Do **not** create OpenForecast fields for every NHiTS parameter.

---

# Pipeline recipe

Create:

```python
recipe = of.Pipeline(
    steps=[
        of.StandardScaler(
            columns="targets",
            per_instance=True,
        ),
        of.Model("nixtla/nhits"),
    ]
)
```

Start with one built-in transform:

```text
StandardScaler
```

That is enough to prove pipeline semantics.

Pipeline transforms should be represented as recipe AST nodes and be serializable.

---

# Ensemble recipe

Implement:

```python
recipe = of.Ensemble(
    models=[
        of.Model("nixtla/nhits"),
        of.Model("nixtla/autoarima"),
    ],
    combine=of.WeightedMean(
        weights=[0.7, 0.3],
    ),
)
```

Also:

```python
combine=of.Mean()
```

The engine, not Nixtla, owns cross-model ensemble composition.

Restrict V1 aggregation to:

```text
point forecasts
matching quantile levels
```

Do not pretend weighted quantiles form a rigorous distribution mixture.

---

# Reduction recipe

Define and serialize now:

```python
recipe = of.Reduction(
    estimator="sklearn/lightgbm",
    strategy="recursive",
    lags=[1, 2, 24, 48, 168],
)
```

Supported strategies in the semantic model:

```text
recursive
direct
multioutput
```

Do **not** execute Reduction yet.

It exists because sktime/Darts will compile it later.

Trying to run it before an eligible provider exists should raise:

```text
UnsupportedRecipeError
```

This is preferable to postponing the protocol design until sktime arrives.

---

# `FitPlan`

Implement:

```python
plan = of.FitPlan(
    validation=of.Holdout(steps=24),
    resources=of.Resources(
        accelerator="auto",
    ),
    seed=42,
)
```

Define a future-facing optional:

```python
search: SearchPlan | None
```

but in V1:

```text
search != None
```

raises:

```text
UnsupportedFitPlan
```

until tuning is implemented.

This preserves the semantic boundary without pretending HPO already works.

---

# `ForecastTask`

Implement:

```python
task = of.ForecastTask(
    horizon=24,
)
```

V1 horizons are **steps**, not `"48h"`.

That avoids DST/calendar ambiguity.

Later duration-based horizons can compile into steps.

---

# `OutputSpec`

Implement:

```python
of.OutputSpec.point()
```

and:

```python
of.OutputSpec.quantiles([0.1, 0.5, 0.9])
```

and reserve:

```python
of.OutputSpec.samples(100)
```

if supported by the model.

---

# Canonical forecast result

I would use a long Arrow representation.

```text
<instance keys>
timestamp
target
kind
quantile
sample
value
```

Examples:

```text
DE 2026-01-03T00 load point    null null 52.3
DE 2026-01-03T00 load quantile 0.1  null 45.8
DE 2026-01-03T00 load quantile 0.5  null 51.9
DE 2026-01-03T00 load quantile 0.9  null 59.4
```

This gives you one stable Arrow schema for univariate, multivariate and probabilistic outputs.

Expose convenience methods:

```python
forecast.to_pandas()
forecast.point()
forecast.quantile(0.5)
forecast.to_wide()
```

---

# Serialization

Every recipe/task must round-trip JSON:

```python
recipe.model_dump_json()
ModelRecipe.model_validate_json(...)
```

This matters because the exact same AST will eventually travel through provider RPC and HTTP.

## Done when

These parse and serialize:

```python
of.Model(...)
of.Pipeline(...)
of.Ensemble(...)
of.Reduction(...)
of.FitPlan(...)
of.ForecastTask(...)
of.OutputSpec(...)
```

without any provider being installed.

---

# Plan 5 — Artifact lifecycle and local model registry

## Goal

Make:

```text
ModelDefinition
      ↓ fit
ModelArtifact
      ↓ forecast
Forecast
```

real.

---

# Artifact identifiers

Every fit creates an immutable revision:

```text
local/electricity-price@01K3X...
```

Optionally create a mutable alias:

```text
local/electricity-price
```

pointing to that revision.

So:

```python
model = of.fit(
    model="...",
    data=train,
    name="electricity-price",
)
```

returns:

```python
ModelHandle(
    ref="local/electricity-price@01K3X..."
)
```

and:

```python
str(model)
```

returns the immutable ref.

---

# Artifact layout

Use platform-specific application storage via `platformdirs`.

Conceptually:

```text
~/.local/share/openforecast/
    models/
        01K3X.../
            manifest.json
            recipe.json
            schema.json
            provider/
                ...
    aliases/
        electricity-price.json
```

The provider directory is opaque to core.

The manifest contains:

```json
{
  "artifact_id": "...",
  "source_model": "nixtla/nhits",
  "recipe": {},
  "provider": "nixtla",
  "provider_version": "...",
  "openforecast_version": "...",
  "protocol_version": 1,
  "training_schema_hash": "...",
  "created_at": "...",
  "metadata": {}
}
```

Provider-specific model files live under:

```text
provider/
```

OpenForecast does not inspect them.

---

# Atomicity

Fit into:

```text
.tmp/<id>/
```

and atomically rename only after success.

A failed training run must never produce a resolvable model ref.

Alias updates must be atomic.

---

# User API

Implement:

```python
model = of.models.get("local/electricity-price")
```

and:

```python
model = of.load("local/electricity-price")
```

if you want the convenience alias.

`ModelHandle` should contain metadata, not loaded heavyweight model objects.

Loading the native artifact only happens in the provider environment.

## Done when

A fake artifact can be:

```text
created
resolved by immutable ref
resolved by alias
deleted safely
round-tripped through manifest serialization
```

---

# Plan 6 — Execution engine and built-in reference provider

This stage proves that OpenForecast itself works before Nixtla exists.

## Goal

Implement:

```python
of.fit(...)
of.forecast(...)
```

end to end using a tiny built-in reference provider.

---

# Provider abstraction

Core should know:

```python
class ProviderClient(Protocol):

    def describe_models(self) -> list[ModelDescriptor]:
        ...

    def fit(
        self,
        recipe: ModelRecipe,
        data: TimeSeriesFrame,
        plan: FitPlan,
        destination: Path,
    ) -> ProviderArtifact:
        ...

    def forecast(
        self,
        artifact: ModelArtifact,
        data: TimeSeriesFrame,
        task: ForecastTask,
        output: OutputSpec,
    ) -> Forecast:
        ...
```

This is a **client to a provider**, not necessarily an in-process Python implementation.

---

# Built-in model

Implement:

```text
builtin/seasonal-naive
```

with parameters:

```python
{
    "season_length": 24
}
```

It should support:

```text
single univariate
single multivariate
panel univariate
panel multivariate
```

by independently forecasting every:

```text
instance × target
```

combination.

That makes it a useful protocol reference implementation.

---

# Engine

Implement:

```python
class Engine:
    def fit(...)
    def forecast(...)
```

Execution sequence for `fit()`:

```text
normalize ModelRecipe
        ↓
resolve leaf model descriptors
        ↓
validate recipe
        ↓
validate TimeSeriesFrame
        ↓
validate capabilities
        ↓
select provider
        ↓
fit
        ↓
persist ModelArtifact
        ↓
return ModelHandle
```

Forecast:

```text
resolve model string / handle
        ↓
determine artifact vs pretrained definition
        ↓
validate lifecycle
        ↓
validate context
        ↓
validate OutputSpec
        ↓
provider forecast
        ↓
normalize Forecast
```

Calling:

```python
of.forecast(
    model="builtin/some-trainable-model",
)
```

must raise `ModelRequiresFit` if its descriptor requires fitting.

---

# Implement pipeline execution

For:

```python
Pipeline(
    StandardScaler(...),
    Model(...)
)
```

core should:

```text
fit scaler
transform data
fit child model
persist scaler parameters
persist child artifact
```

Forecast:

```text
transform context
forecast child
inverse-transform output
```

The resulting artifact is a composite OpenForecast artifact.

---

# Implement ensemble execution

For:

```python
Ensemble(...)
```

fit each child and persist child refs in the parent artifact.

Forecast each child and combine normalized OpenForecast forecasts.

The parent artifact should not copy all child binary payloads if artifact references are sufficient.

---

# User-facing client

At this point implement:

```python
import openforecast as of

model = of.fit(
    model="builtin/seasonal-naive",
    data=train,
    name="baseline",
    params={"season_length": 24},
)

forecast = of.forecast(
    model=model,
    data=context,
    horizon=48,
)
```

Also object-oriented API:

```python
client = of.OpenForecast()

model = client.fit(...)
forecast = client.forecast(...)
```

Top-level functions should call a default local client.

## Done when

A full local fit/persist/reload/forecast test works with **no optional provider installed**.

---

# Plan 7 — Provider process protocol and isolated uv environments

Now build the architectural boundary that Nixtla will use.

## Goal

A provider must run in another Python environment/process without changing `Engine`.

---

# Environment layout

Use:

```text
~/.cache/openforecast/providers/
    nixtla/
        <provider-version>/
            .venv/
```

I would manage this directly rather than relying exclusively on global `uv tool install`, because OpenForecast needs deterministic versioned environments.

Use uv to create/install:

```bash
uv venv <path>
uv pip install --python <path>/bin/python openforecast-nixtla==...
```

For development:

```bash
openforecast providers install nixtla \
    --source ./integrations/nixtla
```

can install the local package.

The provider package itself has its own `uv.lock`.

---

# CLI

Implement:

```bash
openforecast providers list
openforecast providers install nixtla
openforecast providers remove nixtla
openforecast providers inspect nixtla
```

Expected output:

```text
PROVIDER   VERSION   STATUS
builtin    0.1.0     available
nixtla     0.1.0     installed
darts      -         not installed
sktime     -         not installed
```

No Darts/sktime packages need to exist yet.

---

# Provider handshake

Provider executable starts and receives:

```json
{
  "protocol_version": 1,
  "operation": "handshake"
}
```

Returns:

```json
{
  "protocol_version": 1,
  "provider": "nixtla",
  "provider_version": "0.1.0",
  "models": []
}
```

Reject incompatible protocol versions explicitly.

---

# RPC transport

Use:

```text
stdin/stdout JSON Lines
```

for control messages.

Use Arrow IPC bundles for data.

For example:

```json
{
  "protocol_version": 1,
  "operation": "fit",
  "request_id": "...",
  "recipe": {...},
  "fit_plan": {...},
  "data": {
    "path": "/tmp/of-123/frame"
  },
  "artifact_destination": "/tmp/of-123/artifact"
}
```

Provider response:

```json
{
  "request_id": "...",
  "status": "ok",
  "artifact": {
    "path": "/tmp/of-123/artifact"
  }
}
```

Forecast response references a forecast Arrow IPC file.

---

# Error envelope

Do not dump Python tracebacks as protocol semantics.

Use:

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

Provider stderr may contain native logs.

Core translates codes into OpenForecast exceptions.

---

# Logging

Critical rule:

```text
stdout = protocol only
stderr = provider logs
```

Otherwise library logging can corrupt JSON Lines RPC.

---

# Process tests

Create a fake provider package/environment in tests and verify:

```text
handshake
model discovery
fit
forecast
provider crash
malformed JSON
protocol mismatch
timeout
stderr logging
```

## Done when

You can substitute:

```python
BuiltinProviderClient
```

with:

```python
SubprocessProviderClient
```

without changing `Engine`.

---

# Plan 8 — Build the provider conformance suite

Do this **before Nixtla**.

## Goal

Any future provider/model can declare capabilities and automatically prove them.

---

# Conformance matrix

Create reusable tests for:

```text
single-instance univariate
single-instance multivariate
panel univariate
panel multivariate

observed feature
known feature
static feature

point forecast
quantile forecast

fit
artifact save
artifact load
forecast

alias resolution
recipe serialization
```

For each capability:

```text
supported
    → operation must succeed

unsupported
    → OpenForecast validation must reject before native execution
```

That second case is important.

A model claiming:

```python
capabilities.targets.multivariate = False
```

must produce:

```text
UnsupportedDataShape
```

not a random downstream Nixtla exception.

---

# Golden fixtures

Create tiny deterministic datasets:

```text
hourly_single_uni
hourly_single_multi
hourly_panel_uni
hourly_panel_multi
with_observed
with_known
with_static
```

Keep them tiny enough for every integration's CI.

---

# Contract fixtures

Store canonical JSON fixtures for:

```text
ModelDescriptor
ModelRecipe
FitPlan
ModelArtifact manifest
provider handshake
provider fit request
provider forecast request
```

This protects your protocol from accidental breaking changes.

## Done when

The built-in SeasonalNaive provider passes every capability it claims.

Future providers should be able to consume the same conformance harness.

---

# Plan 9 — First Nixtla provider: StatsForecast / AutoARIMA

Now add the first real external provider.

## Goal

Install:

```text
openforecast-nixtla
```

and make:

```python
of.fit("nixtla/autoarima", ...)
```

work through an isolated uv environment.

StatsForecast is a strong first backend because its current API explicitly distinguishes `fit()`/`predict()` from its stateless memory-efficient `forecast()` and provides save/load for fitted model state. ([Nixtla][5])

---

# Integration structure

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

`neuralforecast.py` may remain empty until Plan 10.

---

# Nixtla provider catalog

Initially advertise:

```text
nixtla/autoarima
```

Descriptor approximately:

```yaml
provider: nixtla

lifecycle:
  requires_fit: true
  supports_fit: true

capabilities:
  instances:
    single: true
    panel: true

  targets:
    univariate: true
    multivariate: false

  features:
    observed: false
    known: true
    static: false

  outputs:
    point: true
    quantiles: true
    samples: false
```

Verify actual supported exogenous behavior against StatsForecast and only claim what the adapter implements.

Do not overclaim capabilities because the native library theoretically supports something.

Capabilities describe the **OpenForecast integration**.

---

# Conversion layer

OpenForecast data:

```text
country
timestamp
load
wind_fc
```

becomes internally:

```text
unique_id
ds
y
wind_fc
```

For a single instance, generate an internal reserved ID.

Never mutate the original Arrow tables.

Keep mapping metadata so predictions map back to original OpenForecast instance keys and target names.

StatsForecast's fit API currently accepts custom `id_col`, `time_col`, and `target_col`, so you may avoid physically renaming some fields; nevertheless, that remains provider implementation detail. ([Nixtla][5])

---

# Fit

Compile:

```python
of.Model(
    "nixtla/autoarima",
    params={...},
)
```

into:

```python
AutoARIMA(...)
```

and:

```python
StatsForecast(
    models=[...],
    freq=compiled_frequency,
)
```

then:

```python
sf.fit(...)
```

Persist native state into:

```text
artifact_destination/
    statsforecast.pkl
    provider.json
```

StatsForecast supports saving/restoring the fitted instance directly. ([Nixtla][5])

---

# Forecast

Provider loads:

```text
statsforecast.pkl
```

uses:

```python
sf.predict(
    h=task.horizon,
    X_df=...
)
```

and converts the result to canonical OpenForecast `Forecast`.

---

# API that must work

```python
import openforecast as of

train = of.TimeSeriesFrame.from_pandas(
    history=df,
    time="timestamp",
    frequency="1h",
    instance_keys="country",
    targets="load",
)

model = of.fit(
    model="nixtla/autoarima",
    data=train,
    name="load-arima",
)

print(model.ref)
# local/load-arima@01K...

forecast = of.forecast(
    model=model,
    data=context,
    horizon=24,
)
```

And:

```python
forecast = of.forecast(
    model="local/load-arima",
    data=context,
    horizon=24,
)
```

must give the same result.

This must fail:

```python
of.forecast(
    model="nixtla/autoarima",
    data=context,
    horizon=24,
)
```

with:

```text
ModelRequiresFit
```

and an actionable message.

---

# Conformance

Run at minimum:

```text
single univariate
panel univariate
fit/save/load/forecast
known future exogenous if implemented
point output
quantile output if implemented
unsupported multivariate rejection
```

## Done when

Nixtla is fully uninstallable from root and:

```bash
uv run pytest
```

for the core succeeds even when no Nixtla packages exist.

Then separately:

```bash
openforecast providers install nixtla --source integrations/nixtla
```

enables the model.

---

# Plan 10 — Extend Nixtla provider with NeuralForecast / NHiTS

This is the stage that really validates the abstraction.

## Goal

Add:

```text
nixtla/nhits
```

without changing the OpenForecast data model, lifecycle model, model recipe format or public `fit()` / `forecast()` APIs.

If adding NHiTS requires changing those concepts significantly, treat that as an architectural warning.

NeuralForecast currently expects long-format `unique_id`, `ds`, `y`, and model-specific exogenous lists; the adapter should produce those from OpenForecast's semantic schema. ([Nixtla][3])

---

# Catalog

Advertise:

```text
nixtla/nhits
```

with a JSON Schema for supported model parameters:

```json
{
  "type": "object",
  "properties": {
    "input_size": {"type": "integer", "minimum": 1},
    "max_steps": {"type": "integer", "minimum": 1},
    "learning_rate": {"type": "number", "exclusiveMinimum": 0}
  },
  "additionalProperties": false
}
```

This is intentionally provider/model-specific.

---

# Schema compilation

OpenForecast:

```python
FeatureSpec(
    name="temperature",
    availability="observed",
)
```

compiles to:

```python
hist_exog_list=["temperature"]
```

OpenForecast:

```python
FeatureSpec(
    name="wind_forecast",
    availability="known",
)
```

compiles to:

```python
futr_exog_list=["wind_forecast"]
```

Static features compile to:

```python
stat_exog_list=[...]
```

where supported.

This is precisely where OpenForecast's semantics prove valuable: the public protocol never mentions `hist_exog_list` or `futr_exog_list`. NeuralForecast itself requires these distinctions when constructing models. ([Nixtla][3])

---

# Fit compilation

This:

```python
model = of.fit(
    model=of.Model(
        "nixtla/nhits",
        params={
            "input_size": 168,
            "max_steps": 100,
        },
    ),
    data=train,
    plan=of.FitPlan(seed=42),
    name="load-nhits",
)
```

roughly compiles internally to:

```python
NHITS(
    h=24,
    input_size=168,
    max_steps=100,
    hist_exog_list=[...],
    futr_exog_list=[...],
    stat_exog_list=[...],
)
```

inside the provider.

The `h` question is important.

If NHiTS requires horizon at training time, record that fact in capabilities/model metadata:

```yaml
training:
  horizon_bound_at_fit: true
```

Then:

```python
of.fit(..., horizon=24)
```

must store horizon `24` in the artifact.

Trying:

```python
of.forecast(
    model=model,
    horizon=48,
)
```

should raise:

```text
IncompatibleForecastTask
```

before invoking NeuralForecast.

This is exactly the kind of semantic mismatch OpenForecast should normalize.

---

# Native persistence

Use NeuralForecast's own persistence APIs under the provider artifact directory:

```text
provider/
    neuralforecast/
        ...
```

NeuralForecast currently exposes `save(path=...)` and `load(path=...)` for fitted state. ([Nixtla][6])

OpenForecast continues to know only:

```text
artifact manifest
+
opaque provider directory
```

---

# API that must now work

Panel + future features:

```python
train = of.TimeSeriesFrame.from_pandas(
    history=train_df,
    time="timestamp",
    frequency="1h",
    instance_keys=["country"],
    targets=["load"],
    observed_features=["temperature_actual"],
    known_features=["temperature_forecast", "holiday"],
)

model = of.fit(
    model=of.Model(
        "nixtla/nhits",
        params={
            "input_size": 168,
            "max_steps": 100,
        },
    ),
    data=train,
    horizon=24,
    name="europe-load",
)

context = of.TimeSeriesFrame.from_pandas(
    history=context_df,
    future=future_df,
    time="timestamp",
    frequency="1h",
    instance_keys=["country"],
    targets=["load"],
    observed_features=["temperature_actual"],
    known_features=["temperature_forecast", "holiday"],
)

forecast = of.forecast(
    model=model,
    data=context,
    horizon=24,
    output=of.OutputSpec.quantiles(
        [0.1, 0.5, 0.9]
    ),
)
```

Where the requested quantiles are only allowed if the configured NHiTS loss/output supports them. Capability validation may therefore also depend on fitted artifact configuration.

---

# Pipeline must work with NHiTS

```python
recipe = of.Pipeline(
    steps=[
        of.StandardScaler(
            columns="targets",
            per_instance=True,
        ),
        of.Model(
            "nixtla/nhits",
            params={
                "input_size": 168,
                "max_steps": 100,
            },
        ),
    ]
)

model = of.fit(
    model=recipe,
    data=train,
    horizon=24,
)
```

The scaler belongs to OpenForecast.

NHiTS remains just the leaf implementation.

---

# Ensemble must work across Nixtla families

This should now work:

```python
recipe = of.Ensemble(
    models=[
        of.Model(
            "nixtla/nhits",
            params={"input_size": 168},
        ),
        of.Model(
            "nixtla/autoarima",
        ),
    ],
    combine=of.WeightedMean(
        weights=[0.7, 0.3],
    ),
)

model = of.fit(
    model=recipe,
    data=train,
    horizon=24,
)
```

This is a crucial test.

You now have one OpenForecast artifact representing:

```text
OpenForecast Ensemble
        │
        ├── local/<NHITS artifact>
        │
        └── local/<AutoARIMA artifact>
```

Nixtla itself does not own the ensemble.

OpenForecast does.

---

# Conformance

NHiTS should pass everything it declares:

```text
single univariate
panel univariate
observed features
known features
static features if supported
fit/save/load
horizon-bound validation
point output
quantile output when configured
```

And fail gracefully for unsupported OpenForecast axes.

---

# Plan 11 — Public API stabilization and full V1 end-to-end test

This is where I would call the first OpenForecast milestone complete.

## Goal

Ensure users never need to know about:

```text
provider subprocesses
Arrow IPC paths
Nixtla DataFrames
unique_id
ds
y
StatsForecast
NeuralForecast wrappers
```

unless they intentionally inspect provider metadata.

---

# Final Python API

These must all work.

### Discover

```python
import openforecast as of

of.models.list()
```

Example:

```text
builtin/seasonal-naive
nixtla/autoarima
nixtla/nhits
```

And:

```python
model = of.models.get("nixtla/nhits")

model.lifecycle.requires_fit
# True

model.capabilities.instances.panel
# True
```

---

### Fit via string

```python
model = of.fit(
    model="nixtla/nhits",
    data=train,
    horizon=24,
)
```

---

### Fit via explicit recipe

```python
model = of.fit(
    model=of.Model(
        "nixtla/nhits",
        params={
            "input_size": 168,
        },
    ),
    data=train,
    horizon=24,
)
```

---

### Forecast fitted artifact

```python
forecast = of.forecast(
    model=model,
    data=context,
    horizon=24,
)
```

---

### Forecast by clean string ref

```python
forecast = of.forecast(
    model="local/europe-load",
    data=context,
    horizon=24,
)
```

---

### Pipeline

```python
recipe = of.Pipeline(
    steps=[
        of.StandardScaler(
            columns="targets",
            per_instance=True,
        ),
        of.Model("nixtla/nhits"),
    ],
)
```

---

### Ensemble

```python
recipe = of.Ensemble(
    models=[
        of.Model("nixtla/nhits"),
        of.Model("nixtla/autoarima"),
    ],
    combine=of.Mean(),
)
```

---

### Reduction protocol

This must parse and validate:

```python
recipe = of.Reduction(
    estimator="sklearn/lightgbm",
    strategy="recursive",
    lags=[1, 24, 168],
)
```

but execution should clearly state:

```text
No installed provider supports this Reduction recipe.

Install a provider with reduction support.
```

Later sktime should make this **same recipe** executable.

Do not change its structure when sktime arrives.

---

# CLI smoke test

Add:

```bash
openforecast models list
openforecast models inspect nixtla/nhits

openforecast providers list
openforecast providers inspect nixtla
```

Do not turn the CLI into a second API architecture.

CLI simply calls the same client.

---

# Full acceptance suite

Create one `tests/e2e/test_v1.py` that performs:

```text
install Nixtla provider in isolated environment

discover nixtla/autoarima
discover nixtla/nhits

construct panel TimeSeriesFrame

fit AutoARIMA
persist
reload
forecast

fit NHiTS with tiny max_steps
persist
reload
forecast

fit StandardScaler -> NHiTS pipeline
forecast

fit NHiTS + AutoARIMA ensemble
forecast

resolve artifact via local alias

assert OpenForecast forecast Arrow schema

assert no Nixtla-specific fields escape public objects
```

For CI, use tiny NHiTS training:

```text
max_steps ≈ 1–5
small dataset
```

You're testing integration correctness, not model accuracy.

---

# V1 architectural invariant test

I would literally add a test checking serialized OpenForecast public objects for forbidden terminology:

```text
unique_id
ds
y
hist_exog_list
futr_exog_list
stat_exog_list
```

Those terms are allowed under:

```text
integrations/nixtla/
```

but not inside semantic protocol types.

This sounds slightly extreme, but it's an excellent way of preventing accidental abstraction leakage.

---

# Plan 12 — HTTP/OpenAPI projection after local V1

Only do this after the Nixtla milestone passes.

## Goal

Prove that the OpenForecast protocol can become a remote SaaS API without redesigning it.

The dependency direction is:

```text
OpenForecast semantics
        ↓
Engine
        ↓
HTTP projection
        ↓
OpenAPI
        ↓
remote SDKs
```

not:

```text
OpenAPI
   ↓
OpenForecast semantics
```

---

# Local server

Implement:

```bash
openforecast serve
```

providing:

```text
GET  /v1/models
GET  /v1/models/{ref}

POST /v1/fit
POST /v1/forecast

GET  /v1/artifacts/{ref}
```

Initially this can be a development/local HTTP projection.

Do not yet solve production asynchronous training jobs.

---

# SDK transport abstraction

Refactor `OpenForecast` if necessary to:

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

---

# OpenAPI

Generate OpenAPI from the actual Pydantic request/response models.

Commit:

```text
spec/openapi/openapi.json
```

CI:

```bash
uv run generate-openapi
git diff --exit-code spec/openapi/openapi.json
```

That gives you deterministic API versioning.

Later:

```text
TypeScript
Go
Java
```

clients can be generated from this document.

The Python SDK should remain hand-written because its local transport does things an HTTP-generated SDK cannot do.

---

# Plan 13 — Benchmarking and the beginning of `openforecast/auto`

I would put this immediately after the Nixtla integration rather than before Darts/sktime.

It uses the abstraction you've just built and begins creating the part that could eventually differentiate OpenForecast.

## Implement

```python
result = of.benchmark(
    models=[
        "nixtla/autoarima",
        "nixtla/nhits",
        "builtin/seasonal-naive",
    ],
    data=data,
    validation=of.RollingOrigin(
        horizon=24,
        windows=3,
    ),
    metrics=[
        of.MAE(),
        of.Bias(),
    ],
)
```

Return a normalized Arrow-backed result:

```text
model
fold
metric
value
fit_seconds
forecast_seconds
```

The important thing is that benchmarking consumes:

```text
ModelRecipe
TimeSeriesFrame
FitPlan
ForecastTask
```

rather than having a special Nixtla implementation.

Eventually:

```python
of.fit(
    model="openforecast/auto",
    ...
)
```

can simply use that evaluation infrastructure.

---

# What should deliberately NOT be implemented before the Nixtla milestone

I'd keep these out for now:

```text
Darts
sktime
foundation models
cloud execution
Docker runtimes
remote artifact storage
async training jobs
distributed training
hyperparameter search
hierarchical reconciliation
online updating / partial_fit
irregular time series
custom user Python models
Arrow Flight
gRPC
generated Python SDK
```

But crucially, the architecture should already have clear extension points for them.

---

# The stage dependency graph

The order matters:

```text
PLAN 1
Repository
    │
    ▼
PLAN 2
TimeSeriesFrame + semantic axes
    │
    ▼
PLAN 3
ModelRef + capabilities
    │
    ▼
PLAN 4
Recipe + Fit/Forecast protocols
    │
    ▼
PLAN 5
ModelArtifact + local registry
    │
    ▼
PLAN 6
Engine + built-in reference model
    │
    ▼
PLAN 7
Provider RPC + uv isolation
    │
    ▼
PLAN 8
Conformance suite
    │
    ▼
PLAN 9
Nixtla / StatsForecast / AutoARIMA
    │
    ▼
PLAN 10
Nixtla / NeuralForecast / NHiTS
    │
    ▼
PLAN 11
V1 API stabilization + E2E
    │
    ├─────────────┐
    ▼             ▼
PLAN 12        PLAN 13
HTTP/OpenAPI    Benchmark/Auto
```

I would **not let the agent combine Plans 2–4 into one PR**. Those are the pieces where bad abstractions are expensive to unwind. Plans 2, 3 and 4 should each be reviewed as protocol design changes before provider implementation begins.

The milestone after Plan 11 is quite meaningful: you will have a real local OpenForecast where a user installs the lightweight core, pulls Nixtla into an isolated uv runtime, describes data using OpenForecast-native Arrow semantics, fits either statistical or neural models using clean `"nixtla/..."` strings, persists them as `"local/..."` model refs, composes pipelines/ensembles above the provider, and forecasts through exactly the same universal API. At that point Darts and sktime become **tests of the protocol**, rather than opportunities to redesign it.

[1]: https://arrow.apache.org/docs/python/generated/pyarrow.Schema.html?utm_source=chatgpt.com "pyarrow.Schema — Apache Arrow v25.0.1"
[2]: https://docs.astral.sh/uv/concepts/projects/workspaces/?utm_source=chatgpt.com "Using workspaces | uv - Astral Docs"
[3]: https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/exogenous_variables.html?utm_source=chatgpt.com "Exogenous Variables"
[4]: https://arrow.apache.org/docs/python/ipc.html?utm_source=chatgpt.com "Streaming, Serialization, and IPC — Apache Arrow v25.0.1"
[5]: https://nixtlaverse.nixtla.io/statsforecast/src/core/core.html "StatsForecast core methods and API reference - Nixtla"
[6]: https://nixtlaverse.nixtla.io/neuralforecast/core.html "Core | NeuralForecast - Nixtla"
