# OpenForecast

The unified interface for forecasting.

OpenForecast is a framework-agnostic forecasting API. You describe your data and
the model you want in OpenForecast's own vocabulary, and the library compiles
that into whatever the underlying forecasting library expects — statistical,
neural, or tree-based — behind one stable surface:

```python
import openforecast as of

model = of.fit(model="builtin/seasonal-naive", data=train, params={"season_length": 24})
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
> contracts from Step 5, the recipes, fit plans and forecast tasks from Step 6,
> the artifact lifecycle and local model registry from Step 7, the execution
> engine and its built-in reference provider from Step 8, the provider
> subprocess protocol and isolated uv environments from Step 9, the full
> conformance suite from Step 10, the Nixtla integration from Steps 11 and
> 12 — `nixtla/autoarima` and `nixtla/nhits`, in `integrations/nixtla` — and the
> Darts integration from Step 13 — `darts/theta`, `darts/tide` and
> `darts/nhits`, in `integrations/darts` — and the sktime integration from
> Step 14: `sktime/theta` and `sktime/pooled-trees`, in `integrations/sktime` —
> and the public V1 surface of Step 15, which is the one below and no longer
> moves — the HTTP/OpenAPI projection of Step 16: `openforecast serve`,
> `HttpTransport`, and a generated `spec/openapi/openapi.json` — the
> backtesting and point-in-time evaluation of Step 17: `of.backtest`,
> `of.RollingOrigin`, `of.ForecastOriginValidation` and `of.eligible_models` —
> and the native `TabularView` execution of Step 18:
> `sklearn/hist-gradient-boosting`, in `integrations/sklearn`, which fits
> `estimator.fit(X, y)` on the supervised rows OpenForecast materializes itself —
> and the probabilistic normalization of Step 20: one `Forecast` for point
> forecasts, quantiles and sample paths, `of.PinballLoss`, `of.Coverage` and
> `of.IntervalWidth` over them, and `nixtla/autoarima`'s prediction intervals
> answering `of.OutputSpec.quantiles([...])` — and the first zero-shot foundation
> model of Step 23: `amazon/chronos-2`, in `integrations/chronos`, which
> forecasts from its own reference without ever being fitted.
> `of.fit` and `of.forecast` work end to end today with `builtin/seasonal-naive`,
> both Nixtla models, all three Darts models, both sktime models and the
> scikit-learn estimator, in this process or over the subprocess protocol — the
> global ones trained on real point-in-time vintages, one training sample per
> historical forecast origin, or one supervised row per origin and lead.
> Switching a point-in-time fit between `nixtla/nhits`, `darts/tide`,
> `sktime/pooled-trees` and `sklearn/hist-gradient-boosting` changes the model
> reference and nothing else, which is what
> `tests/e2e/test_v1_experience.py` runs against all four at once — and
> `amazon/chronos-2` joins the same backtest without being fitted at all.
> Step 24 froze that surface: one canonical call per intent, mirrored on a
> client and on the package, and a test that fails if a second one appears.
> See [PLAN.md](PLAN.md) for the 17-step roadmap and
> [PLAN_2.md](PLAN_2.md) for what comes after it.

## The whole workflow

Everything below this section is detail. This is the API:

```python
import openforecast as of

client = of.OpenForecast()

client.models.list()                       # what this build can run
client.models.get("nixtla/autoarima")      # what one reference resolves to

client.fit(...)                            # -> local/de-load@01K...
client.forecast(...)                       # -> Forecast
client.backtest(...)                       # -> BacktestResult
```

Data is built one way, from whatever you have:

```python
data = of.TimeSeriesFrame.from_pandas(...)      # ordinary event-time data
data = of.ForecastDataset.from_pandas(...)      # real forecast vintages
```

And the four operations, end to end:

```python
data = of.TimeSeriesFrame.from_pandas(
    history=df, time="timestamp", frequency="1h",
    instance_keys=["zone"], targets=["load"],
)

model = client.fit("sklearn/hist-gradient-boosting", data=data, horizon=24)
forecast = client.forecast(model, data=data, horizon=24)

result = client.backtest(
    ["sklearn/hist-gradient-boosting", "nixtla/autoarima", "amazon/chronos-2"],
    data=data,
    validation=of.RollingOrigin(horizon=24, windows=5),
    metrics=[of.MAE(), of.Bias()],
)
result.leaderboard("mae")
```

There is one name per intent, and it is the same name everywhere: `fit`, never
`train`; `forecast`, never `predict` or `infer`; `backtest`, never `evaluate` or
`historical_forecasts`. Each of the four operations — the three above and
`eligible_models`, which answers what this data could fit at all — is a method on
a client, and the module-level `of.fit(...)`, `of.forecast(...)`,
`of.backtest(...)` and `of.eligible_models(...)` are the same operations on a
default client that owns the usual store:

```python
of.backtest(models, data, validation=..., metrics=..., client=client)
client.backtest(models, data, validation=..., metrics=...)   # the same call
```

Which client is the only difference, so the signatures are identical apart from
`client=` — asserted in `tests/unit/test_sdk_surface.py` rather than promised, along
with the absence of aliases and the exact contents of `openforecast.__all__`.
`tests/e2e/test_sdk_workflow.py` is this section, executed.

The short forms above are not a convenience layer over a "real" API: a
`Pipeline`, a `FitPlan`, an `OutputSpec` or a `Candidate` is what you write when
you need to say more, and it goes to the same four calls. The machinery behind
them — `ViewPlanner`, `SeriesView`, `SequenceView`, `TabularView`,
`SubprocessProvider`, `ArtifactStore` — is documented throughout this README and
is deliberately not part of the SDK surface: none of it is a name you have to
write to forecast something.

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
| `TabularView`  | individual supervised target rows | HistGradientBoosting, LightGBM, XGBoost |

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

### Supervised rows

A `TabularView` is the third training unit and the one that carries the most
semantics, because it holds no time axis to recover them from. One row is one
`instance × origin × lead`, in three row-aligned tables:

```text
X      the features knowable at the origin
y      what that event time turned out to be
keys   row_id, instance keys, origin_time, event_time, horizon_step
```

The keys are deliberately *not* in `X`. That is what stops an estimator from
being handed a timestamp or a zone as a feature by accident — and it is what
makes a fitted artifact able to forecast an instance it never saw.

Two vintages of the same event time are two rows, and their shared outcome is
repeated rather than reconciled:

```text
X                     y
wind_fc  load_fc      price
NaN      54           80     <- 08:00 forecasting 12:00, wind not published yet
NaN      53           76     <- 08:00 forecasting 13:00
11       55           80     <- 09:00 forecasting 12:00, now it is
12       54           76     <- 09:00 forecasting 13:00
```

Four distinct forecasting examples, because their information vintages differ.
Nothing is deduplicated on `target_time`, and nothing is filled in.
`integrations/sklearn` is what consumes this: `estimator.fit(X, y)`, and
`estimator.predict(X)` at inference, with the reduction — origin, lead, vintage,
truth alignment — already done on OpenForecast's side of the boundary.

## Models

A model is named by a string:

```text
<namespace>/<name>[@revision]

nixtla/nhits
nixtla/autoarima
darts/nhits
sktime/pooled-trees
sklearn/hist-gradient-boosting
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

`of.models.list()` holds what the installed providers advertise: the built-in
reference provider, and whichever integrations have been installed into their
own environments.

## Zero-shot foundation models

There are two model lifecycles, and a descriptor says which one it is:

```text
trainable     of.fit(...) -> local/de-price@01K... -> of.forecast(...)
pretrained    of.forecast(model="amazon/chronos-2", ...)
```

```python
forecast = of.forecast(
    model="amazon/chronos-2",
    data=pit_dataset.at_origin(historical_ref_time),
    horizon=72,
)
```

No fit, no artifact, no second call. Fitting one is refused rather than quietly
accepted:

```python
of.fit(model="amazon/chronos-2", data=data)
# ModelDoesNotSupportFit: amazon/chronos-2 cannot be fitted; it is used
# zero-shot, so forecast with the reference directly
```

Every error carries `error.code` — here `MODEL_DOES_NOT_SUPPORT_FIT` — so a
caller branches on the failure rather than on its prose.

There is no foundation-model data primitive. A pretrained model consumes the
`ForecastContext` a fitted one does, materialized by the same `ViewPlanner` into
the same `ForecastView`, so point-in-time semantics hold without the integration
knowing they exist: OpenForecast owns the information vintage and Chronos
receives the context that vintage implies. What is different is only what is
absent — no horizon bound, no fitted schema, no transforms — and because there
was never a fit to check the declaration against, the capabilities are checked
against the forecast view instead, at the one moment the model is handed data.

That is what puts both lifecycles on one leaderboard:

```python
result = of.backtest(
    models=[
        "sklearn/hist-gradient-boosting",
        "nixtla/nhits",
        "amazon/chronos-2",
    ],
    data=pit_dataset,
    validation=of.ForecastOriginValidation(origins=of.AllOrigins(stride=24), horizon=72),
    metrics=[of.MAE(), of.Bias()],
)
```

The first two are fitted per fold; the third forecasts as it stands. Its rows
report a null `fit_seconds` and an `origin_fidelity` of `pretrained`, because
there were no training origins at all — which is a different thing from a frozen
artifact that may have seen data postdating the early ones.

Chronos-2 can be fine-tuned and this integration does not expose it. Doing so
would mean a training contract, a published artifact and both lifecycles behind
one reference; the lifecycle it declares is the one it has.

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
`of.Reduction(estimator="sklearn/hist-gradient-boosting", strategy="direct", lags=[1, 24, 168])`
expresses generating a tabular problem out of an ordinary event-time series.
A `ForecastDataset` needs none of that: it already carries the features a
supervised row is built from, so `of.fit("sklearn/hist-gradient-boosting",
data=dataset, horizon=72)` is the whole request.

**OpenForecast owns what it can own.** A context length is stated once, as
`WindowPlan(context=168)`, and compiled into `input_size` for Nixtla,
`input_chunk_length` for Darts or `window_length` for sktime. Passing one of
those as a provider parameter is an error that names the field to use instead —
the same for a horizon, a seed, a frequency or a covariate list. Two copies of one number, free to disagree,
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
manifest, a provider subprocess request and an HTTP body; no part of it is
provider-specific.

## Fitted models

A fit produces a resource, not a variable. It is written once, addressed by a
reference, and described by a manifest:

```text
~/.local/share/openforecast/
    models/
        01K5Z6QK3M9TQK1W2E3R4T5Y6U/
            manifest.json     what this is
            recipe.json       what was fitted
            schema.json       the training view's schema
            provider/         opaque
    aliases/
        de-price.json
```

`local/de-price@01K5Z6QK3M9TQK1W2E3R4T5Y6U` is one immutable revision, so
forecasting from it today gives the model it gave a month ago.
`local/de-price` is an alias that follows the latest selected revision, which is
what lets a scheduled forecast job name a model once and pick up retrainings —
and lets a rollback be a pointer move rather than a retrain.

Nothing is published until the fit succeeds. A provider trains into
`.tmp/<artifact-id>` and the directory is renamed into place afterwards, because
a half-written artifact that is nevertheless resolvable would not fail — it
would forecast.

The manifest is what everything except the provider reads:

```json
{
  "training": {
    "view": "sequences",
    "origin_fidelity": "observed",
    "context": 168,
    "horizon": 72,
    "samples": 8832
  }
}
```

It records the artifact id, the source model, the recipe, the provider and its
version, the OpenForecast and protocol versions, the training view and its
origin fidelity, the origin selection, context, horizon and sample count, the
schema the model expects to see again, and any transform that touched the
missing values on the way in. Every one of those is read off the materialized
view rather than reported by the provider, so a manifest cannot describe a fit
that did not happen. `ModelHandle` is a reference plus that manifest and
deliberately nothing else — listing ten artifacts should not deserialize ten
neural networks.

The registry is where a string becomes a state:

```python
from openforecast.registry import ModelRegistry

registry = ModelRegistry()

registry.for_fit("nixtla/nhits")     # a descriptor: plan a fit against this
registry.resolve("local/de-price")   # a handle: forecast with this
registry.resolve("nixtla/nhits")     # ModelRequiresFit
```

That last one is the string lifecycle. Forecasting with a reference that names
an unfitted model is refused rather than quietly fitted on whatever data the
call was handed, which would return a number that looks like a forecast from a
model nobody trained. A model that declares `requires_fit=False` resolves to its
descriptor instead: zero-shot use is something a model states, not something
OpenForecast assumes.

## Fitting and forecasting

```python
import openforecast as of

model = of.fit(
    model="builtin/seasonal-naive",
    data=train,
    params={"season_length": 24},
    name="de-load",
)                                      # local/de-load@01K5Z6QK3M9TQK1W2E3R4T5Y6U

forecast = of.forecast(model="local/de-load", data=context, horizon=48)

forecast.point().to_pandas()
```

A fit is five steps, and none of them branch on who provides the model: the
recipe is normalized, every reference is resolved to a descriptor, the
`ViewPlanner` materializes the view that descriptor's contract names, the view
is checked against what the model declared it can consume, and only then is a
provider started — into a staging directory that is published if, and only if,
the fit succeeded. There is no place in that sequence for `if provider ==
"nixtla"`, because there is nothing left for it to decide.

`builtin/seasonal-naive` is the reference provider: a real local model with a
real contract, so the engine can be proved end to end without a forecasting
library installed. It is held to the same import boundary an external
integration is — its whole surface is `openforecast.views`,
`openforecast.errors`, `openforecast.protocol` and `openforecast.models`, and
the architecture tests check that.

Point-in-time data goes through the same two calls:

```python
model = of.fit(
    model="builtin/seasonal-naive",
    data=forecast_dataset,
    plan=of.FitPlan(origins=of.AtOrigin(ref_time)),
)

forecast = of.forecast(model=model, data=forecast_dataset.at_origin(ref_time), horizon=24)
```

The artifact records `origin_fidelity: observed`, and the provider sees a
`SeriesView` it cannot distinguish from one cut out of event-time data. Asking a
series model to learn from *every* vintage raises `OriginScopeError` — from the
planner, which is the only thing that knows the source type.

Pipelines and ensembles are executed by OpenForecast rather than by any
provider:

```python
model = of.fit(
    model=of.Ensemble(
        models=[
            of.Pipeline(steps=[
                of.StandardScaler(columns="targets"),
                of.Model("builtin/seasonal-naive", params={"season_length": 24}),
            ]),
            of.Model("builtin/seasonal-naive", params={"season_length": 168}),
        ],
        weights=[0.7, 0.3],
    ),
    data=train,
    name="de-load",
)
```

Each leaf is materialized, transformed and fitted into its own directory inside
the one artifact, and the manifest records one training record per leaf, because
two members may consume different views and there is no single materialization
such an artifact could honestly claim. The scaler's statistics are fitted once
and persisted: inference is scaled by those, never by whatever the forecast
context happens to contain, and the forecast comes back on the scale the
caller's data was on.

Members plan independently, which is what makes an ensemble cross-provider
rather than cross-model:

```python
model = of.fit(
    model=of.Ensemble(
        models=[of.Model("nixtla/nhits"), of.Model("sklearn/hist-gradient-boosting")],
        weights=[0.7, 0.3],
    ),
    data=pit_dataset,
    horizon=72,
)
```

`nixtla/nhits` declares a sequence contract and is handed a `SequenceView`;
`sklearn/hist-gradient-boosting` declares a tabular one and is handed a
`TabularView` of the same data. Neither provider learns that an ensemble
exists — both answer a `Forecast`, and that is what is combined. Weights are
relative and default to an equal average; they are fixed rather than learned,
because a weight fitted on data is a second model.

Every member is checked before any of them runs, at fit and at forecast alike: a
member whose contract the data cannot satisfy — a single-origin model asked to
learn from every point-in-time vintage — rejects the whole ensemble rather than
being quietly trained on something else. Quantiles are averaged level by level,
which is *quantile averaging* and not the quantile of the mixture of the members'
distributions; sample paths are not combined at all, since draw *i* of one member
has nothing to do with draw *i* of another.

A forecast is one long Arrow table, whatever was asked for:

```text
zone event_time target kind     quantile sample value

DE   12:00      price  point    null     null   80
DE   12:00      price  quantile 0.1      null   65
```

A wide forecast changes shape with the request — one column per target, or per
target and quantile, or per sample path — and cannot be read by one reader. So
the long table is what a forecast *is*, and the shapes people actually want are
projections of it:

```python
forecast.table          # the long forecast, in canonical column order
forecast.point()        # the point rows, without the columns describing none
forecast.quantile(0.5)  # one level, in the same shape
forecast.sample(7)      # one draw, in the same shape
forecast.to_wide()      # zone, event_time, price_q0.1, price_q0.5, price_q0.9
forecast.to_pandas()    # the long forecast as a DataFrame
```

`quantile` refuses a level that was never asked for rather than interpolating
between the ones that were: a 0.5 derived from a 0.1 and a 0.9 is a different
number from the one the model would have produced.

## Probabilistic forecasts, normalized

What kind of answer to produce is a request, and what a model can answer is a
declaration:

```python
of.OutputSpec.point()
of.OutputSpec.quantiles([0.1, 0.5, 0.9])
of.OutputSpec.samples(200)
```

```yaml
outputs:
  point: true
  quantiles: true
  samples: false
```

A request the model does not declare is refused before the provider is started,
which is the same rule every other capability follows. There are no separate
result classes for the three: `AutoARIMA`'s prediction intervals, a neural
model's sample paths and a naive point forecast all arrive as one `Forecast`
over one long table, so application code downstream of it does not learn which
provider answered or which of the three forms that provider is native in.

OpenForecast performs exactly one conversion between them, in one direction:

```text
samples  ->  quantiles      the draws are the distribution; read it
quantiles -> samples        refused: the paths would have to be invented
point    ->  anything       refused: there is no distribution to read
```

and it is asked for rather than assumed, because how many draws a quantile was
estimated from is part of what it is:

```python
forecast = of.forecast(
    model=model, data=context, horizon=72,
    output=of.OutputSpec.quantiles([0.1, 0.9], from_samples=200),
)
```

The reduction happens in OpenForecast with one estimator, which is what makes
two providers' quantiles comparable rather than each library's own convention.
A deterministic model is never dressed up as a probabilistic one; a calibration
layer that turns point forecasts into distributions is a thing a caller can ask
for explicitly, later, and not something a request quietly triggers.

## Providers in their own environments

Nixtla wants one version of `torch`, Darts wants another, sktime wants
scikit-learn and statsmodels, the scikit-learn integration wants only
scikit-learn, the Chronos integration wants `torch` and `transformers`, and
OpenForecast wants none of it. So an integration is not installed into the OpenForecast
environment at all: it gets its own, built with `uv`, and it is reached over a
subprocess protocol.

```bash
openforecast providers install nixtla
openforecast providers list
openforecast providers inspect nixtla
openforecast providers remove nixtla
```

```text
~/.cache/openforecast/providers/
    nixtla/
        0.1.0/
            environment.json     what the provider said when it was installed
            .venv/
```

An environment is published only once the provider inside it has answered a
handshake, and what it said is written down. That is what makes discovery cheap:
`of.models.list()` reads recorded JSON and starts no process. A process starts
when a model is actually fitted or forecast with — and the handshake is repeated
then, so an environment whose contents changed underneath its record is refused
rather than executed as something it no longer is.

The transport is two channels, and the split is the point:

```text
control    JSON Lines over stdin/stdout — small, ordered, greppable
bulk       Arrow IPC bundles in a directory the message points at
```

```json
{"protocol_version": 1, "operation": "fit", "model": "nixtla/nhits",
 "view": {"kind": "sequences", "path": "/tmp/openforecast-nixtla-x/view"},
 "into": "/…/.tmp/01K5Z…/provider"}
```

A view bundle is the same tables the in-process provider is handed, so reading
one reconstructs a real `SequenceView` — every invariant the view enforces is
enforced again on the far side of the process. A bundle that was truncated in
transit fails to load rather than training on a short window.

```text
sequences/                      tabular/
    schema.json                     schema.json
    provenance.json                 provenance.json
    temporal.arrow                  x.arrow
    samples.arrow                   y.arrow
    static.arrow                    keys.arrow
```

**stdout carries protocol and nothing else.** Forecasting libraries print, so
the serving harness redirects the provider's stdout to stderr for the duration
of every call and writes responses to the stream it captured at startup — a
provider does not have to be careful, it has to be correct. On the engine's
side, a line of stdout that is not a response is a protocol violation rather
than noise to skip.

Writing an integration is therefore the harness plus a provider object:

```python
from openforecast.providers import serve
from openforecast_nixtla.provider import NixtlaProvider

raise SystemExit(serve(NixtlaProvider()))
```

Failures that only exist once there is a boundary are named rather than
discovered: a process that dies is reported with its exit code and the tail of
its log, a request that is never answered has a deadline and the process is
killed, a provider speaking another protocol version is refused, and an error
envelope is re-raised as the error the same failure would have been in-process,
so a caller's handling does not depend on where the model ran.

The engine, meanwhile, learns none of this. A `SubprocessProvider` answers the
same three calls the in-process one does, and `builtin/seasonal-naive` fitted
over the wire produces the same forecast as `builtin/seasonal-naive` fitted
here — which is the test that says the abstraction holds.

## The same semantics, remotely

Where a forecast runs is a client's transport, not a fact about the library:

```python
client = of.OpenForecast(transport=of.LocalTransport())
client = of.OpenForecast(transport=of.HttpTransport("http://localhost:8321"))
```

Both expose `client.models.list()`, `client.models.get(...)`, `client.fit(...)`
and `client.forecast(...)`, and the code above them never branches on which one
it is holding. `LocalTransport` owns an engine and an artifact store;
`HttpTransport` owns a URL. The service is the *same* transport behind a router,
which is why `tests/e2e/test_remote_transport.py` is written as a comparison —
two clients, the same data, the same calls, and the Arrow tables that come back
have to be equal.

```bash
openforecast serve                      # loopback:8321 by default
```

```text
GET  /v1/models
GET  /v1/models/{ref}

POST /v1/fit
POST /v1/forecast

GET  /v1/artifacts/{ref}
```

Control travels as JSON and bulk data as Arrow IPC — the same split the provider
protocol makes. A recipe, a plan and a horizon are Pydantic models and appear in
the OpenAPI document as themselves; a dataset crosses as the Arrow tables it
already holds rather than as nested JSON rows, and is decoded through the
ordinary constructors, so a truncated table fails to load instead of being
fitted as a shorter history. A failure crosses as the failure: a service that
refuses a fit answers with the exception name, and `except of.DataError` means
the same thing on both transports.

`spec/openapi/openapi.json` is generated from those models, committed, and
diffed in CI, which is rule 7 made mechanical:

```bash
uv run generate-openapi
git diff --exit-code spec/openapi/openapi.json
```

Serving needs the extra; calling a service does not.

```bash
pip install 'openforecast[server]'      # to run one
pip install openforecast                # to call one
```

`HttpTransport` is `urllib`, so the core install stays `pydantic`, `pyarrow` and
`platformdirs`, and a remote-only user never installs a web framework.

## Backtesting and point-in-time evaluation

The same models, over the same origins, scored the same way:

```python
result = of.backtest(
    models=[
        "builtin/seasonal-naive",
        "nixtla/autoarima",
        "nixtla/nhits",
        "darts/nhits",
    ],
    data=train,
    validation=of.RollingOrigin(horizon=24, windows=5),
    metrics=[of.MAE(), of.Bias()],
)

result.leaderboard("mae")
result.best("mae")
```

The defining property of the implementation is negative: there is no
backtesting code in it. No Nixtla backtester, no Darts `historical_forecasts`,
no sktime evaluation harness — `of.backtest` is a loop over `of.fit` and
`of.forecast`, because every question a backtest asks was already answered by
the semantic layer. Which is also why it works over any transport: point a
client at a service and the models are fitted and forecast there.

Point-in-time data is the same call with the validation that fits it, and this
is the part worth the whole design:

```python
result = of.backtest(
    models=["nixtla/nhits", "darts/nhits"],
    data=pit_dataset,
    validation=of.ForecastOriginValidation(
        origins=of.OriginsBetween(start, end, stride=24),
        horizon=72,
    ),
    metrics=[of.MAE()],
)
```

At each origin the features come from *that vintage*, the truth comes from the
truth frame, and later vintages are not merely unused — they are absent from the
object the model is handed:

```python
frame.up_to(moment)      # the history, truncated: simulated origins
dataset.up_to(moment)    # the vintages issued by then: observed origins
```

A fold holds the result of one of those, so there is nothing for a bug in the
backtest loop to reach for. `up_to` on an event-time frame keeps the known
features of the truncated rows — a known feature's later values are knowable in
advance, which is what the role means — and moves nothing else.

The result holds two long Arrow tables. The metrics, three of whose columns are
not measurements:

```text
model  fold  origin  metric  value  pairs  fit_seconds  forecast_seconds
       origin_fidelity  provider  artifact
```

`origin_fidelity` is `simulated` or `observed`, read off the artifact the fold
published rather than declared by the backtest — which makes "simulated
historical availability versus true point-in-time availability" a comparison you
can run rather than a caveat you have to remember. `artifact` is the pinned
revision the numbers came from, so a backtest's winner is a reference you can
forecast with. `pairs` says how many outcomes a value was computed over, so a
fold scored on a third of its horizon is visible in the table rather than only in
the metric.

And the predictions those numbers were computed from:

```text
model  fold  instance keys...  origin_time  event_time  horizon_step
       target  kind  quantile  sample  prediction  actual
```

Long over the distribution too, for the same reason a forecast is: one row per
quantile level or sample draw, so the table does not change shape with what was
asked for. Kept rather than dropped, because the metrics are derivable from these
and not the reverse — so the question everyone asks after a backtest is a projection
rather than a second run:

```python
result.predictions                          # every point, Arrow-backed
result.metrics_by("horizon_step")           # does it degrade after 48?
result.metrics_by(["horizon_step", "zone"])
```

The group keys are columns of the prediction table, including your own instance
keys, and an unknown one is an error naming the columns that exist. That is the
whole slicing story; there is no DSL. It is also the larger table by far —
origins × horizon × instances × targets rows per model.

A candidate that is already a revision is evaluated rather than refitted, which
is how you ask whether the model in production has drifted over the last quarter:

```python
result = of.backtest(
    models=["local/de-price@01K...", "nixtla/nhits"],
    data=pit_dataset,
    validation=of.ForecastOriginValidation(origins=of.AllOrigins(stride=24), horizon=72),
    metrics=[of.MAE()],
)
```

Read from the candidate rather than from a mode argument: a pinned revision
names one immutable fit, so it forecasts at every origin with `fit_seconds`
null, while a recipe or a bare reference is fitted per fold. One caveat comes
with mixing them in one table — a frozen artifact was fitted on data that may
postdate the early origins, so its numbers are optimistic beside a candidate
fitted per fold. That is reported rather than refused, the same way
`origin_fidelity` is.

Scoring a distribution is the same call with the output it needs:

```python
result = of.backtest(
    models=["nixtla/autoarima"],
    data=train,
    validation=of.RollingOrigin(horizon=72, windows=5),
    output=of.OutputSpec.quantiles([0.1, 0.5, 0.9]),
    metrics=[of.MAE(), of.PinballLoss(0.9), of.Coverage(), of.IntervalWidth()],
)
```

`of.PinballLoss(0.9)` is the loss the 0.9 quantile is the optimal answer to,
`of.Coverage()` asks how often the outcome fell inside the 0.1 to 0.9 interval
and `of.IntervalWidth()` how wide that interval was — the calibration and
sharpness halves of one question, which is why the first is best *at* its
nominal level rather than highest, and why the second is only readable beside it.
A pinball loss reads the 0.9 of a provider's native quantiles and of another
provider's sample draws identically, and `of.MAE()` scores the median of either;
neither reading invents anything the model did not say. The metrics are checked
against the requested output before the first fit, so a coverage of a point
forecast is refused in the first line of the run rather than after an hour, and a
metric's name carries its parameter — `pinball[0.9]`, `coverage[0.8]` — because
two of them in one backtest are two rows of one table.

`of.eligible_models` is the screening half of `openforecast/auto`:

```python
for entry in of.eligible_models(pit_dataset, horizon=72, plan=plan):
    print(entry)

nixtla/nhits      eligible
nixtla/autoarima  ineligible: a series view holds one forecast origin, but the
                  selection covers 300 vintages
```

Eligibility means exactly one thing — the fit would not be refused — so it
materializes the view the model's contract asks for and checks it against the
capabilities the model declared, which is the same sequence `of.fit` runs. No
provider is started, and an ineligible model comes back with the sentence the fit
would have failed with.

## The architectural invariant

> OpenForecast owns forecasting semantics. Providers only consume
> provider-neutral **execution views**. Point-in-time and ordinary event-time
> data are materialized into those views before crossing the provider boundary.

Two things follow from this, and they shape the whole repository:

**The core never depends on a forecasting framework.** No Nixtla, Darts,
sktime, scikit-learn, PyTorch, JAX or LightGBM — the core install is `pydantic`,
`pyarrow` and `platformdirs`. Integrations depend on OpenForecast, never the reverse, and
each lives in its own distribution under `integrations/` with its own lockfile
and virtual environment, so providers with incompatible dependency graphs
(Torch vs. JAX, say) can coexist without ever meeting.

**No provider branches on where the data came from.** A provider is handed a
`SeriesView`, `SequenceView`, `TabularView` or `ForecastView` — never a
`TimeSeriesFrame` or a `ForecastDataset`. Point-in-time handling lives in the
`ViewPlanner`, once, instead of being re-derived in every integration. A
provider's whole import surface is `openforecast.views`, `openforecast.errors`,
`openforecast.protocol`, `openforecast.models` and `openforecast.providers` —
the last two being how it declares what it provides and how it is served.

[ARCHITECTURE.md](ARCHITECTURE.md) states all seven rules and how each is
enforced.

## Conformance

The invariant above is a claim about behavior, so `tests/conformance/` checks it
as behavior. Named golden datasets — `panel_multivariate`, `pit_varying_vintages`,
`pit_missingness` and the rest — are materialized into all three fit views from
both semantic sources, and every point-in-time property a provider depends on is
asserted on the result:

```text
leakage        origin 09 sees the value 09 published, and never the one 10 did
sample count   100 origins x 3 instances is 300 sequences, and nothing else
missingness    NaN, NaN, 42 is what the feed did, so it is what the view holds
equivalence    identical vintages materialize exactly like event-time data,
               and differ only in OriginFidelity
```

Provider conformance is generated rather than written. A model's descriptor says
which view it trains on, which shapes and feature roles it takes, whether it
learns across origins and what it does about missing values — and the suite
turns each statement into fits that must succeed and requests that must be
refused:

```python
for case in suite.cases_for(descriptor):
    suite.run_case(case, descriptor=descriptor, provider=provider, store=tmp_path)
```

Declaring `view=sequences` therefore buys tests against an event-time frame and
against real forecast vintages, with the provider asserted to have received a
`SequenceView` in both — which is the view boundary, checked rather than assumed.
The built-in reference provider passes every capability it declares, and so do
all four integrations — `integrations/nixtla`, `integrations/darts`,
`integrations/sktime` and `integrations/sklearn` run this suite against their own
descriptors, so `nixtla/autoarima` is fitted from an event-time frame and from
real vintages at a selected origin, and `nixtla/nhits`, `darts/tide`,
`sktime/pooled-trees` and `sklearn/hist-gradient-boosting` from an event-time
frame and from every vintage at once, without any of it being written down. That
the second one costs nothing to hold to the first one's contract is what Step 13
was for; that the third one does too, from a library whose panel and pooling
semantics are explicit and whose horizon is not bound at fit, is what Step 14 was
for; and that the fourth one does too, from a library that is not a forecasting
framework at all and consumes the third view, is what Step 18 was for — with no
case in the suite written for it.

`cases_for` takes optional parameters, which reach every generated fit and may
only name parameters the descriptor already advertises. It is for models whose
defaults are expensive rather than wrong: a neural model's thousand optimization
steps say nothing about whether it consumes a panel.

Each integration runs that suite beside its own library.
`tests/e2e/test_v1_experience.py` is the opposite arrangement and the only place
the four meet: an OpenForecast install that has never heard of any of them,
reaching all four over the subprocess protocol, the way a user's does. It
discovers their models in one catalog, fits `nixtla/autoarima` at one vintage,
fits `nixtla/nhits`, `darts/tide` and `sktime/pooled-trees` across every
historical origin of the same dataset with the same plan, fits
`sklearn/hist-gradient-boosting` on one supervised row per origin and lead of it,
ensembles a Nixtla model with a Darts one, and checks that nothing any of them
calls its own reaches a descriptor, a manifest or a forecast. It needs the
environments installed, so it skips without them:

```bash
openforecast providers install nixtla    # and darts, and sktime, and sklearn
uv run pytest tests/e2e/test_v1_experience.py
```

## Layering

Imports flow in one direction only:

```text
                    protocol/
                        ↓
      data/  models/  recipes/  tasks/
                        ↓
                     views/
                        ↓
      runtime/  registry/  artifacts/  providers/
                        ↓
   client.py  commands/  server/  evaluation/
```

`evaluation/` is in the outermost layer because backtesting is a *user* of
`of.fit` and `of.forecast` rather than something the engine can reach for, which
is exactly why no provider knows it is being backtested.

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

```bash
uv sync --extra server  # add FastAPI and uvicorn, for `openforecast serve`
uv run generate-openapi # regenerate spec/openapi/openapi.json
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
    runtime/     the execution engine, the subprocess transport, uv environments
    providers/   the provider SDK — the client contract, the serving harness —
                 and the built-in reference provider
    protocol/    the provider wire protocol: messages, errors, versions
    commands/    the CLI, including `openforecast serve`
    server/      the HTTP projection: wire models, transports, the FastAPI app
    evaluation/  backtesting, PIT validation strategies, metrics, results
    client.py    the user-facing client

integrations/    provider distributions, each independently versioned
                 nixtla, darts, sktime, sklearn, chronos (the `amazon` provider)
tests/           unit, contract, conformance and e2e suites
spec/            protocol, Arrow and OpenAPI specifications
```

Every package exists with a docstring naming the step that fills it. Nothing is
a stub API — if a name is not implemented yet, it is absent rather than raising
`NotImplementedError`.
