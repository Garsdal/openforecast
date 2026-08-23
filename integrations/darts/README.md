# openforecast-darts

Darts' models as an OpenForecast provider, in their own environment.

```bash
openforecast providers install darts
```

```python
import openforecast as of

model = of.fit(model="darts/theta", data=timeseries, params={"seasonality_period": 24})
forecast = of.forecast(model=model, data=context, horizon=48)
```

This is the integration that exists to check whether the abstraction is
Nixtla-shaped. Switching a point-in-time fit from `nixtla/nhits` to `darts/tide`
changes the model reference and nothing else:

```python
of.fit("nixtla/nhits", data=dataset, horizon=72, plan=plan)
of.fit("darts/tide",   data=dataset, horizon=72, plan=plan)
```

Everything Darts spells differently is compiled inside this distribution.

```text
OpenForecast              Nixtla                    Darts
WindowPlan(context=168)   input_size=168            input_chunk_length=168
horizon=72                h=72                      output_chunk_length=72
one training sample       one unique_id             one TimeSeries in a list
observed feature          hist_exog_list            past_covariates
known feature             futr_exog_list            future_covariates
static feature            stat_exog_list            static_covariates
```

## What it provides

```text
darts/theta    the Theta method, one model per series
darts/tide     dense encoder-decoder over the window, one model over all samples
darts/nhits    multi-rate hierarchical interpolation, one model over all samples
```

`theta` is a *local* model fitted per series; `tide` and `nhits` are *global*
models fitted across every training sample at once. Their descriptors say so,
and the engine reads that rather than asking.

### `darts/theta`

```yaml
training:
  view: series
  origin_scope: single
  horizon_bound_at_fit: false

capabilities:
  instances:  single, panel
  targets:    univariate
  features:   none
  outputs:    point
  missing:    unsupported
```

The same contract `nixtla/autoarima` declares, from another library — because
"local" is a statement about how a model learns, not about whose package it came
from. So the same request is refused in the same way:

```python
of.fit(model="darts/theta", data=forecast_dataset,
       plan=of.FitPlan(origins=of.AllOrigins()))         # OriginScopeError
```

A Theta forecast is a function of the target's own history, so this model
declares no feature support at all. One consequence is worth naming, because it
is about the framework rather than about Darts: a point-in-time dataset holds at
least one feature by construction — an origin and an event time on their own
carry no information — so a model that consumes no feature cannot read vintages
even at a single origin. It is refused for the feature, which is honest and is
also the only local Darts model available at these data sizes: `darts/ARIMA`
takes future covariates but requires 30 observations before it will fit at all,
which the conformance suite's series are shorter than.

### `darts/tide`

```yaml
training:
  view: sequences
  origin_scope: multiple
  context_required: true
  horizon_bound_at_fit: true
  supports_unseen_instances: true

capabilities:
  instances:  single, panel
  targets:    univariate
  features:   observed, known, static
  outputs:    point
  missing:    requires_transform
```

Line for line the declaration of `nixtla/nhits`, which is the claim of this
whole step. So this is the request the local model had to refuse:

```python
model = of.fit(
    model=of.Model("darts/tide", params={"n_epochs": 50}),
    data=forecast_dataset,
    horizon=72,
    plan=of.FitPlan(
        origins=of.AllOrigins(),
        window=of.WindowPlan(context=168),
    ),
    name="de-price",
)

forecast = of.forecast(
    model="local/de-price",
    data=forecast_dataset.at_origin(now),
    horizon=72,
)
```

Every `(instance, origin)` pair of the dataset becomes one training sample of
168 context steps and 72 forecast steps, carrying the values that actually
existed at that origin. A sample is one `TimeSeries` of exactly
`input_chunk_length + output_chunk_length` steps, so the window Darts slides
along it has exactly one position — which is how the "no sample spans two
origins" invariant survives a library that cuts its own windows.

### `darts/nhits`

Same architecture as `nixtla/nhits`, and not the same model:

```yaml
capabilities:
  features:   observed          # and not known, and not static
```

Darts implements NHiTS as a past-covariates model, so it takes no value known
ahead of its event time. That difference costs no code anywhere — it is a
`FeatureCapabilities` declaration, and the engine refuses a known feature by
name before this integration is started. Handing one over as a past covariate
instead would train on it happily while silently ignoring everything it says
about the future, which is exactly the quiet wrong answer the capability
declarations exist to prevent.

It follows that `darts/nhits` cannot read point-in-time vintages today: its one
supported role stops at its own origin by definition, so every sequence sample
carries gaps, and the model declares `requires_transform` rather than filling
them in. `darts/tide` is the model the point-in-time cases run against.

## Layout

```text
src/openforecast_darts/
    __main__.py     the serving harness, two lines
    provider.py     the three provider calls, dispatched
    catalog.py      which models exist, and which adapter runs each
    conversion.py   the views <-> Darts' TimeSeries objects
    parameters.py   a native parameter, as both a schema and a check
    state.py        what an adapter remembers beside the native model
    adapters/
        local_models.py     fitted per series          -> SeriesView
        global_models.py    fitted across all samples  -> SequenceView
```

`TimeSeries`, `past_covariates`, `future_covariates`, `static_covariates`,
`input_chunk_length` and `output_chunk_length` are legal inside this
distribution and nowhere else in OpenForecast. They are constructed in
`conversion.py` on the way into Darts and taken off again on the way out; what
crosses the provider boundary is an execution view and an Arrow table in the
canonical forecast columns.

Darts has no identifier column at all — a panel is a *list* of `TimeSeries`, and
predictions come back as a list in the same order — so where the Nixtla
integration maintains a `unique_id` mapping, this one maintains a position
mapping. Same bookkeeping problem, same place, different spelling.

No adapter imports `darts` at module scope. A handshake — which is what
installing a provider and listing models does — only asks what this integration
advertises, and `darts` pulls in PyTorch and Lightning.

## Development

```bash
uv sync
uv run pytest
```

The tests include the OpenForecast conformance suite, which is generated from
what the descriptors above declare: every capability becomes a fit that must
succeed over both semantic sources, and everything withheld becomes a request
that must be refused. It is the same suite `integrations/nixtla` runs, so
`darts/tide` is held to the point-in-time contract `nixtla/nhits` is held to
without a line of it being restated. The neural models run those generated cases
with `n_epochs=1` — whether a model consumes a panel is not a question a hundred
epochs answer, and the suite only accepts parameters the descriptor already
advertises.
