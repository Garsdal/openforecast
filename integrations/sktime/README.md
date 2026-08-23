# openforecast-sktime

sktime's forecasters as an OpenForecast provider, in their own environment.

```bash
openforecast providers install sktime
```

```python
import openforecast as of

model = of.fit(model="sktime/theta", data=timeseries, params={"sp": 24})
forecast = of.forecast(model=model, data=context, horizon=48)
```

This is the third ecosystem, and it is the one that says out loud what the other
two leave implicit. In sktime, a forecaster handed a panel is *vectorized* over
its instances — many independent fits — unless it is told to pool across them,
which is what makes a model global. Those are exactly OpenForecast's two
training units, named by a library that had to name them:

```python
of.fit("nixtla/nhits",        data=dataset, horizon=72, plan=plan)
of.fit("darts/tide",          data=dataset, horizon=72, plan=plan)
of.fit("sktime/pooled-trees", data=dataset, horizon=72, plan=plan)
```

Everything sktime spells differently is compiled inside this distribution.

```text
OpenForecast              Nixtla             Darts                  sktime
WindowPlan(context=168)   input_size=168     input_chunk_length     window_length=168
horizon=72                h=72               output_chunk_length    ForecastingHorizon(1..72)
one training sample       one unique_id      one TimeSeries         one MultiIndex unit
known feature             futr_exog_list     future_covariates      a column of X
static feature            stat_exog_list     static_covariates      a constant column of X
a global model            —                  a Torch model          pooling="global"
```

## What it provides

```text
sktime/theta          the Theta method, one forecaster per series
sktime/pooled-trees   gradient-boosted trees, reduced recursively and pooled
                      across every training sample
```

`theta` is a *local* model fitted per series; `pooled-trees` is a *global* model
fitted across every sample at once. Their descriptors say so, and the engine
reads that rather than asking.

### `sktime/theta`

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

The same contract `nixtla/autoarima` and `darts/theta` declare, from a third
library — because "local" is a statement about how a model learns, not about
whose package it came from. So the same request is refused in the same way:

```python
of.fit(model="sktime/theta", data=forecast_dataset,
       plan=of.FitPlan(origins=of.AllOrigins()))         # OriginScopeError
```

A Theta forecast is a function of the target's own history, so this model
declares no feature support at all. One consequence is worth naming, because it
is about the framework rather than about sktime: a point-in-time dataset holds at
least one feature by construction — an origin and an event time on their own
carry no information — so a model that consumes no feature cannot read vintages
even at a single origin. It is refused for the feature, which is honest.

`deseasonalize` is off by default here, which is the one place this integration
narrows a library default. sktime's deseasonalization is multiplicative and
raises on a series that touches zero, and a load that goes to zero is an
ordinary series; asking for it back by name puts sktime's own behavior — and its
own refusal — straight back.

### `sktime/pooled-trees`

```yaml
training:
  view: sequences
  origin_scope: multiple
  context_required: true
  horizon_bound_at_fit: false
  supports_unseen_instances: true

capabilities:
  instances:  single, panel
  targets:    univariate
  features:   known, static
  outputs:    point
  missing:    requires_transform
```

So this is the request the local model had to refuse:

```python
model = of.fit(
    model=of.Model("sktime/pooled-trees", params={"max_iter": 300}),
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

Every `(instance, origin)` pair of the dataset becomes one unit of the panel,
holding 168 context steps and 72 forecast steps and carrying the values that
actually existed at that origin:

```text
sample_id  event_time  load   wind_fc
001        00:00       ...    ...
001        01:00       ...    ...
002        00:00       ...    ...
002        01:00       ...    ...
```

`sample_id` is the outer index level and `event_time` the inner one, which is
sktime's explicit panel format. Point-in-time samples share event times by
construction — the same hour is described by several vintages — so the panel
holds one event time in several units and never twice in one, and a pooled
reducer therefore cannot cut a window across the boundary between two origins.

Two things about this model differ from the neural global models of Steps 12 and
13, and both are declarations rather than special cases:

- **The horizon is not bound at fit.** A recursive reduction learns one step and
  rolls, so the artifact answers whatever horizon it is asked for, where
  `darts/tide` bakes 72 into an architecture and refuses 48. The manifest records
  the horizon its samples spanned *and* whether that horizon binds it.
- **There are two feature roles, not three.** sktime has one exogenous frame,
  and a value in it has to exist at the event time being forecast — which an
  observed feature does not. A known feature is a column of `X`, a static
  feature is a column of `X` that is constant within its unit, and an observed
  feature is refused by name before this integration is started.

A forecast at a new origin is an *update*, not a refit: the fitted forecaster is
handed the panel of the origin being asked about with `update_params=False`, so
the parameters are untouched and only the windows the forecast rolls from are
new. That is also what makes an instance the artifact never saw forecastable —
the pooled parameters are shared, and the panel label an instance gets at
inference is positional.

## Layout

```text
src/openforecast_sktime/
    __main__.py     the serving harness, two lines
    provider.py     the three provider calls, dispatched
    catalog.py      which models exist, and which adapter runs each
    conversion.py   the views <-> sktime's pandas containers
    parameters.py   a native parameter, as both a schema and a check
    state.py        what an adapter remembers beside the native model
    adapters/
        local_models.py     fitted per series          -> SeriesView
        panel_models.py     pooled across all samples  -> SequenceView
```

`window_length`, `pooling`, `ForecastingHorizon`, the exogenous `X` and the panel
`MultiIndex` are legal inside this distribution and nowhere else in OpenForecast.
They are constructed in `conversion.py` on the way into sktime and taken off
again on the way out; what crosses the provider boundary is an execution view and
an Arrow table in the canonical forecast columns.

sktime identifies a series by an index level rather than by a column, and an
index level is a bad place to put a caller's instance key — two keys that
stringify the same would silently become one series. So this integration labels
a panel unit by *position* and maps the answer back, where the Nixtla integration
maintains a `unique_id` mapping and the Darts one a list-position mapping. Same
bookkeeping problem, same place, a third spelling.

No adapter imports `sktime` at module scope. A handshake — which is what
installing a provider and listing models does — only asks what this integration
advertises, and importing sktime pulls in its registry, scikit-learn and pandas.

## What this integration is not

It is not a `TabularView` consumer. sktime knows how to turn forecasting into
regression, and OpenForecast already owns that transformation: the `ViewPlanner`
knows the forecast origin, the event time, the lead, the information vintage and
the truth alignment, and a `TabularView` is what it materializes them into.
Reducing through sktime's reduction API instead would put the same semantics in
two places, with the library's version winning silently. `of.Reduction` therefore
stays unexecuted here; a direct `TabularView` consumer is what executes it.

## Development

```bash
uv sync
uv run pytest
```

The tests include the OpenForecast conformance suite, which is generated from
what the descriptors above declare: every capability becomes a fit that must
succeed over both semantic sources, and everything withheld becomes a request
that must be refused. It is the same suite `integrations/nixtla` and
`integrations/darts` run, so `sktime/pooled-trees` is held to the point-in-time
contract `nixtla/nhits` is held to without a line of it being restated. The
boosted model runs those generated cases with `max_iter=5` — whether a model
consumes a panel is not a question a hundred trees answer, and the suite only
accepts parameters the descriptor already advertises.
