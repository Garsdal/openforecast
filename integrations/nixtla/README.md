# openforecast-nixtla

Nixtla's models as an OpenForecast provider, in their own environment.

```bash
openforecast providers install nixtla
```

```python
import openforecast as of

model = of.fit(model="nixtla/autoarima", data=timeseries, params={"season_length": 24})
forecast = of.forecast(model=model, data=context, horizon=48)
```

## What it provides

```text
nixtla/autoarima    order selection over ARIMA models, one model per series
nixtla/nhits        multi-rate hierarchical interpolation, one model over all samples
```

The two are the two halves of the design. `autoarima` is a *local* model fitted
per series; `nhits` is a *global* model fitted across every training sample at
once. Their descriptors say so, and the engine reads that rather than asking.

### `nixtla/autoarima`

```yaml
training:
  view: series
  origin_scope: single
  horizon_bound_at_fit: false

capabilities:
  instances:  single, panel
  targets:    univariate
  features:   known (as exogenous regressors)
  outputs:    point, quantiles
  missing:    unsupported
```

Quantiles are the library's prediction intervals, which is what they already are:
the 0.1 and the 0.9 of the predictive distribution are the bounds of its 80%
interval, and the 0.5 of a symmetric one is the point forecast.

```python
forecast = of.forecast(
    model=model, data=context, horizon=48,
    output=of.OutputSpec.quantiles([0.1, 0.5, 0.9]),
)
```

Samples are not declared and are refused rather than approximated: paths drawn
through fitted interval bounds are paths this model never produced.

Which means point-in-time data is usable at one origin and not across origins:

```python
of.fit(model="nixtla/autoarima", data=forecast_dataset,
       plan=of.FitPlan(origins=of.AtOrigin(ref_time)))   # a SeriesView, so fine

of.fit(model="nixtla/autoarima", data=forecast_dataset,
       plan=of.FitPlan(origins=of.AllOrigins()))         # OriginScopeError
```

AutoARIMA does not learn jointly across historical forecast origins, so the
second request is refused by the engine before this integration is started.

### `nixtla/nhits`

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

Which is the request AutoARIMA had to refuse:

```python
model = of.fit(
    model=of.Model("nixtla/nhits", params={"max_steps": 500}),
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
existed at that origin.

Three things follow from that, and all three are the engine's rather than this
integration's:

- **The window is stated once.** `WindowPlan(context=168)` compiles to
  `input_size=168` and the task's `horizon=72` to `h=72`. Passing either as a
  provider parameter is refused by `of.Model`, because it would be a second copy
  of a number the `ViewPlanner` already had to know.
- **The horizon is bound.** NHiTS learns an output layer of exactly `h` steps,
  so an artifact fitted for 72 answers `IncompatibleForecastTask` when asked for
  48.
- **Missing values are not filled in here.** Point-in-time data is full of real
  gaps — an observed feature has no value past its own origin — and NHiTS cannot
  take one through a gradient step. So the fit is refused, naming the
  `of.Impute` the caller would have to write down. (Executing that step is not
  implemented yet; asking for it says so rather than silently skipping it.)

## Layout

```text
src/openforecast_nixtla/
    __main__.py     the serving harness, two lines
    provider.py     the three provider calls, dispatched
    catalog.py      which models exist, and which adapter runs each
    conversion.py   the views <-> Nixtla's long frame
    parameters.py   a native parameter, as both a schema and a check
    state.py        what an adapter remembers beside the native model
    adapters/
        statsforecast.py    local statistical models    -> SeriesView
        neuralforecast.py   global neural models        -> SequenceView
```

`unique_id`, `ds`, `y`, `hist_exog_list`, `futr_exog_list` and `stat_exog_list`
are legal inside this distribution and nowhere else in OpenForecast. They are
constructed in `conversion.py` on the way into a Nixtla library and taken off
again on the way out; what crosses the provider boundary is an execution view
and an Arrow table in the canonical forecast columns.

Neither adapter imports its library at module scope. A handshake — which is what
installing a provider and listing models does — only asks what this integration
advertises, and `neuralforecast` pulls in PyTorch.

## Development

```bash
uv sync
uv run pytest
```

The tests include the OpenForecast conformance suite, which is generated from
what the descriptors above declare: every capability becomes a fit that must
succeed over both semantic sources, and everything withheld becomes a request
that must be refused. `nhits` runs those generated cases with `max_steps=2` —
whether a model consumes a panel is not a question a thousand optimization steps
answer, and the suite only accepts parameters the descriptor already advertises.
