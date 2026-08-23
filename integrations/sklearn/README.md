# openforecast-sklearn

scikit-learn estimators as an OpenForecast provider, in their own environment.

```bash
openforecast providers install sklearn
```

```python
import openforecast as of

model = of.fit(
    model="sklearn/hist-gradient-boosting",
    data=forecast_dataset,
    horizon=72,
    params={"learning_rate": 0.05, "max_iter": 500, "max_leaf_nodes": 31},
)
forecast = of.forecast(
    model=model, data=forecast_dataset.at_origin(now), horizon=72
)
```

This is the fourth ecosystem and the first one that is not a forecasting
framework at all. scikit-learn has never heard of a forecast origin, an event
time, a lead or an information vintage. It knows a design matrix and a label
vector — which is exactly why it is the integration that proves the
`TabularView` boundary.

```text
ForecastDataset
      ↓
ViewPlanner
      ↓
TabularView          one row per instance × origin × lead
      ↓
scikit-learn         estimator.fit(X, y)
```

The tempting alternative was to reduce through a framework that already knows
how to turn forecasting into regression — sktime's reduction API, for one. That
would put the same semantics in two places, and the library's version would win
silently:

```text
ForecastDataset -> TabularView -> sktime -> sktime reduction -> sklearn estimator
```

OpenForecast already owns the forecast origin, the target time, the lead, the
information vintage, the training row and the truth alignment. Once a
`TabularView` exists, nobody else has to reinterpret them. So this integration
reduces nothing. It receives rows.

## What it provides

```text
sklearn/hist-gradient-boosting   HistGradientBoostingRegressor on supervised rows
```

One estimator, deliberately. Adding `sklearn/random-forest`, `sklearn/ridge` or
`sklearn/extra-trees` should be a table row and a parameter list; if it ever
needs more than that, the `TabularView` is doing less work than it claims to.

`HistGradientBoostingRegressor` is the honest first one because of a single
capability: it routes `NaN` down a learned default branch rather than refusing
it. Point-in-time data is full of real missing values — a feed that had not
published yet at an origin — so an estimator that needs them filled in first
would need an imputation before it could be exercised at all.

### `sklearn/hist-gradient-boosting`

```yaml
ref: sklearn/hist-gradient-boosting
provider: sklearn

lifecycle:
  requires_fit: true

training:
  view: tabular
  origin_scope: multiple
  horizon_bound_at_fit: false
  supports_unseen_instances: true

capabilities:
  instances:  single, panel
  targets:    univariate
  features:   observed, known, static
  outputs:    point
  missing:    native
```

Three of those are worth a sentence each, because they are declarations rather
than implementation details.

- **The horizon is not bound at fit.** One row is one lead, and the lead is not
  a feature, so a fitted estimator answers a row about lead 96 exactly as it
  answers one about lead 3. `nixtla/nhits` bakes its horizon into an
  architecture and refuses a longer one; this model and `sktime/pooled-trees`
  both decline to, for entirely different reasons.
- **An unseen instance is forecastable.** One set of parameters, learned from
  every row. That holds only because the instance keys are *not* in `X`: a
  `TabularView` keeps `row_id`, the instance keys, `origin_time`, `event_time`
  and `horizon_step` in a separate `keys` table, so an estimator cannot have
  been handed a zone or a timestamp as a feature by accident. A caller who wants
  the zone as a feature asks for it as a static one.
- **`observed` is declared, and it is not a column.** A tabular row describes an
  event time *after* its origin, so a measurement has no value there and the
  view does not offer one as a feature. Declaring `observed: false` would
  advertise a refusal OpenForecast does not make — data carrying observed
  features is accepted. What those features hold reaches a row only if the
  caller carries it as a known feature.

## Point-in-time materialization

This is the shape the whole design exists for. Given vintages:

```text
ref_time  target_time  wind_fc  load_fc  price
08:00     12:00        NaN      54       80
08:00     13:00        NaN      53       76
09:00     12:00        11       55       80
09:00     13:00        12       54       76
```

the estimator is handed:

```text
X                     y
wind_fc  load_fc      price
NaN      54           80
NaN      53           76
11       55           80
12       54           76
```

Four rows, and the duplicated targets are intentional. Origin 08:00 and origin
09:00 both forecast 12:00, and they knew different things when they did — so
they are two distinct forecasting examples that happen to share an outcome.
Nothing is deduplicated on `target_time`, and nothing is filled in where the
wind feed had not published yet.

`of.Reduction` is not needed for any of this, and is not what makes it work:

```python
of.fit("sklearn/hist-gradient-boosting", data=forecast_dataset, horizon=72)
```

A `Reduction` is for the *other* case — creating a tabular problem out of an
ordinary event-time series by generating lagged features. That is still to come,
and this integration does not wait for it, because a `ForecastDataset` already
carries the features a supervised row needs.

Target lags are deliberately *not* generated from `ForecastDataset.truth`.
"The event happened before the origin" and "the realized value was actually
available at the origin" are different statements, and an availability model
invented before it is needed would be a leak dressed up as a feature.

## Layout

```text
src/openforecast_sklearn/
    __main__.py     the serving harness, two lines
    provider.py     the three provider calls, dispatched
    catalog.py      which models exist, and which adapter runs each
    adapter.py      the estimator: descriptor, fit(X, y), predict(X)
    conversion.py   views <-> numpy, and the answer's labels
    parameters.py   a native parameter, as both a schema and a check
    state.py        estimator.pkl and metadata.json
```

`conversion.py` is the short one, and its length is the evidence for the claim
above: the other integrations reshape a view into a long frame with `unique_id`
and `ds`, or a `TimeSeries` per series, or a `MultiIndex` panel with an
exogenous frame. Here `TabularView.X` already *is* the design matrix, in the
column order its schema declares, so the conversion is a cast to `float64`.

There is no pandas anywhere on the execution path. A design matrix is a numpy
array, and this is the one integration that needs no DataFrame to talk to its
library.

Two asymmetries carry what work there is:

- **Inference assembles the matrix; training receives it.** A `ForecastView` is
  a history, a future and a static table, so the horizon rows are built in one
  deterministic order — instance-major, ascending event time — and the answer is
  labeled from that order. `predict` returns *n* numbers and no statement about
  what they are about, so the recorded column order is the contract between a
  fit and every forecast made from it.
- **A static feature is a column, repeated.** The `ViewPlanner` broadcasts it
  onto every training row for free; at inference this integration does it by
  hand, against the recorded order, so a mismatch is a refusal rather than a
  shifted matrix.

### Persistence

```text
provider/
    estimator.pkl    scikit-learn's own persistence
    metadata.json    target, column order, feature roles, row count
```

A pickle only reliably loads in the environment that wrote it. That is
acceptable *here* because it lives inside the provider's own directory of an
artifact, beside the environment record naming the versions that produced it,
and nothing outside the provider boundary reads it. Everything that made the fit
an OpenForecast fit is in the JSON next to it — a column order recovered from a
pickle would be a column order only the pickle could explain.

The artifact manifest remains OpenForecast's: the model reference, the recipe,
the provider version, the training view, the origin fidelity, the schema and the
horizon are recorded there whoever executed the fit.

## Development

```bash
uv sync
uv run pytest
```

The tests include the OpenForecast conformance suite, which is generated from
what the descriptor above declares: every capability becomes a fit that must
succeed over both semantic sources, and everything withheld becomes a request
that must be refused before a provider is started. It is the same suite
`integrations/nixtla`, `integrations/darts` and `integrations/sktime` run — so
the first `TabularView` consumer is held to exactly the point-in-time contract
the three sequence models are held to, without a line of it being restated here.
