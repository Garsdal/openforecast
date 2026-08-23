# Working with point-in-time data

A `ForecastDataset` is what production pipelines already emit: every forecast
vintage as it was issued, paired with the outcome it was trying to predict.

```text
zone origin_time event_time wind_fc load_fc
DE   08:00       12:00      10.1    54.2
DE   09:00       12:00      11.7    54.8
DE   10:00       12:00      12.4    55.1
```

Three rows, not one. The same event time appears once per origin and the values
differ between them, which is the whole point: nothing collapses, deduplicates or
forward-fills a vintage.

## Building one from a `(ref_time, target_time)` table

The tables pipelines emit carry both axes at once, with the outcome repeated on
every vintage of the same event time, so there is a constructor that splits them
apart:

```python
import pandas as pd

import openforecast as of

origins = pd.date_range("2026-01-01", periods=120, freq="1h")
rows = [
    {
        "ref_time": origin,
        "target_time": origin + pd.Timedelta(hours=lead),
        "zone": "DE",
        "wind_fc": 10.0 + lead,
        "price": 80.0 + (origin.hour + lead) % 24,
    }
    for origin in origins
    for lead in range(1, 25)
]

dataset = of.ForecastDataset.from_pandas(
    pd.DataFrame(rows),
    origin_time="ref_time",
    event_time="target_time",
    instance_keys=["zone"],
    targets=["price"],
    known_features=["wind_fc"],
    event_frequency="1h",
    origin_frequency="1h",
)

dataset.origins[:2]     # the vintages that exist
```

What comes out is a pair:

| Part | Type | Holds |
| --- | --- | --- |
| `dataset.information` | `PointInTimeFrame` | every vintage, exactly as it was issued |
| `dataset.truth` | `TimeSeriesFrame` | the realized outcome, once per event time |

If the repeated labels disagree between vintages, that is a contradiction in the
source data and raises `InconsistentTruthError` — OpenForecast does not pick one.
A label that is merely *missing* in an earlier vintage is not a disagreement: it
is a label that had not been published yet.

## Lead time is derived, not stored

```python
dataset.information.with_lead_time(unit="hour").table.column_names
```

Storing it would be a third axis free to disagree with the two that already
determine it.

## One inference origin

`ForecastContext` is exactly one origin — the shape production inference always
has:

```python
context = dataset.at_origin(origins[-1])
```

Only that vintage contributes. A feature value revised at 12:00 cannot appear in
the context of the 11:00 origin, and an observed feature holding a value for an
event time after its own origin is rejected outright. Contexts can also be built
straight from live data with `of.ForecastContext.from_pandas(...)`.

## Fitting on vintages

The same two calls as event-time data. What differs is what the artifact records:

```python
client = of.OpenForecast()

model = client.fit(
    "builtin/seasonal-naive",
    data=dataset,
    plan=of.FitPlan(origins=of.LatestOrigin()),
    params={"season_length": 24},
)

model.manifest.training.origin_fidelity      # OriginFidelity.OBSERVED
```

`origin_fidelity` is `observed` here and `simulated` for windows cut out of a
single freshest series. A model trained on the second was told the past was
cleaner than it was, and the artifact has to be able to say so.

`builtin/seasonal-naive` trains on a `SeriesView`, which holds exactly one
forecast origin, so the plan names one. Asking it to learn from *every* vintage
raises `OriginScopeError` — from the planner, which is the only thing that knows
the source type:

```python
try:
    client.fit("builtin/seasonal-naive", data=dataset, plan=of.FitPlan(origins=of.AllOrigins()))
except of.OriginScopeError as error:
    error.code      # 'ORIGIN_SCOPE_ERROR'
```

A sequence or tabular model — `nixtla/nhits`, `darts/tide`,
`sklearn/hist-gradient-boosting` — takes `of.AllOrigins()` and learns one
training sample per historical origin, or one supervised row per origin and lead.

## Forecasting at an origin

```python
forecast = client.forecast(model, data=context, horizon=24)

forecast.point().to_pandas().head(3)
```

The context was materialized from one vintage, so the features the model sees are
the ones that existed then — enforced by the object it is handed rather than by
the code that built it.

## Truncating

```python
dataset.up_to(origins[60])      # the vintages issued by then
```

The counterpart of `TimeSeriesFrame.up_to`, and the operation a point-in-time
backtest is built from: a fold holds the result of one of these, so there is
nothing for a bug in the backtest loop to reach for. See
[Backtesting](backtesting.md).
