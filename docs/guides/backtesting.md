# Backtesting

The same models, over the same origins, scored the same way — and it is a loop
over `fit` and `forecast` rather than any provider's backtester.

```python
import pandas as pd

import openforecast as of

hours = pd.date_range("2026-01-01", periods=24 * 21, freq="1h")
data = of.TimeSeriesFrame.from_pandas(
    history=pd.DataFrame(
        {
            "timestamp": hours,
            "zone": "DE",
            "load": [50.0 + step % 24 for step in range(len(hours))],
        }
    ),
    time="timestamp",
    frequency="1h",
    instance_keys=["zone"],
    targets=["load"],
)

client = of.OpenForecast()
```

## Event-time data: rolling origins

```python
result = client.backtest(
    ["builtin/seasonal-naive"],
    data=data,
    validation=of.RollingOrigin(horizon=24, windows=5),
    metrics=[of.MAE(), of.Bias()],
)

result.leaderboard("mae").to_pandas()
result.best("mae")
```

Each window truncates the history with `up_to` and fits on what was left, so an
origin never sees a value published after it. The models are a list because
comparing them is the point; a candidate may be a reference, a recipe, or an
`of.Candidate` naming one explicitly.

## Point-in-time data: real vintages

The same call with the validation that fits it, and this is the part worth the
whole design:

```python
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

pit = client.backtest(
    ["builtin/seasonal-naive"],
    data=dataset,
    validation=of.ForecastOriginValidation(
        origins=of.OriginsBetween(origins[48], origins[-25], stride=24),
        horizon=24,
    ),
    metrics=[of.MAE()],
    plan=of.FitPlan(origins=of.LatestOrigin()),
)

pit.leaderboard("mae").to_pandas()
```

At each origin the features come from *that vintage*, the truth comes from the
truth frame, and later vintages are not merely unused — they are absent from the
object the model is handed. The `plan` is there because a series model holds one
forecast origin: a sequence or tabular model takes `of.AllOrigins()` and learns
from every vintage up to the fold's origin instead.

## What comes back

Two long Arrow tables. The metrics, three of whose columns are not measurements:

```python
pit.metrics.column_names
```

```text
model  fold  origin  metric  value  pairs  fit_seconds  forecast_seconds
       origin_fidelity  provider  artifact
```

- `origin_fidelity` is `simulated` or `observed`, read off the artifact the fold
  published rather than declared by the backtest — which makes "simulated
  historical availability versus true point-in-time availability" a comparison
  you can run rather than a caveat you have to remember.
- `artifact` is the pinned revision the numbers came from, so a backtest's winner
  is a reference you can forecast with.
- `pairs` says how many outcomes a value was computed over, so a fold scored on a
  third of its horizon is visible in the table rather than only in the metric.

And the predictions those numbers were computed from:

```python
pit.predictions.column_names
```

```text
model  fold  instance keys...  origin_time  event_time  horizon_step
       target  kind  quantile  sample  prediction  actual
```

Kept rather than dropped, because the metrics are derivable from these and not the
reverse — so the question everyone asks after a backtest is a projection rather
than a second run:

```python
pit.metrics_by("horizon_step").to_pandas().head()
pit.metrics_by(["horizon_step", "zone"]).to_pandas().head()
```

The group keys are columns of the prediction table, including your own instance
keys, and an unknown one is an error naming the columns that exist. That is the
whole slicing story; there is no DSL.

## Evaluating a model that is already fitted

A candidate that is already a pinned revision is evaluated rather than refitted,
which is how you ask whether the model in production has drifted:

<!-- docs-exec: skip — names a revision from another example's store -->

```python
result = client.backtest(
    ["local/de-price@01K5Z6QK3M9TQK1W2E3R4T5Y6U", "nixtla/nhits"],
    data=dataset,
    validation=of.ForecastOriginValidation(origins=of.AllOrigins(stride=24), horizon=72),
    metrics=[of.MAE()],
)
```

Read from the candidate rather than from a mode argument: a pinned revision names
one immutable fit, so it forecasts at every origin with `fit_seconds` null, while
a bare reference or a recipe is fitted per fold. One caveat comes with mixing them
in one table — a frozen artifact was fitted on data that may postdate the early
origins, so its numbers are optimistic beside a candidate fitted per fold. That is
reported rather than refused, the same way `origin_fidelity` is.

## Which models could fit this at all

```python
for entry in client.eligible_models(dataset, horizon=24, plan=of.FitPlan(origins=of.LatestOrigin())):
    print(entry)
```

Eligibility means exactly one thing — the fit would not be refused — so it
materializes the view the model's contract asks for and checks it against the
capabilities the model declared, which is the same sequence a fit runs. No
provider is started, and an ineligible model comes back with the sentence the fit
would have failed with.

Scoring a distribution is the same call with the output it needs; see
[Probabilistic forecasts](probabilistic.md).
