# Quickstart

The whole workflow, on one page, against the model every install ships with.
Every block below is executed by the test suite in order, so what you read is
what runs.

## Some data

OpenForecast takes a DataFrame at its edge and stores Arrow from then on. You
describe what the columns *mean* — which is the time axis, which identifies a
series, which is a target — and construction validates that description rather
than repairing it.

```python
import pandas as pd

import openforecast as of

hours = pd.date_range("2026-01-01", periods=24 * 30, freq="1h")
history = pd.DataFrame(
    {
        "timestamp": hours,
        "zone": "DE",
        "load": [50 + 10 * (step % 24) / 24 for step in range(len(hours))],
    }
)

data = of.TimeSeriesFrame.from_pandas(
    history=history,
    time="timestamp",
    frequency="1h",
    instance_keys=["zone"],
    targets=["load"],
)

data.schema.is_panel        # True: the rows are keyed by zone
data.schema.is_univariate   # True: one target
```

## What this build can run

```python
client = of.OpenForecast()

client.models.refs()                            # every model, as references
descriptor = client.models.get("builtin/seasonal-naive")

descriptor.lifecycle.requires_fit               # True
descriptor.training.view                        # ViewKind.SERIES
```

A descriptor is complete enough to plan a fit against on its own: which
execution view to materialize, whether several origins may be learned from
jointly, which feature roles the model takes, what it does about missing values.
No provider process is started to answer any of it.

## Fit

```python
model = client.fit(
    "builtin/seasonal-naive",
    data=data,
    params={"season_length": 24},
    name="de-load",
)

str(model.ref).startswith("local/de-load@")     # True
model.manifest.training.samples                 # what was actually fitted
```

A fit produces a resource rather than a variable: an immutable artifact,
addressed by a reference, described by a manifest. `local/de-load` is an alias
that follows the latest fit; `local/de-load@01K...` is one revision forever.

## Forecast

```python
forecast = client.forecast(model, data=data, horizon=24)

forecast.point().to_pandas().head()
```

A forecast is one long Arrow table whatever was asked for — a point forecast, a
set of quantiles or a set of sample paths — and the shapes people want are
projections of it: `forecast.point()`, `forecast.quantile(0.5)`,
`forecast.sample(7)`, `forecast.to_wide()`, `forecast.to_pandas()`.

## Backtest

```python
result = client.backtest(
    ["builtin/seasonal-naive"],
    data=data,
    validation=of.RollingOrigin(horizon=24, windows=3),
    metrics=[of.MAE(), of.Bias()],
)

result.leaderboard("mae").to_pandas()
result.best("mae")
result.metrics_by("horizon_step").to_pandas()   # does it degrade after 12?
```

`of.backtest` is a loop over `fit` and `forecast` and nothing else — there is no
backtesting code in any provider, and no provider knows it is being backtested.
The result keeps every prediction the metrics were computed from, so slicing is a
projection rather than a second run.

## Which models could fit this at all

```python
for entry in of.eligible_models(data, horizon=24, client=client):
    print(entry)
```

Eligibility means exactly one thing: the fit would not be refused. It
materializes the view the model's contract asks for and checks it against what
the model declared, which is the same sequence `fit` runs — and an ineligible
model comes back with the sentence the fit would have failed with.

## What changes when a real model is installed

The reference

<!-- docs-exec: skip — needs `openforecast providers install nixtla` -->

```python
model = client.fit("nixtla/nhits", data=data, horizon=24)
```

and nothing else. Install the environment first
([Installation](installation.md)), then read
[Fitting](../guides/fitting.md) for the forms a fit takes and
[Point-in-time data](../guides/point-in-time.md) for the part that is worth the
whole design.
