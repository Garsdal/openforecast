# Pipelines and ensembles

Three things stay separate: what to fit, how to fit it, and what to predict. A
recipe is the first of those, and it composes.

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

## A pipeline

Steps, then exactly one model:

```python
model = client.fit(
    of.Pipeline(
        steps=[
            of.StandardScaler(columns="targets"),
            of.Model("builtin/seasonal-naive", params={"season_length": 24}),
        ]
    ),
    data=data,
    name="de-load-scaled",
)

forecast = client.forecast(model, data=data, horizon=24)
forecast.point().to_pandas().head(3)
```

The scaler's statistics are fitted once and persisted: inference is scaled by
those, never by whatever the forecast context happens to contain, and the forecast
comes back on the scale the caller's data was on.

Transforms are ordinary recipe steps, and the ones that touch missing values are
the reason recipes exist at all:

```python
recipe = of.Pipeline(
    steps=[
        of.MissingIndicator(columns="features"),
        of.Impute(columns="features", method="median"),
        of.Model("builtin/seasonal-naive", params={"season_length": 24}),
    ]
)
```

Nothing is imputed silently. A model that cannot consume a missing value declares
`MissingValueSupport.REQUIRES_TRANSFORM`, and the caller writes the indicator and
the imputation down as steps — recorded in the artifact, visible to whoever reads
the forecast later. Putting the indicator *after* the imputation is refused,
because it would come out constant.

## An ensemble

`of.Ensemble` holds *recipes* rather than models, so an ensemble of pipelines
needs no new vocabulary:

```python
model = client.fit(
    of.Ensemble(
        models=[
            of.Pipeline(
                steps=[
                    of.StandardScaler(columns="targets"),
                    of.Model("builtin/seasonal-naive", params={"season_length": 24}),
                ]
            ),
            of.Model("builtin/seasonal-naive", params={"season_length": 168}),
        ],
        weights=[0.7, 0.3],
    ),
    data=data,
    name="de-load-ensemble",
)

len(model.manifest.training_records)     # one per leaf
```

Each leaf is materialized, transformed and fitted into its own directory inside
the one artifact, and the manifest records one training record per leaf — because
two members may consume different views, and there is no single materialization
such an artifact could honestly claim.

Weights are relative and default to an equal average. They are fixed rather than
learned, because a weight fitted on data is a second model.

## Cross-provider by construction

Members plan independently, which is what makes an ensemble cross-*provider*
rather than cross-*model*:

<!-- docs-exec: skip — needs `openforecast providers install nixtla` and `sklearn` -->

```python
model = client.fit(
    of.Ensemble(
        models=[of.Model("nixtla/nhits"), of.Model("sklearn/hist-gradient-boosting")],
        weights=[0.7, 0.3],
    ),
    data=dataset,
    horizon=72,
)
```

`nixtla/nhits` declares a sequence contract and is handed a `SequenceView`;
`sklearn/hist-gradient-boosting` declares a tabular one and is handed a
`TabularView` of the same data. Neither provider learns that an ensemble exists —
both answer a `Forecast`, and that is what is combined.

Every member is checked before any of them runs, at fit and at forecast alike: a
member whose contract the data cannot satisfy — a single-origin model asked to
learn from every vintage — rejects the whole ensemble rather than being quietly
trained on something else.

Quantiles are averaged level by level, which is *quantile averaging* and not the
quantile of the mixture of the members' distributions. Sample paths are not
combined at all, since draw *i* of one member has nothing to do with draw *i* of
another.

## Reductions

Generating a tabular problem out of an ordinary event-time series is a recipe
node too:

```python
recipe = of.Reduction(
    estimator="sklearn/hist-gradient-boosting",
    strategy="direct",
    lags=[1, 24, 168],
)
```

A `ForecastDataset` needs none of that: it already carries the features a
supervised row is built from, so
`client.fit("sklearn/hist-gradient-boosting", data=dataset, horizon=72)` is the
whole request.

## Recipes are data

```python
payload = recipe.model_dump(mode="json")
of.parse_recipe(payload) == recipe      # True
```

Tagged by `kind`, serializable, and the same JSON that reaches an artifact
manifest, a provider subprocess request and an HTTP body. No part of it is
provider-specific.
