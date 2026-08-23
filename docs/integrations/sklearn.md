# scikit-learn

`integrations/sklearn`, published as `openforecast-sklearn`. Not a forecasting
framework at all, which is the point: it consumes the third execution view.

```bash
openforecast providers install sklearn
```

## Models

| Reference | Trains on | Notable |
| --- | --- | --- |
| `sklearn/hist-gradient-boosting` | `TabularView` — individual supervised rows | reads `NaN` as a branch rather than as an error, so `MissingValueSupport.NATIVE` |

The estimator was chosen for that last capability: a point-in-time design matrix
is full of missing values, because a feature that had not been published at an
origin is missing *as information*. An estimator that treats it as a branch needs
no imputation written down at all.

## What crosses the boundary

`estimator.fit(X, y)` at fit and `estimator.predict(X)` at inference, with the
reduction — origin, lead, vintage, truth alignment — already done on
OpenForecast's side:

```text
X      the features knowable at the origin
y      what that event time turned out to be
keys   row_id, instance keys, origin_time, event_time, horizon_step
```

The keys are deliberately not in `X`. That is what stops a timestamp or a zone
being handed to the estimator as a feature by accident, and it is what makes the
fitted artifact able to forecast an instance it never saw.

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

## Fitting one

A `ForecastDataset` already carries the features a supervised row is built from,
so there is nothing else to say:

<!-- docs-exec: skip — needs `openforecast providers install sklearn` -->

```python
import openforecast as of

client = of.OpenForecast()

model = client.fit("sklearn/hist-gradient-boosting", data=dataset, horizon=72)
```

For an ordinary event-time series, the lags and the strategy are a recipe node
instead:

<!-- docs-exec: skip — needs `openforecast providers install sklearn` -->

```python
model = client.fit(
    of.Reduction(
        estimator="sklearn/hist-gradient-boosting",
        strategy="direct",
        lags=[1, 24, 168],
    ),
    data=timeseries,
    horizon=72,
)
```

## Terminology

This integration adds nothing to the list of forbidden provider spellings, which
is itself the point: `X` and `y` are what a `TabularView` already calls its own
tables. `params` names the estimator's own knobs — `max_iter`, `learning_rate`,
`max_leaf_nodes`, `min_samples_leaf`, `l2_regularization` — and nothing that
OpenForecast owns.
