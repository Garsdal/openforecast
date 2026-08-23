# Concepts in five minutes

Five ideas carry the whole library. Each has a page of its own under
[Concepts](../concepts/data-model.md); this is the shape of them.

## 1. Two ways to describe data, and no third

`TimeSeriesFrame` is ordinary `instance × event_time × variable` data — what
happened, once per event time. `ForecastDataset` is real forecast vintages:
`instance × origin_time × event_time × variable`, paired with the truth it was
trying to predict. Vintages are deliberately *not* expressible as optional fields
on the first one, because a representation that can hold either quietly holds
neither.

Features carry two orthogonal axes rather than a list of categories: `kind`
(temporal or static) and `availability` (observed only up to the origin, or known
into the future). What that implies — a known feature's later values stay
knowable when you truncate history — falls out of the axes instead of being
special-cased.

## 2. Providers see execution views, never your data

A view is named after the *training unit* it holds rather than after a model
family:

| View | Training unit | Typical models |
| --- | --- | --- |
| `SeriesView` | one complete time series | ARIMA, ETS, Theta |
| `SequenceView` | many context → horizon sequences | NHiTS, TFT, PatchTST |
| `TabularView` | individual supervised target rows | HistGradientBoosting, LightGBM |

`ForecastView` is the inference counterpart of all three: one origin, one
horizon. The `ViewPlanner` is the only thing in OpenForecast that knows which
semantic source it is materializing from, so point-in-time handling lives in one
place instead of being re-derived in every integration.

## 3. A model is a string; a fit is a resource

```text
<namespace>/<name>[@revision]

nixtla/nhits
amazon/chronos-2
local/de-price
local/de-price@01K5Z6QK3M9TQK1W2E3R4T5Y6U
```

The string is a name, not a state — whether anything has been fitted is a
question for the registry. That is what lets a provider model and your own fitted
artifact appear in the same argument position. A fit publishes an immutable
artifact with a manifest describing what was actually materialized, and an
unpinned `local/de-price` is an alias that follows the latest revision.

Two lifecycles exist, and a descriptor says which one a model has: trainable
(`fit` then `forecast`) and pretrained (`forecast` with the reference directly).
Fitting a pretrained model is refused rather than quietly accepted.

## 4. One name per intent

`fit`, never `train`. `forecast`, never `predict` or `infer`. `backtest`, never
`evaluate` or `historical_forecasts`. Each of the four operations — those three
and `eligible_models` — is a method on a client, and the module-level function
beside it is that method on a default client:

<!-- docs-exec: skip — illustrative: the two lines are one call written two ways -->

```python
of.backtest(models, data, validation=..., metrics=..., client=client)
client.backtest(models, data, validation=..., metrics=...)   # the same call
```

Which client runs it is the only difference, and a test asserts the signatures
are identical apart from `client=`.

## 5. Nothing is invented on your behalf

A context length is stated once, as `WindowPlan(context=168)`, and compiled into
whatever the provider calls it; passing the provider's spelling instead is an
error that names the field to use. A missing value in point-in-time data is
information, so it is never silently imputed — a model that cannot consume one
declares that, and the caller writes the transform down as a recipe step, where
the artifact records it. A point forecast is never dressed up as a distribution.
A quantile that was never requested is not interpolated from the ones that were.

Every failure carries a code:

```python
import pandas as pd

import openforecast as of

data = of.TimeSeriesFrame.from_pandas(
    history=pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=48, freq="1h"),
            "zone": "DE",
            "load": range(48),
        }
    ),
    time="timestamp",
    frequency="1h",
    instance_keys=["zone"],
    targets=["load"],
)

try:
    of.forecast(model="builtin/seasonal-naive", data=data, horizon=24)
except of.ModelRequiresFit as error:
    error.code      # 'MODEL_REQUIRES_FIT'
```

so a caller branches on the failure rather than on its prose.
