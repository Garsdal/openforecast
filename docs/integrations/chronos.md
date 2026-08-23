# Chronos

`integrations/chronos`, published as `openforecast-chronos`, providing the
`amazon` namespace. The distribution is named after the library and the models
after who publishes them, because Amazon publishes more than one forecasting
model.

```bash
openforecast providers install amazon
```

## Models

| Reference | Lifecycle | Trains on | Notable |
| --- | --- | --- | --- |
| `amazon/chronos-2` | pretrained | nothing | point forecasts and quantiles, `MissingValueSupport.NATIVE`, panel |

The reference is deliberately the same string as the checkpoint on the Hugging
Face Hub: a reference a user already knows should not have to be translated.

## No fit

<!-- docs-exec: skip — needs `openforecast providers install amazon` -->

```python
import openforecast as of

forecast = of.forecast(
    model="amazon/chronos-2",
    data=dataset.at_origin(origin),
    horizon=72,
)
```

No fit, no artifact, no second call. Fitting one is refused rather than quietly
accepted:

<!-- docs-exec: skip — needs `openforecast providers install amazon` -->

```python
of.fit(model="amazon/chronos-2", data=data)
# ModelDoesNotSupportFit: amazon/chronos-2 cannot be fitted; it is used
# zero-shot, so forecast with the reference directly
```

There is no foundation-model data primitive. A pretrained model consumes the
`ForecastContext` a fitted one does, materialized by the same `ViewPlanner` into
the same `ForecastView`, so point-in-time semantics hold without the integration
knowing they exist: OpenForecast owns the information vintage and Chronos
receives the context that vintage implies.

What is different is only what is absent — no horizon bound, no fitted schema, no
transforms — and because there was never a fit to check the declaration against,
the capabilities are checked against the forecast view instead, at the one moment
the model is handed data.

## On one leaderboard with the rest

<!-- docs-exec: skip — needs several provider environments installed -->

```python
result = of.backtest(
    models=["sklearn/hist-gradient-boosting", "nixtla/nhits", "amazon/chronos-2"],
    data=dataset,
    validation=of.ForecastOriginValidation(origins=of.AllOrigins(stride=24), horizon=72),
    metrics=[of.MAE(), of.Bias()],
)
```

The first two are fitted per fold; the third forecasts as it stands. Its rows
report a null `fit_seconds` and an `origin_fidelity` of `pretrained`, because
there were no training origins at all — which is a different thing from a frozen
artifact that may have seen data postdating the early origins.

## Weights, and what CI runs

A checkpoint is a download, and none of the generated conformance assertions are
about the numbers, so the integration's ordinary test run answers a stand-in
pipeline. A separate CI job runs the real weights, so that a change to the
library's own API — a renamed keyword, a different tensor layout — fails there
rather than in a user's environment.

```bash
cd integrations/chronos
uv run pytest                 # the generated conformance suite, stand-in pipeline
uv run pytest -m weights      # the real checkpoint
```

## Fine-tuning

Chronos-2 can be fine-tuned and this integration does not expose it. Doing so
would mean a training contract, a published artifact and both lifecycles behind
one reference; the lifecycle a model declares is the one it has.
