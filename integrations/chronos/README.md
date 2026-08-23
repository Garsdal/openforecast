# openforecast-chronos

Amazon's Chronos foundation models as an OpenForecast provider, in their own
environment.

```bash
openforecast providers install amazon
```

```python
import openforecast as of

forecast = of.forecast(
    model="amazon/chronos-2",
    data=pit_dataset.at_origin(now),
    horizon=72,
)
```

No `of.fit`. That is the whole of what this integration adds.

## The second model lifecycle

Every integration before this one advertises models that have to be trained:

```text
of.fit(...)  ->  local/de-price@01K...  ->  of.forecast(...)
```

Chronos-2 is pretrained, so its descriptor declares no training contract at all
and the reference itself is what a forecast names. Fitting it is refused with a
structured `MODEL_DOES_NOT_SUPPORT_FIT`, rather than accepted and quietly
ignored:

```python
of.fit(model="amazon/chronos-2", data=data)
# ModelDoesNotSupportFit: amazon/chronos-2 cannot be fitted; it is used
# zero-shot, so forecast with the reference directly
```

## Nothing else changes

There is no `FoundationView`, no `PretrainedFrame` and no second forecasting
call. A zero-shot forecast goes down the path a fitted one does:

```text
ForecastContext
      ↓
ViewPlanner
      ↓
ForecastView          one origin, its context and the horizon it asks about
      ↓
Chronos-2             a provider call, with no fitted state behind it
      ↓
Forecast              the same Arrow table, point or quantile
```

Which means point-in-time semantics are free. OpenForecast owns the information
vintage, so `pit_dataset.at_origin(t)` gives Chronos exactly what was knowable at
`t` — the same information the fitted models were given at that origin — and a
backtest compares the two honestly:

```python
result = of.backtest(
    models=[
        "sklearn/hist-gradient-boosting",
        "nixtla/nhits",
        "amazon/chronos-2",
    ],
    data=pit_dataset,
    validation=of.ForecastOriginValidation(
        origins=of.AllOrigins(stride=24),
        horizon=72,
    ),
    metrics=[of.MAE(), of.Bias()],
)
```

The two fitted candidates are trained per fold; Chronos forecasts as it stands.
Its rows report `fit_seconds` as null and `origin_fidelity` as `pretrained`,
because there was no fit at any origin — which is a different thing from a
frozen artifact that may have seen data postdating the early ones.

## What it maps

| OpenForecast | Chronos-2 |
| --- | --- |
| target history | `target` |
| observed feature | `past_covariates` |
| known feature | `past_covariates` and `future_covariates` |
| static feature | not supported, so not declared |
| `OutputSpec.point()` | the median of `predict_quantiles` |
| `OutputSpec.quantiles([...])` | `predict_quantiles(quantile_levels=[...])` |
| missing value | a `NaN`, passed through unchanged |

A point forecast is the median rather than a mean: Chronos-2 predicts quantiles
and nothing else, and the median is the level it was trained to emit and the one
a point metric reads a distribution at.

## Fine-tuning

Chronos-2 supports it and this integration does not expose it. Fine-tuning would
mean a training contract, a published artifact and both lifecycles for one
reference — worth having, and not what this step is about.

## Names

The provider is `amazon`, because a provider name is the namespace of the models
it advertises and the model is published as `amazon/chronos-2`. The distribution
is `openforecast-chronos`, after the library it wraps. They are the only pair in
the repository that disagree; `INTEGRATION_NAMES` in
`openforecast.runtime.environments` is where that is written down.

## Testing

```bash
uv run pytest             # the boundary, against a stand-in pipeline
uv run pytest -m weights  # the real checkpoint, downloaded from Hugging Face
```

The default run touches no network: the conformance suite is generated from the
descriptor and exercised against a pipeline that answers the shape of the
question. The `weights` marker is what runs Chronos itself.
