# Nixtla

`integrations/nixtla`, published as `openforecast-nixtla`, wrapping
`statsforecast` and `neuralforecast`.

```bash
openforecast providers install nixtla
openforecast providers inspect nixtla
```

## Models

| Reference | Trains on | Notable |
| --- | --- | --- |
| `nixtla/autoarima` | `SeriesView` — one series, one origin | native prediction intervals, so `of.OutputSpec.quantiles([...])` is answered rather than reduced |
| `nixtla/nhits` | `SequenceView` — many origins at once | panel, horizon bound at fit, forecasts an instance it never saw |

`nixtla/autoarima` declares `MissingValueSupport.UNSUPPORTED`, and
`nixtla/nhits` `REQUIRES_TRANSFORM` — so a fit on data holding gaps needs the
transform written down as a recipe step:

<!-- docs-exec: skip — needs `openforecast providers install nixtla` -->

```python
import openforecast as of

client = of.OpenForecast()

model = client.fit(
    of.Pipeline(
        steps=[
            of.MissingIndicator(columns="features"),
            of.Impute(columns="features", method="median"),
            of.Model("nixtla/nhits", params={"max_steps": 500}),
        ]
    ),
    data=dataset,
    plan=of.FitPlan(origins=of.AllOrigins(), window=of.WindowPlan(context=168)),
    horizon=72,
)
```

## What OpenForecast owns and Nixtla does not see

| You write | Compiled to |
| --- | --- |
| `WindowPlan(context=168)` | `input_size` |
| `ForecastTask(horizon=72)` | the model's horizon |
| `FitPlan(seed=42)` | the library's seeding |
| feature roles on the schema | `hist_exog_list`, `futr_exog_list`, `stat_exog_list` |
| instance keys, event time, target | `unique_id`, `ds`, `y` |

Passing one of the right-hand spellings as a provider parameter is an error that
names the field to use instead. Two copies of one number, free to disagree, with
the provider's spelling winning silently, is not a convenience — and none of those
spellings appears anywhere in a descriptor, a manifest, a wire message or a
forecast, which a test asserts over the public surface rather than by review.

## Parameters

`params` reaches the library unchanged, and only names the descriptor advertises:
`max_steps`, `learning_rate`, `batch_size`, `windows_batch_size`,
`num_lr_decays`, `dropout_prob_theta`, `activation`, `pooling_mode`,
`interpolation_mode`, `scaler_type`, `exclude_insample_y` and
`drop_last_loader` for `nixtla/nhits`; the ARIMA search bounds and
`season_length` for `nixtla/autoarima`.

<!-- docs-exec: skip — needs `openforecast providers install nixtla` -->

```python
descriptor = of.models.get("nixtla/nhits")

[parameter.name for parameter in descriptor.parameters]
```

## Conformance

The integration runs the generated conformance suite beside its own library:
every statement in its descriptors becomes fits that must succeed and requests
that must be refused. `nixtla/autoarima` is fitted from an event-time frame and
from real forecast vintages at one selected origin; `nixtla/nhits` from an
event-time frame and from every vintage at once. None of those cases is written
down — declaring the contract is what buys them.
