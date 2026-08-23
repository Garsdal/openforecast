# sktime

`integrations/sktime`, published as `openforecast-sktime`.

```bash
openforecast providers install sktime
```

## Models

| Reference | Trains on | Notable |
| --- | --- | --- |
| `sktime/theta` | `SeriesView` — one series, one origin | local, fitted per series |
| `sktime/pooled-trees` | `SequenceView` — every origin at once | gradient-boosted trees, reduced recursively and pooled across the panel; the horizon is *not* bound at fit |

`sktime/pooled-trees` takes known and static features but not observed ones, and
declares `MissingValueSupport.REQUIRES_TRANSFORM`.

<!-- docs-exec: skip — needs `openforecast providers install sktime` -->

```python
import openforecast as of

client = of.OpenForecast()

model = client.fit(
    "sktime/pooled-trees",
    data=dataset,
    plan=of.FitPlan(origins=of.AllOrigins(), window=of.WindowPlan(context=168)),
)

forecast = client.forecast(model, data=dataset.at_origin(origin), horizon=48)
```

The horizon is absent from the fit and present at the forecast, which is what
`horizon_bound_at_fit=False` means: the same artifact answers 24 steps and 48.

## What OpenForecast owns and sktime does not see

| You write | Compiled to |
| --- | --- |
| `WindowPlan(context=168)` | `window_length` |
| `ForecastTask(horizon=72)` | a `ForecastingHorizon` |
| a panel `TimeSeriesFrame` or `ForecastDataset` | sktime's hierarchical index and its pooling settings |

`window_length`, `pooling` and `ForecastingHorizon` are therefore legal inside
`integrations/sktime` and nowhere else.

## One deliberate default

`sktime/theta` deseasonalizes multiplicatively, which is undefined on a series
that touches zero — and a load that goes to zero is a perfectly ordinary series.
So the integration sets `deseasonalize=False` by default and a caller who wants
it asks for it by name. The alternative is a model that refuses data for a reason
nobody stated.

## Why this integration exists

sktime is the library whose panel and pooling semantics are *explicit*, and whose
horizon is not bound at fit time. Holding it to the same contracts as Nixtla and
Darts — with no case in the conformance suite written for it — is what showed the
contracts describe forecasting rather than one library's idea of it.
