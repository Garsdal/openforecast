# Darts

`integrations/darts`, published as `openforecast-darts`.

```bash
openforecast providers install darts
```

## Models

| Reference | Trains on | Notable |
| --- | --- | --- |
| `darts/theta` | `SeriesView` — one series, one origin | local, fitted per series |
| `darts/tide` | `SequenceView` | panel, horizon bound at fit, unseen instances forecastable |
| `darts/nhits` | `SequenceView` | the same contract, a different architecture |

The global models declare `MissingValueSupport.REQUIRES_TRANSFORM`, so gaps are
handled by recipe steps the artifact records rather than inside the provider.

<!-- docs-exec: skip — needs `openforecast providers install darts` -->

```python
import openforecast as of

client = of.OpenForecast()

model = client.fit(
    "darts/tide",
    data=dataset,
    plan=of.FitPlan(origins=of.AllOrigins(), window=of.WindowPlan(context=168)),
    horizon=72,
)
```

## What OpenForecast owns and Darts does not see

| You write | Compiled to |
| --- | --- |
| `WindowPlan(context=168)` | `input_chunk_length` |
| `ForecastTask(horizon=72)` | `output_chunk_length` |
| `observed_features` on the schema | `past_covariates` |
| `known_features` on the schema | `future_covariates` |

Those four spellings name concepts OpenForecast already owns, so they are legal
inside `integrations/darts` and nowhere else. A `params={"input_chunk_length":
168}` is refused with the field to use instead.

## Why this integration exists

Nixtla and Darts spell the same ideas differently and want incompatible versions
of `torch`. Two facts follow, and they are the two the provider boundary was built
for:

- switching a point-in-time fit from `nixtla/nhits` to `darts/tide` changes the
  model reference and nothing else — the data, the plan, the context length, the
  origin selection and the resulting `Forecast` are the same objects
- the two libraries never meet. Each has its own environment, its own lockfile
  and its own subprocess; an OpenForecast install has neither of them in its
  dependency tree

`tests/e2e/test_v1_experience.py` is where both are true at once: one install that
has never heard of either library, reaching both over the protocol, ensembling a
Nixtla model with a Darts one, and checking that nothing either of them calls its
own reaches a descriptor, a manifest or a forecast.
