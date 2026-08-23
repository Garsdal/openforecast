# Fitting

A fit takes what to fit, the data, and — when the model needs more than the
defaults — how to fit it. It returns a handle to an immutable artifact.

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

## The short form

```python
model = client.fit("builtin/seasonal-naive", data=data, params={"season_length": 24})

model.ref                    # local/...@01K...
model.manifest.provider      # 'builtin'
```

`params` are the provider's own knobs, and only the ones its descriptor
advertises. A parameter naming something OpenForecast owns — a context length, a
horizon, a seed, a frequency, a covariate list — is refused with the field to use
instead. Two copies of one number, free to disagree, with the provider's spelling
winning silently, is not a convenience.

## Naming a fit

```python
model = client.fit(
    "builtin/seasonal-naive",
    data=data,
    params={"season_length": 24},
    name="de-load",
)

str(model.ref).startswith("local/de-load@")     # True
```

`local/de-load` is an alias that follows the latest fit; the pinned
`local/de-load@01K...` is one revision forever. That is what lets a scheduled job
name a model once and pick up retrainings — and lets a rollback be a pointer move
rather than a retrain.

## How to fit it: `FitPlan`

```python
plan = of.FitPlan(
    origins=of.AllOrigins(stride=1),
    window=of.WindowPlan(context=168),
    seed=42,
)
```

| Field | Says |
| --- | --- |
| `origins` | which forecast origins to learn from |
| `window` | how much history one training sample conditions on |
| `seed` | how reproducibly |
| `resources` | on what — CPU, GPU, how many devices |

The context length is stated once, in steps of the data's frequency, and compiled
into `input_size` for Nixtla, `input_chunk_length` for Darts or `window_length`
for sktime. The four origin selections — `of.AllOrigins()`, `of.LatestOrigin()`,
`of.AtOrigin(t)`, `of.OriginsBetween(start, end, stride=12)` — mean the same
thing on both semantic sources: on a `TimeSeriesFrame` they pick among the origins
that can be simulated, on a `ForecastDataset` among the vintages that exist.

## What to fit: recipes

A recipe is what you write when a bare reference is not enough. It is a
serializable AST, and it goes to the same `fit` call:

```python
recipe = of.Pipeline(
    steps=[
        of.StandardScaler(columns="targets"),
        of.Model("builtin/seasonal-naive", params={"season_length": 24}),
    ]
)

model = client.fit(recipe, data=data, name="de-load-scaled")
```

Pipelines and ensembles are executed by OpenForecast rather than by any provider,
and every transform is recorded in the manifest — visible to whoever reads the
forecast later. See [Pipelines and ensembles](ensembles.md) for combining models,
and `of.parse_recipe` for reading one back from JSON.

The missing-value steps are part of the same vocabulary and are recorded in the
artifact, but they are not executable yet — a pipeline that silently skipped one
would be indistinguishable from one that ran it, so asking for it raises
`UnsupportedPlanError` rather than fitting without it:

<!-- docs-exec: skip — `MissingIndicator` is declared and recorded, and not executable yet -->

```python
model = client.fit(
    of.Pipeline(
        steps=[
            of.MissingIndicator(columns="features"),
            of.Impute(columns="features", method="median"),
            of.Model("builtin/seasonal-naive", params={"season_length": 24}),
        ]
    ),
    data=data,
)
```

Putting the indicator *after* the imputation is refused whichever way that lands,
because it would come out constant.

## What a fit actually does

Five steps, none of which branch on who provides the model:

1. the recipe is normalized
2. every reference is resolved to a descriptor
3. the `ViewPlanner` materializes the view that descriptor's contract names
4. the view is checked against what the model declared it can consume
5. a provider is started — into a staging directory that is published if, and
   only if, the fit succeeded

Nothing is published until the fit succeeds, because a half-written artifact that
is nevertheless resolvable would not fail: it would forecast.

## What comes back

```python
plain = client.fit("builtin/seasonal-naive", data=data, params={"season_length": 24})
manifest = plain.manifest

manifest.training.view              # which execution view was materialized
manifest.training.origin_fidelity   # observed vintages, or simulated windows
manifest.training.samples           # how many training units there were
manifest.openforecast_version
```

Every one of those is read off the materialized view rather than reported by the
provider, so a manifest cannot describe a fit that did not happen. `ModelHandle`
is a reference plus that manifest and deliberately nothing else — listing ten
artifacts should not deserialize ten neural networks. A pipeline or an ensemble
records one training record per leaf instead, in `manifest.training_records`,
because two members may consume different views and there is no single
materialization such an artifact could honestly claim.

## Fitting a pretrained model is refused

```python
try:
    client.fit("amazon/chronos-2", data=data)
except of.ModelError as error:
    error.code      # 'UNKNOWN_MODEL_ERROR' without the environment installed,
                    # 'MODEL_DOES_NOT_SUPPORT_FIT' with it
```

A zero-shot model has no training contract to satisfy, so it forecasts from its
own reference directly — see [Model lifecycles](../concepts/model-lifecycle.md).
