# Model lifecycles

## A string is a name, not a state

```text
<namespace>/<name>[@revision]

nixtla/nhits
amazon/chronos-2
local/de-price
local/de-price@01K5Z6QK3M9TQK1W2E3R4T5Y6U
```

Whether anything has been fitted is a question for the registry, not for the
string. That is what lets a provider model and your own fitted artifact appear in
the same argument position — a backtest can compare `nixtla/nhits` with
`local/de-price@01K...` without a mode argument saying which kind each one is.

## Two lifecycles, declared

```text
trainable     fit(...) -> local/de-price@01K... -> forecast(...)
pretrained    forecast(model="amazon/chronos-2", ...)
```

A descriptor says which one a model has, and the mismatches are refused rather
than absorbed:

- forecasting with a reference that names an unfitted trainable model raises
  `ModelRequiresFit`, rather than quietly fitting one on whatever data the call
  was handed
- fitting a pretrained model raises `ModelDoesNotSupportFit`, whose message names
  what to do instead

Zero-shot use is something a model states, not something OpenForecast assumes —
and both lifecycles land on one leaderboard, because a backtest reads the
candidate rather than a mode: the trainable ones are fitted per fold, the
pretrained one forecasts as it stands with a null `fit_seconds` and an
`origin_fidelity` of `pretrained`.

There is no foundation-model data primitive. A pretrained model consumes the same
`ForecastContext` a fitted one does, materialized by the same `ViewPlanner` into
the same `ForecastView`, so point-in-time semantics hold without the integration
knowing they exist. What is different is only what is absent — no horizon bound,
no fitted schema, no transforms — and because there was never a fit to check the
declaration against, the capabilities are checked against the forecast view
instead, at the one moment the model is handed data.

## A fit is a resource

```text
~/.local/share/openforecast/
    models/
        01K5Z6QK3M9TQK1W2E3R4T5Y6U/
            manifest.json     what this is
            recipe.json       what was fitted
            schema.json       the training view's schema
            provider/         opaque
    aliases/
        de-price.json
```

`local/de-price@01K5Z6QK3M9TQK1W2E3R4T5Y6U` is one immutable revision, so
forecasting from it today gives the model it gave a month ago.
`local/de-price` is an alias that follows the latest selected revision, which is
what lets a scheduled forecast job name a model once and pick up retrainings — and
lets a rollback be a pointer move rather than a retrain.

Nothing is published until the fit succeeds. A provider trains into
`.tmp/<artifact-id>` and the directory is renamed into place afterwards, because a
half-written artifact that is nevertheless resolvable would not fail — it would
forecast.

## The manifest is what everything except the provider reads

```json
{
  "training": {
    "view": "sequences",
    "origin_fidelity": "observed",
    "context": 168,
    "horizon": 72,
    "samples": 8832
  }
}
```

It records the artifact id, the source model, the recipe, the provider and its
version, the OpenForecast and protocol versions, the training view and its origin
fidelity, the origin selection, context, horizon and sample count, the schema the
model expects to see again, and any transform that touched the missing values on
the way in.

Every one of those is read off the materialized view rather than reported by the
provider, so a manifest cannot describe a fit that did not happen. `ModelHandle`
is a reference plus that manifest and deliberately nothing else — listing ten
artifacts should not deserialize ten neural networks. An ensemble records one
training record per leaf, because two members may consume different views and
there is no single materialization such an artifact could honestly claim.

## The registry is where a string becomes a state

```python
from openforecast.registry import ModelRegistry

registry = ModelRegistry()

registry.for_fit("builtin/seasonal-naive")     # a descriptor: plan a fit against this
```

<!-- docs-exec: skip — resolves references that exist only in a real store -->

```python
registry.resolve("local/de-price")             # a handle: forecast with this
registry.resolve("nixtla/nhits")               # ModelRequiresFit
```

`ModelRegistry` is machinery rather than user vocabulary: `of.fit` and
`of.forecast` go through it, and a caller who only forecasts never has to name it.
It is documented here because the lifecycle above is easiest to see in the two
methods that answer different questions about the same string.
