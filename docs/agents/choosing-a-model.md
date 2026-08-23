# Choosing a model

Do not try models until one works. A descriptor is complete enough to decide
against on its own, and `of.eligible_models` decides against it for you.

## Ask what would be refused

```python
import openforecast as of
import pandas as pd

hours = pd.date_range("2026-01-01", periods=24 * 20, freq="1h")
data = of.TimeSeriesFrame.from_pandas(
    history=pd.DataFrame(
        {"timestamp": hours, "zone": "DE", "load": [50.0 + step % 24 for step in range(len(hours))]}
    ),
    time="timestamp",
    frequency="1h",
    instance_keys=["zone"],
    targets=["load"],
)

for entry in of.eligible_models(data, horizon=24):
    print(entry.model, entry.eligible, entry.reason)
```

Eligibility means exactly one thing: **the fit would not be refused**. It
materializes the execution view the model's contract asks for and checks it
against the capabilities the model declared — the same sequence `of.fit` runs —
so an ineligible model comes back with the sentence the fit would have failed
with. No provider process is started.

Narrow the question with `models=[...]` when you already have candidates, and
pass the `plan` you intend to fit with, since a plan is part of what makes a
model eligible.

## Read the descriptor

```python
descriptor = of.models.get("builtin/seasonal-naive")

descriptor.lifecycle.requires_fit         # True -> fit it; False -> forecast with the ref
descriptor.training.view                  # which execution view it consumes
descriptor.training.origin_scope          # may it learn from several origins at once?
descriptor.training.horizon_bound_at_fit  # must `horizon` be given to `fit`?
descriptor.capabilities.instances.panel   # many series in one model?
descriptor.capabilities.outputs.quantiles # can it answer a distribution?
descriptor.capabilities.missing_values    # UNSUPPORTED, REQUIRES_TRANSFORM, NATIVE
descriptor.parameters_schema              # the provider's own parameters, as JSON Schema
```

The whole catalog, as data:

```bash
openforecast models list --json
openforecast models get nixtla/nhits --json
```

## What each declaration decides

| Declaration | What it means for your request |
| --- | --- |
| `lifecycle.requires_fit` | `False` — forecast with the reference; fitting it is refused |
| `training` is `None` | pretrained only; there is no fit to plan |
| `training.view` | `series` holds one origin, `sequences` and `tabular` hold many |
| `training.origin_scope` | `SINGLE` — asking it to learn from every vintage raises `ORIGIN_SCOPE_ERROR` |
| `training.horizon_bound_at_fit` | `True` — pass `horizon=` to `fit`, and forecast that far and no further |
| `training.context_required` | `True` — pass `of.FitPlan(window=of.WindowPlan(context=168))` |
| `capabilities.instances.panel` | `False` — one series per fit; a panel is refused |
| `capabilities.features.known` | `False` — a known feature in the data is refused rather than dropped |
| `capabilities.outputs.quantiles` | `False` — `of.OutputSpec.quantiles([...])` is refused |
| `capabilities.missing_values` | `REQUIRES_TRANSFORM` — write `MissingIndicator` and `Impute` into the recipe |

Defaults are the conservative ones: a descriptor that declares nothing describes
a single-series, univariate, point-forecast model that cannot see a missing
value. A capability is something a provider states.

## If nothing is eligible

A build with no provider environments installed has one model,
`builtin/seasonal-naive`, which is a real model with a real contract and not a
placeholder. Reaching a neural, tree-based or zero-shot model means installing
its environment:

```bash
openforecast providers list
openforecast providers install nixtla
```

Each integration gets its own virtual environment and is reached over a
subprocess protocol, so their dependencies never have to agree.
[Installation](../getting-started/installation.md) lists what each one ships.

## Then swap the string

The reason to choose from a catalog rather than to commit to a library is that
the choice is one string:

<!-- docs-exec: skip — needs `openforecast providers install nixtla` and sklearn -->

```python
model = of.fit("nixtla/nhits", data=data, horizon=24)
model = of.fit("sklearn/hist-gradient-boosting", data=data, horizon=24)
```

Nothing else about the surrounding code changes — not the data, not the plan, not
how the forecast is read. See [Fitting](../guides/fitting.md) for the forms a fit
takes.
