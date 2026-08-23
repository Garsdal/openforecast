# Plans, tasks and outputs

*Generated from the code by `uv run generate-reference`. Do not edit by hand.*

How to fit it, what to predict, and what kind of answer to produce.

## `Accelerator`

*Enumeration — `openforecast.tasks.plan`*

| Member | Value |
| --- | --- |
| `AUTO` | `'auto'` |
| `CPU` | `'cpu'` |
| `GPU` | `'gpu'` |

## `AllOrigins`

*Pydantic model — `openforecast.tasks.origins`*

Every origin the data holds, optionally thinned by ``stride``.

A stride of 12 on hourly vintages trains on two origins a day. It thins the
samples rather than the horizon each sample covers, so the sequences stay
exactly as long — this is a way to spend less compute on highly overlapping
origins, not a way to change what a sample means.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `mode` | `Literal[OriginMode.ALL]` | `OriginMode.ALL` |  |
| `stride` | `int` | `1` |  |

## `AtOrigin`

*Pydantic model — `openforecast.tasks.origins`*

Exactly one named origin.

Matched exactly rather than approximately. Answering for 10:00 when 11:00
was asked for would train the model on a vintage the caller never named.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `mode` | `Literal[OriginMode.AT]` | `OriginMode.AT` |  |
| `origin` | `datetime` | *required* |  |

## `FitPlan`

*Pydantic model — `openforecast.tasks.plan`*

How to fit: which origins, how much context, how reproducibly, on what.

Not *what* to fit — that is the recipe — and not *what to predict*, which is
the forecast task. Keeping the three apart is what lets the same recipe be
fitted at one origin and at every origin without being rewritten.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `origins` | `AllOrigins \| LatestOrigin \| AtOrigin \| OriginsBetween` | `AllOrigins(mode=OriginMode.ALL, stride=1)` |  |
| `window` | `WindowPlan \| None` | `None` |  |
| `seed` | `int \| None` | `None` |  |
| `resources` | `Resources` | `Resources(accelerator=Accelerator.AUTO, devices=None)` |  |
| `search` | `SearchPlan \| None` | `None` |  |

## `ForecastTask`

*Pydantic model — `openforecast.tasks.forecast`*

How far ahead to forecast, from whatever origin is being asked about.

In V1 a horizon is a count of steps of the data's frequency, not a duration:
24 on hourly data is a day, and on daily data it is more than three weeks.
The frequency is declared on the data, so restating it here would let the two
disagree about what "24" meant.

The origin is deliberately absent. At fit time the origins come from the
plan, and at inference time the context *is* one origin — a task that also
named one could contradict the data it was handed.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `horizon` | `int` | *required* |  |

## `LatestOrigin`

*Pydantic model — `openforecast.tasks.origins`*

Only the freshest origin — the one production inference would run at.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `mode` | `Literal[OriginMode.LATEST]` | `OriginMode.LATEST` |  |

## `OriginSelection`

*Type alias — `openforecast.tasks.origins`*

```python
OriginSelection(*args, **kwargs)
```

Runtime representation of an annotated type.

At its core 'Annotated[t, dec1, dec2, ...]' is an alias for the type 't'
with extra annotations. The alias behaves like a normal typing alias.
Instantiating is the same as instantiating the underlying type; binding
it to types is also the same.

The metadata itself is stored in a '__metadata__' attribute as a tuple.

One of: `AllOrigins`, `LatestOrigin`, `AtOrigin`, `OriginsBetween`.

## `OriginsBetween`

*Pydantic model — `openforecast.tasks.origins`*

Every origin in a closed interval, optionally thinned by ``stride``.

The bounds are inclusive, and an interval the data does not cover is an
error rather than an empty fit: it almost always means the caller is
reasoning about a different dataset than the one they passed.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `mode` | `Literal[OriginMode.BETWEEN]` | `OriginMode.BETWEEN` |  |
| `start` | `datetime` | *required* |  |
| `end` | `datetime` | *required* |  |
| `stride` | `int` | `1` |  |

## `OutputKind`

*Enumeration — `openforecast.tasks.forecast`*

| Member | Value |
| --- | --- |
| `POINT` | `'point'` |
| `QUANTILES` | `'quantiles'` |
| `SAMPLES` | `'samples'` |

## `OutputSpec`

*Pydantic model — `openforecast.tasks.forecast`*

What kind of forecast to produce.

The fields are ``levels`` and ``draws`` rather than ``quantiles`` and
``samples`` because the constructors own those two names — and the
constructors are the interface:

```python
of.OutputSpec.quantiles([0.1, 0.5, 0.9])   # spec.levels
of.OutputSpec.samples(100)                 # spec.draws
```

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `kind` | `OutputKind` | `OutputKind.POINT` |  |
| `levels` | `tuple[float, ...]` | `()` |  |
| `draws` | `int \| None` | `None` |  |

## `Resources`

*Pydantic model — `openforecast.tasks.plan`*

What hardware a fit may use.

Deliberately thin. A provider knows how to talk to its own accelerators; the
point of naming them here is that the *request* is OpenForecast's, so two
providers do not need two different spellings of "use the GPU".

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `accelerator` | `Accelerator` | `Accelerator.AUTO` |  |
| `devices` | `int \| None` | `None` |  |

## `WindowPlan`

*Pydantic model — `openforecast.tasks.plan`*

How much history one training sample conditions on.

Steps of the data's frequency, not a duration: 168 on hourly data is a week,
and the same number on daily data is half a year. The frequency lives on the
data, so saying it again here would let the two disagree.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `context` | `int` | *required* |  |
