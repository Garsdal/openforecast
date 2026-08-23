# Recovering from errors

Every failure, at every boundary, carries the same three fields. Branch on the
code; read the message; use the details.

```python
import pandas as pd

import openforecast as of

hours = pd.date_range("2026-01-01", periods=48, freq="1h")
data = of.TimeSeriesFrame.from_pandas(
    history=pd.DataFrame({"timestamp": hours, "load": [50.0 + step % 24 for step in range(48)]}),
    time="timestamp",
    frequency="1h",
    targets=["load"],
)

try:
    of.forecast(model="builtin/seasonal-naive", data=data, horizon=24)
except of.OpenForecastError as error:
    print(error.code)      # 'MODEL_REQUIRES_FIT'
    print(error.message)   # the sentence a person reads
    print(error.details)   # {'model': 'builtin/seasonal-naive'}
```

The same three come back from `openforecast ... --json` on stderr, from the HTTP
API as `{"error": {...}}`, and from a provider over its own wire protocol. The
codes are declared and frozen by a test, because recovery branches on the code
and a message has to stay free to be rewritten.

## What each code means, and what to do

| Code | What to do |
| --- | --- |
| `MODEL_NOT_FOUND` | `openforecast models list` — the reference is not in this build's catalog |
| `MODEL_REQUIRES_FIT` | `of.fit` it first, then forecast with the `local/...` reference that comes back |
| `MODEL_DOES_NOT_SUPPORT_FIT` | pretrained: drop the fit and forecast with the reference directly |
| `PROVIDER_NOT_INSTALLED` | `openforecast providers install <provider>` |
| `PROVIDER_EXECUTION_FAILED` | the provider process failed; the details carry its exit code and the tail of its log |
| `INVALID_MODEL_REF` | the reference is not `namespace/name[@revision]` |
| `UNSUPPORTED_DATA_SHAPE` | the model declared it cannot take this panel or multivariate target; pick another |
| `UNSUPPORTED_FEATURE` | the model does not accept a feature role present in the data; drop the column or pick another model |
| `UNSUPPORTED_OUTPUT` | the model cannot answer this `OutputSpec`; ask for `point()`, or `quantiles(..., from_samples=N)` from a sampling model |
| `ORIGIN_SCOPE_ERROR` | a series model cannot learn from many origins; select one with `of.LatestOrigin()` or `of.AtOrigin(t)` |
| `INCOMPATIBLE_FORECAST_TASK` | the horizon or the context does not match what the artifact was fitted with |
| `INVALID_DATA` | construction refused the data; the message names the row or column |
| `INCONSISTENT_TRUTH` | the same event time carries contradicting outcomes across vintages; fix the source |
| `INVALID_FREQUENCY` | the timestamps are off the declared grid, or the frequency string is not one |
| `INVALID_RECIPE` | the recipe is not executable — for example a forecast asked for with a recipe rather than a fitted model |
| `INVALID_MODEL_PARAMETERS` | a parameter the provider does not advertise, or one OpenForecast owns (see below) |
| `UNSUPPORTED_PLAN` | the plan asks for something not implemented; the message names the alternative |
| `INVALID_SCHEMA` | a description of the data that cannot be true |
| `INVALID_ARTIFACT` | the artifact on disk does not match its manifest |
| `DUPLICATE_MODEL` | two providers registered the same reference |

## The three refusals worth expecting

**A parameter OpenForecast owns.** A context length, a horizon, a seed, a
frequency and a covariate list are stated once in OpenForecast's own vocabulary
and compiled into each library's spelling. Passing `input_size`,
`input_chunk_length` or `window_length` as a provider parameter is
`INVALID_MODEL_PARAMETERS`, and the message names the field to use instead —
`of.FitPlan(window=of.WindowPlan(context=168))`.

**A missing value the model cannot see.** If the descriptor says
`missing_values` is `REQUIRES_TRANSFORM`, write the transforms down as recipe
steps rather than pre-filling the data:

```python
recipe = of.Pipeline(
    steps=[
        of.MissingIndicator(columns="features"),
        of.Impute(columns="features", method="median"),
        of.Model("builtin/seasonal-naive", params={"season_length": 24}),
    ]
)
```

They are recorded in the artifact, so whoever reads the forecast later can see
what was done. Putting the indicator *after* the imputation is refused, because
it would come out constant.

**A distribution from a model that does not have one.** `samples -> quantiles` is
the only conversion OpenForecast performs, and it is asked for rather than
assumed:

```python
of.OutputSpec.quantiles([0.1, 0.9], from_samples=200)
```

`quantiles -> samples` and `point -> anything` are refused: the paths would have
to be invented.

## Do not retry blindly

A refusal is a statement about the request, not a transient condition. Every
capability is checked before a provider process starts, so retrying the same
request produces the same error. The two exceptions are
`PROVIDER_EXECUTION_FAILED`, which is a real failure in someone else's process,
and a missing environment, which `openforecast providers install` fixes.

`openforecast doctor` answers whether the installation can forecast at all, which
is the thing to check before concluding that a request is wrong.
