# Working with event-time data

`TimeSeriesFrame` is ordinary `instance × event_time × variable` data: what
happened, once per event time. Three Arrow tables — history, future and static —
against one schema.

## Building one

```python
import pandas as pd

import openforecast as of

hours = pd.date_range("2026-01-01", periods=24 * 14, freq="1h")
history = pd.DataFrame(
    {
        "timestamp": hours,
        "country": "DE",
        "load": [50.0 + step % 24 for step in range(len(hours))],
        "temperature_actual": [10.0 + step % 12 for step in range(len(hours))],
    }
)

frame = of.TimeSeriesFrame.from_pandas(
    history=history,
    time="timestamp",
    frequency="1h",
    instance_keys=["country"],
    targets=["load"],
    observed_features=["temperature_actual"],
)

frame.schema.is_panel        # True
frame.schema.is_univariate   # True
```

Every column you name is given a role, and the roles are what the rest of the
library plans against:

| Role | Means |
| --- | --- |
| `targets` | what is being forecast |
| `observed_features` | known only up to the origin — a measurement |
| `known_features` | known into the future — a calendar, a weather forecast |
| `static_features` | constant within an instance — a capacity, a market |

`kind` (temporal or static) and `availability` (observed or known) are the two
axes underneath that table, and the interesting categories are derived from them
rather than enumerated — which is why there is no `PANEL_MULTIVARIATE` anywhere
in the API.

## Values known into the future

A known feature may hold values past the last observation, and those go in the
future table:

```python
future = pd.DataFrame(
    {
        "timestamp": pd.date_range(hours[-1] + pd.Timedelta(hours=1), periods=24, freq="1h"),
        "country": "DE",
        "temperature_forecast": [11.0] * 24,
    }
)

frame = of.TimeSeriesFrame.from_pandas(
    history=history.assign(temperature_forecast=12.0),
    future=future,
    time="timestamp",
    frequency="1h",
    instance_keys=["country"],
    targets=["load"],
    observed_features=["temperature_actual"],
    known_features=["temperature_forecast"],
)
```

A target or an observed feature in the future table is an error: it would mean
the data claims to know something it cannot.

## Construction validates and never repairs

Each of these is refused rather than fixed, because each silently changes what
the data means:

- duplicate instance/time rows
- timestamps off the declared frequency grid
- targets or observed features in the future table
- static features that vary within an instance

Gaps and missing values are preserved exactly as they are. A missing observation
is information, not a defect to be filled in.

```python
try:
    of.TimeSeriesFrame.from_pandas(
        history=pd.concat([history, history.head(1)]),
        time="timestamp",
        frequency="1h",
        instance_keys=["country"],
        targets=["load"],
        observed_features=["temperature_actual"],
    )
except of.DataError as error:
    error.code      # 'DATA_ERROR'
```

## Storing and reading it back

```python
frame.write("de-load")
frame = of.TimeSeriesFrame.read("de-load")
```

Arrow in, Arrow out: nothing is re-inferred on the way back, so the schema you
declared is the schema a fit sees weeks later.

## Truncating history

`up_to` is what makes an origin honest — the history as it stood at a moment,
with everything later absent from the object rather than merely unused:

```python
earlier = frame.up_to(hours[24 * 7])

earlier.schema.instance_keys
```

The known features of the truncated rows are kept, because a known feature's
later values are knowable in advance and that is what the role means. Nothing
else moves. This is the operation a backtest over event-time data is built from,
and its point-in-time counterpart is `ForecastDataset.up_to` — see
[Point-in-time data](point-in-time.md).
