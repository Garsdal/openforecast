# Forecasting

```python
import pandas as pd

import openforecast as of

hours = pd.date_range("2026-01-01", periods=24 * 14, freq="1h")
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
model = client.fit("builtin/seasonal-naive", data=data, params={"season_length": 24}, name="de-load")
```

## Naming the model

Three things are the same argument: the handle a fit returned, a pinned revision,
and the alias that follows the latest one.

```python
forecast = client.forecast(model, data=data, horizon=24)
same = client.forecast("local/de-load", data=data, horizon=24)
```

A reference naming a model that was never fitted raises `ModelRequiresFit` rather
than quietly fitting one on whatever data the call was handed — which would
return a number that looks like a forecast from a model nobody trained. A model
that declares `requires_fit=False` resolves to its descriptor instead, and
forecasts zero-shot.

## What a forecast is

One long Arrow table, whatever was asked for:

```text
zone event_time target kind     quantile sample value

DE   12:00      load   point    null     null   80
DE   12:00      load   quantile 0.1      null   65
```

A wide forecast changes shape with the request — one column per target, or per
target and quantile, or per sample path — and cannot be read by one reader. So the
long table is what a forecast *is*, and the shapes people want are projections
of it:

```python
forecast.table              # the long forecast, in canonical column order
forecast.point()            # the point rows, without the columns describing none
forecast.to_wide()          # zone, event_time, load
forecast.to_pandas()        # the long forecast as a DataFrame
```

<!-- docs-exec: skip — `builtin/seasonal-naive` declares point output only, so there is no level to read -->

```python
forecast.quantile(0.5)      # one level, in the same shape
forecast.sample(7)          # one draw, in the same shape
```

`quantile` refuses a level that was never asked for rather than interpolating
between the ones that were: a 0.5 derived from a 0.1 and a 0.9 is a different
number from the one the model would have produced. See
[Probabilistic forecasts](probabilistic.md) for asking for the levels in the
first place.

## Forecasting at a chosen origin

The forecast starts after what the data it is handed knows, so an earlier origin
is an earlier view of the data rather than an argument:

```python
context = data.up_to(hours[24 * 10])
earlier_model = client.fit("builtin/seasonal-naive", data=context, params={"season_length": 24})
earlier = client.forecast(earlier_model, data=context, horizon=24)

earlier.point().to_pandas().head(3)
```

`builtin/seasonal-naive` remembers one series and continues it, so it is fitted on
the truncated frame here. A global model fitted once forecasts any origin its
schema still matches, and the truncation is the only thing that changes.

For a `ForecastDataset` that is `dataset.at_origin(t)` — one vintage, and only
that vintage, as [Point-in-time data](point-in-time.md) describes.

## Where it runs

Nothing above changes when the model runs somewhere else. A client's transport is
the only thing that decides that:

<!-- docs-exec: skip — needs a service listening on 8321 -->

```python
remote = of.OpenForecast(transport=of.HttpTransport("http://localhost:8321"))
forecast = remote.forecast("local/de-load", data=data, horizon=24)
```

Control travels as JSON and bulk data as Arrow IPC, and the Arrow tables that come
back are equal to the local ones — which is written as a comparison in
`tests/e2e/test_remote_transport.py` rather than promised here. Start a service
with `openforecast serve` from an install that has the `server` extra.
