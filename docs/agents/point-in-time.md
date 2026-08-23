# Point-in-time rules

This is the one part of the library where a request that looks right can be
wrong, so it is stated as rules. The concepts behind them are in
[Point-in-time semantics](../concepts/point-in-time.md); what follows is what to
do.

## Which representation

| You have | Use |
| --- | --- |
| one row per (series, timestamp) — what happened | `of.TimeSeriesFrame` |
| one row per (series, **origin**, timestamp) — what was known when | `of.ForecastDataset` |

If your table has two time columns — `ref_time` and `target_time`,
`forecast_time` and `valid_time`, `issued_at` and `for` — it is the second one.
Do not collapse it to the freshest vintage first: that throws away the only
information the model would have actually had.

## Construct it

```python
from datetime import datetime, timedelta

import pyarrow as pa

import openforecast as of

origins = [datetime(2026, 1, 1) + timedelta(hours=step) for step in range(72)]
rows = [(o, o + timedelta(hours=lead), lead) for o in origins for lead in range(1, 25)]

dataset = of.ForecastDataset.from_arrow(
    pa.table(
        {
            "ref_time": pa.array([o for o, _, _ in rows], type=pa.timestamp("us")),
            "target_time": pa.array([e for _, e, _ in rows], type=pa.timestamp("us")),
            "zone": pa.array(["DE"] * len(rows)),
            "wind_fc": pa.array([12.0 - 0.1 * lead for _, _, lead in rows]),
            "price": pa.array([80.0 + e.hour for _, e, _ in rows]),
        }
    ),
    origin_time="ref_time",
    event_time="target_time",
    instance_keys=["zone"],
    targets=["price"],
    known_features=["wind_fc"],
    event_frequency="1h",
    origin_frequency="1h",
)

len(dataset.origins)                    # 72 vintages
dataset.truth.history.num_rows          # one outcome per event time, not per row
```

`from_pandas` takes the same arguments and a DataFrame. The constructor splits
the table into the two things it holds: every vintage exactly as issued, and the
realized outcome once per event time.

## The rules

1. **The same event time appears once per origin.** Three origins forecasting
   noon are three rows with three different feature values, and all three are
   true. Nothing deduplicates on the event time.
2. **Lead time is derived, never stored.** Ask for it with
   `dataset.information.with_lead_time(unit="hour")`. A third column would be
   free to disagree with the two axes that already determine it.
3. **A missing value is information.** It means the feature had not been
   published at that origin. It is not filled in.
4. **Disagreeing labels are an error.** If the repeated outcome column contradicts
   itself between vintages of the same event time, that is
   `INCONSISTENT_TRUTH` — OpenForecast does not pick one. A label merely *absent*
   from an earlier vintage is not a disagreement.
5. **One origin is one inference request.** `dataset.at_origin(t)` is a
   `ForecastContext`: only that vintage contributes, and a value revised later
   cannot appear in it.
6. **Truncate with `up_to`, never by hand.** `dataset.up_to(t)` keeps the
   vintages issued by then; `frame.up_to(t)` truncates an event-time history.
   A fold built from either has nothing later in it to leak.

## Fit and forecast

```python
model = of.fit(
    "builtin/seasonal-naive",
    data=dataset,
    plan=of.FitPlan(origins=of.LatestOrigin()),
    params={"season_length": 24},
    name="de-price",
)

model.manifest.training.origin_fidelity        # 'observed': the origins were real

forecast = of.forecast(model, data=dataset.at_origin(dataset.origins[-1]), horizon=24)
```

`origin_fidelity` is the field to check when comparing results. `observed` means
the origins were real forecast vintages; `simulated` means they were cut out of
one freshest series, which tells the model the past was cleaner than it was.
It is read off the materialized view rather than declared, so it cannot describe
a fit that did not happen.

## Origin selection means the same thing on both sources

```python
of.AllOrigins(stride=1)
of.LatestOrigin()
of.AtOrigin("2026-01-03T00:00:00Z")
of.OriginsBetween("2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z", stride=12)
```

On a `TimeSeriesFrame` these pick among origins that can be simulated; on a
`ForecastDataset`, among the vintages that exist. So the same `of.FitPlan` works
on both and only the recorded fidelity differs.

A model whose `training.origin_scope` is `SINGLE` — a series model — cannot learn
from every vintage at once, and asking it to raises `ORIGIN_SCOPE_ERROR`. Either
select one origin, as above, or choose a model whose view holds many.

## Backtesting on vintages

<!-- docs-exec: skip — needs `openforecast providers install nixtla` -->

```python
result = of.backtest(
    models=["nixtla/nhits", "amazon/chronos-2"],
    data=dataset,
    validation=of.ForecastOriginValidation(
        origins=of.OriginsBetween(start, end, stride=24),
        horizon=24,
    ),
    metrics=[of.MAE()],
)
```

At each origin the features come from *that vintage* and the truth comes from the
truth frame. Later vintages are not merely unused — they are absent from the
object the model is handed. See [Backtesting](../guides/backtesting.md) for what
the result holds.
