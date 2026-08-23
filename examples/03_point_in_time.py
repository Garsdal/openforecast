"""Real forecast vintages: what was actually known at each origin, kept apart.

    uv run examples/03_point_in_time.py

This is the part worth the whole design. A `ForecastDataset` is the table a
production pipeline already emits — one row per (origin, event time) — and
nothing in OpenForecast collapses, deduplicates or forward-fills a vintage.
"""

from datetime import datetime, timedelta

import pyarrow as pa

import openforecast as of

# -- data -------------------------------------------------------------------
# 120 hourly origins, each carrying a 24-hour-ahead vintage. The *known feature*
# varies by vintage, which is what makes this point-in-time data rather than a
# reshaped event-time table: the 08:00 view of noon and the 10:00 view of noon
# are different numbers, and both are true. The target depends only on the event
# time, because there is one outcome however many times it was forecast.

origins = [datetime(2026, 1, 1) + timedelta(hours=step) for step in range(120)]
rows = [
    (origin, origin + timedelta(hours=lead), lead) for origin in origins for lead in range(1, 25)
]

dataset = of.ForecastDataset.from_arrow(
    pa.table(
        {
            "ref_time": pa.array([origin for origin, _, _ in rows], type=pa.timestamp("us")),
            "target_time": pa.array([event for _, event, _ in rows], type=pa.timestamp("us")),
            "zone": pa.array(["DE"] * len(rows)),
            # Forecast wind, revised at every origin: a shorter lead is sharper.
            "wind_fc": pa.array([12.0 - 0.1 * lead for _, _, lead in rows]),
            # The outcome, a function of the event time alone.
            "price": pa.array([80.0 + event.hour for _, event, _ in rows]),
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

# What comes out is a pair: every vintage as issued, and the realized outcome
# once per event time. If the repeated labels had disagreed, that would be a
# contradiction in the source data and `InconsistentTruthError`, not a guess.
print("origins:", len(dataset.origins), "information rows:", dataset.information.table.num_rows)
print("truth rows:", dataset.truth.history.num_rows)

# Lead time is derived rather than stored: it is already determined by the two
# axes, and a third column would be free to disagree with them.
print("with lead time:", dataset.information.with_lead_time(unit="hour").table.column_names)

# -- fitting on vintages ----------------------------------------------------
# The same call as event-time data. `builtin/seasonal-naive` trains on a
# `SeriesView`, which holds exactly one forecast origin, so the plan names one.

client = of.OpenForecast()

model = client.fit(
    "builtin/seasonal-naive",
    data=dataset,
    plan=of.FitPlan(origins=of.LatestOrigin()),
    params={"season_length": 24},
    name="de-price",
)

record = model.manifest.training
if record is not None:
    # `observed`, not `simulated`: the origins were real, not windows cut out of
    # one freshest series. A model trained on the second was told the past was
    # cleaner than it was, and the artifact has to be able to say so.
    print("origin fidelity:", record.origin_fidelity, "source:", record.source)

# Asking a series model to learn from *every* vintage is refused by the planner,
# which is the only thing that knows the source type — and the refusal carries a
# code rather than only prose.
try:
    client.fit(
        "builtin/seasonal-naive",
        data=dataset,
        plan=of.FitPlan(origins=of.AllOrigins()),
        params={"season_length": 24},
    )
except of.OriginScopeError as error:
    print("refused:", error.code, "-", error.message)

# A sequence or tabular model — `nixtla/nhits`, `darts/tide`,
# `sklearn/hist-gradient-boosting` — takes `of.AllOrigins()` instead and learns
# one training sample per historical origin. The reference is the only thing
# that changes.

# -- forecasting at one origin ----------------------------------------------
# `ForecastContext` is exactly one origin, which is the shape production
# inference always has. Only that vintage contributes: a feature value revised
# later cannot appear here, enforced by the object rather than by convention.

context = dataset.at_origin(origins[-1])
forecast = client.forecast(model, data=context, horizon=24)

for row in forecast.point().to_pylist()[:3]:
    print(row)

# -- truncating -------------------------------------------------------------
# The operation a point-in-time backtest is built from: a fold holds the result
# of one of these, so there is nothing for a bug in the loop to reach for.

earlier = dataset.up_to(origins[60])
print("vintages issued by then:", len(earlier.origins))
