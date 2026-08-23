"""Comparing models over the same origins, and slicing the result afterwards.

    uv run examples/04_backtest.py

`of.backtest` is a loop over `fit` and `forecast` and nothing else. No provider
contains a backtester, and no provider learns that it is being backtested.
"""

from datetime import datetime, timedelta

import pyarrow as pa

import openforecast as of

# -- data -------------------------------------------------------------------

hours = [datetime(2026, 1, 1) + timedelta(hours=step) for step in range(24 * 40)]
data = of.TimeSeriesFrame.from_arrow(
    pa.table(
        {
            "timestamp": pa.array(hours, type=pa.timestamp("us")),
            "zone": pa.array(["DE"] * len(hours)),
            "load": pa.array(
                [50.0 + (step % 24) + 0.05 * (step // 24) for step in range(len(hours))]
            ),
        }
    ),
    time="timestamp",
    frequency="1h",
    instance_keys=["zone"],
    targets=["load"],
)

client = of.OpenForecast()

# -- which models could fit this at all -------------------------------------
# Eligibility means exactly one thing: the fit would not be refused. It
# materializes the view each contract asks for and checks it, which is the same
# sequence a fit runs — and an ineligible model comes back with the sentence the
# fit would have failed with. No provider is started.

for entry in client.eligible_models(data, horizon=24):
    print(entry)

# -- rolling origins over event-time data -----------------------------------
# The candidates are a list because comparing them is the point. Each is a
# reference, a recipe, or an `of.Candidate` naming one explicitly — and a recipe
# is where parameters live, so these two are the same model told two things about
# the season it should repeat. Two candidates resolving to one reference need
# names, because otherwise their rows could not be told apart; the backtest says
# so rather than picking one.

result = client.backtest(
    [
        of.Candidate(
            of.Model("builtin/seasonal-naive", params={"season_length": 24}), name="daily"
        ),
        of.Candidate(
            of.Model("builtin/seasonal-naive", params={"season_length": 168}), name="weekly"
        ),
    ],
    data=data,
    validation=of.RollingOrigin(horizon=24, windows=4),
    metrics=[of.MAE(), of.Bias()],
)

print(result.leaderboard("mae").to_pylist())
print("best:", result.best("mae"))

# Each window truncates the history with `up_to` and fits on what was left, so
# no origin ever sees a value published after it.

# -- what comes back --------------------------------------------------------
# Two long Arrow tables. Three columns of the metrics table are not measurements:
# `origin_fidelity` (read off the artifact the fold published, not declared by
# the backtest), `artifact` (the pinned revision the numbers came from, so a
# winner is a reference you can forecast with) and `pairs` (how many outcomes a
# value was computed over).

print("metrics:", result.metrics.column_names)
print("predictions:", result.predictions.column_names)

# The predictions are kept because the metrics are derivable from them and not
# the reverse — so the question everyone asks after a backtest is a projection
# rather than a second run.

for row in result.metrics_by("horizon_step").to_pylist()[:4]:
    print(row)

# -- point-in-time origins --------------------------------------------------
# The same call with the validation that fits real vintages. At each origin the
# features come from *that* vintage; later ones are not merely unused, they are
# absent from the object the model is handed. See 03_point_in_time.py.

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
            "wind_fc": pa.array([12.0 - 0.1 * lead for _, _, lead in rows]),
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

pit = client.backtest(
    [of.Model("builtin/seasonal-naive", params={"season_length": 24})],
    data=dataset,
    validation=of.ForecastOriginValidation(
        origins=of.OriginsBetween(origins[48], origins[-25], stride=24),
        horizon=24,
    ),
    metrics=[of.MAE()],
    # A series model holds one forecast origin. A sequence or tabular model takes
    # `of.AllOrigins()` and learns from every vintage up to the fold's origin.
    plan=of.FitPlan(origins=of.LatestOrigin()),
)

print(pit.leaderboard("mae").to_pylist())
print("fidelity:", pit.metrics.column("origin_fidelity").unique())
