"""Many series in one frame, one fit, and per-series numbers out the other end.

    uv run examples/02_panel.py

A panel is not a different API. `instance_keys` says which columns identify a
series, and everything downstream — the fit, the forecast, the backtest slicing —
is keyed by them without being told again.
"""

from datetime import datetime, timedelta

import pyarrow as pa

import openforecast as of

# -- data -------------------------------------------------------------------
# Three zones with the same daily shape and different levels and trends. One
# long table, one row per (zone, hour) — never one column per series.

LEVEL = {"DE": 50.0, "FR": 62.0, "NL": 41.0}
TREND = {"DE": 0.2, "FR": 0.0, "NL": 0.8}

hours = [datetime(2026, 1, 1) + timedelta(hours=step) for step in range(24 * 30)]
rows = [(zone, hour, step) for zone in LEVEL for step, hour in enumerate(hours)]

data = of.TimeSeriesFrame.from_arrow(
    pa.table(
        {
            "zone": pa.array([zone for zone, _, _ in rows]),
            "timestamp": pa.array([hour for _, hour, _ in rows], type=pa.timestamp("us")),
            "load": pa.array(
                [LEVEL[zone] + (step % 24) + TREND[zone] * (step // 24) for zone, _, step in rows]
            ),
        }
    ),
    time="timestamp",
    frequency="1h",
    instance_keys=["zone"],
    targets=["load"],
)

print("instances:", data.schema.instance_keys, "->", data.history.num_rows, "rows")

# -- one fit over the whole panel -------------------------------------------
# `builtin/seasonal-naive` is a *local* model: it trains on one complete series
# at a time, so a panel is fitted series by series inside a single artifact and
# the forecast comes back labeled with the zone it is about.

client = of.OpenForecast()

model = client.fit(
    "builtin/seasonal-naive",
    data=data,
    params={"season_length": 24},
    name="eu-load",
)

forecast = client.forecast(model, data=data, horizon=24)
point = forecast.point()

print("forecast rows:", point.num_rows, "columns:", point.column_names)
print("zones forecast:", point.column("zone").unique())

# -- per-series scores ------------------------------------------------------
# The backtest keeps every prediction the metrics were computed from, so "which
# zone is hardest" is a projection of one run rather than three runs. NL trends
# fastest here, and a seasonal naive forecast is exactly what a trend defeats.

result = client.backtest(
    # A candidate is a reference or a recipe, and a recipe is where parameters
    # live: a bare reference would be backtested on the model's defaults.
    [of.Model("builtin/seasonal-naive", params={"season_length": 24})],
    data=data,
    validation=of.RollingOrigin(horizon=24, windows=3),
    metrics=[of.MAE(), of.Bias()],
)

print("overall:", result.leaderboard("mae").to_pylist())
for row in result.metrics_by("zone").to_pylist():
    print(row)

# The group keys are columns of the prediction table, your own instance keys
# included, so `metrics_by(["zone", "horizon_step"])` is the same question asked
# more finely. An unknown key is an error naming the columns that exist.
