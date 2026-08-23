"""Fit, forecast, and read the artifact back — the whole loop, smallest version.

    uv run examples/01_quickstart.py

Everything here works on a core install and the model every build ships with, so
there is nothing to download and no provider environment to create first.
"""

from datetime import datetime, timedelta

import pyarrow as pa

import openforecast as of

# -- data -------------------------------------------------------------------
# Thirty days of hourly load with a daily shape, generated rather than fetched:
# an example that needs a file is an example that stops working.

hours = [datetime(2026, 1, 1) + timedelta(hours=step) for step in range(24 * 30)]
history = pa.table(
    {
        "timestamp": pa.array(hours, type=pa.timestamp("us")),
        "zone": pa.array(["DE"] * len(hours)),
        "load": pa.array([50.0 + (step % 24) + (step // 24) * 0.1 for step in range(len(hours))]),
    }
)

data = of.TimeSeriesFrame.from_arrow(
    history,
    time="timestamp",
    frequency="1h",
    instance_keys=["zone"],
    targets=["load"],
)

print(f"{data.history.num_rows} rows, panel={data.schema.is_panel}")

# -- what this build can run ------------------------------------------------
# The catalog answers from declarations. No provider process is started.

client = of.OpenForecast()

print("models:", [str(ref) for ref in client.models.refs()])

descriptor = client.models.get("builtin/seasonal-naive")
print("requires a fit:", descriptor.lifecycle.requires_fit)
print("can answer:", descriptor.capabilities.outputs)

# -- fit --------------------------------------------------------------------
# A fit produces a resource rather than a variable: an immutable artifact
# addressed by a reference. `local/de-load` follows the latest fit; the pinned
# `local/de-load@01K...` is one revision forever.

model = client.fit(
    "builtin/seasonal-naive",
    data=data,
    params={"season_length": 24},
    name="de-load",
)

print("fitted:", model.ref)
print("by provider:", model.manifest.provider, model.manifest.provider_version)

# -- forecast ---------------------------------------------------------------
# One `Forecast` whatever was asked for. `point()` is a projection of it, and so
# are `quantile(0.5)`, `sample(7)`, `to_wide()` and `to_pandas()`.

forecast = client.forecast(model, data=data, horizon=24)

for row in forecast.point().to_pylist()[:3]:
    print(row)

# The reference is the only thing that changes when a real model is installed:
#
#     openforecast providers install nixtla
#     client.fit("nixtla/nhits", data=data, horizon=24)
