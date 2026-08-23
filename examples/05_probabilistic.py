"""Asking for a distribution, and being refused when the model has none.

    uv run examples/05_probabilistic.py

What kind of answer to produce is a request; what a model can answer is a
declaration. Both are explicit and neither is inferred, so this example runs on a
core install: the model every build ships with declares point output only, and
the refusal is as much of the protocol as the quantiles are.
"""

from datetime import datetime, timedelta

import pyarrow as pa

import openforecast as of

hours = [datetime(2026, 1, 1) + timedelta(hours=step) for step in range(24 * 30)]
data = of.TimeSeriesFrame.from_arrow(
    pa.table(
        {
            "timestamp": pa.array(hours, type=pa.timestamp("us")),
            "zone": pa.array(["DE"] * len(hours)),
            "price": pa.array([80.0 + (step % 24) for step in range(len(hours))]),
        }
    ),
    time="timestamp",
    frequency="1h",
    instance_keys=["zone"],
    targets=["price"],
)

client = of.OpenForecast()

# -- the three requests -----------------------------------------------------
# There is one `OutputSpec` and three ways to build it. There are no separate
# result classes to match: a point forecast, a set of quantiles and a set of
# sample paths all arrive as one `Forecast` over one long table, so code
# downstream never learns which of the three its provider is native in.

print(of.OutputSpec.point())
print(of.OutputSpec.quantiles([0.1, 0.5, 0.9]))
print(of.OutputSpec.samples(200))

# -- what a model declares --------------------------------------------------

for descriptor in client.models.list():
    outputs = descriptor.capabilities.outputs
    print(f"{descriptor.ref}: point={outputs.point} quantiles={outputs.quantiles}")

# -- the refusal ------------------------------------------------------------
# Checked against the declaration before any provider is started, and reported
# with a stable code and machine-readable details rather than only prose. An
# agent recovers by reading `error.code`, never by matching the sentence.

model = client.fit(
    "builtin/seasonal-naive", data=data, params={"season_length": 24}, name="de-price"
)

try:
    client.forecast(model, data=data, horizon=24, output=of.OutputSpec.quantiles([0.1, 0.9]))
except of.UnsupportedOutput as error:
    print("refused:", error.code)
    print("details:", error.details)

# A deterministic model is never dressed up as a probabilistic one. Turning point
# forecasts into a distribution is a calibration layer a caller can ask for
# explicitly, and never something a request quietly triggers.

# -- the one conversion -----------------------------------------------------
#
#     samples   -> quantiles     the draws are the distribution; read it
#     quantiles -> samples       refused: the paths would have to be invented
#     point     -> anything      refused: there is no distribution to read
#
# And it is asked for rather than assumed, because how many draws a quantile was
# estimated from is part of what that quantile is:

print(of.OutputSpec.quantiles([0.1, 0.9], from_samples=200))

# The reduction happens in OpenForecast with one estimator, which is what makes
# two providers' quantiles comparable rather than each library's own convention.

# -- with a probabilistic model installed -----------------------------------
# `openforecast providers install nixtla` adds `nixtla/autoarima`, whose
# prediction intervals answer a quantile request. Everything below is the same
# three calls; only the reference changed.

probabilistic = [
    descriptor for descriptor in client.models.list() if descriptor.capabilities.outputs.quantiles
]

if not probabilistic:
    print("no quantile-capable model installed; try: openforecast providers install nixtla")
else:
    chosen = probabilistic[0]
    print("using", chosen.ref)
    # Whether there is a fit at all is the lifecycle's answer, not this script's:
    # `nixtla/autoarima` is fitted, `amazon/chronos-2` answers quantiles zero-shot.
    # See 07_zero_shot.py.
    target = (
        client.fit(chosen.ref, data=data, horizon=24, name="de-price-probabilistic")
        if chosen.lifecycle.requires_fit
        else chosen.ref
    )
    forecast = client.forecast(
        target, data=data, horizon=24, output=of.OutputSpec.quantiles([0.1, 0.5, 0.9])
    )
    print(forecast.quantile(0.9).to_pylist()[:2])
    # zone, event_time, price_q0.1, price_q0.5, price_q0.9
    print(forecast.to_wide().column_names)

    # The metrics that read a distribution. `Coverage` and `IntervalWidth` are the
    # calibration and sharpness halves of one question, which is why the first is
    # best *at* its nominal level rather than highest, and why the second is only
    # readable beside it. They are checked against the requested output before the
    # first fit, so a coverage of a point forecast fails in the first line of the
    # run rather than after an hour.
    result = client.backtest(
        [chosen.ref],
        data=data,
        validation=of.RollingOrigin(horizon=24, windows=3),
        output=of.OutputSpec.quantiles([0.1, 0.5, 0.9]),
        metrics=[of.MAE(), of.PinballLoss(0.9), of.Coverage(), of.IntervalWidth()],
    )
    print(result.leaderboard("pinball[0.9]").to_pylist())
