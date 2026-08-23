"""A pretrained model forecasts without a fit — and the lifecycle says which.

    uv run examples/07_zero_shot.py

Whether a model has to be fitted and whether it can be are two questions, and a
descriptor answers both. This example reads the answer off the catalog rather
than off the model's name, so it works whichever foundation model is installed —
and says what is missing when none is.
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

# -- the lifecycle, read rather than guessed --------------------------------
# Two flags, not one: a statistical forecaster must be fitted and can be, and a
# pretrained model may be usable zero-shot and fine-tunable, or usable zero-shot
# and frozen. Collapsing them would make the third combination inexpressible.

for descriptor in client.models.list():
    lifecycle = descriptor.lifecycle
    print(
        f"{descriptor.ref}: zero_shot={lifecycle.is_zero_shot} "
        f"requires_fit={lifecycle.requires_fit} supports_fit={lifecycle.supports_fit}"
    )

zero_shot = [
    descriptor.ref for descriptor in client.models.list() if descriptor.lifecycle.is_zero_shot
]

# -- the mirror image -------------------------------------------------------
# A trainable model asked to forecast from its own reference is refused, with the
# code that says what to do about it. This is the same check `amazon/chronos-2`
# passes: one lifecycle declaration, read in both directions.

try:
    client.forecast("builtin/seasonal-naive", data=data, horizon=24)
except of.ModelRequiresFit as error:
    print("refused:", error.code, "-", error.message)

# -- the zero-shot call -----------------------------------------------------
# No fit, no artifact, no second call — and no foundation-model data primitive
# either. A pretrained model consumes the same data a fitted one does,
# materialized by the same `ViewPlanner` into the same `ForecastView`, so
# point-in-time semantics hold without the integration knowing they exist.

if not zero_shot:
    print("no pretrained model installed; try: openforecast providers install amazon")
    print("then the whole call is:")
    print('    of.forecast(model="amazon/chronos-2", data=data, horizon=24)')
else:
    ref = zero_shot[0]
    forecast = client.forecast(ref, data=data, horizon=24)
    print(forecast.point().to_pylist()[:3])

    # Fitting one is refused rather than quietly accepted, unless it declares
    # `supports_fit` — the lifecycle a model declares is the one it has.
    descriptor = client.models.get(ref)
    if not descriptor.lifecycle.supports_fit:
        try:
            client.fit(ref, data=data, horizon=24)
        except of.ModelDoesNotSupportFit as error:
            print("refused:", error.code, "-", error.message)

    # On one leaderboard with the rest: fitted candidates are refit per fold, a
    # pretrained one forecasts as it stands. Its rows report a null `fit_seconds`
    # and an `origin_fidelity` of `pretrained`, which is a different thing from a
    # frozen artifact that may have seen data postdating the early origins.
    result = client.backtest(
        [ref, of.Model("builtin/seasonal-naive", params={"season_length": 24})],
        data=data,
        validation=of.RollingOrigin(horizon=24, windows=3),
        metrics=[of.MAE(), of.Bias()],
    )
    print(result.leaderboard("mae").to_pylist())
