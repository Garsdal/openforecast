"""Recipes: what to fit, composed — pipelines, transforms and ensembles.

    uv run examples/06_ensemble.py

Three things stay separate: what to fit, how to fit it, and what to predict. A
recipe is the first of those, it composes, and it goes to the same `fit` call a
bare reference does.
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
            "load": pa.array([50.0 + (step % 24) for step in range(len(hours))]),
        }
    ),
    time="timestamp",
    frequency="1h",
    instance_keys=["zone"],
    targets=["load"],
)

client = of.OpenForecast()

# -- a pipeline -------------------------------------------------------------
# Steps, then exactly one model. The scaler's statistics are fitted once and
# persisted, so inference is scaled by those and never by whatever the forecast
# context happened to contain — and the forecast comes back on the caller's scale.

model = client.fit(
    of.Pipeline(
        steps=[
            of.StandardScaler(columns="targets"),
            of.Model("builtin/seasonal-naive", params={"season_length": 24}),
        ]
    ),
    data=data,
    name="de-load-scaled",
)

forecast = client.forecast(model, data=data, horizon=24)
print(forecast.point().to_pylist()[:2])

# -- transforms are ordinary steps ------------------------------------------
# Nothing is imputed silently. A model that cannot consume a missing value
# declares `MissingValueSupport.REQUIRES_TRANSFORM`, and then the caller writes
# the indicator and the imputation down — recorded in the artifact, visible to
# whoever reads the forecast later. Putting the indicator *after* the imputation
# is refused, because it would come out constant.

recipe = of.Pipeline(
    steps=[
        of.MissingIndicator(columns="features"),
        of.Impute(columns="features", method="median"),
        of.Model("builtin/seasonal-naive", params={"season_length": 24}),
    ]
)

# -- an ensemble ------------------------------------------------------------
# `of.Ensemble` holds *recipes* rather than models, so an ensemble of pipelines
# needs no new vocabulary. Weights are relative and default to an equal average;
# they are fixed rather than learned, because a weight fitted on data is a
# second model.

ensemble = client.fit(
    of.Ensemble(
        models=[
            of.Pipeline(
                steps=[
                    of.StandardScaler(columns="targets"),
                    of.Model("builtin/seasonal-naive", params={"season_length": 24}),
                ]
            ),
            of.Model("builtin/seasonal-naive", params={"season_length": 168}),
        ],
        weights=[0.7, 0.3],
    ),
    data=data,
    name="de-load-ensemble",
)

# One training record per leaf: two members may consume different views, and
# there is no single materialization such an artifact could honestly claim.
print("leaves:", len(ensemble.manifest.training_records))
for member in ensemble.manifest.training_records:
    print(" ", member.view, member.source, member.samples, "samples")

print(client.forecast(ensemble, data=data, horizon=24).point().to_pylist()[:2])

# Members plan independently, which is what makes an ensemble cross-*provider*
# rather than cross-*model*: `nixtla/nhits` declares a sequence contract and is
# handed a `SequenceView`, `sklearn/hist-gradient-boosting` declares a tabular one
# and is handed a `TabularView` of the same data, and neither provider learns that
# an ensemble exists. Every member is checked before any of them runs, so a member
# whose contract the data cannot satisfy rejects the whole ensemble rather than
# being quietly trained on something else.
#
# Quantiles are averaged level by level, which is quantile averaging and not the
# quantile of the mixture. Sample paths are not combined at all, since draw *i* of
# one member has nothing to do with draw *i* of another.

# -- recipes are data -------------------------------------------------------
# Tagged by `kind`, serializable, and the same JSON that reaches an artifact
# manifest, a provider subprocess request and an HTTP body. No part of it is
# provider-specific.

payload = recipe.model_dump(mode="json")
print("round-trips:", of.parse_recipe(payload) == recipe)
print(payload)
