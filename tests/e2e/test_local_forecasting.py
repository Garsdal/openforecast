"""Step 8's "done when", from the public API and nothing else.

> A completely local built-in model can fit, persist, reload and forecast from
> the public API.

Everything here goes through ``of.fit`` and ``of.forecast`` with the built-in
provider — no stub, no patching, no reaching into the store. Each test reloads
through a second client where reloading is the point, because a fitted model
that only works in the process that fitted it is not an artifact.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import openforecast as of
from openforecast.errors import DataError, ModelRequiresFit

MODEL = "builtin/seasonal-naive"
START = datetime(2026, 1, 1)
HOUR = timedelta(hours=1)


def at(step: int) -> datetime:
    return START + HOUR * step


@pytest.fixture
def client(tmp_path: Path) -> of.OpenForecast:
    return of.OpenForecast(store=tmp_path / "openforecast")


def frame(
    *,
    zones: tuple[str, ...] = ("DE", "FR"),
    targets: tuple[str, ...] = ("load",),
    periods: int = 12,
) -> of.TimeSeriesFrame:
    """A panel whose value at hour ``t`` is ``t`` plus an offset per zone."""
    rows: list[dict[str, Any]] = []
    for offset, zone in enumerate(zones):
        for step in range(periods):
            row: dict[str, Any] = {"zone": zone, "timestamp": at(step)}
            for index, target in enumerate(targets):
                row[target] = float(step + offset * 100 + index * 10)
            rows.append(row)
    return of.TimeSeriesFrame.from_pandas(
        history=pd.DataFrame(rows),
        time="timestamp",
        frequency="1h",
        instance_keys=["zone"],
        targets=list(targets),
    )


def single_series(periods: int = 12) -> of.TimeSeriesFrame:
    """The same data without instance keys: one series, not a panel of one."""
    rows = [{"timestamp": at(step), "load": float(step)} for step in range(periods)]
    return of.TimeSeriesFrame.from_pandas(
        history=pd.DataFrame(rows), time="timestamp", frequency="1h", targets=["load"]
    )


def dataset() -> of.ForecastDataset:
    """Real vintages: every origin published its own view of every event time."""
    rows = [
        {
            "zone": "DE",
            "ref_time": at(origin),
            "target_time": at(event),
            "price": float(event),
            "wind_fc": float(origin * 100 + event),
        }
        for origin in range(12)
        for event in range(12)
    ]
    return of.ForecastDataset.from_pandas(
        pd.DataFrame(rows),
        origin_time="ref_time",
        event_time="target_time",
        instance_keys=["zone"],
        targets=["price"],
        known_features=["wind_fc"],
        event_frequency="1h",
        origin_frequency="1h",
    )


def values(forecast: of.Forecast) -> list[Any]:
    return forecast.point().column("value").to_pylist()


# -- the whole lifecycle ----------------------------------------------------


def test_fit_persist_reload_forecast(client: of.OpenForecast, tmp_path: Path) -> None:
    """The Step 8 acceptance test, in one place."""
    data = frame()

    handle = client.fit(model=MODEL, data=data, params={"season_length": 4}, name="de-load")

    assert str(handle.ref).startswith("local/de-load@")
    assert (handle.provider_path / "seasonal-naive.json").is_file()

    # A different client, sharing only the directory on disk.
    reloaded = of.OpenForecast(store=tmp_path / "openforecast")
    forecast = reloaded.forecast(model="local/de-load", data=data, horizon=4)

    assert values(forecast) == [8.0, 9.0, 10.0, 11.0, 108.0, 109.0, 110.0, 111.0]
    assert forecast.origin_time == at(11)
    assert forecast.model == str(handle.ref)


def test_the_module_functions_are_the_same_client(
    client: of.OpenForecast, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``of.fit`` and ``of.forecast`` delegate to one default client."""
    from openforecast import client as client_module

    monkeypatch.setattr(client_module, "_default", client)
    data = frame()

    handle = of.fit(model=MODEL, data=data, params={"season_length": 2}, name="module")

    assert values(of.forecast(model=handle, data=data, horizon=2)) == [10.0, 11.0, 110.0, 111.0]
    assert client_module.default_client() is client


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (frame(zones=("DE",)), [10.0, 11.0]),
        (single_series(), [10.0, 11.0]),
        (frame(targets=("load", "wind")), [10.0, 20.0, 11.0, 21.0, 110.0, 120.0, 111.0, 121.0]),
    ],
    ids=["panel-univariate", "single-univariate", "panel-multivariate"],
)
def test_it_forecasts_every_shape_the_semantic_model_can_express(
    client: of.OpenForecast, data: of.TimeSeriesFrame, expected: list[float]
) -> None:
    handle = client.fit(MODEL, data, params={"season_length": 2})

    assert values(client.forecast(handle, data, horizon=2)) == expected


def test_an_alias_follows_the_latest_fit(client: of.OpenForecast) -> None:
    """A scheduled job names a model once; retraining moves the pointer."""
    first = client.fit(MODEL, frame(), params={"season_length": 4}, name="de-load")
    second = client.fit(MODEL, frame(), params={"season_length": 1}, name="de-load")

    latest = client.forecast("local/de-load", frame(), horizon=2)
    pinned = client.forecast(str(first.ref), frame(), horizon=2)

    assert first.ref != second.ref
    assert values(latest) == [11.0, 11.0, 111.0, 111.0]
    assert values(pinned) == [8.0, 9.0, 108.0, 109.0]


def test_forecasting_with_an_unfitted_model_is_not_a_fit(client: of.OpenForecast) -> None:
    with pytest.raises(ModelRequiresFit):
        client.forecast(MODEL, frame(), horizon=2)


# -- point-in-time through the same API -------------------------------------


def test_a_series_model_fits_one_vintage_of_point_in_time_data(
    client: of.OpenForecast,
) -> None:
    """The public API is identical; only the origin selection says which vintage."""
    data = dataset()

    handle = client.fit(
        MODEL,
        data,
        params={"season_length": 2},
        plan=of.FitPlan(origins=of.AtOrigin(at(11))),
        name="de-price",
    )

    assert handle.training.origin_fidelity == "observed"
    assert handle.training.source == "forecast_dataset"

    forecast = client.forecast(handle, data.at_origin(at(11)), horizon=2)
    assert values(forecast) == [10.0, 11.0]


def test_a_series_model_cannot_learn_across_vintages(client: of.OpenForecast) -> None:
    """Raised by the planner, which is the only thing that knows the source type."""
    with pytest.raises(of.OriginScopeError):
        client.fit(MODEL, dataset(), plan=of.FitPlan(origins=of.AllOrigins()))


def test_a_point_in_time_dataset_is_not_an_inference_origin(client: of.OpenForecast) -> None:
    handle = client.fit(MODEL, dataset(), plan=of.FitPlan(origins=of.LatestOrigin()))

    with pytest.raises(DataError, match="at_origin"):
        client.forecast(handle, dataset(), horizon=2)


# -- recipes OpenForecast executes itself -----------------------------------


def test_a_pipeline_scales_and_unscales_around_the_model(client: of.OpenForecast) -> None:
    """The forecast comes back on the caller's scale, not the model's."""
    data = frame()
    recipe = of.Pipeline(
        steps=(
            of.StandardScaler(columns=of.ColumnSet.TARGETS),
            of.Model(MODEL, params={"season_length": 4}),
        )
    )

    handle = client.fit(recipe, data, name="scaled")

    assert handle.is_composite
    assert handle.manifest.provider == "openforecast"
    assert values(client.forecast(handle, data, horizon=4)) == pytest.approx(
        [8.0, 9.0, 10.0, 11.0, 108.0, 109.0, 110.0, 111.0]
    )


def test_an_ensemble_combines_two_fitted_models(client: of.OpenForecast) -> None:
    data = frame()
    recipe = of.Ensemble(
        models=(
            of.Model(MODEL, params={"season_length": 4}),
            of.Model(MODEL, params={"season_length": 1}),
        ),
        combine=of.WeightedMean(weights=(3, 1)),
    )

    handle = client.fit(recipe, data, name="blend")

    assert len(handle.training_records) == 2
    # 0.75 * [8, 9] + 0.25 * [11, 11]
    assert values(client.forecast("local/blend", data, horizon=2)) == pytest.approx(
        [8.75, 9.5, 108.75, 109.5]
    )


def test_an_ensemble_of_pipelines_needs_no_extra_vocabulary(client: of.OpenForecast) -> None:
    data = frame()
    recipe = of.Ensemble(
        models=(
            of.Pipeline(
                steps=(
                    of.StandardScaler(columns=of.ColumnSet.TARGETS),
                    of.Model(MODEL, params={"season_length": 4}),
                )
            ),
            of.Model(MODEL, params={"season_length": 4}),
        )
    )

    handle = client.fit(recipe, data, name="nested")

    assert values(client.forecast(handle, data, horizon=2)) == pytest.approx(
        [8.0, 9.0, 108.0, 109.0]
    )


# -- what the artifact says -------------------------------------------------


def test_the_artifact_describes_the_fit_that_actually_happened(
    client: of.OpenForecast,
) -> None:
    handle = client.fit(MODEL, frame(), params={"season_length": 4}, name="de-load")
    artifact = client.engine.store.read(handle.ref)

    assert artifact.recipe == of.Model(MODEL, params={"season_length": 4})
    assert handle.manifest.provider == "builtin"
    assert str(handle.manifest.source_model) == MODEL
    assert handle.training.view == "series"
    assert handle.training.samples == 2
    assert handle.serves_horizon(1000)


def test_a_horizon_a_series_model_binds_nothing_to_is_always_servable(
    client: of.OpenForecast,
) -> None:
    """A local model is asked for a horizon at inference, so any horizon is fine."""
    handle = client.fit(MODEL, frame(), params={"season_length": 2})

    assert len(values(client.forecast(handle, frame(), horizon=48))) == 96


def test_an_artifact_answers_for_every_model_it_holds(client: of.OpenForecast) -> None:
    """A composite serves a horizon only if all of its members do."""
    handle = client.fit(
        of.Ensemble(models=(of.Model(MODEL, params={"season_length": 2}), of.Model(MODEL))),
        frame(),
        name="blend",
    )

    assert handle.serves_horizon(5)
    assert all(record.horizon is None for record in handle.training_records)
