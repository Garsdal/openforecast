"""Step 17's "done when", through the public API and the built-in provider.

> Backtesting is entirely built on ``ModelRecipe``, ``ForecastDataset`` /
> ``TimeSeriesFrame``, the ``ViewPlanner``, ``FitPlan``, ``ForecastTask`` and
> ``Forecast``, with no Nixtla/Darts/sktime-specific backtesting
> implementation.

So everything here goes through ``of.backtest`` and ``of.eligible_models``, and
the only model involved is the reference provider — which is the point rather
than a limitation: if backtesting needed anything a provider has to supply, it
could not be proved with a model that supplies nothing.

The numbers are exact rather than approximate. ``builtin/seasonal-naive`` with
``season_length=1`` repeats the last observed value, and the golden data rises by
one per step, so at horizon *h* the error of step *k* is exactly *k* — which
makes the MAE of a fold arithmetic rather than something a regression would have
to accept as given.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import openforecast as of
from openforecast.errors import DataError, RecipeError

MODEL = "builtin/seasonal-naive"
START = datetime(2026, 1, 1)
HOUR = timedelta(hours=1)


def at(step: int) -> datetime:
    return START + HOUR * step


@pytest.fixture
def client(tmp_path: Path) -> of.OpenForecast:
    return of.OpenForecast(store=tmp_path / "openforecast")


def frame(periods: int = 16, zones: tuple[str, ...] = ("DE",)) -> of.TimeSeriesFrame:
    """A history whose value at hour ``t`` is ``t``, plus 100 per zone."""
    rows: list[dict[str, Any]] = [
        {"zone": zone, "timestamp": at(step), "load": float(step + offset * 100)}
        for offset, zone in enumerate(zones)
        for step in range(periods)
    ]
    return of.TimeSeriesFrame.from_pandas(
        history=pd.DataFrame(rows),
        time="timestamp",
        frequency="1h",
        instance_keys=["zone"],
        targets=["load"],
    )


def dataset(origins: int = 16) -> of.ForecastDataset:
    """Real vintages: every origin published its own view of every event time."""
    rows: list[dict[str, Any]] = [
        {
            "zone": "DE",
            "ref_time": at(origin),
            "target_time": at(event),
            "price": float(event),
            "wind_fc": float(origin * 100 + event),
        }
        for origin in range(origins)
        for event in range(origins)
    ]
    return of.ForecastDataset.from_pandas(
        pd.DataFrame(rows),
        origin_time="ref_time",
        event_time="target_time",
        targets=["price"],
        instance_keys=["zone"],
        known_features=["wind_fc"],
        event_frequency="1h",
        origin_frequency="1h",
    )


def values(table: Any, column: str) -> list[Any]:
    found: list[Any] = table.column(column).to_pylist()
    return found


# -- backtesting event-time data -------------------------------------------


def test_it_scores_every_model_at_every_origin(client: of.OpenForecast) -> None:
    result = of.backtest(
        models=[MODEL, of.Candidate(of.Model(MODEL, params={"season_length": 2}), name="sn-2")],
        data=frame(),
        validation=of.RollingOrigin(horizon=3, windows=2),
        metrics=[of.MAE(), of.Bias()],
        client=client,
    )

    assert result.models == (MODEL, "sn-2")
    assert result.metrics == ("mae", "bias")
    assert result.num_rows == 2 * 2 * 2  # models x folds x metrics
    assert result.origins == (at(9), at(12))


def test_the_measurements_are_the_arithmetic_they_should_be(client: of.OpenForecast) -> None:
    """Repeating the last value on data rising by one: errors of 1, 2, 3."""
    result = of.backtest(
        models=[MODEL],
        data=frame(),
        validation=of.RollingOrigin(horizon=3, windows=1),
        metrics=[of.MAE(), of.RMSE(), of.Bias()],
        client=client,
    )
    scored = dict(zip(values(result.table, "metric"), values(result.table, "value"), strict=True))

    assert scored["mae"] == pytest.approx(2.0)
    assert scored["rmse"] == pytest.approx((14 / 3) ** 0.5)
    # Every forecast is below what happened, so the bias is negative.
    assert scored["bias"] == pytest.approx(-2.0)
    assert values(result.table, "pairs") == [3, 3, 3]


def test_every_row_says_what_would_make_it_incomparable(client: of.OpenForecast) -> None:
    result = of.backtest(
        models=[MODEL],
        data=frame(),
        validation=of.RollingOrigin(horizon=2, windows=1),
        metrics=[of.MAE()],
        client=client,
    )

    assert values(result.table, "origin_fidelity") == ["simulated"]
    assert values(result.table, "provider") == ["builtin"]
    assert values(result.table, "fit_seconds")[0] >= 0.0


def test_a_backtest_leaves_an_artifact_you_can_forecast_with(client: of.OpenForecast) -> None:
    """The `artifact` column is a reference, not a receipt."""
    result = of.backtest(
        models=[MODEL],
        data=frame(),
        validation=of.RollingOrigin(horizon=2, windows=1),
        metrics=[of.MAE()],
        client=client,
    )
    reference = values(result.table, "artifact")[0]

    forecast = client.forecast(reference, frame().up_to(at(13)), horizon=2)

    assert forecast.model == reference
    assert values(forecast.point(), "value") == [13.0, 13.0]


def test_an_unanswerable_event_time_is_dropped_rather_than_scored(
    client: of.OpenForecast,
) -> None:
    """A model answering NaN has said it does not know, and `pairs` says how often.

    ``builtin/seasonal-naive`` repeats what happened a season ago, so a gap in
    the history comes back as a gap in the forecast. Scoring it would make the
    metric NaN — one unanswerable event time destroying a whole fold, with
    nothing in the result saying which one it was.
    """
    rows: list[dict[str, Any]] = [
        {"zone": "DE", "timestamp": at(step), "load": None if step == 8 else float(step)}
        for step in range(12)
    ]
    data = of.TimeSeriesFrame.from_pandas(
        history=pd.DataFrame(rows),
        time="timestamp",
        frequency="1h",
        instance_keys=["zone"],
        targets=["load"],
    )

    result = of.backtest(
        models=[of.Model(MODEL, params={"season_length": 4})],
        data=data,
        validation=of.RollingOrigin(horizon=4, windows=1),
        metrics=[of.MAE()],
        client=client,
    )

    assert values(result.table, "pairs") == [3]
    assert values(result.table, "value") == [pytest.approx(4.0)]


def test_a_panel_is_scored_over_every_instance(client: of.OpenForecast) -> None:
    result = of.backtest(
        models=[MODEL],
        data=frame(zones=("DE", "FR")),
        validation=of.RollingOrigin(horizon=2, windows=1),
        metrics=[of.MAE()],
        client=client,
    )

    assert values(result.table, "pairs") == [4]
    assert values(result.table, "value") == [pytest.approx(1.5)]


# -- backtesting real vintages ---------------------------------------------


def test_point_in_time_data_is_backtested_at_the_origins_it_holds(
    client: of.OpenForecast,
) -> None:
    result = of.backtest(
        models=[MODEL],
        data=dataset(),
        validation=of.ForecastOriginValidation(
            horizon=2, origins=of.OriginsBetween(at(8), at(12), stride=2)
        ),
        metrics=[of.MAE()],
        plan=of.FitPlan(origins=of.LatestOrigin()),
        client=client,
    )

    assert result.origins == (at(8), at(10), at(12))
    assert values(result.table, "origin_fidelity") == ["observed"] * 3


def test_a_fit_at_a_historical_origin_never_saw_a_later_vintage(
    client: of.OpenForecast,
) -> None:
    """The Step 17 guarantee, asserted where it is observable: the manifest.

    The artifact of each fold records the origins it was materialized from, and
    the fold selected the *latest* one — which has to be the fold's own origin
    rather than the freshest vintage in the dataset.
    """
    result = of.backtest(
        models=[MODEL],
        data=dataset(),
        validation=of.ForecastOriginValidation(horizon=2, origins=of.OriginsBetween(at(8), at(10))),
        metrics=[of.MAE()],
        plan=of.FitPlan(origins=of.LatestOrigin()),
        client=client,
    )

    for origin, reference in zip(
        values(result.table, "origin"), values(result.table, "artifact"), strict=True
    ):
        handle = client.artifact(reference)
        forecast = client.forecast(reference, dataset().at_origin(origin), horizon=1)
        assert handle.training.origin_fidelity == "observed"
        assert handle.training.source == "forecast_dataset"
        # The value it repeats is the last outcome it could have known.
        assert values(forecast.point(), "value") == [float(_step(origin))]


def _step(moment: datetime) -> int:
    return round((moment - START) / HOUR)


def test_the_same_models_can_be_compared_across_the_two_fidelities(
    client: of.OpenForecast,
) -> None:
    """Simulated availability against true point-in-time availability.

    The comparison Step 17 asks for, and the reason ``origin_fidelity`` is a
    column: the same model, the same origins and the same horizon, backtested
    once against vintages and once against the truth frame on its own.
    """
    data = dataset()
    origins = of.OriginsBetween(at(8), at(10))

    observed = of.backtest(
        models=[MODEL],
        data=data,
        validation=of.ForecastOriginValidation(horizon=2, origins=origins),
        metrics=[of.MAE()],
        plan=of.FitPlan(origins=of.LatestOrigin()),
        client=client,
    )
    simulated = of.backtest(
        models=[MODEL],
        # Cut to 12:00 so that the rolling windows land on the same three
        # origins the vintages were selected at: a rolling origin counts back
        # from the end of the history it is given.
        data=data.truth.up_to(at(12)),
        validation=of.RollingOrigin(horizon=2, windows=3, stride=1),
        metrics=[of.MAE()],
        client=client,
    )

    assert set(values(observed.table, "origin_fidelity")) == {"observed"}
    assert set(values(simulated.table, "origin_fidelity")) == {"simulated"}
    assert observed.origins == simulated.origins
    # This model conditions on no feature, so the two agree on the numbers —
    # which is what makes the fidelity column the only difference between them.
    assert values(observed.table, "value") == values(simulated.table, "value")


# -- what a backtest refuses -----------------------------------------------


def test_two_candidates_with_the_same_name_are_refused(client: of.OpenForecast) -> None:
    """Their rows could not be told apart, so neither could their conclusions."""
    with pytest.raises(RecipeError, match="name more than one candidate"):
        of.backtest(
            models=[MODEL, of.Model(MODEL, params={"season_length": 2})],
            data=frame(),
            validation=of.RollingOrigin(horizon=2, windows=1),
            metrics=[of.MAE()],
            client=client,
        )


def test_a_fitted_artifact_is_not_a_candidate(client: of.OpenForecast) -> None:
    handle = client.fit(MODEL, frame(), name="already-fitted")

    with pytest.raises(RecipeError, match="fitted artifact, not a candidate"):
        of.backtest(
            models=[handle],
            data=frame(),
            validation=of.RollingOrigin(horizon=2, windows=1),
            metrics=[of.MAE()],
            client=client,
        )


def test_a_pinned_revision_is_not_a_candidate_either(client: of.OpenForecast) -> None:
    """A revision names one fit; a candidate is fitted once per fold."""
    handle = client.fit(MODEL, frame(), name="pinned")

    with pytest.raises(RecipeError, match="pin fitted revisions"):
        of.backtest(
            models=[str(handle.ref)],
            data=frame(),
            validation=of.RollingOrigin(horizon=2, windows=1),
            metrics=[of.MAE()],
            client=client,
        )


def test_a_backtest_without_a_metric_measures_nothing(client: of.OpenForecast) -> None:
    with pytest.raises(RecipeError, match="at least one metric"):
        of.backtest(
            models=[MODEL],
            data=frame(),
            validation=of.RollingOrigin(horizon=2, windows=1),
            metrics=[],
            client=client,
        )


def test_a_horizon_the_truth_does_not_reach_is_reported_rather_than_scored(
    client: of.OpenForecast,
) -> None:
    """Scoring the part that exists is fine; scoring nothing at all is not."""
    with pytest.raises(DataError, match="nothing to score"):
        of.backtest(
            models=[MODEL],
            data=dataset(origins=6),
            validation=of.ForecastOriginValidation(horizon=2, origins=of.AtOrigin(at(5))),
            metrics=[of.MAE()],
            plan=of.FitPlan(origins=of.LatestOrigin()),
            client=client,
        )


# -- the plan a candidate is fitted with ------------------------------------


def test_a_window_is_not_carried_to_a_model_that_cannot_bind_one(
    client: of.OpenForecast,
) -> None:
    """What makes one ``plan=`` comparable across model families.

    ``of.fit`` refuses a ``WindowPlan`` handed to a series model, correctly. A
    backtest's plan is a template over candidates that deliberately do not
    share a contract, so the window reaches the ones that size samples with it.
    """
    plan = of.FitPlan(window=of.WindowPlan(context=4))

    with pytest.raises(RecipeError, match="sizes no context window"):
        client.fit(MODEL, frame(), horizon=2, plan=plan)

    result = of.backtest(
        models=[MODEL],
        data=frame(),
        validation=of.RollingOrigin(horizon=2, windows=1),
        metrics=[of.MAE()],
        plan=plan,
        client=client,
    )

    assert result.num_rows == 1


def test_a_candidate_can_state_the_plan_it_needs(client: of.OpenForecast) -> None:
    result = of.backtest(
        models=[
            of.Candidate(MODEL, name="every-origin", plan=of.FitPlan(origins=of.AllOrigins())),
        ],
        data=frame(),
        validation=of.RollingOrigin(horizon=2, windows=1),
        metrics=[of.MAE()],
        plan=of.FitPlan(origins=of.LatestOrigin()),
        client=client,
    )

    assert result.models == ("every-origin",)


# -- eligibility: the `openforecast/auto` foundation ------------------------


def test_eligibility_is_answered_for_the_whole_catalog(client: of.OpenForecast) -> None:
    found = of.eligible_models(frame(), horizon=4, client=client)

    assert [str(entry.model) for entry in found] == [MODEL]
    assert all(entry.eligible for entry in found)
    assert all(entry.reason is None for entry in found)


def test_a_series_model_is_ineligible_for_multi_origin_learning(
    client: of.OpenForecast,
) -> None:
    """The plan's own example: rule out AutoARIMA where every vintage is a sample."""
    found = of.eligible_models(
        dataset(), horizon=4, plan=of.FitPlan(origins=of.AllOrigins()), client=client
    )

    assert not found[0].eligible
    assert found[0].reason is not None
    assert "one forecast origin" in found[0].reason


def test_eligibility_is_a_statement_about_a_plan_and_data_together(
    client: of.OpenForecast,
) -> None:
    """The same model, the same data, one origin instead of every origin."""
    found = of.eligible_models(
        dataset(), horizon=4, plan=of.FitPlan(origins=of.LatestOrigin()), client=client
    )

    assert found[0].eligible


def test_a_model_that_cannot_be_given_this_data_says_which_property(
    client: of.OpenForecast,
) -> None:
    """The refusal is the sentence the fit would have failed with."""
    found = of.eligible_models(frame(), horizon=4, models=["builtin/seasonal-naive"], client=client)

    assert found[0].eligible
    assert str(found[0]) == f"{MODEL} eligible"


def test_ineligibility_prints_its_reason(client: of.OpenForecast) -> None:
    found = of.eligible_models(dataset(), plan=of.FitPlan(origins=of.AllOrigins()), client=client)

    assert str(found[0]).startswith(f"{MODEL} ineligible: ")


def test_eligibility_fits_nothing(client: of.OpenForecast) -> None:
    """A descriptor is complete enough to plan against, so nothing is executed."""
    of.eligible_models(frame(), horizon=4, client=client)

    assert client.engine.store.list() == ()
