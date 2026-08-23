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
from openforecast.models import (
    FeatureCapabilities,
    InstanceCapabilities,
    MissingValueSupport,
    ModelCapabilities,
    ModelCatalog,
    ModelLifecycle,
    OutputCapabilities,
    TargetCapabilities,
    TrainingContract,
)
from openforecast.runtime import ProviderRegistry
from tests import providers

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
    assert result.metric_names == ("mae", "bias")
    assert result.metrics.num_rows == 2 * 2 * 2  # models x folds x metrics
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
    scored = dict(
        zip(values(result.metrics, "metric"), values(result.metrics, "value"), strict=True)
    )

    assert scored["mae"] == pytest.approx(2.0)
    assert scored["rmse"] == pytest.approx((14 / 3) ** 0.5)
    # Every forecast is below what happened, so the bias is negative.
    assert scored["bias"] == pytest.approx(-2.0)
    assert values(result.metrics, "pairs") == [3, 3, 3]


def test_every_row_says_what_would_make_it_incomparable(client: of.OpenForecast) -> None:
    result = of.backtest(
        models=[MODEL],
        data=frame(),
        validation=of.RollingOrigin(horizon=2, windows=1),
        metrics=[of.MAE()],
        client=client,
    )

    assert values(result.metrics, "origin_fidelity") == ["simulated"]
    assert values(result.metrics, "provider") == ["builtin"]
    assert values(result.metrics, "fit_seconds")[0] >= 0.0


def test_a_backtest_leaves_an_artifact_you_can_forecast_with(client: of.OpenForecast) -> None:
    """The `artifact` column is a reference, not a receipt."""
    result = of.backtest(
        models=[MODEL],
        data=frame(),
        validation=of.RollingOrigin(horizon=2, windows=1),
        metrics=[of.MAE()],
        client=client,
    )
    reference = values(result.metrics, "artifact")[0]

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

    assert values(result.metrics, "pairs") == [3]
    assert values(result.metrics, "value") == [pytest.approx(4.0)]


def test_a_panel_is_scored_over_every_instance(client: of.OpenForecast) -> None:
    result = of.backtest(
        models=[MODEL],
        data=frame(zones=("DE", "FR")),
        validation=of.RollingOrigin(horizon=2, windows=1),
        metrics=[of.MAE()],
        client=client,
    )

    assert values(result.metrics, "pairs") == [4]
    assert values(result.metrics, "value") == [pytest.approx(1.5)]


# -- the predictions the metrics came from ----------------------------------


def test_the_result_holds_every_prediction_as_well_as_the_metrics(
    client: of.OpenForecast,
) -> None:
    """One row per model, fold, instance, event time and target."""
    result = of.backtest(
        models=[MODEL],
        data=frame(zones=("DE", "FR")),
        validation=of.RollingOrigin(horizon=3, windows=2),
        metrics=[of.MAE()],
        client=client,
    )

    assert result.instance_keys == ("zone",)
    assert result.predictions.column_names == [
        "model",
        "fold",
        "zone",
        "origin_time",
        "event_time",
        "horizon_step",
        "target",
        "kind",
        "quantile",
        "sample",
        "prediction",
        "actual",
    ]
    # folds x instances x horizon x targets.
    assert result.predictions.num_rows == 2 * 2 * 3 * 1
    assert set(values(result.predictions, "horizon_step")) == {1, 2, 3}
    assert set(values(result.predictions, "zone")) == {"DE", "FR"}


def test_a_metric_can_be_regrouped_by_horizon_step_without_re_running_anything(
    client: of.OpenForecast,
) -> None:
    """The question a fold-level table cannot answer: does it degrade with horizon?

    Repeating the last observed value on data rising by one is wrong by exactly
    *k* at step *k*, so the degradation is arithmetic — and pooled over the two
    folds, because a horizon group holds every origin's step *k*.
    """
    result = of.backtest(
        models=[MODEL],
        data=frame(),
        validation=of.RollingOrigin(horizon=3, windows=2),
        metrics=[of.MAE()],
        client=client,
    )

    grouped = result.metrics_by("horizon_step")

    assert values(grouped, "model") == [MODEL] * 3
    assert values(grouped, "horizon_step") == [1, 2, 3]
    assert values(grouped, "value") == [1.0, 2.0, 3.0]
    assert values(grouped, "pairs") == [2, 2, 2]


def test_a_grouping_can_name_an_instance_key(client: of.OpenForecast) -> None:
    """The keys are prediction columns like any other, under the caller's names."""
    result = of.backtest(
        models=[MODEL],
        data=frame(zones=("DE", "FR")),
        validation=of.RollingOrigin(horizon=2, windows=1),
        metrics=[of.MAE()],
        client=client,
    )

    grouped = result.metrics_by(["zone", "horizon_step"])

    assert values(grouped, "zone") == ["DE", "DE", "FR", "FR"]
    assert values(grouped, "horizon_step") == [1, 2, 1, 2]
    assert values(grouped, "value") == [1.0, 2.0, 1.0, 2.0]


def test_a_grouping_by_something_the_predictions_do_not_hold_is_refused(
    client: of.OpenForecast,
) -> None:
    result = of.backtest(
        models=[MODEL],
        data=frame(),
        validation=of.RollingOrigin(horizon=2, windows=1),
        metrics=[of.MAE()],
        client=client,
    )

    with pytest.raises(RecipeError, match="not columns of the prediction table"):
        result.metrics_by("provider")


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
    assert values(result.metrics, "origin_fidelity") == ["observed"] * 3


def test_a_cross_view_ensemble_is_backtested_like_any_other_candidate(
    tmp_path: Path,
) -> None:
    """Step 21.6: no ensemble-specific evaluation logic, because there is none to add.

    The candidate is an ensemble of a model that learns from sequences and one
    that learns from tabular rows, over real vintages, fitted and scored by the
    ordinary lifecycle: one row per fold in the result, named once, exactly as a
    bare model reference would be.
    """
    deep = providers.descriptor(
        "sequences", training=TrainingContract.sequences(supports_unseen_instances=True)
    )
    trees = providers.descriptor("trees", provider="other", training=TrainingContract.tabular())
    client = of.OpenForecast(
        store=tmp_path / "crossing",
        catalog=ModelCatalog((deep, trees)),
        providers=ProviderRegistry(
            [
                providers.StubProvider(models=(deep,), value=10.0),
                providers.StubProvider(name="other", models=(trees,), value=20.0),
            ]
        ),
    )
    recipe = of.Ensemble(models=(of.Model("stub/sequences"), of.Model("other/trees")))

    result = of.backtest(
        models=[of.Candidate(recipe, name="blend")],
        data=dataset(),
        validation=of.ForecastOriginValidation(
            horizon=2, origins=of.OriginsBetween(at(8), at(12), stride=2)
        ),
        metrics=[of.MAE()],
        plan=of.FitPlan(window=of.WindowPlan(context=3)),
        client=client,
    )

    assert result.models == ("blend",)
    assert result.metrics.num_rows == 3  # one metric per fold, not per member
    assert values(result.metrics, "origin_fidelity") == ["observed"] * 3
    # Every member answered its own constant, and what was scored is their mean.
    assert set(values(result.predictions, "prediction")) == {15.0}


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
        values(result.metrics, "origin"), values(result.metrics, "artifact"), strict=True
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

    assert set(values(observed.metrics, "origin_fidelity")) == {"observed"}
    assert set(values(simulated.metrics, "origin_fidelity")) == {"simulated"}
    assert observed.origins == simulated.origins
    # This model conditions on no feature, so the two agree on the numbers —
    # which is what makes the fidelity column the only difference between them.
    assert values(observed.metrics, "value") == values(simulated.metrics, "value")


# -- frozen artifacts -------------------------------------------------------


def test_a_frozen_revision_is_evaluated_rather_than_refitted(client: of.OpenForecast) -> None:
    """Whether the model in production has drifted, asked as a backtest.

    The artifact is fitted once, on the history up to 09:00, and then evaluated
    at a later origin without being refitted. ``builtin/seasonal-naive`` with
    ``season_length=1`` repeats the last value it *was fitted on* — 9 — so the
    errors at 13, 14 and 15 are 4, 5 and 6, which is exactly the staleness a
    frozen revision has and a per-fold fit does not.
    """
    frozen = client.fit(MODEL, frame().up_to(at(9)), name="in-production")
    published = len(client.engine.store.list())

    result = of.backtest(
        models=[str(frozen.ref)],
        data=frame(),
        validation=of.RollingOrigin(horizon=3, windows=1),
        metrics=[of.MAE()],
        client=client,
    )

    assert result.models == (str(frozen.ref),)
    assert values(result.metrics, "artifact") == [str(frozen.ref)]
    # No fit happened, which is said by a null rather than by a zero.
    assert values(result.metrics, "fit_seconds") == [None]
    assert values(result.metrics, "value") == [pytest.approx(5.0)]
    # And nothing new was published: there was nothing to publish.
    assert len(client.engine.store.list()) == published


def test_the_handle_a_fit_returned_is_the_revision_it_names(client: of.OpenForecast) -> None:
    """``of.backtest(models=[handle])`` evaluates that artifact, as the string would."""
    frozen = client.fit(MODEL, frame().up_to(at(9)), name="in-production")

    result = of.backtest(
        models=[frozen],
        data=frame(),
        validation=of.RollingOrigin(horizon=3, windows=1),
        metrics=[of.MAE()],
        client=client,
    )

    assert result.models == (str(frozen.ref),)
    assert values(result.metrics, "fit_seconds") == [None]


def test_a_frozen_revision_and_a_fitted_candidate_share_one_table(
    client: of.OpenForecast,
) -> None:
    """Mixing them is the caller's judgement, and the table says which was which."""
    frozen = client.fit(MODEL, frame().up_to(at(9)), name="in-production")

    result = of.backtest(
        models=[of.Candidate(str(frozen.ref), name="frozen"), of.Candidate(MODEL, name="refitted")],
        data=frame(),
        validation=of.RollingOrigin(horizon=3, windows=1),
        metrics=[of.MAE()],
        client=client,
    )

    scored = dict(
        zip(values(result.metrics, "model"), values(result.metrics, "value"), strict=True)
    )
    timed = dict(
        zip(values(result.metrics, "model"), values(result.metrics, "fit_seconds"), strict=True)
    )

    assert timed["frozen"] is None
    assert timed["refitted"] >= 0.0
    # The stale artifact repeats 9 and the refitted one repeats 12, which is why
    # the plan says the two numbers are not comparable.
    assert scored["frozen"] == pytest.approx(5.0)
    assert scored["refitted"] == pytest.approx(2.0)


def test_a_plan_on_a_frozen_candidate_would_do_nothing(client: of.OpenForecast) -> None:
    """There is no fit for it to configure, so it is refused rather than ignored."""
    frozen = client.fit(MODEL, frame().up_to(at(9)), name="in-production")

    with pytest.raises(RecipeError, match="the plan on this candidate would do nothing"):
        of.backtest(
            models=[of.Candidate(str(frozen.ref), plan=of.FitPlan(origins=of.LatestOrigin()))],
            data=frame(),
            validation=of.RollingOrigin(horizon=3, windows=1),
            metrics=[of.MAE()],
            client=client,
        )


def test_parameters_on_a_frozen_candidate_would_do_nothing_either(
    client: of.OpenForecast,
) -> None:
    frozen = client.fit(MODEL, frame().up_to(at(9)), name="in-production")

    with pytest.raises(RecipeError, match="already part of it"):
        of.backtest(
            models=[of.Model(str(frozen.ref), params={"season_length": 2})],
            data=frame(),
            validation=of.RollingOrigin(horizon=3, windows=1),
            metrics=[of.MAE()],
            client=client,
        )


# -- backtesting a distribution ---------------------------------------------

# The reference provider is deterministic, so a probabilistic backtest needs a
# model that declares quantiles or samples. The stub does, and what is under test
# is exactly what does *not* change: one metric list, one prediction table, one
# set of columns, whichever of the two forms the provider is native in.
PROBABILISTIC = "stub/series"


def probabilistic_client(tmp_path: Path, *, quantiles: bool, samples: bool) -> of.OpenForecast:
    descriptor = providers.descriptor(
        "series",
        capabilities=ModelCapabilities(
            instances=InstanceCapabilities(single=True, panel=True),
            targets=TargetCapabilities(univariate=True, multivariate=True),
            features=FeatureCapabilities(observed=True, known=True, static=True),
            outputs=OutputCapabilities(point=True, quantiles=quantiles, samples=samples),
            missing_values=MissingValueSupport.NATIVE,
        ),
    )
    return of.OpenForecast(
        store=tmp_path / "probabilistic",
        catalog=ModelCatalog((descriptor,)),
        providers=ProviderRegistry([providers.StubProvider(models=(descriptor,), value=8.0)]),
    )


def probabilistic_metrics() -> list[of.Metric]:
    return [of.MAE(), of.PinballLoss(0.9), of.Coverage(), of.IntervalWidth()]


def test_a_quantile_backtest_scores_the_distribution_it_asked_for(tmp_path: Path) -> None:
    result = of.backtest(
        models=[PROBABILISTIC],
        data=frame(),
        validation=of.RollingOrigin(horizon=2, windows=2),
        output=of.OutputSpec.quantiles([0.1, 0.5, 0.9]),
        metrics=probabilistic_metrics(),
        client=probabilistic_client(tmp_path, quantiles=True, samples=False),
    )

    assert result.metric_names == ("mae", "pinball[0.9]", "coverage[0.8]", "interval_width[0.8]")
    # The stub answers value + (level - 0.5) * 10, so the 0.1 to 0.9 interval is
    # eight wide at every event time — the one number here that is arithmetic.
    widths = [
        value
        for metric, value in zip(
            values(result.metrics, "metric"), values(result.metrics, "value"), strict=True
        )
        if metric == "interval_width[0.8]"
    ]
    assert widths == [pytest.approx(8.0)] * 2
    coverages = [
        value
        for metric, value in zip(
            values(result.metrics, "metric"), values(result.metrics, "value"), strict=True
        )
        if metric == "coverage[0.8]"
    ]
    assert all(0.0 <= coverage <= 1.0 for coverage in coverages)


def test_the_prediction_table_holds_one_row_per_level(tmp_path: Path) -> None:
    result = of.backtest(
        models=[PROBABILISTIC],
        data=frame(),
        validation=of.RollingOrigin(horizon=2, windows=2),
        output=of.OutputSpec.quantiles([0.1, 0.5, 0.9]),
        metrics=[of.MAE()],
        client=probabilistic_client(tmp_path, quantiles=True, samples=False),
    )

    # folds x horizon x levels, of one instance and one target.
    assert result.predictions.num_rows == 2 * 2 * 3
    assert set(values(result.predictions, "kind")) == {"quantile"}
    assert set(values(result.predictions, "quantile")) == {0.1, 0.5, 0.9}
    assert set(values(result.predictions, "sample")) == {None}
    # A point metric read the median of it: two outcomes per fold, not six rows.
    assert values(result.metrics, "pairs") == [2, 2]


def test_a_sample_model_answers_the_same_backtest(tmp_path: Path) -> None:
    """The claim of Step 20: the caller's code does not change with the provider."""
    result = of.backtest(
        models=[PROBABILISTIC],
        data=frame(),
        validation=of.RollingOrigin(horizon=2, windows=2),
        output=of.OutputSpec.samples(4),
        metrics=probabilistic_metrics(),
        client=probabilistic_client(tmp_path, quantiles=False, samples=True),
    )

    assert result.metric_names == ("mae", "pinball[0.9]", "coverage[0.8]", "interval_width[0.8]")
    assert set(values(result.predictions, "kind")) == {"sample"}
    assert set(values(result.predictions, "sample")) == {0, 1, 2, 3}
    assert all(value is not None for value in values(result.metrics, "value"))


def test_quantiles_of_a_sample_model_are_reduced_before_they_are_scored(
    tmp_path: Path,
) -> None:
    result = of.backtest(
        models=[PROBABILISTIC],
        data=frame(),
        validation=of.RollingOrigin(horizon=2, windows=2),
        output=of.OutputSpec.quantiles([0.1, 0.5, 0.9], from_samples=4),
        metrics=probabilistic_metrics(),
        client=probabilistic_client(tmp_path, quantiles=False, samples=True),
    )

    assert set(values(result.predictions, "kind")) == {"quantile"}
    assert set(values(result.predictions, "quantile")) == {0.1, 0.5, 0.9}


def test_a_probabilistic_metric_is_regrouped_from_the_predictions_too(
    tmp_path: Path,
) -> None:
    result = of.backtest(
        models=[PROBABILISTIC],
        data=frame(),
        validation=of.RollingOrigin(horizon=2, windows=2),
        output=of.OutputSpec.quantiles([0.1, 0.5, 0.9]),
        metrics=[of.IntervalWidth()],
        client=probabilistic_client(tmp_path, quantiles=True, samples=False),
    )

    by_step = result.metrics_by("horizon_step")

    assert values(by_step, "horizon_step") == [1, 2]
    assert values(by_step, "value") == [pytest.approx(8.0)] * 2
    # Two folds pooled inside each horizon group, as for any other metric.
    assert values(by_step, "pairs") == [2, 2]


def test_a_metric_that_cannot_score_the_requested_output_is_refused_before_any_fit(
    tmp_path: Path,
) -> None:
    """Discovering it after an hour of fits would be discovering it too late."""
    client = probabilistic_client(tmp_path, quantiles=True, samples=False)

    with pytest.raises(RecipeError, match="point forecast is not one"):
        of.backtest(
            models=[PROBABILISTIC],
            data=frame(),
            validation=of.RollingOrigin(horizon=2, windows=1),
            metrics=[of.Coverage()],
            client=client,
        )
    with pytest.raises(RecipeError, match=r"0.5 is not among the levels"):
        of.backtest(
            models=[PROBABILISTIC],
            data=frame(),
            validation=of.RollingOrigin(horizon=2, windows=1),
            output=of.OutputSpec.quantiles([0.1, 0.9]),
            metrics=[of.MAE()],
            client=client,
        )


def test_a_deterministic_model_is_not_asked_for_a_distribution(
    client: of.OpenForecast,
) -> None:
    """The capability check of the engine, reached through a backtest."""
    with pytest.raises(DataError, match="cannot produce a quantiles forecast"):
        of.backtest(
            models=[MODEL],
            data=frame(),
            validation=of.RollingOrigin(horizon=2, windows=1),
            output=of.OutputSpec.quantiles([0.1, 0.5, 0.9]),
            metrics=[of.MAE()],
            client=client,
        )


# -- what a backtest refuses -----------------------------------------------


def test_a_pinned_revision_inside_a_recipe_is_refused(client: of.OpenForecast) -> None:
    """A revision on its own is evaluated; a revision as a *step* cannot be fitted."""
    frozen = client.fit(MODEL, frame().up_to(at(9)), name="in-production")

    with pytest.raises(RecipeError, match="inside a recipe that is fitted per fold"):
        of.backtest(
            models=[
                of.Pipeline(
                    steps=(
                        of.StandardScaler(columns=of.ColumnSet.TARGETS),
                        of.Model(str(frozen.ref)),
                    )
                )
            ],
            data=frame(),
            validation=of.RollingOrigin(horizon=2, windows=1),
            metrics=[of.MAE()],
            client=client,
        )


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

    assert result.metrics.num_rows == 1


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


# -- pretrained candidates: Step 23's "done when" ---------------------------

PRETRAINED = "stub/pretrained"


def mixed_client(tmp_path: Path) -> of.OpenForecast:
    """A catalog holding both lifecycles, so one backtest can compare them."""
    trainable = providers.descriptor("series", provider="other")
    pretrained = providers.descriptor("pretrained", lifecycle=ModelLifecycle.pretrained())
    return of.OpenForecast(
        store=tmp_path / "mixed",
        catalog=ModelCatalog((trainable, pretrained)),
        providers=ProviderRegistry(
            [
                providers.StubProvider(models=(pretrained,), value=3.0),
                providers.StubProvider(name="other", models=(trainable,), value=5.0),
            ]
        ),
    )


def test_a_pretrained_model_is_backtested_beside_a_fitted_one(tmp_path: Path) -> None:
    """The claim of Step 23: one interface, two lifecycles, the same origins.

    The zero-shot candidate is not fitted at any fold and the trainable one is,
    and both are scored on exactly what was knowable at each origin — which is
    what makes the comparison worth making.
    """
    result = of.backtest(
        models=[PRETRAINED, "other/series"],
        data=frame(),
        validation=of.RollingOrigin(horizon=2, windows=2),
        metrics=[of.MAE()],
        client=mixed_client(tmp_path),
    )

    assert set(result.models) == {PRETRAINED, "other/series"}
    assert result.metrics.num_rows == 2 * 2
    fitted = _rows_for(result, "other/series")
    zero_shot = _rows_for(result, PRETRAINED)
    assert all(seconds is not None for seconds in fitted["fit_seconds"])
    assert all(seconds is None for seconds in zero_shot["fit_seconds"])


def test_a_pretrained_candidate_leaves_no_artifact_and_names_itself(
    tmp_path: Path,
) -> None:
    """``artifact`` is what produced the number, and here that is the reference."""
    client = mixed_client(tmp_path)
    result = of.backtest(
        models=[PRETRAINED],
        data=frame(),
        validation=of.RollingOrigin(horizon=2, windows=2),
        metrics=[of.MAE()],
        client=client,
    )

    rows = _rows_for(result, PRETRAINED)
    assert set(rows["artifact"]) == {PRETRAINED}
    assert set(rows["provider"]) == {"stub"}
    assert client.engine.store.list() == ()


def test_a_pretrained_candidate_reports_that_it_had_no_training_origins(
    tmp_path: Path,
) -> None:
    """Not ``simulated`` and not ``observed``: there were no training origins.

    A frozen revision and a pretrained model both report a null ``fit_seconds``,
    and they do not mean the same thing about the numbers beside them. This is
    the column that says which one it was.
    """
    result = of.backtest(
        models=[PRETRAINED],
        data=frame(),
        validation=of.RollingOrigin(horizon=2, windows=1),
        metrics=[of.MAE()],
        client=mixed_client(tmp_path),
    )

    assert values(result.metrics, "origin_fidelity") == ["pretrained"]


def test_a_plan_on_a_pretrained_candidate_would_do_nothing(tmp_path: Path) -> None:
    with pytest.raises(RecipeError, match="used zero-shot"):
        of.backtest(
            models=[of.Candidate(PRETRAINED, plan=of.FitPlan(seed=1))],
            data=frame(),
            validation=of.RollingOrigin(horizon=2, windows=1),
            metrics=[of.MAE()],
            client=mixed_client(tmp_path),
        )


def test_parameters_on_a_pretrained_candidate_would_do_nothing_either(
    tmp_path: Path,
) -> None:
    with pytest.raises(RecipeError, match="already part of it"):
        of.backtest(
            models=[of.Candidate(of.Model(PRETRAINED, params={"anything": 1}))],
            data=frame(),
            validation=of.RollingOrigin(horizon=2, windows=1),
            metrics=[of.MAE()],
            client=mixed_client(tmp_path),
        )


def test_a_pretrained_model_over_point_in_time_data_is_scored_at_every_vintage(
    tmp_path: Path,
) -> None:
    """Step 23.7: no training folds, and the information vintage still holds."""
    result = of.backtest(
        models=[PRETRAINED],
        data=dataset(),
        validation=of.ForecastOriginValidation(
            horizon=2, origins=of.OriginsBetween(at(8), at(12), stride=2)
        ),
        metrics=[of.MAE()],
        client=mixed_client(tmp_path),
    )

    assert result.origins == (at(8), at(10), at(12))
    assert values(result.metrics, "origin_fidelity") == ["pretrained"] * 3


def _rows_for(result: of.BacktestResult, model: str) -> dict[str, list[Any]]:
    table = result.metrics
    keep = [index for index, name in enumerate(values(table, "model")) if name == model]
    return {
        name: [values(table, name)[index] for index in keep]
        for name in ("fit_seconds", "artifact", "provider")
    }


# -- eligibility: the `openforecast/auto` foundation ------------------------


def test_a_pretrained_model_is_ineligible_because_there_is_no_fit_to_refuse(
    tmp_path: Path,
) -> None:
    """Eligibility screens fits, and a zero-shot model has none to screen.

    Reported rather than hidden, and the reason says where to go instead: it is
    not that the data is wrong for the model, it is that the question is.
    """
    found = of.eligible_models(frame(), horizon=4, client=mixed_client(tmp_path))
    zero_shot = next(entry for entry in found if str(entry.model) == PRETRAINED)

    assert not zero_shot.eligible
    assert zero_shot.reason is not None
    assert "used zero-shot" in zero_shot.reason
    # And the model beside it in the same catalog is screened as usual.
    assert next(entry for entry in found if str(entry.model) == "other/series").eligible


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
