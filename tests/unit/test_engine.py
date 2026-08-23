"""The engine: what it decides, what it refuses, and what it never asks.

Most of this runs against a stub provider rather than the built-in model,
because what is under test is the sequence — resolve, materialize, check,
execute, publish — and not what any model computes. The one thing asserted about
the provider is what it was handed: a view the descriptor asked for, and nothing
else.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

import openforecast as of
from openforecast.artifacts import ArtifactStore
from openforecast.errors import (
    DataError,
    IncompatibleForecastTask,
    ProviderError,
    RecipeError,
    UnsupportedPlanError,
)
from openforecast.models import (
    FeatureCapabilities,
    InstanceCapabilities,
    MissingValueSupport,
    ModelCapabilities,
    ModelCatalog,
    ModelRef,
    OriginScope,
    OutputCapabilities,
    TargetCapabilities,
    TrainingContract,
)
from openforecast.protocol import ForecastColumn
from openforecast.runtime import (
    Engine,
    ProviderRegistry,
    leaves,
    normalize_forecast_context,
    normalize_recipe,
)
from openforecast.views import SequenceView, SeriesView
from tests import artifacts, providers

SERIES = "stub/series"
SEQUENCES = "stub/sequences"


@pytest.fixture
def provider() -> providers.StubProvider:
    return providers.StubProvider(
        models=(
            providers.descriptor("series"),
            providers.descriptor(
                "sequences",
                training=TrainingContract.sequences(supports_unseen_instances=True),
            ),
        )
    )


@pytest.fixture
def engine(tmp_path: Path, provider: providers.StubProvider) -> Engine:
    catalog = ModelCatalog(provider.descriptors())
    return Engine(
        store=ArtifactStore(tmp_path),
        catalog=catalog,
        providers=ProviderRegistry([provider]),
    )


def frame(**overrides: Any) -> of.TimeSeriesFrame:
    return artifacts.frame(**overrides)


# -- normalizing what a caller passed ---------------------------------------


def test_a_model_reference_and_parameters_are_the_short_spelling_of_a_recipe() -> None:
    recipe = normalize_recipe("builtin/seasonal-naive", {"season_length": 24})

    assert recipe == of.Model("builtin/seasonal-naive", params={"season_length": 24})


def test_parameters_alongside_a_recipe_would_name_them_twice() -> None:
    with pytest.raises(RecipeError, match="already carries its own"):
        normalize_recipe(of.Model("builtin/seasonal-naive"), {"season_length": 24})


def test_leaves_are_walked_in_the_order_they_are_fitted() -> None:
    """The transforms of an enclosing pipeline travel with every leaf."""
    recipe = of.Pipeline(
        steps=(
            of.StandardScaler(columns=of.ColumnSet.TARGETS),
            of.Ensemble(models=(of.Model(SERIES), of.Model(SEQUENCES))),
        )
    )

    found = leaves(recipe)

    assert [str(leaf.model.ref) for leaf in found] == [SERIES, SEQUENCES]
    assert all(
        leaf.transforms == (of.StandardScaler(columns=of.ColumnSet.TARGETS),) for leaf in found
    )


def test_a_reduction_is_writable_but_not_yet_executable() -> None:
    with pytest.raises(UnsupportedPlanError, match="not executable yet"):
        leaves(
            of.Reduction(
                estimator=ModelRef.parse("lightgbm/regressor"),
                strategy=of.ReductionStrategy.DIRECT,
                lags=(1,),
            )
        )


def test_a_frame_forecasts_from_the_end_of_its_history() -> None:
    """The only origin a frame can describe without discarding data."""
    context = normalize_forecast_context(frame())

    assert context.origin_time == artifacts.at(7)


def test_a_point_in_time_dataset_is_not_narrowed_to_an_origin_silently() -> None:
    with pytest.raises(DataError, match="at_origin"):
        normalize_forecast_context(artifacts.dataset())


def test_a_context_already_names_its_origin() -> None:
    context = of.ForecastContext(origin_time=artifacts.at(7), frame=frame())

    assert normalize_forecast_context(context) is context
    with pytest.raises(DataError, match="a context is one origin"):
        normalize_forecast_context(context, origin_time=artifacts.at(6))


# -- what the provider is handed --------------------------------------------


def test_the_provider_is_handed_the_view_its_contract_asked_for(
    engine: Engine, provider: providers.StubProvider
) -> None:
    engine.fit(SERIES, frame())
    engine.fit(SEQUENCES, frame(), horizon=2, plan=of.FitPlan(window=of.WindowPlan(context=3)))

    assert isinstance(provider.fits[0].view, SeriesView)
    assert isinstance(provider.fits[1].view, SequenceView)


def test_the_provider_is_handed_the_seed_and_the_parameters(
    engine: Engine, provider: providers.StubProvider
) -> None:
    engine.fit(SERIES, frame(), plan=of.FitPlan(seed=7), params={"anything": 1})

    assert provider.fits[0].params == {"anything": 1}
    assert provider.fits[0].seed == 7


def test_a_leaf_model_owns_the_artifacts_provider_directory(
    engine: Engine, provider: providers.StubProvider
) -> None:
    """Step 7's promise: one model means ``provider/`` is the provider's alone."""
    handle = engine.fit(SERIES, frame())

    # The fit wrote into the staging directory the artifact was published from,
    # which is the same directory under a different name.
    assert provider.fits[0].into.name == handle.provider_path.name
    assert provider.fits[0].into.parent.name == handle.artifact_id
    assert (handle.provider_path / providers.STATE_FILENAME).is_file()
    assert not handle.is_composite
    assert handle.manifest.provider == "stub"


def test_a_composite_gives_every_leaf_a_directory_of_its_own(
    engine: Engine, provider: providers.StubProvider
) -> None:
    handle = engine.fit(of.Ensemble(models=(of.Model(SERIES), of.Model(SERIES))), frame())

    directories = [call.into for call in provider.fits]
    assert directories[0] != directories[1]
    assert sorted(path.name for path in handle.provider_path.iterdir()) == ["leaf-0", "leaf-1"]
    assert (handle.provider_path / "leaf-0" / "state" / providers.STATE_FILENAME).is_file()
    assert handle.is_composite
    assert handle.manifest.provider == "openforecast"


# -- what the engine refuses ------------------------------------------------


def test_a_model_that_cannot_take_a_panel_is_refused_before_it_is_started(
    tmp_path: Path,
) -> None:
    single = providers.descriptor(
        "single",
        capabilities=ModelCapabilities(instances=InstanceCapabilities(single=True, panel=False)),
    )
    provider = providers.StubProvider(models=(single,))
    engine = Engine(
        store=ArtifactStore(tmp_path),
        catalog=ModelCatalog((single,)),
        providers=ProviderRegistry([provider]),
    )

    with pytest.raises(DataError, match="cannot be fitted on a panel"):
        engine.fit("stub/single", frame())
    assert provider.fits == []


def test_a_model_that_cannot_see_missing_values_is_refused(tmp_path: Path) -> None:
    """And the message says what would make it fittable, rather than filling them in."""
    strict = providers.descriptor(
        "strict",
        capabilities=ModelCapabilities(
            instances=InstanceCapabilities(single=True, panel=True),
            missing_values=MissingValueSupport.REQUIRES_TRANSFORM,
        ),
    )
    engine = Engine(
        store=ArtifactStore(tmp_path),
        catalog=ModelCatalog((strict,)),
        providers=ProviderRegistry([providers.StubProvider(models=(strict,))]),
    )
    history = artifacts.factories.history(instances=("DE",), instance_key="zone", periods=8)
    history.loc[history.index[-1], "load"] = artifacts.factories.NAN
    gapped = of.TimeSeriesFrame.from_pandas(
        history=history,
        time="timestamp",
        frequency="1h",
        instance_keys=["zone"],
        targets=["load"],
    )

    with pytest.raises(DataError, match="of.Impute"):
        engine.fit("stub/strict", gapped)


def test_a_series_model_cannot_learn_from_every_vintage(engine: Engine) -> None:
    """The one branch on source type lives in the planner, and this is it."""
    with pytest.raises(of.OriginScopeError, match="one forecast origin"):
        engine.fit(SERIES, artifacts.dataset())


def test_a_horizon_a_model_was_not_fitted_for_is_refused(engine: Engine) -> None:
    handle = engine.fit(
        SEQUENCES, frame(), horizon=2, plan=of.FitPlan(window=of.WindowPlan(context=3))
    )

    with pytest.raises(IncompatibleForecastTask, match="horizon bound"):
        engine.forecast(handle, frame(), horizon=3)


def test_a_model_that_binds_no_horizon_is_asked_for_any(tmp_path: Path) -> None:
    """The shape of the training samples is not a promise about the horizon.

    A sequence model that learns one step and rolls — a pooled reduction, say —
    is fitted from samples that span a horizon and is bound to none of it. So the
    manifest records the two separately, and the engine refuses on the binding
    rather than on the sample shape.
    """
    rolling = providers.descriptor(
        "rolling", training=TrainingContract.sequences(horizon_bound_at_fit=False)
    )
    engine = Engine(
        store=ArtifactStore(tmp_path),
        catalog=ModelCatalog((rolling,)),
        providers=ProviderRegistry([providers.StubProvider(models=(rolling,))]),
    )

    handle = engine.fit(
        "stub/rolling", frame(), horizon=2, plan=of.FitPlan(window=of.WindowPlan(context=3))
    )

    assert handle.training.horizon == 2
    assert not handle.training.horizon_bound
    assert handle.serves_horizon(3)
    assert engine.forecast(handle, frame(), horizon=3).table.num_rows > 0


def test_a_context_declaring_other_targets_is_refused(engine: Engine) -> None:
    """A model answers what it was fitted for, or it is not the model to ask."""
    handle = engine.fit(SERIES, frame())
    extra = of.TimeSeriesFrame.from_pandas(
        history=artifacts.factories.history(
            instances=("DE", "FR"), instance_key="zone", periods=8, targets=("load", "wind")
        ),
        time="timestamp",
        frequency="1h",
        instance_keys=["zone"],
        targets=["load", "wind"],
    )

    with pytest.raises(DataError, match="was fitted to forecast"):
        engine.forecast(handle, extra, horizon=2)


def test_data_the_model_was_not_fitted_on_is_refused(engine: Engine) -> None:
    handle = engine.fit(SERIES, frame())
    daily = of.TimeSeriesFrame.from_pandas(
        history=artifacts.factories.history(
            instances=("DE", "FR"), instance_key="zone", periods=8, step=timedelta(days=1)
        ),
        time="timestamp",
        frequency="1d",
        instance_keys=["zone"],
        targets=["load"],
    )

    with pytest.raises(DataError, match="was fitted on 1h data"):
        engine.forecast(handle, daily, horizon=2)


def test_forecasting_with_a_recipe_asks_for_something_that_was_never_fitted(
    engine: Engine,
) -> None:
    with pytest.raises(RecipeError, match="fitted model"):
        engine.forecast(
            of.Ensemble(models=(of.Model(SERIES), of.Model(SERIES))), frame(), horizon=1
        )


def test_forecasting_with_an_unfitted_reference_refuses_to_fit_one(engine: Engine) -> None:
    with pytest.raises(of.ModelRequiresFit):
        engine.forecast(SERIES, frame(), horizon=1)


def test_a_provider_that_answers_a_different_question_is_caught(tmp_path: Path) -> None:
    """A short answer looks exactly like a correct one, so it is checked."""
    descriptor = providers.descriptor("series")
    provider = providers.StubProvider(models=(descriptor,), corrupt=lambda table: table.slice(0, 1))
    engine = Engine(
        store=ArtifactStore(tmp_path),
        catalog=ModelCatalog((descriptor,)),
        providers=ProviderRegistry([provider]),
    )
    handle = engine.fit(SERIES, frame())

    with pytest.raises(ProviderError, match="was asked for"):
        engine.forecast(handle, frame(), horizon=2)


def test_an_output_the_model_cannot_produce_is_refused(engine: Engine) -> None:
    handle = engine.fit(SERIES, frame())

    with pytest.raises(DataError, match="cannot produce a quantiles forecast"):
        engine.forecast(handle, frame(), horizon=2, output=of.OutputSpec.quantiles([0.5]))


# -- probabilistic output ---------------------------------------------------


def probabilistic_engine(
    tmp_path: Path, *, quantiles: bool = False, samples: bool = False
) -> Engine:
    """An engine over one stub model declaring exactly these output capabilities."""
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
    return Engine(
        store=ArtifactStore(tmp_path),
        catalog=ModelCatalog((descriptor,)),
        providers=ProviderRegistry([providers.StubProvider(models=(descriptor,))]),
    )


def test_a_model_that_declares_quantiles_is_asked_for_them(tmp_path: Path) -> None:
    engine = probabilistic_engine(tmp_path, quantiles=True)
    handle = engine.fit(SERIES, frame())

    forecast = engine.forecast(
        handle, frame(), horizon=2, output=of.OutputSpec.quantiles([0.1, 0.5, 0.9])
    )

    assert forecast.kind is of.OutputKind.QUANTILES
    assert forecast.quantile_levels == (0.1, 0.5, 0.9)
    assert forecast.table.column(ForecastColumn.SAMPLE.value).null_count == forecast.num_rows


def test_a_model_that_declares_samples_is_asked_for_the_draws(tmp_path: Path) -> None:
    engine = probabilistic_engine(tmp_path, samples=True)
    handle = engine.fit(SERIES, frame())

    forecast = engine.forecast(handle, frame(), horizon=2, output=of.OutputSpec.samples(3))

    assert forecast.kind is of.OutputKind.SAMPLES
    assert forecast.sample_indices == (0, 1, 2)


def test_quantiles_of_a_sample_model_are_reduced_by_openforecast(tmp_path: Path) -> None:
    """The provider draws and OpenForecast reduces, with one estimator for all of them."""
    engine = probabilistic_engine(tmp_path, samples=True)
    handle = engine.fit(SERIES, frame())

    forecast = engine.forecast(
        handle,
        frame(),
        horizon=2,
        output=of.OutputSpec.quantiles([0.1, 0.9], from_samples=4),
    )

    assert forecast.kind is of.OutputKind.QUANTILES
    assert forecast.quantile_levels == (0.1, 0.9)
    # The stub draws value, value+1, value+2, value+3, so the 0.9 of four draws
    # is 2.7 above the first one — computed here rather than by the provider.
    low: list[Any] = forecast.quantile(0.1).column("value").to_pylist()
    high: list[Any] = forecast.quantile(0.9).column("value").to_pylist()
    assert max(high) - min(low) == pytest.approx(2.4)


def test_a_point_model_is_not_asked_for_samples_to_make_quantiles_from(
    tmp_path: Path,
) -> None:
    """The conversion is samples to quantiles, never a distribution around a point."""
    engine = probabilistic_engine(tmp_path)
    handle = engine.fit(SERIES, frame())

    with pytest.raises(DataError, match="cannot produce a quantiles forecast"):
        engine.forecast(
            handle, frame(), horizon=2, output=of.OutputSpec.quantiles([0.5], from_samples=10)
        )


def test_a_quantile_request_of_a_sample_model_says_how_to_ask_for_it(tmp_path: Path) -> None:
    engine = probabilistic_engine(tmp_path, samples=True)
    handle = engine.fit(SERIES, frame())

    with pytest.raises(DataError, match=r"from_samples=n"):
        engine.forecast(handle, frame(), horizon=2, output=of.OutputSpec.quantiles([0.5]))


def test_a_provider_answering_levels_nobody_asked_for_is_caught(tmp_path: Path) -> None:
    """A 0.9 where a 0.95 was asked for is scored as a 0.95 by everything downstream."""
    descriptor = providers.descriptor(
        "series",
        capabilities=ModelCapabilities(
            instances=InstanceCapabilities(single=True, panel=True),
            features=FeatureCapabilities(observed=True, known=True, static=True),
            outputs=OutputCapabilities(point=True, quantiles=True),
            missing_values=MissingValueSupport.NATIVE,
        ),
    )
    relabel = pa.array([0.5, 0.5], type=pa.float64())
    provider = providers.StubProvider(
        models=(descriptor,),
        corrupt=lambda table: table.set_column(
            table.column_names.index(ForecastColumn.QUANTILE.value),
            ForecastColumn.QUANTILE.value,
            relabel,
        ),
    )
    engine = Engine(
        store=ArtifactStore(tmp_path),
        catalog=ModelCatalog((descriptor,)),
        providers=ProviderRegistry([provider]),
    )
    handle = engine.fit(SERIES, frame(periods=6, instances=("DE",)))

    with pytest.raises(ProviderError, match=r"asked for the quantiles \[0.1, 0.9\]"):
        engine.forecast(
            handle,
            frame(periods=6, instances=("DE",)),
            horizon=1,
            output=of.OutputSpec.quantiles([0.1, 0.9]),
        )


def test_a_provider_that_is_not_installed_is_named(tmp_path: Path) -> None:
    descriptor = providers.descriptor("series")
    engine = Engine(
        store=ArtifactStore(tmp_path),
        catalog=ModelCatalog((descriptor,)),
        providers=ProviderRegistry(),
    )

    with pytest.raises(ProviderError, match="no provider named 'stub'"):
        engine.fit(SERIES, frame())


# -- what a fit records -----------------------------------------------------


def test_the_manifest_records_the_view_that_was_actually_materialized(
    engine: Engine,
) -> None:
    handle = engine.fit(
        SEQUENCES,
        frame(),
        horizon=2,
        plan=of.FitPlan(window=of.WindowPlan(context=3), seed=11),
        name="de-load",
    )

    training = handle.training
    assert (training.view, training.context, training.horizon) == ("sequences", 3, 2)
    assert training.origin_fidelity == "simulated"
    assert training.seed == 11
    assert str(handle.ref).startswith("local/de-load@")


def test_a_point_in_time_fit_is_recorded_as_observed(engine: Engine) -> None:
    """The same view type, and the only difference is what it says about itself."""
    handle = engine.fit(
        SEQUENCES,
        artifacts.dataset(),
        horizon=2,
        plan=of.FitPlan(origins=of.AllOrigins(stride=2), window=of.WindowPlan(context=3)),
    )

    assert handle.training.origin_fidelity == "observed"
    assert handle.training.origins == of.AllOrigins(stride=2)


def test_a_composite_records_one_training_record_per_leaf(engine: Engine) -> None:
    handle = engine.fit(
        of.Ensemble(models=(of.Model(SERIES), of.Model(SERIES))), frame(), name="both"
    )

    assert len(handle.training_records) == 2
    assert handle.manifest.source_model is None
    with pytest.raises(of.ArtifactError, match="training_records"):
        _ = handle.training


def test_an_ensemble_is_only_as_combinable_as_its_least_capable_member(
    tmp_path: Path,
) -> None:
    """Every child contract has to be satisfiable by the data, before anything runs.

    A member that cannot consume the source is not a partial ensemble — the
    combination would be an average over a model that was never fitted. So the
    refusal happens while the leaves are being materialized, which is before the
    first provider is started and long before an artifact exists.
    """
    single_only = providers.descriptor(
        "single-only",
        capabilities=ModelCapabilities(
            instances=InstanceCapabilities(single=True, panel=False),
            targets=TargetCapabilities(univariate=True),
            features=FeatureCapabilities(known=True),
            missing_values=MissingValueSupport.NATIVE,
        ),
    )
    provider = providers.StubProvider(models=(providers.descriptor("series"), single_only))
    store = ArtifactStore(tmp_path)
    engine = Engine(
        store=store,
        catalog=ModelCatalog(provider.descriptors()),
        providers=ProviderRegistry([provider]),
    )

    with pytest.raises(DataError, match="panel"):
        engine.fit(of.Ensemble(models=(of.Model(SERIES), of.Model("stub/single-only"))), frame())

    assert provider.fits == []
    assert not list((store.root / "models").glob("*"))


def test_an_ensemble_averages_its_members(tmp_path: Path) -> None:
    """The combination is OpenForecast's, so no provider sees more than its own."""
    descriptor = providers.descriptor("series")
    catalog = ModelCatalog((descriptor,))
    store = ArtifactStore(tmp_path)
    weak = providers.StubProvider(models=(descriptor,), value=10.0)
    engine = Engine(store=store, catalog=catalog, providers=ProviderRegistry([weak]))

    handle = engine.fit(
        of.Ensemble(
            models=(of.Model(SERIES), of.Model(SERIES, params={"other": 1})),
            combine=of.WeightedMean(weights=(3, 1)),
        ),
        frame(),
    )
    forecast = engine.forecast(handle, frame(), horizon=2)

    assert set(forecast.table.column(ForecastColumn.VALUE.value).to_pylist()) == {10.0}


def test_the_forecast_is_labeled_with_what_produced_it(engine: Engine) -> None:
    handle = engine.fit(SERIES, frame(), name="de-load")

    forecast = engine.forecast("local/de-load", frame(), horizon=2)

    assert forecast.model == str(handle.ref)
    assert forecast.origin_time == artifacts.at(7)
    assert forecast.horizon == 2
    assert forecast.event_times == (artifacts.at(8), artifacts.at(9))


def test_a_failed_fit_publishes_nothing(tmp_path: Path) -> None:
    """A resolvable artifact whose provider state is missing would forecast."""

    class Failing(providers.StubProvider):
        def fit(self, **kwargs: Any) -> None:
            raise RuntimeError("the library exploded")

    descriptor = providers.descriptor("series")
    store = ArtifactStore(tmp_path)
    engine = Engine(
        store=store,
        catalog=ModelCatalog((descriptor,)),
        providers=ProviderRegistry([Failing(models=(descriptor,))]),
    )

    with pytest.raises(RuntimeError, match="exploded"):
        engine.fit(SERIES, frame())
    assert store.list() == ()
    assert list(store.staging_root.iterdir()) == []


def test_the_engine_says_what_it_can_execute(engine: Engine) -> None:
    assert "providers=1" in repr(engine)
    assert isinstance(engine.registry.catalog, ModelCatalog)


def test_a_sequence_model_learning_across_origins_uses_every_vintage(
    engine: Engine, provider: providers.StubProvider
) -> None:
    """Point-in-time data reaches a global model as ordinary training samples."""
    engine.fit(
        SEQUENCES,
        artifacts.dataset(),
        horizon=2,
        plan=of.FitPlan(window=of.WindowPlan(context=2)),
    )

    view = provider.fits[0].view
    assert isinstance(view, SequenceView)
    assert len(view.origins) > 1
    assert view.provenance.is_observed


def test_the_single_origin_scope_is_the_contracts_business(tmp_path: Path) -> None:
    """A sequences model may still declare that it learns from one origin."""
    descriptor = providers.descriptor(
        "one-origin",
        training=TrainingContract.sequences(origin_scope=OriginScope.SINGLE),
    )
    engine = Engine(
        store=ArtifactStore(tmp_path),
        catalog=ModelCatalog((descriptor,)),
        providers=ProviderRegistry([providers.StubProvider(models=(descriptor,))]),
    )

    plan = of.FitPlan(origins=of.AtOrigin(artifacts.at(5)), window=of.WindowPlan(context=2))
    handle = engine.fit("stub/one-origin", frame(), horizon=2, plan=plan)

    assert handle.training.origins == of.AtOrigin(artifacts.at(5))


def test_a_feature_the_model_cannot_consume_is_named(tmp_path: Path) -> None:
    blind = providers.descriptor(
        "blind",
        capabilities=ModelCapabilities(
            instances=InstanceCapabilities(single=True, panel=True),
            targets=TargetCapabilities(univariate=True, multivariate=True),
            features=FeatureCapabilities(),
            missing_values=MissingValueSupport.NATIVE,
        ),
    )
    engine = Engine(
        store=ArtifactStore(tmp_path),
        catalog=ModelCatalog((blind,)),
        providers=ProviderRegistry([providers.StubProvider(models=(blind,))]),
    )

    with pytest.raises(DataError, match=r"cannot be given the features \['temp_fc'\]"):
        engine.fit("stub/blind", frame())


def test_a_forecast_answers_in_the_canonical_columns(engine: Engine) -> None:
    handle = engine.fit(SERIES, frame())

    table = engine.forecast(handle, frame(), horizon=2).table

    assert table.column_names[0] == "zone"
    assert isinstance(table, pa.Table)
