"""The model-construction IR: what can be written down, and what cannot.

The assertions worth reading twice are the ones that reject something. A recipe
is the only record of what produced an artifact, so anything ambiguous —
a parameter that duplicates a concept OpenForecast owns, a missing indicator
that would come out constant, an ensemble whose weights do not match its
members — has to fail where it is written rather than where it is executed.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

import openforecast as of
from openforecast.errors import RecipeError
from openforecast.models import ModelRef
from openforecast.recipes import (
    ColumnSet,
    ImputeMethod,
    Recipe,
    RecipeKind,
    ReductionStrategy,
    estimator_refs,
    parse_recipe,
)

LIGHTGBM = ModelRef.parse("lightgbm/regressor")


def nhits(**params: Any) -> of.Model:
    return of.Model("nixtla/nhits", params=params)


# -- leaf models -----------------------------------------------------------


def test_a_model_is_named_positionally_like_a_string() -> None:
    model = of.Model("nixtla/nhits", params={"max_steps": 500})

    assert model.ref == ModelRef.parse("nixtla/nhits")
    assert model.params == {"max_steps": 500}
    assert model.kind is RecipeKind.MODEL


def test_a_model_can_also_be_named_by_keyword() -> None:
    assert of.Model(ref="nixtla/nhits") == of.Model("nixtla/nhits")


def test_a_model_named_twice_is_not_resolved_by_picking_one() -> None:
    with pytest.raises(RecipeError, match=r"both positionally and by keyword"):
        of.Model("nixtla/nhits", ref="darts/nhits")


def test_a_model_may_name_a_fitted_revision() -> None:
    """An ensemble of already-fitted artifacts is a legitimate recipe."""
    model = of.Model("local/de-price@01K5Z6QK3M9TQK1W2E3R4T5Y6U")

    assert model.ref.is_pinned


def test_a_malformed_reference_is_rejected_where_it_is_written() -> None:
    with pytest.raises(of.ModelRefError, match=r"names no provider"):
        of.Model("nhits")


@pytest.mark.parametrize(
    ("name", "instead"),
    [
        ("input_size", "WindowPlan"),
        ("input_chunk_length", "WindowPlan"),
        ("context_length", "WindowPlan"),
        ("h", "ForecastTask"),
        ("horizon", "ForecastTask"),
        ("output_chunk_length", "ForecastTask"),
        ("random_state", "FitPlan"),
        ("seed", "FitPlan"),
        ("freq", "frequency"),
        ("hist_exog_list", "observed features"),
        ("futr_exog_list", "known features"),
        ("stat_exog_list", "static features"),
    ],
)
def test_a_parameter_naming_something_openforecast_owns_is_refused(name: str, instead: str) -> None:
    """Two copies of one number, and the provider's spelling would win silently."""
    with pytest.raises(RecipeError, match=r"OpenForecast owns") as raised:
        nhits(**{name: 168})

    assert instead in str(raised.value)


def test_provider_parameters_openforecast_does_not_own_pass_through() -> None:
    model = nhits(max_steps=500, dropout=0.1, scaler_type="robust")

    assert model.params == {"max_steps": 500, "dropout": 0.1, "scaler_type": "robust"}


def test_parameters_that_cannot_be_serialized_are_refused() -> None:
    """A recipe is recorded in a manifest and sent to a provider as JSON."""
    with pytest.raises(RecipeError, match=r"survive serialization"):
        nhits(callback=object())


# -- pipelines -------------------------------------------------------------


def test_a_pipeline_ends_in_its_estimator() -> None:
    pipeline = of.Pipeline(
        steps=(
            of.MissingIndicator(columns=ColumnSet.FEATURES),
            of.Impute(columns=ColumnSet.FEATURES, method=ImputeMethod.MEDIAN),
            of.StandardScaler(columns=ColumnSet.TARGETS),
            nhits(),
        )
    )

    assert pipeline.estimator == nhits()
    assert [step.kind for step in pipeline.transforms] == [
        RecipeKind.MISSING_INDICATOR,
        RecipeKind.IMPUTE,
        RecipeKind.STANDARD_SCALER,
    ]


def test_a_pipeline_that_forecasts_nothing_is_not_a_recipe() -> None:
    with pytest.raises(RecipeError, match=r"must end in something that forecasts"):
        of.Pipeline(steps=(of.StandardScaler(columns=ColumnSet.TARGETS),))


def test_a_transform_after_the_estimator_has_nothing_to_transform() -> None:
    with pytest.raises(RecipeError, match=r"must be its last step"):
        of.Pipeline(steps=(nhits(), of.StandardScaler(columns=ColumnSet.TARGETS)))


def test_a_pipeline_holds_one_estimator() -> None:
    with pytest.raises(RecipeError, match=r"holds one estimator"):
        of.Pipeline(steps=(nhits(), of.Model("nixtla/autoarima")))


def test_a_pipeline_does_not_nest() -> None:
    """Flattening it would change nothing, so a nested one is a recipe written twice."""
    inner = of.Pipeline(steps=(nhits(),))
    with pytest.raises(ValidationError):
        of.Pipeline(steps=(inner,))  # pyright: ignore[reportArgumentType]


def test_an_ensemble_can_be_a_pipeline_step() -> None:
    ensemble = of.Ensemble(models=(nhits(), of.Model("nixtla/autoarima")))
    pipeline = of.Pipeline(steps=(of.StandardScaler(columns=ColumnSet.TARGETS), ensemble))

    assert pipeline.estimator == ensemble


def test_an_indicator_after_an_imputation_of_the_same_columns_is_refused() -> None:
    """The indicator would be constant, discarding what it was added to record."""
    with pytest.raises(RecipeError, match=r"would be\s+constant"):
        of.Pipeline(
            steps=(
                of.Impute(columns=ColumnSet.FEATURES, method=ImputeMethod.MEDIAN),
                of.MissingIndicator(columns=ColumnSet.FEATURES),
                nhits(),
            )
        )


def test_an_indicator_after_an_imputation_of_other_columns_is_fine() -> None:
    pipeline = of.Pipeline(
        steps=(
            of.Impute(columns=("wind_fc",), method=ImputeMethod.MEDIAN),
            of.MissingIndicator(columns=("solar_fc",)),
            nhits(),
        )
    )

    assert len(pipeline.transforms) == 2


def test_an_imputation_of_unknown_columns_is_treated_as_overlapping() -> None:
    """A role is resolved against a schema the recipe has not met, so it may overlap."""
    with pytest.raises(RecipeError, match=r"would be\s+constant"):
        of.Pipeline(
            steps=(
                of.Impute(columns=ColumnSet.FEATURES, method=ImputeMethod.ZERO),
                of.MissingIndicator(columns=("wind_fc",)),
                nhits(),
            )
        )


# -- transforms ------------------------------------------------------------


def test_a_column_role_keeps_a_recipe_portable() -> None:
    scaler = of.StandardScaler(columns=ColumnSet.TARGETS)

    assert scaler.column_set is ColumnSet.TARGETS
    assert scaler.explicit_columns is None


def test_explicit_columns_are_kept_as_given() -> None:
    indicator = of.MissingIndicator(columns=("wind_fc", "solar_fc"))

    assert indicator.explicit_columns == ("wind_fc", "solar_fc")
    assert indicator.column_set is None


def test_a_bare_column_name_is_not_a_role() -> None:
    """``columns=ColumnSet.TARGETS`` stays unambiguous even if a column is called that."""
    with pytest.raises(RecipeError, match=r"is not a column role"):
        of.StandardScaler(columns="load")  # pyright: ignore[reportArgumentType]


def test_a_transform_must_name_some_column() -> None:
    with pytest.raises(RecipeError, match=r"names no columns"):
        of.StandardScaler(columns=())


def test_a_transform_cannot_name_one_column_twice() -> None:
    with pytest.raises(of.SchemaError, match=r"duplicate transform column"):
        of.Impute(columns=("wind_fc", "wind_fc"), method=ImputeMethod.MEAN)


def test_an_imputation_states_its_method() -> None:
    """No default: which fill is right depends on what the column means."""
    with pytest.raises(ValidationError):
        of.Impute(columns=ColumnSet.FEATURES)  # pyright: ignore[reportCallIssue]


def test_an_imputation_method_can_be_named_by_string() -> None:
    assert (
        of.Impute(columns=ColumnSet.FEATURES, method=ImputeMethod.MEDIAN).method
        is ImputeMethod.MEDIAN
    )


def test_a_missing_indicator_cannot_overwrite_the_column_it_describes() -> None:
    with pytest.raises(RecipeError, match=r"non-empty suffix"):
        of.MissingIndicator(columns=ColumnSet.FEATURES, suffix=" ")


def test_a_lead_time_feature_names_its_column_and_unit() -> None:
    feature = of.LeadTimeFeature(name="lead_hours", unit=of.FrequencyUnit.HOUR)

    assert (feature.name, feature.unit) == ("lead_hours", of.FrequencyUnit.HOUR)


def test_origin_calendar_features_name_the_columns_they_add() -> None:
    features = of.OriginCalendarFeatures(hour=True, weekday=True)

    assert features.columns == ("origin_hour", "origin_weekday")


def test_origin_calendar_features_must_request_something() -> None:
    with pytest.raises(RecipeError, match=r"at least one of hour, weekday or month"):
        of.OriginCalendarFeatures()


# -- ensembles -------------------------------------------------------------


def test_an_ensemble_averages_its_members_by_default() -> None:
    ensemble = of.Ensemble(models=(nhits(), of.Model("nixtla/autoarima")))

    assert ensemble.combine == of.Mean()


def test_an_ensemble_of_one_is_that_one() -> None:
    with pytest.raises(RecipeError, match=r"at least two members"):
        of.Ensemble(models=(nhits(),))


def test_weights_are_relative_and_normalize() -> None:
    assert of.WeightedMean(weights=(7, 3)).normalized == pytest.approx((0.7, 0.3))


def test_a_zero_weighted_member_is_left_out_rather_than_ignored() -> None:
    with pytest.raises(RecipeError, match=r"must be positive"):
        of.WeightedMean(weights=(1.0, 0.0))


def test_one_weight_per_member() -> None:
    with pytest.raises(RecipeError, match=r"one weight per member"):
        of.Ensemble(
            models=(nhits(), of.Model("nixtla/autoarima")),
            combine=of.WeightedMean(weights=(0.5, 0.3, 0.2)),
        )


def test_an_ensemble_composes_recipes_rather_than_models() -> None:
    ensemble = of.Ensemble(
        models=(
            of.Pipeline(steps=(of.StandardScaler(columns=ColumnSet.TARGETS), nhits())),
            of.Ensemble(models=(of.Model("darts/nhits"), of.Model("nixtla/autoarima"))),
        )
    )

    assert estimator_refs(ensemble) == (
        ModelRef.parse("nixtla/nhits"),
        ModelRef.parse("darts/nhits"),
        ModelRef.parse("nixtla/autoarima"),
    )


# -- reductions ------------------------------------------------------------


def test_a_reduction_names_an_estimator_a_strategy_and_its_lags() -> None:
    reduction = of.Reduction(
        estimator=LIGHTGBM,
        strategy=ReductionStrategy.DIRECT,
        lags=(1, 24, 168),
    )

    assert reduction.estimator == ModelRef.parse("lightgbm/regressor")
    assert reduction.strategy is ReductionStrategy.DIRECT
    assert reduction.lags == (1, 24, 168)
    assert estimator_refs(reduction) == (ModelRef.parse("lightgbm/regressor"),)


@pytest.mark.parametrize("strategy", list(ReductionStrategy))
def test_every_strategy_is_expressible_before_any_of_them_executes(
    strategy: ReductionStrategy,
) -> None:
    """Execution lands in Step 14; a recipe you cannot write down cannot be stored."""
    assert of.Reduction(estimator=LIGHTGBM, strategy=strategy, lags=(1,)).strategy is strategy


def test_a_reduction_needs_something_to_condition_on() -> None:
    with pytest.raises(RecipeError, match=r"at least one lag"):
        of.Reduction(estimator=LIGHTGBM, strategy=ReductionStrategy.DIRECT, lags=())


@pytest.mark.parametrize(
    ("lags", "message"),
    [
        ((0, 24), r"therefore positive"),
        ((-1,), r"therefore positive"),
        ((24, 24), r"duplicate lags"),
        ((168, 24), r"must be ascending"),
    ],
)
def test_lags_are_a_canonical_ascending_set(lags: tuple[int, ...], message: str) -> None:
    with pytest.raises(RecipeError, match=message):
        of.Reduction(estimator=LIGHTGBM, strategy=ReductionStrategy.DIRECT, lags=lags)


# -- serialization ---------------------------------------------------------


def recipes() -> list[Recipe]:
    nested = of.Ensemble(
        models=(
            of.Pipeline(steps=(of.StandardScaler(columns=ColumnSet.TARGETS), nhits())),
            of.Ensemble(models=(of.Model("darts/nhits"), of.Model("nixtla/autoarima"))),
        )
    )
    return [
        nested,
        nhits(max_steps=500),
        of.Pipeline(
            steps=(
                of.MissingIndicator(columns=ColumnSet.FEATURES),
                of.Impute(columns=("wind_fc",), method=ImputeMethod.MEDIAN),
                of.LeadTimeFeature(name="lead_hours", unit=of.FrequencyUnit.HOUR),
                of.OriginCalendarFeatures(hour=True, weekday=True),
                of.StandardScaler(columns=ColumnSet.TARGETS, per_instance=False),
                nhits(),
            )
        ),
        of.Ensemble(
            models=(nhits(), of.Model("nixtla/autoarima")),
            combine=of.WeightedMean(weights=(0.7, 0.3)),
        ),
        of.Reduction(estimator=LIGHTGBM, strategy=ReductionStrategy.RECURSIVE, lags=(1, 24)),
    ]


@pytest.mark.parametrize("recipe", recipes(), ids=lambda recipe: str(recipe.kind))
def test_every_recipe_round_trips_through_json(recipe: Recipe) -> None:
    """The property Step 6 is done when: a recipe survives a boundary intact."""
    restored = parse_recipe(json.loads(recipe.model_dump_json()))

    assert restored == recipe
    assert type(restored) is type(recipe)


def test_a_recipe_is_tagged_so_a_reader_never_has_to_guess() -> None:
    payload = json.loads(of.Ensemble(models=(nhits(), of.Model("darts/nhits"))).model_dump_json())

    assert payload["kind"] == "ensemble"
    assert payload["combine"] == {"combine": "mean"}
    assert [member["kind"] for member in payload["models"]] == ["model", "model"]


def test_a_recipe_is_frozen_so_the_manifest_matches_what_was_fitted() -> None:
    with pytest.raises(ValidationError):
        nhits().ref = ModelRef.parse("darts/nhits")


def test_an_unknown_node_kind_is_not_silently_accepted() -> None:
    with pytest.raises(ValidationError):
        parse_recipe({"kind": "magic"})
