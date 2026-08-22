"""Lifecycle, training contract and capabilities.

These three answer, between them, everything the engine needs before it starts a
provider: whether the model has to be fitted, what to materialize for it, and
whether the data it would be handed is data it can accept.
"""

from __future__ import annotations

import pytest

from openforecast import FeatureSpec, SchemaError
from openforecast.models import (
    FeatureCapabilities,
    InstanceCapabilities,
    MissingValueSupport,
    ModelCapabilities,
    ModelLifecycle,
    OriginScope,
    OutputCapabilities,
    TargetCapabilities,
    TrainingContract,
    ViewKind,
)

# -- lifecycle --------------------------------------------------------------


def test_the_ordinary_model_must_be_fitted_and_can_be() -> None:
    lifecycle = ModelLifecycle.trainable()

    assert lifecycle.requires_fit
    assert lifecycle.supports_fit
    assert not lifecycle.supports_update
    assert not lifecycle.is_zero_shot


def test_a_pretrained_model_may_be_frozen_or_tunable() -> None:
    """The combination that a single ``requires_fit`` flag could not express."""
    frozen = ModelLifecycle.pretrained()
    tunable = ModelLifecycle.pretrained(supports_fit=True)

    assert frozen.is_zero_shot and not frozen.supports_fit
    assert tunable.is_zero_shot and tunable.supports_fit


def test_a_model_that_must_be_fitted_but_cannot_be_is_rejected() -> None:
    with pytest.raises(SchemaError, match="could never be used"):
        ModelLifecycle(requires_fit=True, supports_fit=False)


def test_a_model_that_cannot_be_fitted_cannot_be_updated() -> None:
    with pytest.raises(SchemaError, match="no artifact to update"):
        ModelLifecycle(requires_fit=False, supports_fit=False, supports_update=True)


# -- the training contract --------------------------------------------------


def test_the_three_contracts_of_the_plan() -> None:
    """AutoARIMA, NHiTS and a LightGBM reduction, as Step 5 specifies them."""
    autoarima = TrainingContract.series()
    assert autoarima.view is ViewKind.SERIES
    assert autoarima.origin_scope is OriginScope.SINGLE
    assert not autoarima.horizon_bound_at_fit
    assert not autoarima.learns_across_origins

    nhits = TrainingContract.sequences(supports_unseen_instances=True)
    assert nhits.view is ViewKind.SEQUENCES
    assert nhits.origin_scope is OriginScope.MULTIPLE
    assert nhits.context_required
    assert nhits.horizon_bound_at_fit
    assert nhits.supports_unseen_instances
    assert nhits.learns_across_origins

    lightgbm = TrainingContract.tabular()
    assert lightgbm.view is ViewKind.TABULAR
    assert lightgbm.origin_scope is OriginScope.MULTIPLE
    assert not lightgbm.context_required


def test_a_contract_cannot_name_the_inference_view() -> None:
    with pytest.raises(SchemaError, match="inference counterpart"):
        TrainingContract(view=ViewKind.FORECAST, origin_scope=OriginScope.SINGLE)


def test_a_series_model_cannot_learn_across_origins() -> None:
    """One time axis cannot hold two vintages, so the contract cannot claim it does.

    This is the declaration-time half of the ``OriginScopeError`` a user meets
    when they hand point-in-time data to AutoARIMA with ``AllOrigins()``.
    """
    with pytest.raises(SchemaError, match="one complete time series"):
        TrainingContract(view=ViewKind.SERIES, origin_scope=OriginScope.MULTIPLE)


def test_a_series_model_binds_no_horizon_at_fit() -> None:
    with pytest.raises(SchemaError, match="carries no horizon"):
        TrainingContract(
            view=ViewKind.SERIES, origin_scope=OriginScope.SINGLE, horizon_bound_at_fit=True
        )


def test_a_series_model_cannot_forecast_an_unseen_instance() -> None:
    with pytest.raises(SchemaError, match="no fitted model"):
        TrainingContract(
            view=ViewKind.SERIES, origin_scope=OriginScope.SINGLE, supports_unseen_instances=True
        )


@pytest.mark.parametrize("view", [ViewKind.SERIES, ViewKind.TABULAR])
def test_only_a_sequence_model_can_require_a_context_length(view: ViewKind) -> None:
    """Neither of the other views sizes a context window, so requiring one is a lie."""
    with pytest.raises(SchemaError, match="sizes no context window"):
        TrainingContract(view=view, origin_scope=OriginScope.SINGLE, context_required=True)


def test_a_contract_round_trips_through_json() -> None:
    contract = TrainingContract.sequences(supports_unseen_instances=True)

    assert TrainingContract.model_validate_json(contract.model_dump_json()) == contract
    assert contract.model_dump()["view"] == "sequences"


# -- capabilities -----------------------------------------------------------


def test_the_default_declaration_is_the_conservative_one() -> None:
    """A capability a provider has is one it states, never one it is assumed to have."""
    capabilities = ModelCapabilities()

    assert capabilities.instances.single and not capabilities.instances.panel
    assert capabilities.targets.univariate and not capabilities.targets.multivariate
    assert not any(
        (capabilities.features.observed, capabilities.features.known, capabilities.features.static)
    )
    assert capabilities.outputs.point
    assert capabilities.missing_values is MissingValueSupport.UNSUPPORTED
    assert not capabilities.tolerates_missing_values
    assert not capabilities.requires_missing_value_transform


def test_instance_and_target_capabilities_answer_about_a_dataset() -> None:
    instances = InstanceCapabilities(single=False, panel=True)
    assert instances.supports(is_panel=True)
    assert not instances.supports(is_panel=False)

    targets = TargetCapabilities(univariate=True, multivariate=False)
    assert targets.supports(1)
    assert not targets.supports(2)


def test_a_target_count_below_one_is_not_a_capability_question() -> None:
    with pytest.raises(SchemaError, match="at least one target"):
        TargetCapabilities().supports(0)


def test_feature_capabilities_name_the_features_a_model_cannot_be_given() -> None:
    """Reported rather than raised: the caller knows which model and which data."""
    capabilities = FeatureCapabilities(known=True, static=True)
    features = (
        FeatureSpec.known("wind_fc"),
        FeatureSpec.observed("temperature_actual"),
        FeatureSpec.static("capacity"),
    )

    assert capabilities.supports(features[0])
    assert not capabilities.supports(features[1])
    assert capabilities.unsupported(features) == ("temperature_actual",)
    assert FeatureCapabilities(observed=True, known=True, static=True).unsupported(features) == ()


def test_a_model_must_support_something_on_every_axis() -> None:
    with pytest.raises(SchemaError, match="single series"):
        InstanceCapabilities(single=False, panel=False)
    with pytest.raises(SchemaError, match="univariate"):
        TargetCapabilities(univariate=False, multivariate=False)
    with pytest.raises(SchemaError, match="point forecasts"):
        OutputCapabilities(point=False, quantiles=False, samples=False)


def test_missing_value_support_is_declared_rather_than_assumed() -> None:
    native = ModelCapabilities(missing_values=MissingValueSupport.NATIVE)
    transform = ModelCapabilities(missing_values=MissingValueSupport.REQUIRES_TRANSFORM)

    assert native.tolerates_missing_values
    assert not native.requires_missing_value_transform
    assert transform.requires_missing_value_transform
    assert not transform.tolerates_missing_values
