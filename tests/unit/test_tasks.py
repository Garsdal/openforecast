"""Origin selection, fit plans and forecast tasks.

The origin selections are the part with real behavior: they are what turns
"train on everything" into a concrete list of vintages, and they have to mean the
same thing whether those vintages are real or simulated. Everything else here is
a declaration, and the tests are mostly about what it refuses to declare.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest
from pydantic import TypeAdapter, ValidationError

import openforecast as of
from openforecast.errors import ProviderError, RecipeError, UnsupportedPlanError
from openforecast.models import OutputCapabilities
from openforecast.tasks import (
    Accelerator,
    OriginMode,
    OriginSelection,
    OutputKind,
    SearchPlan,
    SearchStrategy,
)

START = datetime(2026, 1, 1, 0, 0, 0)
HOUR = timedelta(hours=1)


def at(step: int) -> datetime:
    return START + HOUR * step


ORIGINS = [at(step) for step in range(6)]


# -- origin selection ------------------------------------------------------


def test_every_origin_is_the_default() -> None:
    assert of.AllOrigins().select(ORIGINS) == tuple(ORIGINS)
    assert of.FitPlan().origins == of.AllOrigins()


def test_a_stride_thins_the_origins() -> None:
    assert of.AllOrigins(stride=2).select(ORIGINS) == (at(0), at(2), at(4))


def test_the_latest_origin_is_the_newest_one() -> None:
    assert of.LatestOrigin().select(ORIGINS) == (at(5),)


def test_origins_arrive_in_order_however_they_were_given() -> None:
    """A vintage's position in the source table says nothing about its time."""
    shuffled = [at(3), at(0), at(5), at(0)]

    assert of.AllOrigins().select(shuffled) == (at(0), at(3), at(5))


def test_one_origin_is_matched_exactly() -> None:
    assert of.AtOrigin(at(2)).select(ORIGINS) == (at(2),)


def test_an_origin_that_does_not_exist_is_not_approximated() -> None:
    """Answering for 10:00 when 11:00 was asked for would train on another vintage."""
    with pytest.raises(of.DataError, match=r"no origin"):
        of.AtOrigin(at(2) + timedelta(minutes=30)).select(ORIGINS)


def test_an_origin_can_be_named_as_a_timestamp_string() -> None:
    assert of.AtOrigin("2026-01-01T02:00:00").select(ORIGINS) == (at(2),)


def test_a_range_is_closed_at_both_ends() -> None:
    assert of.OriginsBetween(at(1), at(3)).select(ORIGINS) == (at(1), at(2), at(3))


def test_a_range_can_be_thinned_too() -> None:
    assert of.OriginsBetween(at(0), at(5), stride=2).select(ORIGINS) == (at(0), at(2), at(4))


def test_a_range_that_covers_nothing_is_an_error_rather_than_an_empty_fit() -> None:
    with pytest.raises(of.DataError, match=r"no origin between"):
        of.OriginsBetween(at(40), at(50)).select(ORIGINS)


def test_a_range_cannot_end_before_it_starts() -> None:
    with pytest.raises(RecipeError, match=r"ends before it starts"):
        of.OriginsBetween(at(3), at(1))


def test_a_selection_over_no_origins_says_so() -> None:
    with pytest.raises(of.DataError, match=r"no forecast origins"):
        of.AllOrigins().select([])


def test_a_selection_named_twice_is_not_resolved_by_picking_one() -> None:
    with pytest.raises(RecipeError, match=r"both positionally and by keyword"):
        of.AtOrigin(at(1), origin=at(2))


def test_a_selection_cannot_carry_a_field_that_means_nothing_to_it() -> None:
    """A stride on the latest origin, a range on a single one: not writable."""
    with pytest.raises(ValidationError):
        of.LatestOrigin(stride=2)  # pyright: ignore[reportCallIssue]
    with pytest.raises(ValidationError):
        of.AtOrigin(at(1), end=at(3))


SELECTIONS: list[OriginSelection] = [
    of.AllOrigins(stride=12),
    of.LatestOrigin(),
    of.AtOrigin(at(1)),
    of.OriginsBetween(at(1), at(3), stride=2),
]


@pytest.mark.parametrize("selection", SELECTIONS, ids=lambda selection: str(selection.mode))
def test_a_selection_round_trips_through_json(selection: OriginSelection) -> None:
    adapter: TypeAdapter[OriginSelection] = TypeAdapter(OriginSelection)
    restored = adapter.validate_json(adapter.dump_json(selection))

    assert restored == selection
    assert type(restored) is type(selection)


def test_the_wire_form_is_tagged_by_mode() -> None:
    payload = json.loads(of.AllOrigins(stride=12).model_dump_json())

    assert payload == {"mode": OriginMode.ALL, "stride": 12}


# -- window and fit plan ---------------------------------------------------


def test_a_window_sizes_the_context_of_a_sample() -> None:
    plan = of.FitPlan(window=of.WindowPlan(context=168))

    assert plan.context == 168


def test_a_plan_without_a_window_sizes_nothing() -> None:
    assert of.FitPlan().context is None


def test_a_context_of_zero_steps_is_no_context() -> None:
    with pytest.raises(ValidationError):
        of.WindowPlan(context=0)


def test_a_plan_records_how_reproducible_it_is() -> None:
    plan = of.FitPlan(seed=42, resources=of.Resources(accelerator=Accelerator.GPU, devices=2))

    assert plan.seed == 42
    assert plan.resources.accelerator is Accelerator.GPU
    assert plan.resources.devices == 2


def test_a_cpu_fit_has_no_devices_to_allocate() -> None:
    with pytest.raises(RecipeError, match=r"no devices to allocate"):
        of.Resources(accelerator=Accelerator.CPU, devices=4)


def test_hyperparameter_search_is_reserved_and_refused_loudly() -> None:
    """Accepting it and fitting once would look exactly like a search that ran."""
    search = SearchPlan(strategy=SearchStrategy.GRID, space={"max_steps": [100, 500]})

    with pytest.raises(UnsupportedPlanError, match=r"not implemented yet"):
        of.FitPlan(search=search)


def test_a_reserved_search_plan_is_still_a_checked_declaration() -> None:
    assert SearchPlan(strategy=SearchStrategy.RANDOM, space={"lr": [0.1]}, trials=8).strategy is (
        SearchStrategy.RANDOM
    )
    with pytest.raises(RecipeError, match=r"how many trials"):
        SearchPlan(strategy=SearchStrategy.RANDOM, space={"lr": [0.1]})
    with pytest.raises(RecipeError, match=r"exhaustive"):
        SearchPlan(strategy=SearchStrategy.GRID, space={"lr": [0.1]}, trials=8)
    with pytest.raises(RecipeError, match=r"at least one parameter"):
        SearchPlan(strategy=SearchStrategy.GRID, space={})
    with pytest.raises(RecipeError, match=r"no candidate values"):
        SearchPlan(strategy=SearchStrategy.GRID, space={"lr": []})


def test_a_plan_round_trips_through_json() -> None:
    plan = of.FitPlan(
        origins=of.OriginsBetween(at(1), at(4), stride=2),
        window=of.WindowPlan(context=168),
        seed=7,
        resources=of.Resources(accelerator=Accelerator.GPU),
    )

    assert of.FitPlan.model_validate_json(plan.model_dump_json()) == plan


def test_a_plan_is_frozen_so_the_manifest_matches_what_was_fitted() -> None:
    with pytest.raises(ValidationError):
        of.FitPlan().seed = 1


# -- forecast task and output ----------------------------------------------


def test_a_task_counts_steps_of_the_data_frequency() -> None:
    assert of.ForecastTask(24).horizon == of.ForecastTask(horizon=24).horizon == 24


def test_a_horizon_named_twice_is_not_resolved_by_picking_one() -> None:
    with pytest.raises(RecipeError, match=r"both positionally and by keyword"):
        of.ForecastTask(24, horizon=48)


def test_a_task_forecasts_at_least_one_step() -> None:
    with pytest.raises(ValidationError):
        of.ForecastTask(0)


def test_a_task_does_not_name_an_origin() -> None:
    """At fit time the plan holds the origins; at inference the context is one."""
    with pytest.raises(ValidationError):
        of.ForecastTask(24, origin=at(1))


def test_the_three_output_kinds() -> None:
    assert of.OutputSpec.point() == of.OutputSpec()
    assert of.OutputSpec.quantiles([0.1, 0.5, 0.9]).levels == (0.1, 0.5, 0.9)
    assert of.OutputSpec.samples(100).draws == 100
    assert of.OutputSpec.point().is_probabilistic is False
    assert of.OutputSpec.samples(100).is_probabilistic is True


@pytest.mark.parametrize(
    ("levels", "message"),
    [
        ([0.0, 0.5], r"strictly between 0 and 1"),
        ([0.5, 1.0], r"strictly between 0 and 1"),
        ([0.5, 0.5], r"duplicate quantile levels"),
        ([0.9, 0.1], r"must be ascending"),
        ([], r"must name the levels"),
    ],
)
def test_quantile_levels_are_checked(levels: list[float], message: str) -> None:
    with pytest.raises(RecipeError, match=message):
        of.OutputSpec.quantiles(levels)


def test_a_kind_cannot_carry_another_kind_fields() -> None:
    with pytest.raises(RecipeError, match=r"does not take quantile levels"):
        of.OutputSpec(kind=OutputKind.POINT, levels=(0.5,))
    with pytest.raises(RecipeError, match=r"does not take a draw count"):
        of.OutputSpec(kind=OutputKind.POINT, draws=10)
    with pytest.raises(RecipeError, match=r"how many draws"):
        of.OutputSpec(kind=OutputKind.SAMPLES)


def test_an_output_request_is_checked_against_what_a_model_declares() -> None:
    """Quantiles are not downgraded to a point forecast, nor derived unasked."""
    point_only = OutputCapabilities()
    probabilistic = OutputCapabilities(point=True, quantiles=True, samples=True)

    assert of.OutputSpec.point().is_supported_by(point_only)
    assert not of.OutputSpec.quantiles([0.5]).is_supported_by(point_only)
    assert not of.OutputSpec.samples(10).is_supported_by(point_only)
    assert of.OutputSpec.quantiles([0.5]).is_supported_by(probabilistic)
    assert of.OutputSpec.samples(10).is_supported_by(probabilistic)


# -- quantiles of samples ---------------------------------------------------


def test_a_model_that_only_draws_samples_can_be_asked_for_quantiles_of_them() -> None:
    """The one safe conversion of Step 20, and it has to be asked for."""
    draws_only = OutputCapabilities(point=False, quantiles=False, samples=True)

    assert not of.OutputSpec.quantiles([0.1, 0.9]).is_supported_by(draws_only)
    assert of.OutputSpec.quantiles([0.1, 0.9], from_samples=200).is_supported_by(draws_only)


def test_quantiles_of_samples_are_executed_as_a_sample_forecast() -> None:
    """The provider draws; OpenForecast reduces, with one estimator for all of them."""
    spec = of.OutputSpec.quantiles([0.1, 0.9], from_samples=200)

    assert spec.derived_from_samples
    assert spec.as_executed() == of.OutputSpec.samples(200)
    assert of.OutputSpec.quantiles([0.1, 0.9]).as_executed() == of.OutputSpec.quantiles([0.1, 0.9])
    assert not of.OutputSpec.quantiles([0.1, 0.9]).derived_from_samples


def test_the_row_kind_of_a_request_is_singular_and_round_trips() -> None:
    """One row of a forecast is one number, so a request for quantiles holds quantile rows."""
    for kind in OutputKind:
        assert OutputKind.of_row(kind.row_kind) is kind
    assert OutputKind.QUANTILES.row_kind == "quantile"
    with pytest.raises(ProviderError, match=r"not a kind a forecast row can hold"):
        OutputKind.of_row("quantiles")


def test_an_output_spec_round_trips_through_json() -> None:
    spec = of.OutputSpec.quantiles([0.1, 0.9])

    assert of.OutputSpec.model_validate_json(spec.model_dump_json()) == spec
