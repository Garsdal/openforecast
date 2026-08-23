"""What a ``ForecastView`` becomes, and what the answer is labeled with.

The conformance suite checks the shape of an answer. It cannot check that the
right *numbers* went in, because a stand-in pipeline that read the covariate
column as the target would answer the right shape from the wrong input and pass
every generated case. This is the file that pins that down.
"""

from __future__ import annotations

import math

import golden
import pyarrow as pa
import pytest
from openforecast_chronos import conversion

from openforecast.errors import ProviderError
from openforecast.protocol import ForecastColumn
from openforecast.views import forecast_columns

LEVELS = (0.1, 0.5, 0.9)


def column(table: object, name: str) -> list[object]:
    values: list[object] = table.column(name).to_pylist()  # pyright: ignore[reportAttributeAccessIssue]
    return values


# -- what the model is handed ------------------------------------------------


def test_a_single_series_becomes_one_input_holding_its_context() -> None:
    view = golden.view()
    (only,) = conversion.inputs_for(view, "load")

    assert only.target.tolist() == [float(step) for step in range(golden.HISTORY)]
    assert only.past_covariates == {}
    assert only.future_covariates == {}
    assert only.as_mapping() == {"target": pytest.approx(only.target)}


def test_a_panel_becomes_one_input_per_instance_in_the_views_order() -> None:
    view = golden.view(instances=("DE", "FR", "NL"))
    inputs = conversion.inputs_for(view, "load")

    assert len(inputs) == len(view.instances)
    assert [entry.target[0] for entry in inputs] == [0.0, 100.0, 200.0]
    # Each context is the whole history of *that* instance, not of the table.
    assert all(entry.target.shape[0] == golden.HISTORY for entry in inputs)


def test_an_observed_feature_is_a_past_covariate_and_nothing_more() -> None:
    """It has no value past the origin, so there is nothing to put in the future slot."""
    view = golden.view(observed=True)
    (only,) = conversion.inputs_for(view, "load")

    assert set(only.past_covariates) == {"temp"}
    assert only.future_covariates == {}
    assert only.past_covariates["temp"].tolist() == [step / 2 for step in range(golden.HISTORY)]


def test_a_known_feature_is_in_both_slots() -> None:
    """Which is the requirement: a future covariate must also be a past one."""
    view = golden.view(known=True)
    (only,) = conversion.inputs_for(view, "load")

    assert set(only.past_covariates) == {"temp_fc"}
    assert set(only.future_covariates) == {"temp_fc"}
    assert only.past_covariates["temp_fc"].shape[0] == golden.HISTORY
    assert only.future_covariates["temp_fc"].shape[0] == golden.HORIZON
    assert only.future_covariates["temp_fc"].tolist() == [
        (golden.HISTORY + step) / 4 for step in range(golden.HORIZON)
    ]


def test_both_roles_travel_together() -> None:
    view = golden.view(observed=True, known=True)
    (only,) = conversion.inputs_for(view, "load")

    assert set(only.past_covariates) == {"temp", "temp_fc"}
    assert set(only.future_covariates) == {"temp_fc"}
    assert set(only.as_mapping()) == {"target", "past_covariates", "future_covariates"}


def test_a_history_in_any_row_order_is_read_in_time_order() -> None:
    """A context is a sequence, and a transport does not decide what order it is in."""
    view = golden.view()
    shuffled = view.history.take([3, 0, 5, 1, 4, 2])
    scrambled = type(view)(
        origin_time=view.origin_time,
        history=shuffled,
        future=view.future,
        metadata=view.metadata,
    )

    (only,) = conversion.inputs_for(scrambled, "load")
    assert only.target.tolist() == [float(step) for step in range(golden.HISTORY)]


def test_a_second_target_is_refused_by_name() -> None:
    with pytest.raises(ProviderError, match="one target at a time and was given 2"):
        conversion.single_target(("load", "wind"))


# -- what comes back ---------------------------------------------------------


def _answer(view: object, levels: tuple[float | None, ...], value: float = 1.0) -> object:
    instances = len(view.instances)  # pyright: ignore[reportAttributeAccessIssue]
    steps = len(view.event_times)  # pyright: ignore[reportAttributeAccessIssue]
    predicted = [
        [[value + position for _ in levels] for _ in range(steps)] for position in range(instances)
    ]
    return conversion.answer(view, predicted, target="load", levels=list(levels))


def test_a_point_answer_is_one_row_per_instance_and_event_time() -> None:
    view = golden.view(instances=("DE", "FR"))
    table = _answer(view, (None,))

    assert table.column_names == list(forecast_columns(("zone",)))
    assert table.num_rows == 2 * golden.HORIZON
    assert set(column(table, ForecastColumn.KIND.value)) == {"point"}
    assert all(level is None for level in column(table, ForecastColumn.QUANTILE.value))
    assert set(column(table, ForecastColumn.TARGET.value)) == {"load"}
    assert set(column(table, "zone")) == {"DE", "FR"}


def test_a_quantile_answer_is_one_row_per_level() -> None:
    view = golden.view()
    table = _answer(view, LEVELS)

    assert table.num_rows == golden.HORIZON * len(LEVELS)
    assert set(column(table, ForecastColumn.KIND.value)) == {"quantile"}
    assert sorted(set(column(table, ForecastColumn.QUANTILE.value))) == list(LEVELS)
    assert all(draw is None for draw in column(table, ForecastColumn.SAMPLE.value))


def test_the_event_times_are_the_ones_the_view_asked_about() -> None:
    view = golden.view()
    table = _answer(view, (None,))

    assert column(table, ForecastColumn.EVENT_TIME.value) == list(view.event_times)


def test_a_short_answer_is_a_provider_that_did_not_answer() -> None:
    view = golden.view()
    predicted = [[[1.0]] * (golden.HORIZON - 1)]

    with pytest.raises(ProviderError, match="answered 2 steps"):
        conversion.answer(view, predicted, target="load", levels=[None])


def test_an_answer_about_the_wrong_number_of_instances_is_refused() -> None:
    view = golden.view(instances=("DE", "FR"))
    predicted = [[[1.0]] * golden.HORIZON]

    with pytest.raises(ProviderError, match="asked about 2 instances and answered 1"):
        conversion.answer(view, predicted, target="load", levels=[None])


def test_an_answer_with_the_wrong_number_of_levels_is_refused() -> None:
    view = golden.view()
    predicted = [[[1.0, 2.0]] * golden.HORIZON]

    with pytest.raises(ProviderError, match="answered 2 values per step"):
        conversion.answer(view, predicted, target="load", levels=[0.1, 0.5, 0.9])


# -- missing values ----------------------------------------------------------


def test_a_missing_observation_reaches_the_model_as_a_nan() -> None:
    """Nothing is filled in: Chronos reads a NaN as an unobserved step."""
    view = golden.view()
    values = view.history.column("load").to_pylist()
    values[2] = None
    patched = view.history.set_column(
        view.history.column_names.index("load"), "load", pa.array(values, type=pa.float64())
    )
    holed = type(view)(
        origin_time=view.origin_time,
        history=patched,
        future=view.future,
        metadata=view.metadata,
    )

    (only,) = conversion.inputs_for(holed, "load")
    assert math.isnan(only.target[2])
    assert not any(math.isnan(only.target[step]) for step in (0, 1, 3, 4, 5))
