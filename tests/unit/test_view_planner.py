"""Materialization: the same view types out of both semantic sources.

The leakage assertions here are the point of the whole package. A poisoned
feature value in a later vintage must be unreachable from an earlier origin, and
an observed feature must never carry a value for an event time that had not
happened yet at the origin being materialized.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

import openforecast as of
from openforecast import (
    DataError,
    ForecastContext,
    ForecastDataset,
    OriginScopeError,
    RecipeError,
    SchemaError,
    TimeSeriesFrame,
)
from openforecast.models import TrainingContract
from openforecast.views import (
    AllOrigins,
    AtOrigin,
    LatestOrigin,
    OriginFidelity,
    OriginsBetween,
    SequenceView,
    SeriesView,
    SourceKind,
    TabularView,
    ViewKind,
    ViewPlanner,
    ViewRequest,
)
from tests import factories

START = datetime(2026, 1, 1, 0, 0, 0)
HOUR = timedelta(hours=1)

#: A value no real feed would produce, so a leaked vintage is unmistakable.
POISON = 999999.0

planner = ViewPlanner()


def at(step: int) -> datetime:
    return START + HOUR * step


def missing(value: float | None) -> bool:
    """Null and ``NaN`` are the two spellings of "no value here"."""
    return value is None or math.isnan(value)


def frame(**overrides: Any) -> TimeSeriesFrame:
    """A two-instance panel with one target, one observed and one known feature."""
    options: dict[str, Any] = {
        "instances": ("DE", "FR"),
        "instance_key": "zone",
        "periods": 8,
        "observed": ["temp"],
        "known": ["temp_fc"],
    }
    options.update(overrides)
    return TimeSeriesFrame.from_pandas(
        history=factories.history(**options),
        time="timestamp",
        frequency="1h",
        instance_keys=["zone"],
        targets=["load"],
        observed_features=["temp"],
        known_features=["temp_fc"],
    )


def dataset(rows: list[dict[str, Any]], **overrides: Any) -> ForecastDataset:
    options: dict[str, Any] = {
        "origin_time": "ref_time",
        "event_time": "target_time",
        "instance_keys": ["zone"],
        "targets": ["price"],
        "known_features": ["wind_fc"],
        "event_frequency": "1h",
        "origin_frequency": "1h",
    }
    options.update(overrides)
    return ForecastDataset.from_pandas(pd.DataFrame(rows), **options)


def row(
    origin: int, event: int, *, price: float | None = None, wind: float = 10.0
) -> dict[str, Any]:
    return {
        "zone": "DE",
        "ref_time": at(origin),
        "target_time": at(event),
        "price": float(event) if price is None else price,
        "wind_fc": wind,
    }


def vintages(*, origins: int, span: int, wind: Any = None) -> list[dict[str, Any]]:
    """Every origin describes every event time, so the two sources are comparable."""
    return [
        row(origin, event, wind=float(event) if wind is None else wind)
        for origin in range(origins)
        for event in range(span)
    ]


def series_view(data: object, **request: Any) -> SeriesView:
    view = planner.fit_view(data, ViewRequest(kind=ViewKind.SERIES, **request))
    assert isinstance(view, SeriesView)
    return view


def sequence_view(data: object, **request: Any) -> SequenceView:
    view = planner.fit_view(data, ViewRequest(kind=ViewKind.SEQUENCES, **request))
    assert isinstance(view, SequenceView)
    return view


def tabular_view(data: object, **request: Any) -> TabularView:
    view = planner.fit_view(data, ViewRequest(kind=ViewKind.TABULAR, **request))
    assert isinstance(view, TabularView)
    return view


# -- TimeSeriesFrame -------------------------------------------------------


def test_a_time_series_frame_becomes_one_series_per_instance() -> None:
    view = series_view(frame())

    assert len(view.series_ids) == 2
    assert view.temporal.num_rows == 16
    assert view.series.column("zone").to_pylist() == ["DE", "FR"]
    assert view.schema.origin_time is None
    assert view.provenance.origin_fidelity is OriginFidelity.SIMULATED
    assert view.provenance.source is SourceKind.TIME_SERIES


def test_a_time_series_frame_becomes_simulated_sequences() -> None:
    view = sequence_view(frame(), context=3, horizon=2)

    # Origins 2..5 of 0..7 are the ones with three context and two forecast steps.
    assert view.origins == (at(2), at(3), at(4), at(5))
    assert len(view.sample_ids) == 8
    assert view.temporal.num_rows == 8 * 5
    assert view.provenance.origin_fidelity is OriginFidelity.SIMULATED


def test_a_simulated_origin_does_not_hand_over_observations_it_did_not_have() -> None:
    """The forecast half of a window is the future: an observed feature is blank there."""
    view = sequence_view(frame(), context=3, horizon=2)
    first = view.temporal.slice(0, 5).to_pydict()

    assert first["temp"][:3] == [0.0, 0.5, 1.0]
    assert first["temp"][3:] == [None, None]
    assert None not in first["temp_fc"]


def test_a_time_series_frame_becomes_supervised_rows() -> None:
    view = tabular_view(frame(), horizon=2)

    assert view.X.column_names == ["temp_fc"]
    assert view.y.column_names == ["load"]
    assert set(view.keys.column("horizon_step").to_pylist()) == {1, 2}
    # Origins 0..6 reach one step ahead, 0..5 reach two: 13 rows per instance.
    assert view.num_rows == 26


def test_the_static_features_of_a_panel_travel_with_the_training_unit() -> None:
    with_static = TimeSeriesFrame.from_pandas(
        history=factories.history(
            instances=("DE", "FR"),
            instance_key="zone",
            periods=8,
            known=["temp_fc"],
            static={"capacity": {"DE": 1.0, "FR": 2.0}},
        ),
        time="timestamp",
        frequency="1h",
        instance_keys=["zone"],
        targets=["load"],
        known_features=["temp_fc"],
        static_features=["capacity"],
    )
    view = sequence_view(with_static, context=2, horizon=2)
    assert view.static is not None
    assert view.static.column_names == ["sample_id", "capacity"]
    assert view.static.num_rows == len(view.sample_ids)
    assert set(view.static.column("capacity").to_pylist()) == {1.0, 2.0}


def test_a_window_longer_than_the_data_is_an_error_rather_than_an_empty_view() -> None:
    with pytest.raises(DataError, match=r"no origin has 20 context steps"):
        sequence_view(frame(), context=20, horizon=2)


# -- ForecastDataset -------------------------------------------------------


def test_a_point_in_time_dataset_becomes_observed_sequences() -> None:
    view = sequence_view(dataset(vintages(origins=6, span=6)), context=2, horizon=2)
    assert view.provenance.origin_fidelity is OriginFidelity.OBSERVED
    assert view.provenance.source is SourceKind.FORECAST_DATASET
    assert view.origins == (at(1), at(2), at(3))


def test_one_instance_and_origin_is_one_sample() -> None:
    """Step 10's arithmetic: 3 instances over the eligible origins."""
    rows = [
        {**row(origin, event), "zone": zone}
        for zone in ("DE", "FR", "NL")
        for origin in range(5)
        for event in range(5)
    ]
    view = sequence_view(dataset(rows), context=1, horizon=1)
    # Origins 0..3 can carry one context and one forecast step; 4 cannot.
    assert view.origins == (at(0), at(1), at(2), at(3))
    assert len(view.sample_ids) == 3 * 4
    assert view.samples.num_rows == 3 * 4


def test_a_later_vintage_cannot_reach_an_earlier_origin() -> None:
    """The leakage sentinel: origin 2 poisons event 3, and origin 1 must not see it."""
    rows = [
        *[row(0, event, wind=10.0) for event in range(4)],
        *[row(1, event, wind=20.0) for event in range(4)],
        *[row(2, event, wind=POISON) for event in range(4)],
    ]
    view = sequence_view(dataset(rows), context=2, horizon=1, origins=AtOrigin(at(1)))
    values = view.temporal.column("wind_fc").to_pylist()

    assert 20.0 in values
    assert POISON not in values


def test_each_sample_reads_the_features_of_its_own_origin() -> None:
    rows = [
        *[row(0, event, wind=100.0 + event) for event in range(4)],
        *[row(1, event, wind=200.0 + event) for event in range(4)],
        *[row(2, event, wind=300.0 + event) for event in range(4)],
    ]
    view = tabular_view(dataset(rows), horizon=1)
    by_origin = dict(
        zip(
            view.keys.column("origin_time").to_pylist(),
            view.X.column("wind_fc").to_pylist(),
            strict=True,
        )
    )
    assert by_origin == {at(0): 101.0, at(1): 202.0, at(2): 303.0}


def test_missingness_survives_materialization() -> None:
    """An availability that improves between vintages is information, not noise.

    Event 4 is unknown at origins 1 and 2 and known at origin 3. All three rows
    survive, spelled exactly as the source spelled them.
    """

    def wind(origin: int, event: int) -> float:
        if event != 4:
            return 10.0
        return 42.0 if origin == 3 else math.nan

    rows = [
        row(origin, event, wind=wind(origin, event))
        for origin in (1, 2, 3)
        for event in range(origin + 1, origin + 4)
    ]
    view = tabular_view(dataset(rows), horizon=3)
    values = {
        origin: value
        for origin, event, value in zip(
            view.keys.column("origin_time").to_pylist(),
            view.keys.column("event_time").to_pylist(),
            view.X.column("wind_fc").to_pylist(),
            strict=True,
        )
        if event == at(4)
    }
    assert missing(values[at(1)])
    assert missing(values[at(2)])
    assert values[at(3)] == 42.0


def test_a_series_view_of_point_in_time_data_needs_one_origin() -> None:
    built = dataset(vintages(origins=4, span=4))
    with pytest.raises(OriginScopeError, match=r"one forecast origin"):
        series_view(built)


def test_a_series_view_of_one_selected_origin_is_that_vintage() -> None:
    built = dataset(vintages(origins=4, span=4))
    view = series_view(built, origins=AtOrigin(at(2)))
    assert view.schema.origin_time == at(2)
    # The series stops at its origin: a series carries no future.
    assert view.temporal.column("event_time").to_pylist() == [at(0), at(1), at(2)]
    assert view.provenance.origin_fidelity is OriginFidelity.OBSERVED


# -- event-time and point-in-time equivalence -------------------------------


def equivalent_sources() -> tuple[TimeSeriesFrame, ForecastDataset]:
    """The same numbers, once as a plain series and once as identical vintages.

    Only known features are compared, so that both sources really do describe
    the same value at every event time: this is the case where the two views can
    only differ in provenance.
    """
    rows = vintages(origins=6, span=6)
    built = dataset(rows)
    history = pd.DataFrame(
        [
            {
                "zone": "DE",
                "target_time": at(event),
                "price": float(event),
                "wind_fc": float(event),
            }
            for event in range(6)
        ]
    )
    plain = TimeSeriesFrame.from_pandas(
        history=history,
        time="target_time",
        frequency="1h",
        instance_keys=["zone"],
        targets=["price"],
        known_features=["wind_fc"],
    )
    return plain, built


def test_identical_vintages_materialize_into_identical_sequences() -> None:
    plain, built = equivalent_sources()

    from_event_time = sequence_view(plain, context=2, horizon=2)
    from_vintages = sequence_view(built, context=2, horizon=2)

    assert from_event_time.schema == from_vintages.schema
    assert from_event_time.temporal.equals(from_vintages.temporal)
    assert from_event_time.samples.equals(from_vintages.samples)
    # Everything matches except where the origins came from.
    assert from_event_time != from_vintages
    assert from_event_time.provenance.origin_fidelity is OriginFidelity.SIMULATED
    assert from_vintages.provenance.origin_fidelity is OriginFidelity.OBSERVED


def test_materialization_is_deterministic() -> None:
    plain, _ = equivalent_sources()
    assert sequence_view(plain, context=2, horizon=2) == sequence_view(plain, context=2, horizon=2)


# -- origin selection ------------------------------------------------------


def test_a_stride_thins_the_origins() -> None:
    built = dataset(vintages(origins=6, span=6))
    view = sequence_view(built, context=1, horizon=1, origins=AllOrigins(stride=2))
    assert view.origins == (at(0), at(2), at(4))


def test_the_latest_origin_is_the_newest_vintage() -> None:
    built = dataset(vintages(origins=6, span=7))
    view = sequence_view(built, context=1, horizon=1, origins=LatestOrigin())
    assert view.origins == (at(5),)


def test_origins_between_two_moments_are_bounded_at_both_ends() -> None:
    built = dataset(vintages(origins=6, span=6))
    view = sequence_view(built, context=1, horizon=1, origins=OriginsBetween(at(1), at(3)))
    assert view.origins == (at(1), at(2), at(3))


def test_an_origin_that_does_not_exist_is_not_approximated() -> None:
    built = dataset(vintages(origins=3, span=3))
    with pytest.raises(DataError, match=r"no origin"):
        sequence_view(built, context=1, horizon=1, origins=AtOrigin(at(90)))


def test_a_range_with_no_origin_in_it_is_an_error() -> None:
    built = dataset(vintages(origins=3, span=3))
    with pytest.raises(DataError, match=r"no origin between"):
        sequence_view(built, context=1, horizon=1, origins=OriginsBetween(at(40), at(50)))


def test_a_selection_cannot_mix_its_modes() -> None:
    """Each selection is its own type, so a mixed one cannot be written at all."""
    with pytest.raises(ValidationError, match=r"start"):
        AtOrigin(at(1), start=at(0))


# -- the forecast view -----------------------------------------------------


def context_at(origin: int) -> ForecastContext:
    return dataset(vintages(origins=6, span=8)).at_origin(at(origin))


def test_a_forecast_view_trims_the_history_to_the_trained_context() -> None:
    view = planner.forecast_view(
        context_at(4), ViewRequest(kind=ViewKind.FORECAST, horizon=2, context=3)
    )
    assert view.origin_time == at(4)
    assert view.history.column("event_time").to_pylist() == [at(2), at(3), at(4)]
    assert view.event_times == (at(5), at(6))
    assert view.metadata.context == 3


def test_a_forecast_view_keeps_the_whole_history_when_no_context_is_bound() -> None:
    view = planner.forecast_view(context_at(4), ViewRequest(kind=ViewKind.FORECAST, horizon=2))
    assert view.history.num_rows == 5
    assert view.metadata.context is None


def test_a_history_shorter_than_the_trained_context_is_an_error() -> None:
    with pytest.raises(DataError, match=r"steps short"):
        planner.forecast_view(
            context_at(1), ViewRequest(kind=ViewKind.FORECAST, horizon=2, context=5)
        )


def test_the_event_time_axis_is_renamed_to_the_view_vocabulary() -> None:
    view = planner.forecast_view(context_at(3), ViewRequest(kind=ViewKind.FORECAST, horizon=1))
    assert "target_time" not in view.history.column_names
    assert "event_time" in view.history.column_names


def test_a_horizon_the_context_says_nothing_about_is_still_asked_about() -> None:
    """Beyond the vintage's own reach the known features are missing, not invented."""
    view = planner.forecast_view(context_at(5), ViewRequest(kind=ViewKind.FORECAST, horizon=4))
    values = view.future.column("wind_fc").to_pylist()
    assert view.event_times == (at(6), at(7), at(8), at(9))
    assert values[:2] == [6.0, 7.0]
    assert values[2:] == [None, None]


# -- routing and validation ------------------------------------------------


def test_a_forecast_context_is_not_a_training_dataset() -> None:
    with pytest.raises(DataError, match=r"cannot build an execution view"):
        series_view(context_at(3))


def test_a_source_dataset_is_not_an_inference_context() -> None:
    with pytest.raises(DataError, match=r"materialized from a ForecastContext"):
        planner.forecast_view(
            frame(),  # pyright: ignore[reportArgumentType]
            ViewRequest(kind=ViewKind.FORECAST, horizon=2),
        )


def test_fit_view_does_not_build_forecast_views() -> None:
    with pytest.raises(SchemaError, match=r"materialized by forecast_view"):
        planner.fit_view(frame(), ViewRequest(kind=ViewKind.FORECAST, horizon=2))


def test_a_sequence_request_without_a_context_length_is_incomplete() -> None:
    with pytest.raises(SchemaError, match=r"needs a context length"):
        ViewRequest(kind=ViewKind.SEQUENCES, horizon=2)


def test_a_tabular_request_without_a_horizon_is_incomplete() -> None:
    with pytest.raises(SchemaError, match=r"needs a horizon"):
        ViewRequest(kind=ViewKind.TABULAR)


def test_a_series_request_binds_neither_context_nor_horizon() -> None:
    with pytest.raises(SchemaError, match=r"binds neither"):
        ViewRequest(kind=ViewKind.SERIES, horizon=2)


def test_a_tabular_request_binds_no_context_length() -> None:
    with pytest.raises(SchemaError, match=r"binds no context length"):
        ViewRequest(kind=ViewKind.TABULAR, horizon=2, context=4)


# -- what a contract, a plan and a task jointly ask for --------------------


def test_a_contract_and_a_plan_become_one_request() -> None:
    """The translation Step 8's engine performs, so the engine can stay trivial."""
    request = ViewRequest.for_contract(
        TrainingContract.sequences(),
        plan=of.FitPlan(origins=of.LatestOrigin(), window=of.WindowPlan(context=168)),
        task=of.ForecastTask(72),
    )

    assert request == ViewRequest(
        kind=ViewKind.SEQUENCES,
        context=168,
        horizon=72,
        origins=of.LatestOrigin(),
    )


def test_a_series_contract_asks_for_neither_context_nor_horizon() -> None:
    request = ViewRequest.for_contract(
        TrainingContract.series(),
        plan=of.FitPlan(origins=of.AtOrigin(at(2))),
        task=of.ForecastTask(24),
    )

    assert request == ViewRequest(kind=ViewKind.SERIES, origins=of.AtOrigin(at(2)))


def test_a_window_handed_to_a_series_model_is_refused_rather_than_dropped() -> None:
    """It was written by someone expecting it to have an effect."""
    with pytest.raises(RecipeError, match=r"sizes no context window"):
        ViewRequest.for_contract(
            TrainingContract.series(),
            plan=of.FitPlan(window=of.WindowPlan(context=168)),
        )


def test_a_sequence_model_is_not_given_a_default_context_length() -> None:
    with pytest.raises(RecipeError, match=r"cannot be\s+given a default context length"):
        ViewRequest.for_contract(TrainingContract.sequences(), task=of.ForecastTask(24))


def test_a_window_handed_to_a_tabular_model_is_refused() -> None:
    with pytest.raises(RecipeError, match=r"binds no context length"):
        ViewRequest.for_contract(
            TrainingContract.tabular(),
            plan=of.FitPlan(window=of.WindowPlan(context=168)),
            task=of.ForecastTask(24),
        )


def test_a_bounded_view_needs_a_task_to_bound_it() -> None:
    with pytest.raises(RecipeError, match=r"needs a horizon"):
        ViewRequest.for_contract(TrainingContract.tabular())


def test_a_plan_defaults_to_every_origin() -> None:
    request = ViewRequest.for_contract(TrainingContract.tabular(), task=of.ForecastTask(24))

    assert request.origins == of.AllOrigins()


def test_the_request_a_contract_produces_materializes() -> None:
    """The translation is only useful if the planner accepts what it produces."""
    built = dataset(vintages(origins=4, span=4))
    request = ViewRequest.for_contract(
        TrainingContract.sequences(),
        plan=of.FitPlan(origins=of.AllOrigins(), window=of.WindowPlan(context=1)),
        task=of.ForecastTask(1),
    )

    view = planner.fit_view(built, request)

    assert isinstance(view, SequenceView)
    assert view.provenance.origin_fidelity is OriginFidelity.OBSERVED
