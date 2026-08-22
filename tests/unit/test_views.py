"""The execution view types themselves: what they accept and what they refuse.

The planner is tested separately. Here the tables are built by hand, so a view
that would silently accept a malformed sample shows up as a missing error rather
than as a strange model.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pyarrow as pa
import pytest

from openforecast import DataError, FeatureSpec, SchemaError
from openforecast.views import (
    ForecastView,
    ForecastViewMetadata,
    OriginFidelity,
    SequenceView,
    SequenceViewSchema,
    SeriesView,
    SeriesViewSchema,
    SourceKind,
    TabularView,
    TabularViewSchema,
    ViewKind,
    ViewProvenance,
)
from openforecast.views.base import opaque_id

START = datetime(2026, 1, 1, 0, 0, 0)
HOUR = timedelta(hours=1)

SIMULATED = ViewProvenance(source=SourceKind.TIME_SERIES, origin_fidelity=OriginFidelity.SIMULATED)
OBSERVED = ViewProvenance(
    source=SourceKind.FORECAST_DATASET, origin_fidelity=OriginFidelity.OBSERVED
)


def moments(count: int, *, after: int = 0) -> list[datetime]:
    return [START + HOUR * (after + index) for index in range(count)]


def table(**columns: list[Any]) -> pa.Table:
    return pa.table(columns)


# -- SeriesView ------------------------------------------------------------


def series_schema(**overrides: Any) -> SeriesViewSchema:
    options: dict[str, Any] = {
        "frequency": "1h",
        "targets": ("load",),
        "instance_keys": ("zone",),
    }
    options.update(overrides)
    return SeriesViewSchema(**options)


def test_a_series_view_holds_one_table_per_instance() -> None:
    view = SeriesView(
        temporal=table(series_id=["a", "a", "a"], event_time=moments(3), load=[1.0, 2.0, 3.0]),
        series=table(series_id=["a"], zone=["DE"]),
        schema=series_schema(),
        provenance=SIMULATED,
    )
    assert view.kind is ViewKind.SERIES
    assert view.series_ids == ("a",)
    assert view.temporal.column_names == ["series_id", "event_time", "load"]
    assert view.static is None
    assert "simulated" in repr(view)


def test_a_series_id_with_no_key_row_cannot_be_mapped_back() -> None:
    with pytest.raises(DataError, match=r"absent from series"):
        SeriesView(
            temporal=table(series_id=["a", "b"], event_time=moments(2), load=[1.0, 2.0]),
            series=table(series_id=["a"], zone=["DE"]),
            schema=series_schema(),
            provenance=SIMULATED,
        )


def test_a_key_row_with_no_data_is_an_announced_series_that_was_never_built() -> None:
    with pytest.raises(DataError, match=r"absent from temporal"):
        SeriesView(
            temporal=table(series_id=["a"], event_time=moments(1), load=[1.0]),
            series=table(series_id=["a", "b"], zone=["DE", "FR"]),
            schema=series_schema(),
            provenance=SIMULATED,
        )


def test_declared_static_features_need_a_static_table() -> None:
    with pytest.raises(DataError, match=r"no static table"):
        SeriesView(
            temporal=table(series_id=["a"], event_time=moments(1), load=[1.0]),
            series=table(series_id=["a"], zone=["DE"]),
            schema=series_schema(features=(FeatureSpec.static("capacity"),)),
            provenance=SIMULATED,
        )


def test_an_undeclared_static_table_is_rejected_rather_than_ignored() -> None:
    with pytest.raises(DataError, match=r"declares no static features"):
        SeriesView(
            temporal=table(series_id=["a"], event_time=moments(1), load=[1.0]),
            series=table(series_id=["a"], zone=["DE"]),
            schema=series_schema(),
            provenance=SIMULATED,
            static=table(series_id=["a"], capacity=[1.0]),
        )


def test_a_view_column_name_cannot_be_reused_for_a_target() -> None:
    with pytest.raises(SchemaError, match=r"reserves"):
        series_schema(targets=("event_time",))


def test_a_series_view_schema_cannot_claim_another_kind() -> None:
    with pytest.raises(SchemaError, match=r"cannot declare kind"):
        series_schema(kind=ViewKind.SEQUENCES)


def test_an_empty_view_is_not_a_view() -> None:
    with pytest.raises(DataError, match=r"at least one row"):
        SeriesView(
            temporal=table(series_id=[], event_time=[], load=[]),
            series=table(series_id=[], zone=[]),
            schema=series_schema(),
            provenance=SIMULATED,
        )


# -- SequenceView ----------------------------------------------------------


def sequence_schema(**overrides: Any) -> SequenceViewSchema:
    options: dict[str, Any] = {
        "frequency": "1h",
        "context": 2,
        "horizon": 2,
        "targets": ("price",),
        "instance_keys": ("zone",),
    }
    options.update(overrides)
    return SequenceViewSchema(**options)


def sequence_samples(**overrides: Any) -> dict[str, list[Any]]:
    times = moments(4)
    options: dict[str, list[Any]] = {
        "sample_id": ["s"],
        "zone": ["DE"],
        "origin_time": [times[1]],
        "context_start": [times[0]],
        "context_end": [times[1]],
        "forecast_start": [times[2]],
        "forecast_end": [times[3]],
    }
    options.update(overrides)
    return options


def test_a_sequence_view_spans_exactly_its_declared_window() -> None:
    view = SequenceView(
        temporal=table(sample_id=["s"] * 4, event_time=moments(4), price=[1.0, 2.0, 3.0, 4.0]),
        samples=table(**sequence_samples()),
        schema=sequence_schema(),
        provenance=OBSERVED,
    )
    assert view.schema.length == 4
    assert view.sample_ids == ("s",)
    assert view.origins == (START + HOUR,)
    assert view.provenance.is_observed


def test_a_sample_short_of_its_window_is_not_padded() -> None:
    with pytest.raises(DataError, match=r"holds 3 event times"):
        SequenceView(
            temporal=table(sample_id=["s"] * 3, event_time=moments(3), price=[1.0, 2.0, 3.0]),
            samples=table(**sequence_samples()),
            schema=sequence_schema(),
            provenance=OBSERVED,
        )


def test_a_sample_off_the_frequency_grid_is_rejected() -> None:
    times = [START, START + HOUR, START + HOUR * 2, START + HOUR * 4]
    with pytest.raises(DataError, match=r"expected the 4 steps"):
        SequenceView(
            temporal=table(sample_id=["s"] * 4, event_time=times, price=[1.0, 2.0, 3.0, 4.0]),
            samples=table(**sequence_samples()),
            schema=sequence_schema(),
            provenance=OBSERVED,
        )


def test_an_origin_that_is_not_the_last_context_step_is_rejected() -> None:
    """The origin is where the context ends; anything else spans two origins."""
    samples = sequence_samples(origin_time=[START])
    with pytest.raises(DataError, match=r"declares bounds"):
        SequenceView(
            temporal=table(sample_id=["s"] * 4, event_time=moments(4), price=[1.0] * 4),
            samples=table(**samples),
            schema=sequence_schema(),
            provenance=OBSERVED,
        )


def test_sequence_static_features_are_keyed_by_sample() -> None:
    view = SequenceView(
        temporal=table(sample_id=["s"] * 4, event_time=moments(4), price=[1.0] * 4),
        samples=table(**sequence_samples()),
        schema=sequence_schema(features=(FeatureSpec.static("capacity"),)),
        provenance=OBSERVED,
        static=table(sample_id=["s"], capacity=[7.0]),
    )
    assert view.static is not None
    assert view.static.column_names == ["sample_id", "capacity"]


# -- TabularView -----------------------------------------------------------


def tabular_schema(**overrides: Any) -> TabularViewSchema:
    options: dict[str, Any] = {
        "frequency": "1h",
        "horizon": 2,
        "targets": ("price",),
        "instance_keys": ("zone",),
        "features": (FeatureSpec.known("wind_fc"),),
    }
    options.update(overrides)
    return TabularViewSchema(**options)


def tabular_keys(**overrides: Any) -> dict[str, list[Any]]:
    options: dict[str, list[Any]] = {
        "row_id": ["r1", "r2"],
        "zone": ["DE", "DE"],
        "origin_time": [START, START],
        "event_time": moments(2, after=1),
        "horizon_step": [1, 2],
    }
    options.update(overrides)
    return options


def test_a_tabular_view_is_row_aligned() -> None:
    view = TabularView(
        X=table(wind_fc=[1.0, 2.0]),
        y=table(price=[10.0, 20.0]),
        keys=table(**tabular_keys()),
        schema=tabular_schema(),
        provenance=OBSERVED,
    )
    assert view.num_rows == 2
    assert view.kind is ViewKind.TABULAR
    assert view.origins == (START,)


def test_unaligned_tables_are_rejected() -> None:
    with pytest.raises(DataError, match=r"row-aligned"):
        TabularView(
            X=table(wind_fc=[1.0]),
            y=table(price=[10.0, 20.0]),
            keys=table(**tabular_keys()),
            schema=tabular_schema(),
            provenance=OBSERVED,
        )


def test_a_horizon_step_must_be_the_distance_from_the_origin() -> None:
    with pytest.raises(DataError, match=r"horizon step"):
        TabularView(
            X=table(wind_fc=[1.0, 2.0]),
            y=table(price=[10.0, 20.0]),
            keys=table(**tabular_keys(horizon_step=[1, 1])),
            schema=tabular_schema(),
            provenance=OBSERVED,
        )


def test_a_repeated_row_id_identifies_two_rows() -> None:
    with pytest.raises(DataError, match=r"duplicate row_id"):
        TabularView(
            X=table(wind_fc=[1.0, 2.0]),
            y=table(price=[10.0, 20.0]),
            keys=table(**tabular_keys(row_id=["r", "r"])),
            schema=tabular_schema(),
            provenance=OBSERVED,
        )


def test_a_tabular_view_cannot_declare_observed_features() -> None:
    """At a positive lead an observed feature has no value, so it is not a column."""
    with pytest.raises(SchemaError, match=r"explicit lag feature"):
        tabular_schema(features=(FeatureSpec.observed("wind"),))


# -- ForecastView ----------------------------------------------------------


def forecast_metadata(**overrides: Any) -> ForecastViewMetadata:
    options: dict[str, Any] = {
        "frequency": "1h",
        "horizon": 2,
        "targets": ("price",),
        "instance_keys": ("zone",),
        "features": (FeatureSpec.known("wind_fc"),),
    }
    options.update(overrides)
    return ForecastViewMetadata(**options)


def forecast_view(**overrides: Any) -> ForecastView:
    options: dict[str, Any] = {
        "origin_time": START + HOUR,
        "history": table(
            zone=["DE", "DE"],
            event_time=moments(2),
            price=[1.0, 2.0],
            wind_fc=[3.0, 4.0],
        ),
        "future": table(zone=["DE", "DE"], event_time=moments(2, after=2), wind_fc=[5.0, 6.0]),
        "metadata": forecast_metadata(),
    }
    options.update(overrides)
    return ForecastView(**options)


def test_a_forecast_view_names_the_event_times_it_asks_about() -> None:
    view = forecast_view()
    assert view.kind is ViewKind.FORECAST
    assert view.event_times == (START + HOUR * 2, START + HOUR * 3)
    assert view.instances == (("DE",),)


def test_history_after_the_origin_is_leakage() -> None:
    with pytest.raises(DataError, match=r"after the origin"):
        forecast_view(origin_time=START)


def test_the_future_must_be_exactly_the_horizon() -> None:
    with pytest.raises(DataError, match=r"exactly the 2 horizon steps"):
        forecast_view(
            future=table(zone=["DE"], event_time=moments(1, after=2), wind_fc=[5.0]),
        )


def test_a_declared_context_length_must_be_covered_exactly() -> None:
    with pytest.raises(DataError, match=r"exactly the 3 context steps"):
        forecast_view(metadata=forecast_metadata(context=3))


# -- identifiers -----------------------------------------------------------


def test_ids_are_deterministic_and_do_not_reveal_their_inputs() -> None:
    first = opaque_id(("DE",), START)
    assert first == opaque_id(("DE",), START)
    assert first != opaque_id(("FR",), START)
    assert first != opaque_id(("DE",), START + HOUR)
    assert "DE" not in first
    assert "2026" not in first


# -- static tables ---------------------------------------------------------


def test_a_static_table_holds_one_row_per_series() -> None:
    with pytest.raises(DataError, match=r"one row per series"):
        SeriesView(
            temporal=table(series_id=["a", "b"], event_time=moments(2), load=[1.0, 2.0]),
            series=table(series_id=["a", "b"], zone=["DE", "FR"]),
            schema=series_schema(features=(FeatureSpec.static("capacity"),)),
            provenance=SIMULATED,
            static=table(series_id=["a"], capacity=[1.0]),
        )


def test_a_static_table_holds_one_row_per_sample() -> None:
    with pytest.raises(DataError, match=r"one row per sample"):
        SequenceView(
            temporal=table(sample_id=["s"] * 4, event_time=moments(4), price=[1.0] * 4),
            samples=table(**sequence_samples()),
            schema=sequence_schema(features=(FeatureSpec.static("capacity"),)),
            provenance=OBSERVED,
            static=table(sample_id=["s", "t"], capacity=[1.0, 2.0]),
        )


def test_a_forecast_view_static_table_holds_one_row_per_instance() -> None:
    with pytest.raises(DataError, match=r"one row per instance"):
        forecast_view(
            metadata=forecast_metadata(
                features=(FeatureSpec.known("wind_fc"), FeatureSpec.static("capacity"))
            ),
            static=table(zone=["DE", "FR"], capacity=[1.0, 2.0]),
        )


def test_a_forecast_view_needs_the_static_features_it_declares() -> None:
    with pytest.raises(DataError, match=r"no static table"):
        forecast_view(
            metadata=forecast_metadata(
                features=(FeatureSpec.known("wind_fc"), FeatureSpec.static("capacity"))
            )
        )


def test_an_undeclared_static_table_is_not_silently_carried() -> None:
    with pytest.raises(DataError, match=r"declares no static features"):
        forecast_view(static=table(zone=["DE"], capacity=[1.0]))


# -- comparison ------------------------------------------------------------


def test_views_of_different_types_are_not_comparable() -> None:
    view = forecast_view()
    assert view != object()
    assert view == forecast_view()


def test_a_column_cannot_be_both_a_target_and_a_feature() -> None:
    with pytest.raises(SchemaError, match=r"both a target and a feature"):
        series_schema(targets=("load",), features=(FeatureSpec.known("load"),))
