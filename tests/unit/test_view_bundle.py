"""View bundles: what a provider in another process is actually handed.

A bundle is the only representation of a view that crosses a process boundary,
so the property that matters is that nothing is lost on the way: the view that
comes back has to equal the view that went out, tables, schema, provenance and
all. Everything else here is a bundle that is not one — a missing table, a kind
nobody declared — failing to load rather than loading as something else.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from openforecast import DataError, FeatureAvailability, FeatureKind, FeatureSpec, Frequency
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
    ViewProvenance,
    read_answer,
    read_fit_view,
    read_forecast_view,
    read_view,
    write_answer,
    write_view,
)
from openforecast.views.bundle import SCHEMA_FILENAME

START = datetime(2026, 1, 1, 0, 0, 0)
HOUR = timedelta(hours=1)

SIMULATED = ViewProvenance(source=SourceKind.TIME_SERIES, origin_fidelity=OriginFidelity.SIMULATED)
OBSERVED = ViewProvenance(
    source=SourceKind.FORECAST_DATASET, origin_fidelity=OriginFidelity.OBSERVED
)

HOURLY = Frequency.parse("1h")
KNOWN = FeatureSpec(name="wind_fc", availability=FeatureAvailability.KNOWN)
STATIC = FeatureSpec(name="capacity", kind=FeatureKind.STATIC)


def moments(count: int, *, after: int = 0) -> list[datetime]:
    return [START + HOUR * (after + index) for index in range(count)]


def table(**columns: list[Any]) -> pa.Table:
    return pa.table(columns)


def series_view() -> SeriesView:
    return SeriesView(
        temporal=table(
            series_id=["a"] * 4,
            event_time=moments(4),
            load=[1.0, 2.0, None, 4.0],
            wind_fc=[0.5, 0.6, 0.7, 0.8],
        ),
        series=table(series_id=["a"], zone=["DE"]),
        schema=SeriesViewSchema(
            frequency=HOURLY, targets=("load",), instance_keys=("zone",), features=(KNOWN, STATIC)
        ),
        provenance=SIMULATED,
        static=table(series_id=["a"], capacity=[10.0]),
    )


def sequence_view() -> SequenceView:
    return SequenceView(
        temporal=table(sample_id=["s"] * 4, event_time=moments(4), load=[1.0, 2.0, 3.0, 4.0]),
        samples=table(
            sample_id=["s"],
            zone=["DE"],
            origin_time=[START + HOUR],
            context_start=[START],
            context_end=[START + HOUR],
            forecast_start=[START + HOUR * 2],
            forecast_end=[START + HOUR * 3],
        ),
        schema=SequenceViewSchema(
            frequency=HOURLY, context=2, horizon=2, targets=("load",), instance_keys=("zone",)
        ),
        provenance=OBSERVED,
    )


def tabular_view() -> TabularView:
    return TabularView(
        X=table(wind_fc=[0.5, 0.6]),
        y=table(load=[1.0, 2.0]),
        keys=table(
            row_id=["r1", "r2"],
            zone=["DE", "DE"],
            origin_time=[START, START],
            event_time=[START + HOUR, START + HOUR * 2],
            horizon_step=[1, 2],
        ),
        schema=TabularViewSchema(
            frequency=HOURLY,
            horizon=2,
            targets=("load",),
            instance_keys=("zone",),
            features=(KNOWN,),
        ),
        provenance=OBSERVED,
    )


def forecast_view() -> ForecastView:
    return ForecastView(
        origin_time=START + HOUR,
        history=table(zone=["DE", "DE"], event_time=moments(2), load=[1.0, 2.0]),
        future=table(zone=["DE", "DE"], event_time=moments(2, after=2)),
        metadata=ForecastViewMetadata(
            frequency=HOURLY, horizon=2, targets=("load",), instance_keys=("zone",)
        ),
    )


@pytest.mark.parametrize(
    "view",
    [series_view(), sequence_view(), tabular_view(), forecast_view()],
    ids=["series", "sequences", "tabular", "forecast"],
)
def test_a_view_survives_the_round_trip_it_makes_to_a_provider(view: Any, tmp_path: Path) -> None:
    write_view(view, tmp_path / "view")

    assert read_view(tmp_path / "view") == view


def test_a_bundle_says_which_view_it_holds(tmp_path: Path) -> None:
    """Nothing has to tell the reader what it is opening."""
    for name, view in (
        ("series", series_view()),
        ("sequences", sequence_view()),
        ("tabular", tabular_view()),
        ("forecast", forecast_view()),
    ):
        written = write_view(view, tmp_path / name)
        assert read_view(written).kind == name


def test_a_missing_static_table_is_not_read_back_from_an_earlier_write(tmp_path: Path) -> None:
    write_view(series_view(), tmp_path / "view")
    without = SeriesView(
        temporal=table(series_id=["a"] * 2, event_time=moments(2), load=[1.0, 2.0]),
        series=table(series_id=["a"], zone=["DE"]),
        schema=SeriesViewSchema(frequency=HOURLY, targets=("load",), instance_keys=("zone",)),
        provenance=SIMULATED,
    )

    write_view(without, tmp_path / "view")

    assert read_view(tmp_path / "view") == without


def test_a_truncated_bundle_fails_to_load_rather_than_training_on_less(tmp_path: Path) -> None:
    """Every invariant the view enforces is enforced again on the far side."""
    written = write_view(sequence_view(), tmp_path / "view")
    temporal = pa.ipc.open_file(str(written / "temporal.arrow")).read_all()
    path = str(written / "temporal.arrow")
    with pa.OSFile(path, "wb") as sink, pa.ipc.new_file(sink, temporal.schema) as writer:
        writer.write_table(temporal.slice(0, 3))

    with pytest.raises(DataError, match=r"holds 3 event times, expected the 4"):
        read_view(written)


def test_a_bundle_missing_a_table_is_not_a_bundle(tmp_path: Path) -> None:
    written = write_view(tabular_view(), tmp_path / "view")
    (written / "y.arrow").unlink()

    with pytest.raises(DataError, match=r"y.arrow is missing"):
        read_view(written)


def test_a_bundle_missing_its_schema_is_not_a_bundle(tmp_path: Path) -> None:
    (tmp_path / "view").mkdir()

    with pytest.raises(DataError, match=r"schema.json is missing"):
        read_view(tmp_path / "view")


def test_a_bundle_declaring_no_known_kind_is_refused(tmp_path: Path) -> None:
    written = write_view(series_view(), tmp_path / "view")
    (written / SCHEMA_FILENAME).write_text('{"kind": "tensors"}', encoding="utf-8")

    with pytest.raises(DataError, match=r"declares kind 'tensors'"):
        read_view(written)


def test_a_bundle_that_is_not_json_is_refused(tmp_path: Path) -> None:
    written = write_view(series_view(), tmp_path / "view")
    (written / SCHEMA_FILENAME).write_text("{", encoding="utf-8")

    with pytest.raises(DataError, match=r"is not valid JSON"):
        read_view(written)


def test_a_training_bundle_and_a_forecast_bundle_are_not_interchangeable(tmp_path: Path) -> None:
    fit = write_view(series_view(), tmp_path / "fit")
    inference = write_view(forecast_view(), tmp_path / "inference")

    with pytest.raises(DataError, match=r"holds a forecast view"):
        read_fit_view(inference)
    with pytest.raises(DataError, match=r"holds a series view"):
        read_forecast_view(fit)


def test_an_answer_travels_back_as_one_arrow_file(tmp_path: Path) -> None:
    answer = table(zone=["DE"], event_time=[START], value=[80.0])

    written = write_answer(answer, tmp_path / "nested" / "answer.arrow")

    assert read_answer(written).equals(answer)


def test_an_answer_that_was_never_written_is_not_an_empty_forecast(tmp_path: Path) -> None:
    with pytest.raises(DataError, match=r"no forecast was written"):
        read_answer(tmp_path / "answer.arrow")
