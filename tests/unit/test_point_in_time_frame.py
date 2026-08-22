from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pytest

from openforecast import DataError, PointInTimeFrame
from openforecast.data.point_in_time import SCHEMA_FILENAME, TABLE_FILENAME
from tests import factories

ORIGIN = datetime(2026, 1, 1, 8, 0, 0)
EVENT = datetime(2026, 1, 1, 12, 0, 0)
HOUR = timedelta(hours=1)


def vintages(rows: list[dict[str, Any]], **overrides: Any) -> PointInTimeFrame:
    """A frame built row by row, so a test can state exactly what it means."""
    options: dict[str, Any] = {
        "origin_time": "ref_time",
        "event_time": "target_time",
        "event_frequency": "1h",
        "instance_keys": ["zone"],
        "known_features": ["wind_fc"],
    }
    options.update(overrides)
    return PointInTimeFrame.from_pandas(pd.DataFrame(rows), **options)


def row(origin: int, event: int, wind: float = 10.0, zone: str = "DE") -> dict[str, Any]:
    return {
        "zone": zone,
        "ref_time": ORIGIN + HOUR * origin,
        "target_time": EVENT + HOUR * event,
        "wind_fc": wind,
    }


# -- the core shape --------------------------------------------------------


def test_the_same_event_time_is_kept_once_per_origin() -> None:
    """Three vintages of one event time are three rows, not one."""
    frame = vintages([row(0, 0, 10.1), row(1, 0, 11.7), row(2, 0, 12.4)])

    assert frame.table.num_rows == 3
    assert frame.origins == (ORIGIN, ORIGIN + HOUR, ORIGIN + 2 * HOUR)
    assert frame.event_times == (EVENT,)
    assert frame.table.column("wind_fc").to_pylist() == [10.1, 11.7, 12.4]


def test_the_table_is_stored_in_canonical_order_and_drops_undeclared_columns() -> None:
    frame = vintages(
        [{**row(0, 0), "load_actual": None, "note": "ignored"}],
        observed_features=["load_actual"],
    )
    assert frame.table.column_names == [
        "zone",
        "ref_time",
        "target_time",
        "load_actual",
        "wind_fc",
    ]


def test_a_declared_column_must_be_present() -> None:
    with pytest.raises(DataError, match="missing declared columns"):
        vintages([row(0, 0)], known_features=["wind_fc", "solar_fc"])


def test_a_duplicate_origin_and_event_is_rejected() -> None:
    with pytest.raises(DataError, match="duplicate instance/origin/event rows"):
        vintages([row(0, 0, 10.0), row(0, 0, 11.0)])


def test_the_same_origin_and_event_may_repeat_across_instances() -> None:
    frame = vintages([row(0, 0, zone="DE"), row(0, 0, zone="FR")])
    assert frame.instances == (("DE",), ("FR",))


def test_missingness_that_improves_between_vintages_is_preserved() -> None:
    """``NaN, NaN, 42`` says when the feed caught up, so it must survive intact."""
    frame = vintages([row(0, 0, math.nan), row(1, 0, math.nan), row(2, 0, 42.0)])
    # pandas spells a missing float as NaN and pyarrow converts that to a null.
    assert frame.table.column("wind_fc").to_pylist() == [None, None, 42.0]


def test_an_arrow_nan_stays_a_nan() -> None:
    """Arrow can hold NaN as a value distinct from null, and neither is repaired."""
    table = pa.table(
        {
            "ref_time": pa.array([ORIGIN, ORIGIN + HOUR], type=pa.timestamp("us")),
            "target_time": pa.array([EVENT, EVENT], type=pa.timestamp("us")),
            "wind_fc": pa.array([math.nan, None], type=pa.float64()),
        }
    )
    frame = PointInTimeFrame.from_arrow(
        table,
        origin_time="ref_time",
        event_time="target_time",
        event_frequency="1h",
        known_features=["wind_fc"],
    )
    values: list[Any] = frame.table.column("wind_fc").to_pylist()
    assert math.isnan(values[0])
    assert values[1] is None


def test_event_times_must_sit_on_the_declared_grid() -> None:
    off_grid = row(0, 0)
    off_grid["target_time"] = EVENT + timedelta(minutes=30)
    with pytest.raises(DataError, match="event time has 1 timestamps"):
        vintages([row(0, 0), off_grid])


def test_origin_times_are_only_gridded_when_a_frequency_is_declared() -> None:
    irregular = row(0, 0)
    irregular["ref_time"] = ORIGIN + timedelta(minutes=30)
    assert vintages([row(1, 0), irregular]).table.num_rows == 2
    with pytest.raises(DataError, match="origin time has 1 timestamps"):
        vintages([row(1, 0), irregular], origin_frequency="1h")


def test_null_keys_and_axes_are_rejected() -> None:
    missing_zone = row(0, 0)
    missing_zone["zone"] = None
    with pytest.raises(DataError, match="null values in instance key 'zone'"):
        vintages([row(0, 1), missing_zone])

    missing_origin = row(0, 0)
    missing_origin["ref_time"] = None
    with pytest.raises(DataError, match="null values in origin time 'ref_time'"):
        vintages([row(1, 1), missing_origin])


def test_a_time_axis_must_be_a_timestamp() -> None:
    with pytest.raises(DataError, match="must be a timestamp"):
        vintages([{**row(0, 0), "ref_time": 8}])


def test_the_two_axes_must_share_a_time_zone() -> None:
    """The axes are compared to each other on every lead time."""
    aware = row(0, 0)
    aware["ref_time"] = ORIGIN.replace(tzinfo=UTC)
    with pytest.raises(DataError, match="must share a time zone"):
        vintages([aware])


def test_a_table_is_required() -> None:
    with pytest.raises(DataError, match="must be a pyarrow.Table"):
        PointInTimeFrame.from_arrow(
            "not a table",  # pyright: ignore[reportArgumentType]
            origin_time="ref_time",
            event_time="target_time",
            event_frequency="1h",
            known_features=["wind_fc"],
        )


# -- leakage ---------------------------------------------------------------


def test_an_observed_feature_cannot_hold_a_value_past_its_own_origin() -> None:
    """At that origin, nobody could have measured it yet."""
    # Origin 12:00, event 13:00: one hour beyond what the origin could know.
    leaked = {**row(4, 1), "load_actual": 55.0}
    with pytest.raises(DataError, match="observed feature 'load_actual' has 1 values"):
        vintages(
            [{**row(4, 0), "load_actual": 50.0}, leaked],
            observed_features=["load_actual"],
        )


def test_an_observed_feature_may_be_measured_up_to_its_origin_and_missing_after() -> None:
    frame = vintages(
        [
            {**row(4, 0), "load_actual": 50.0},  # origin 12:00, event 12:00
            {**row(4, 1), "load_actual": math.nan},  # origin 12:00, event 13:00
        ],
        observed_features=["load_actual"],
    )
    assert frame.table.column("load_actual").to_pylist() == [50.0, None]


def test_at_origin_selects_only_that_vintage() -> None:
    frame = vintages([row(0, 0, 10.0), row(1, 0, 20.0), row(2, 0, 999999.0)])
    vintage = frame.at_origin(ORIGIN + HOUR)

    assert vintage.origins == (ORIGIN + HOUR,)
    assert vintage.table.column("wind_fc").to_pylist() == [20.0]


def test_at_origin_accepts_an_iso_string() -> None:
    frame = vintages([row(0, 0), row(1, 0)])
    assert frame.at_origin("2026-01-01T09:00:00").origins == (ORIGIN + HOUR,)


def test_at_origin_will_not_guess_a_nearby_vintage() -> None:
    frame = vintages([row(0, 0), row(2, 0)])
    with pytest.raises(DataError, match="no origin 2026-01-01T09:00:00 in this data"):
        frame.at_origin(ORIGIN + HOUR)


def test_at_origin_rejects_an_unparseable_timestamp() -> None:
    frame = vintages([row(0, 0)])
    with pytest.raises(DataError, match="cannot parse origin_time"):
        frame.at_origin("yesterday")


# -- lead time -------------------------------------------------------------


def test_lead_time_is_derived_rather_than_stored() -> None:
    frame = vintages([row(0, 0), row(1, 0), row(2, 0)])
    assert "lead_time" not in frame.table.column_names

    with_lead = frame.with_lead_time("hour")
    assert with_lead.table.column("lead_time").to_pylist() == [4, 3, 2]
    assert with_lead.schema.feature_names == ("wind_fc", "lead_time")
    assert with_lead.schema.known_features[-1].name == "lead_time"


def test_a_lead_time_may_be_negative() -> None:
    """A vintage can describe event times before its own origin."""
    past = row(6, 0)  # origin 14:00, event 12:00
    assert vintages([past]).with_lead_time().table.column("lead_time").to_pylist() == [-2]


def test_lead_time_can_be_named() -> None:
    frame = vintages([row(0, 0)]).with_lead_time("hour", name="lead_hours")
    assert frame.table.column_names[-1] == "lead_hours"


def test_lead_time_will_not_round() -> None:
    frame = vintages([row(0, 0)], event_frequency="30m")
    with pytest.raises(DataError, match="not a whole number of 1d steps"):
        frame.with_lead_time("1d")


def test_lead_time_refuses_to_overwrite_a_column() -> None:
    frame = vintages([{**row(0, 0), "lead_time": 4}], known_features=["wind_fc", "lead_time"])
    with pytest.raises(DataError, match="already has that column"):
        frame.with_lead_time()


# -- construction and serialization ----------------------------------------


def test_from_pandas_builds_the_same_frame_as_from_arrow() -> None:
    frame = pd.DataFrame([row(0, 0), row(1, 0)])
    assert PointInTimeFrame.from_pandas(
        frame,
        origin_time="ref_time",
        event_time="target_time",
        event_frequency="1h",
        instance_keys=["zone"],
        known_features=["wind_fc"],
    ) == PointInTimeFrame.from_arrow(
        pa.Table.from_pandas(frame, preserve_index=False),
        origin_time="ref_time",
        event_time="target_time",
        event_frequency="1h",
        instance_keys=["zone"],
        known_features=["wind_fc"],
    )


def test_from_pandas_rejects_something_that_is_not_a_dataframe() -> None:
    with pytest.raises(DataError, match="not a pandas DataFrame"):
        PointInTimeFrame.from_pandas(
            [row(0, 0)],
            origin_time="ref_time",
            event_time="target_time",
            event_frequency="1h",
            known_features=["wind_fc"],
        )


def panel_frame() -> PointInTimeFrame:
    return PointInTimeFrame.from_pandas(
        factories.point_in_time(
            instances=("DE", "FR"),
            instance_key="zone",
            origins=3,
            horizon=4,
            known=("wind_fc", "solar_fc"),
            observed=("load_actual",),
        ),
        origin_time="ref_time",
        event_time="target_time",
        event_frequency="1h",
        origin_frequency="1h",
        instance_keys=["zone"],
        observed_features=["load_actual"],
        known_features=["wind_fc", "solar_fc"],
    )


def test_round_trip_preserves_everything(tmp_path: Path) -> None:
    frame = panel_frame()
    frame.write(tmp_path / "pit")
    restored = PointInTimeFrame.read(tmp_path / "pit")

    assert restored == frame
    assert restored.table.equals(frame.table)
    assert {path.name for path in (tmp_path / "pit").iterdir()} == {
        SCHEMA_FILENAME,
        TABLE_FILENAME,
    }


def test_round_trip_preserves_missingness(tmp_path: Path) -> None:
    frame = vintages([row(0, 0, math.nan), row(1, 0, 42.0)])
    frame.write(tmp_path)
    assert PointInTimeFrame.read(tmp_path).table.column("wind_fc").to_pylist() == [None, 42.0]


def test_reading_a_frame_revalidates_it(tmp_path: Path) -> None:
    panel_frame().write(tmp_path)
    schema = (tmp_path / SCHEMA_FILENAME).read_text(encoding="utf-8")
    (tmp_path / SCHEMA_FILENAME).write_text(
        schema.replace('"unit": "hour"', '"unit": "day"'), encoding="utf-8"
    )
    with pytest.raises(DataError, match="do not sit on the 1d grid"):
        PointInTimeFrame.read(tmp_path)


@pytest.mark.parametrize("missing", [SCHEMA_FILENAME, TABLE_FILENAME])
def test_reading_an_incomplete_directory_fails(tmp_path: Path, missing: str) -> None:
    panel_frame().write(tmp_path)
    (tmp_path / missing).unlink()
    with pytest.raises(DataError, match=f"{missing} is missing"):
        PointInTimeFrame.read(tmp_path)


# -- dunder ----------------------------------------------------------------


def test_frames_compare_by_value() -> None:
    assert vintages([row(0, 0)]) == vintages([row(0, 0)])
    assert vintages([row(0, 0)]) != vintages([row(0, 0, 99.0)])
    assert vintages([row(0, 0)]) != "not a frame"


def test_repr_names_the_shape() -> None:
    text = repr(panel_frame())
    assert "panel" in text
    assert "origins=3" in text
