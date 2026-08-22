from __future__ import annotations

import json
from pathlib import Path

import pytest

from openforecast import DataError, TimeSeriesFrame
from openforecast.data.frame import (
    FUTURE_FILENAME,
    HISTORY_FILENAME,
    SCHEMA_FILENAME,
    STATIC_FILENAME,
)
from tests import factories


def full_frame() -> TimeSeriesFrame:
    return TimeSeriesFrame.from_pandas(
        factories.history(
            instances=("DE", "FR"),
            instance_key="country",
            periods=6,
            targets=("load", "price"),
            observed=("temperature_actual",),
            known=("temperature_forecast",),
            static={"capacity": {"DE": 80.0, "FR": 60.0}},
        ),
        time="timestamp",
        frequency="1h",
        instance_keys=["country"],
        targets=["load", "price"],
        observed_features=["temperature_actual"],
        known_features=["temperature_forecast"],
        static_features=["capacity"],
        future=factories.future(
            instances=("DE", "FR"),
            instance_key="country",
            periods=3,
            known=("temperature_forecast",),
        ),
    )


def test_round_trip_preserves_everything(tmp_path: Path) -> None:
    frame = full_frame()
    frame.write(tmp_path / "frame")
    restored = TimeSeriesFrame.read(tmp_path / "frame")

    assert restored == frame
    assert restored.schema == frame.schema
    assert restored.history.equals(frame.history)
    assert restored.future is not None and frame.future is not None
    assert restored.future.equals(frame.future)
    assert restored.static is not None and frame.static is not None
    assert restored.static.equals(frame.static)


def test_write_lays_out_the_documented_files(tmp_path: Path) -> None:
    full_frame().write(tmp_path)
    written = {path.name for path in tmp_path.iterdir()}
    assert written == {SCHEMA_FILENAME, HISTORY_FILENAME, FUTURE_FILENAME, STATIC_FILENAME}
    schema = json.loads((tmp_path / SCHEMA_FILENAME).read_text(encoding="utf-8"))
    assert schema["frequency"] == {"unit": "hour", "step": 1}
    assert schema["instance_keys"] == ["country"]


def test_absent_tables_are_not_written(tmp_path: Path) -> None:
    frame = TimeSeriesFrame.from_pandas(
        factories.history(periods=3),
        time="timestamp",
        frequency="1h",
        targets=["load"],
    )
    frame.write(tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == {SCHEMA_FILENAME, HISTORY_FILENAME}
    assert TimeSeriesFrame.read(tmp_path) == frame


def test_rewriting_removes_a_stale_table(tmp_path: Path) -> None:
    """A reader must never pick up the future of a previous write."""
    full_frame().write(tmp_path)
    assert (tmp_path / FUTURE_FILENAME).is_file()

    plain = TimeSeriesFrame.from_pandas(
        factories.history(periods=3),
        time="timestamp",
        frequency="1h",
        targets=["load"],
    )
    plain.write(tmp_path)
    assert not (tmp_path / FUTURE_FILENAME).is_file()
    assert TimeSeriesFrame.read(tmp_path) == plain


def test_write_creates_missing_parents(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "frame"
    assert full_frame().write(target) == target
    assert (target / SCHEMA_FILENAME).is_file()


def test_reading_a_frame_revalidates_it(tmp_path: Path) -> None:
    full_frame().write(tmp_path)
    # A schema.json edited to claim a frequency the data does not satisfy must
    # not load just because the Arrow files are intact.
    schema = json.loads((tmp_path / SCHEMA_FILENAME).read_text(encoding="utf-8"))
    schema["frequency"] = {"unit": "day", "step": 1}
    (tmp_path / SCHEMA_FILENAME).write_text(json.dumps(schema), encoding="utf-8")
    with pytest.raises(DataError, match="do not sit on the 1d grid"):
        TimeSeriesFrame.read(tmp_path)


@pytest.mark.parametrize("missing", [SCHEMA_FILENAME, HISTORY_FILENAME])
def test_reading_an_incomplete_directory_fails(tmp_path: Path, missing: str) -> None:
    full_frame().write(tmp_path)
    (tmp_path / missing).unlink()
    with pytest.raises(DataError, match=f"{missing} is missing"):
        TimeSeriesFrame.read(tmp_path)


def test_reading_a_missing_directory_fails(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="is not a TimeSeriesFrame"):
        TimeSeriesFrame.read(tmp_path / "absent")


def test_frames_compare_by_value(tmp_path: Path) -> None:
    frame = full_frame()
    assert frame == full_frame()
    assert frame != TimeSeriesFrame.from_pandas(
        factories.history(periods=3),
        time="timestamp",
        frequency="1h",
        targets=["load"],
    )
    assert frame != "not a frame"
