from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from openforecast import (
    DataError,
    ForecastDataset,
    Frequency,
    InconsistentTruthError,
    PointInTimeFrame,
    TimeSeriesFrame,
)
from openforecast.data.forecast_dataset import INFORMATION_DIRNAME, TRUTH_DIRNAME
from tests import factories

ORIGIN = datetime(2026, 1, 1, 8, 0, 0)
EVENT = datetime(2026, 1, 1, 12, 0, 0)
HOUR = timedelta(hours=1)

#: A value no real feed would produce, so a leaked vintage is unmistakable.
POISON = 999999.0


def row(origin: int, event: int, *, price: float = 80.0, wind: float = 10.0) -> dict[str, Any]:
    return {
        "zone": "DE",
        "ref_time": ORIGIN + HOUR * origin,
        "target_time": EVENT + HOUR * event,
        "price": price,
        "wind_fc": wind,
    }


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


# -- splitting information from truth --------------------------------------


def test_a_label_repeated_across_vintages_becomes_one_truth_row() -> None:
    built = dataset([row(0, 0, price=80.0), row(1, 0, price=80.0)])

    assert built.truth.history.num_rows == 1
    assert built.truth.history.to_pydict() == {
        "zone": ["DE"],
        "target_time": [EVENT],
        "price": [80.0],
    }
    assert built.information.table.num_rows == 2


def test_disagreeing_labels_are_a_contradiction_rather_than_a_choice() -> None:
    with pytest.raises(InconsistentTruthError, match=r"target 'price' disagrees") as raised:
        dataset([row(0, 0, price=80.0), row(1, 0, price=81.0)])
    assert "80.0" in str(raised.value)
    assert "81.0" in str(raised.value)


def test_a_missing_label_is_no_information_rather_than_a_disagreement() -> None:
    """A vintage issued before the outcome was published simply has no label yet."""
    built = dataset([row(0, 0, price=math.nan), row(1, 0, price=80.0)])
    assert built.truth.history.column("price").to_pylist() == [80.0]


def test_an_event_no_vintage_ever_labelled_stays_missing() -> None:
    built = dataset([row(0, 0, price=math.nan), row(1, 0, price=math.nan)])
    assert built.truth.history.column("price").to_pylist() == [None]


def test_targets_are_kept_out_of_the_information_frame() -> None:
    built = dataset([row(0, 0)])
    assert "price" not in built.information.table.column_names
    assert built.information.table.column_names == ["zone", "ref_time", "target_time", "wind_fc"]


def test_vintage_specific_feature_values_are_all_kept() -> None:
    built = dataset([row(0, 0, wind=10.1), row(1, 0, wind=11.7), row(2, 0, wind=12.4)])
    assert built.information.table.column("wind_fc").to_pylist() == [10.1, 11.7, 12.4]
    assert built.origins == (ORIGIN, ORIGIN + HOUR, ORIGIN + 2 * HOUR)


def test_a_missing_target_column_is_reported() -> None:
    with pytest.raises(DataError, match=r"missing declared columns \['load'\]"):
        dataset([row(0, 0)], targets=["load"])


def test_static_features_are_lifted_into_the_truth_frame() -> None:
    built = dataset(
        [{**row(0, 0), "capacity": 80.0}, {**row(1, 0), "capacity": 80.0}],
        static_features=["capacity"],
    )
    assert built.truth.static is not None
    assert built.truth.static.to_pydict() == {"zone": ["DE"], "capacity": [80.0]}
    assert "capacity" not in built.information.table.column_names


def test_a_static_feature_that_varies_within_an_instance_is_not_static() -> None:
    with pytest.raises(DataError, match="varies within an instance"):
        dataset(
            [{**row(0, 0), "capacity": 80.0}, {**row(1, 0), "capacity": 90.0}],
            static_features=["capacity"],
        )


def test_a_panel_dataset_keeps_its_instances_apart() -> None:
    built = ForecastDataset.from_pandas(
        factories.point_in_time(instances=("DE", "FR"), instance_key="zone"),
        origin_time="ref_time",
        event_time="target_time",
        instance_keys=["zone"],
        targets=["price"],
        known_features=["wind_fc"],
        event_frequency="1h",
        origin_frequency="1h",
    )
    assert built.instances == (("DE",), ("FR",))
    assert built.truth.instances == (("DE",), ("FR",))
    assert built.targets == ("price",)


# -- pairing the two frames -----------------------------------------------


def parts() -> tuple[PointInTimeFrame, TimeSeriesFrame]:
    built = dataset([row(0, 0), row(1, 0)])
    return built.information, built.truth


def test_the_two_frames_must_agree_on_their_instance_keys() -> None:
    information, _ = parts()
    truth = TimeSeriesFrame.from_pandas(
        pd.DataFrame([{"target_time": EVENT, "price": 80.0}]),
        time="target_time",
        frequency="1h",
        targets=["price"],
    )
    with pytest.raises(DataError, match="must share their instance keys"):
        ForecastDataset(information=information, truth=truth)


def test_the_two_frames_must_agree_on_the_event_axis() -> None:
    information, _ = parts()
    truth = TimeSeriesFrame.from_pandas(
        pd.DataFrame([{"zone": "DE", "moment": EVENT, "price": 80.0}]),
        time="moment",
        frequency="1h",
        instance_keys=["zone"],
        targets=["price"],
    )
    with pytest.raises(DataError, match="must be the same column"):
        ForecastDataset(information=information, truth=truth)


def test_the_two_frames_must_agree_on_the_event_frequency() -> None:
    information, truth = parts()
    daily = TimeSeriesFrame(
        history=truth.history,
        schema=truth.schema.model_copy(update={"frequency": Frequency.parse("1d")}),
    )
    with pytest.raises(DataError, match="must agree"):
        ForecastDataset(information=information, truth=daily)


def test_a_column_cannot_be_both_a_target_and_an_information_feature() -> None:
    information, truth = parts()
    renamed = TimeSeriesFrame(
        history=truth.history.rename_columns(["zone", "target_time", "wind_fc"]),
        schema=truth.schema.model_copy(update={"targets": ("wind_fc",)}),
    )
    with pytest.raises(DataError, match="both a truth target and an information feature"):
        ForecastDataset(information=information, truth=renamed)


def test_truth_cannot_hold_an_instance_no_origin_ever_covered() -> None:
    information, _ = parts()
    truth = TimeSeriesFrame.from_pandas(
        pd.DataFrame(
            [
                {"zone": "DE", "target_time": EVENT, "price": 80.0},
                {"zone": "FR", "target_time": EVENT, "price": 70.0},
            ]
        ),
        time="target_time",
        frequency="1h",
        instance_keys=["zone"],
        targets=["price"],
    )
    with pytest.raises(DataError, match="instances absent from information"):
        ForecastDataset(information=information, truth=truth)


# -- at_origin -------------------------------------------------------------


def leaking_dataset() -> ForecastDataset:
    """Three vintages of the 12:00 event, the newest one poisoned.

    Each origin also covers its own hour, so every context has a history:

    ```text
    origin 08:00 -> event 12:00 -> wind 10
    origin 09:00 -> event 12:00 -> wind 20
    origin 10:00 -> event 12:00 -> wind 999999
    ```
    """
    return dataset(
        [
            row(0, -4, price=40.0, wind=1.0),  # origin 08:00, event 08:00
            row(0, 0, wind=10.0),
            row(1, -3, price=50.0, wind=2.0),  # origin 09:00, event 09:00
            row(1, 0, wind=20.0),
            row(2, -2, price=60.0, wind=3.0),  # origin 10:00, event 10:00
            row(2, 0, wind=POISON),
        ]
    )


def test_a_later_vintage_never_leaks_backward() -> None:
    context = leaking_dataset().at_origin(ORIGIN + HOUR)
    values = context.future.column("wind_fc").to_pylist() if context.future else []

    assert 20.0 in values
    assert POISON not in values


def test_no_column_of_a_context_holds_a_later_vintage() -> None:
    context = leaking_dataset().at_origin(ORIGIN)
    for table in (context.history, context.future):
        if table is None:
            continue
        for name in table.column_names:
            assert POISON not in table.column(name).to_pylist()


def test_at_origin_splits_history_from_future_at_the_origin() -> None:
    built = ForecastDataset.from_pandas(
        factories.point_in_time(origins=3, horizon=4),
        origin_time="ref_time",
        event_time="target_time",
        targets=["price"],
        known_features=["wind_fc"],
        event_frequency="1h",
        origin_frequency="1h",
    )
    origin = factories.START + timedelta(hours=2)
    context = built.at_origin(origin)

    past: list[Any] = context.history.column("target_time").to_pylist()
    assert context.future is not None
    upcoming: list[Any] = context.future.column("target_time").to_pylist()

    assert context.origin_time == origin
    assert max(past) <= origin
    assert min(upcoming) > origin


def test_the_history_of_a_context_carries_truth_the_vintage_never_mentioned() -> None:
    """Target history behind the origin comes from truth; its features stay null."""
    built = dataset(
        [
            row(0, -2, price=70.0),  # origin 08:00, event 10:00
            row(4, 0, price=80.0),  # origin 12:00, event 12:00
        ]
    )
    context = built.at_origin(ORIGIN + 4 * HOUR)
    history = context.history.to_pydict()

    assert history["target_time"] == [EVENT - 2 * HOUR, EVENT]
    assert history["price"] == [70.0, 80.0]
    # The 12:00 vintage says nothing about the 10:00 event, and nothing invents it.
    assert history["wind_fc"] == [None, 10.0]


def test_a_context_carries_the_static_features_of_the_truth_frame() -> None:
    built = dataset([{**row(0, -4), "capacity": 80.0}], static_features=["capacity"])
    context = built.at_origin(ORIGIN)
    assert context.static is not None
    assert context.static.to_pydict() == {"zone": ["DE"], "capacity": [80.0]}
    assert context.schema.has_static_features


def test_at_origin_rejects_an_origin_that_does_not_exist() -> None:
    with pytest.raises(DataError, match="no origin"):
        leaking_dataset().at_origin(ORIGIN + 12 * HOUR)


def test_an_origin_with_nothing_knowable_yet_is_an_error() -> None:
    """A vintage entirely ahead of its origin and no truth behind it forecasts nothing."""
    built = dataset([row(0, 0)])
    with pytest.raises(DataError, match="nothing is knowable at origin"):
        built.at_origin(ORIGIN)


# -- serialization ---------------------------------------------------------


def full_dataset() -> ForecastDataset:
    return ForecastDataset.from_pandas(
        factories.point_in_time(
            instances=("DE", "FR"),
            instance_key="zone",
            origins=4,
            horizon=3,
            known=("wind_fc", "solar_fc"),
            observed=("load_actual",),
            static={"capacity": {"DE": 80.0, "FR": 60.0}},
        ),
        origin_time="ref_time",
        event_time="target_time",
        instance_keys=["zone"],
        targets=["price"],
        observed_features=["load_actual"],
        known_features=["wind_fc", "solar_fc"],
        static_features=["capacity"],
        event_frequency="1h",
        origin_frequency="1h",
    )


def test_round_trip_preserves_everything(tmp_path: Path) -> None:
    built = full_dataset()
    built.write(tmp_path / "dataset")
    restored = ForecastDataset.read(tmp_path / "dataset")

    assert restored == built
    assert restored.information.table.equals(built.information.table)
    assert restored.truth == built.truth


def test_write_lays_out_the_two_frames(tmp_path: Path) -> None:
    full_dataset().write(tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == {INFORMATION_DIRNAME, TRUTH_DIRNAME}


def test_reading_an_incomplete_directory_fails(tmp_path: Path) -> None:
    full_dataset().write(tmp_path)
    for path in sorted((tmp_path / TRUTH_DIRNAME).iterdir()):
        path.unlink()
    (tmp_path / TRUTH_DIRNAME).rmdir()
    with pytest.raises(DataError, match=f"{TRUTH_DIRNAME}/ is missing"):
        ForecastDataset.read(tmp_path)


# -- dunder ----------------------------------------------------------------


def test_datasets_compare_by_value() -> None:
    assert dataset([row(0, 0)]) == dataset([row(0, 0)])
    assert dataset([row(0, 0)]) != dataset([row(0, 0, wind=99.0)])
    assert dataset([row(0, 0)]) != "not a dataset"


def test_repr_names_the_shape() -> None:
    text = repr(full_dataset())
    assert "origins=4" in text
    assert "targets=['price']" in text
