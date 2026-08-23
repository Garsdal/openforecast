"""The bulk channel: what a dataset is once it has crossed a network.

The claim being tested is narrow and load-bearing — a payload is decoded through
the *ordinary* constructors, so a frame that arrives is a frame that would have
been legal to build here. A truncated table therefore fails to load rather than
being fitted as a shorter history, and a schema that no longer matches its data
is refused on the far side exactly as it would have been on this one.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import pytest

import openforecast as of
from openforecast.errors import DataError
from openforecast.server.wire import (
    DataKind,
    FitBody,
    ForecastContextPayload,
    ForecastDatasetPayload,
    PointInTimePayload,
    TimeSeriesPayload,
    decode_data,
    decode_table,
    encode_data,
    encode_table,
)

START = datetime(2026, 1, 1)
HOUR = timedelta(hours=1)


def at(step: int) -> datetime:
    return START + HOUR * step


def frame(*, with_future: bool = False) -> of.TimeSeriesFrame:
    rows: list[dict[str, Any]] = []
    for offset, zone in enumerate(("DE", "FR")):
        for step in range(12):
            row: dict[str, Any] = {
                "zone": zone,
                "timestamp": at(step),
                "load": float(step + offset * 100),
            }
            if with_future:
                row["wind_fc"] = float(step)
            rows.append(row)
    history = pd.DataFrame(rows)
    future = (
        pd.DataFrame(
            [
                {"zone": zone, "timestamp": at(step), "wind_fc": float(step)}
                for zone in ("DE", "FR")
                for step in range(12, 16)
            ]
        )
        if with_future
        else None
    )
    return of.TimeSeriesFrame.from_pandas(
        history=history,
        time="timestamp",
        frequency="1h",
        instance_keys=["zone"],
        targets=["load"],
        future=future,
        known_features=["wind_fc"] if with_future else [],
    )


def dataset() -> of.ForecastDataset:
    rows: list[dict[str, Any]] = [
        {
            "zone": "DE",
            "ref_time": at(origin),
            "target_time": at(event),
            "price": float(event),
            "wind_fc": float(origin * 100 + event),
        }
        for origin in range(8)
        for event in range(8)
    ]
    return of.ForecastDataset.from_pandas(
        pd.DataFrame(rows),
        origin_time="ref_time",
        event_time="target_time",
        targets=["price"],
        instance_keys=["zone"],
        known_features=["wind_fc"],
        event_frequency="1h",
        origin_frequency="1h",
    )


@pytest.mark.parametrize("with_future", [False, True])
def test_a_time_series_frame_survives_the_round_trip(with_future: bool) -> None:
    """Including the tables it does *not* have: absent is not empty."""
    original = frame(with_future=with_future)
    payload = encode_data(original)

    assert isinstance(payload, TimeSeriesPayload)
    assert payload.kind is DataKind.TIME_SERIES
    assert (payload.future is not None) is with_future
    assert decode_data(payload) == original


def test_a_point_in_time_frame_survives_the_round_trip() -> None:
    original = dataset().information
    payload = encode_data(original)

    assert isinstance(payload, PointInTimePayload)
    assert decode_data(payload) == original


def test_a_forecast_dataset_survives_the_round_trip() -> None:
    """Both halves, and the axes they have to agree on."""
    original = dataset()
    payload = encode_data(original)

    assert isinstance(payload, ForecastDatasetPayload)
    assert decode_data(payload) == original


def test_a_forecast_context_carries_the_origin_it_is_split_at() -> None:
    """A context is one origin; losing it would forecast from another one."""
    original = dataset().at_origin(at(5))
    payload = encode_data(original)

    assert isinstance(payload, ForecastContextPayload)
    assert payload.origin_time == at(5)
    assert decode_data(payload) == original


def test_anything_else_is_refused_rather_than_coerced() -> None:
    with pytest.raises(DataError, match="cannot send list to a forecasting service"):
        encode_data([1, 2, 3])


def test_a_truncated_table_fails_to_load_rather_than_arriving_short() -> None:
    """The whole reason a payload is decoded through the constructor."""
    payload = encode_data(frame())
    assert isinstance(payload, TimeSeriesPayload)
    damaged = payload.model_copy(update={"history": payload.history[:40]})

    with pytest.raises(DataError, match="history"):
        decode_data(damaged)


def test_a_payload_that_is_not_base64_is_data_rather_than_a_crash() -> None:
    with pytest.raises(DataError, match="not valid base64"):
        decode_table("not base64 at all!!", "history")


def test_a_schema_that_no_longer_describes_its_data_is_refused_on_arrival() -> None:
    """The far side re-validates; it does not trust what the near side said."""
    payload = encode_data(frame())
    assert isinstance(payload, TimeSeriesPayload)
    lying = payload.model_copy(
        update={"data_schema": payload.data_schema.model_copy(update={"targets": ("nonexistent",)})}
    )

    with pytest.raises(DataError):
        decode_data(lying)


def test_a_table_round_trips_through_arrow_ipc_unchanged() -> None:
    original = frame().history

    assert decode_table(encode_table(original), "history").equals(original)


def test_a_request_body_holds_a_recipe_as_itself_rather_than_as_a_string() -> None:
    """Control is JSON: a recipe is readable in a log, a training set is not."""
    body = FitBody(
        model=of.Pipeline(
            steps=(
                of.StandardScaler(columns=of.ColumnSet.TARGETS),
                of.Model("builtin/seasonal-naive", params={"season_length": 4}),
            )
        ),
        data=encode_data(frame()),
        horizon=4,
        plan=of.FitPlan(origins=of.LatestOrigin()),
    )
    serialized = body.model_dump(mode="json")

    assert serialized["model"]["kind"] == "pipeline"
    assert serialized["plan"]["origins"]["mode"] == "latest"
    assert isinstance(serialized["data"]["history"], str)
