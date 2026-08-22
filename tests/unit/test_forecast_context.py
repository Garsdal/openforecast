from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from openforecast import DataError, ForecastContext, TimeSeriesFrame
from tests import factories

ORIGIN = datetime(2026, 1, 1, 5, 0, 0)  # the sixth hour of the factory history
HOUR = timedelta(hours=1)


def context(**overrides: object) -> ForecastContext:
    options: dict[str, object] = {
        "origin_time": ORIGIN,
        "event_time": "timestamp",
        "frequency": "1h",
        "instance_keys": ["zone"],
        "targets": ["price"],
        "observed_features": ["load_actual"],
        "known_features": ["wind_fc"],
        "history": factories.history(
            instances=("DE", "FR"),
            instance_key="zone",
            periods=6,
            targets=("price",),
            observed=("load_actual",),
            known=("wind_fc",),
        ),
        "future": factories.future(
            instances=("DE", "FR"),
            instance_key="zone",
            periods=3,
            known=("wind_fc",),
        ),
    }
    options.update(overrides)
    return ForecastContext.from_pandas(**options)  # pyright: ignore[reportArgumentType]


def test_a_context_is_one_frame_and_one_origin() -> None:
    built = context()

    assert built.origin_time == ORIGIN
    assert isinstance(built.frame, TimeSeriesFrame)
    assert built.schema.targets == ("price",)
    assert built.instances == (("DE",), ("FR",))
    assert built.history.num_rows == 12
    assert built.future is not None and built.future.num_rows == 6
    assert built.static is None


def test_the_origin_may_be_an_iso_string() -> None:
    assert context(origin_time="2026-01-01T05:00:00").origin_time == ORIGIN


def test_an_unparseable_origin_is_rejected() -> None:
    with pytest.raises(DataError, match="cannot parse origin_time"):
        context(origin_time="the day before yesterday")


def test_history_may_not_reach_past_the_origin() -> None:
    """A value after the origin is one nobody had yet."""
    with pytest.raises(DataError, match="history holds 2 event times after the origin"):
        context(origin_time=ORIGIN - 2 * HOUR)


def test_the_future_must_begin_after_the_origin() -> None:
    with pytest.raises(DataError, match="future holds 1 event times at or before the origin"):
        context(origin_time=ORIGIN + HOUR)


def test_the_event_time_at_the_origin_belongs_to_history() -> None:
    """The origin itself is knowable, so the boundary is inclusive on the left."""
    built = context()
    assert ORIGIN in built.history.column("timestamp").to_pylist()
    assert built.future is not None
    assert ORIGIN not in built.future.column("timestamp").to_pylist()


def test_a_context_needs_no_future_at_all() -> None:
    built = context(future=None, known_features=[])
    assert built.future is None


def test_static_features_are_carried_through() -> None:
    built = context(
        history=factories.history(
            instances=("DE", "FR"),
            instance_key="zone",
            periods=6,
            targets=("price",),
            observed=("load_actual",),
            known=("wind_fc",),
            static={"capacity": {"DE": 80.0, "FR": 60.0}},
        ),
        static_features=["capacity"],
    )
    assert built.static is not None
    assert built.static.to_pydict() == {"zone": ["DE", "FR"], "capacity": [80.0, 60.0]}


def test_a_frame_is_required() -> None:
    with pytest.raises(DataError, match="must be a TimeSeriesFrame"):
        ForecastContext(
            origin_time=ORIGIN,
            frame="not a frame",  # pyright: ignore[reportArgumentType]
        )


def test_the_underlying_frame_is_still_validated() -> None:
    duplicated = pd.concat([factories.history(periods=2)] * 2, ignore_index=True)
    with pytest.raises(DataError, match="duplicate instance/time rows"):
        ForecastContext.from_pandas(
            duplicated,
            origin_time=ORIGIN,
            event_time="timestamp",
            frequency="1h",
            targets=["load"],
        )


def test_contexts_compare_by_value() -> None:
    assert context() == context()
    assert context() != context(future=None, known_features=[])
    assert context() != "not a context"


def test_repr_names_the_origin() -> None:
    assert "2026-01-01T05:00:00" in repr(context())
