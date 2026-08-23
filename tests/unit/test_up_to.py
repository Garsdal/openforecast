"""``up_to``: the semantic sources, as they stood at a moment.

The two truncations are what make a historical origin something a model can be
fitted at without being told about the future, and they are tested here rather
than through a backtest because a leak in either would be invisible in a metric.

The point-in-time cases lean on the golden property of ``tests.factories``: a
known feature's value names the origin that produced it, so a value from a later
vintage is identifiable rather than merely suspicious.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import pytest

import openforecast as of
from openforecast.errors import DataError
from tests.factories import HOUR, START, history, point_in_time


def at(step: int) -> datetime:
    return START + HOUR * step


def frame(**kwargs: Any) -> of.TimeSeriesFrame:
    rows = history(instances=("DE", "FR"), instance_key="zone", periods=8, **kwargs)
    return of.TimeSeriesFrame.from_pandas(
        history=rows,
        time="timestamp",
        frequency="1h",
        instance_keys=["zone"],
        targets=["load"],
        observed_features=list(kwargs.get("observed", ())),
        known_features=list(kwargs.get("known", ())),
        static_features=list(kwargs.get("static", {})),
    )


def dataset(**kwargs: Any) -> of.ForecastDataset:
    rows = point_in_time(origins=5, horizon=4, **kwargs)
    return of.ForecastDataset.from_pandas(
        rows,
        origin_time="ref_time",
        event_time="target_time",
        targets=["price"],
        event_frequency="1h",
        origin_frequency="1h",
        known_features=["wind_fc"],
    )


def times(table: Any, column: str = "timestamp") -> list[datetime]:
    values: list[datetime] = table.column(column).to_pylist()
    return values


# -- event-time data --------------------------------------------------------


def test_it_keeps_the_history_up_to_and_including_the_moment() -> None:
    past = frame().up_to(at(4))

    assert max(times(past.history)) == at(4)
    assert len(times(past.history)) == 10  # five event times, two instances


def test_a_target_after_the_moment_is_gone_rather_than_masked() -> None:
    """The whole point: a fit at this origin cannot reach what happened next."""
    past = frame().up_to(at(4))

    assert at(5) not in times(past.history)
    assert past.future is None


def test_known_features_of_later_event_times_survive_as_the_future() -> None:
    """A known feature is knowable in advance, so truncating history is not losing it."""
    past = frame(known=("wind_fc",)).up_to(at(4))

    assert past.future is not None
    assert sorted(set(times(past.future))) == [at(step) for step in (5, 6, 7)]
    assert "wind_fc" in past.future.column_names
    assert "load" not in past.future.column_names


def test_observed_features_do_not_become_knowable_by_being_truncated_around() -> None:
    past = frame(observed=("measured",), known=("wind_fc",)).up_to(at(4))

    assert "measured" in past.history.column_names
    assert past.future is not None
    assert "measured" not in past.future.column_names


def test_an_existing_future_table_and_the_truncated_history_are_one_future() -> None:
    """Both say the same thing about the same feature, so both are carried."""
    data = of.TimeSeriesFrame.from_pandas(
        history=history(instances=("DE",), periods=8, known=("wind_fc",)),
        future=pd.DataFrame(
            [{"timestamp": at(step), "wind_fc": 99.0} for step in (8, 9)],
        ),
        time="timestamp",
        frequency="1h",
        targets=["load"],
        known_features=["wind_fc"],
    )

    past = data.up_to(at(4))

    assert past.future is not None
    assert sorted(set(times(past.future))) == [at(step) for step in (5, 6, 7, 8, 9)]


def test_static_rows_follow_the_instances_that_are_left() -> None:
    data = of.TimeSeriesFrame.from_pandas(
        history=history(
            instances=("DE", "FR"),
            instance_key="zone",
            periods=8,
            static={"capacity": {"DE": 1.0, "FR": 2.0}},
        ),
        time="timestamp",
        frequency="1h",
        instance_keys=["zone"],
        targets=["load"],
        static_features=["capacity"],
    )

    past = data.up_to(at(4))

    assert past.static is not None
    assert past.static.num_rows == 2


def test_a_moment_before_the_history_is_an_error_rather_than_an_empty_frame() -> None:
    with pytest.raises(DataError, match="at or before"):
        frame().up_to(at(-1))


def test_the_moment_may_be_written_as_a_timestamp_string() -> None:
    assert frame().up_to(at(4).isoformat()) == frame().up_to(at(4))


# -- point-in-time data -----------------------------------------------------


def test_only_the_vintages_issued_by_then_remain() -> None:
    past = dataset().up_to(at(2))

    assert past.origins == (at(0), at(1), at(2))


def test_a_later_vintage_is_absent_rather_than_unused() -> None:
    """The guarantee backtesting rests on, stated as a property of the object.

    ``wind_fc`` names the origin that produced it, so a value from the 03:00
    vintage is recognizable. Nothing downstream can reach one, because the
    truncated dataset does not hold one.
    """
    full = dataset()
    past = full.up_to(at(2))

    issued: list[Any] = past.information.table.column("ref_time").to_pylist()
    kept: list[Any] = past.information.table.column("wind_fc").to_pylist()
    everything: list[Any] = full.information.table.column("wind_fc").to_pylist()

    assert max(issued) == at(2)
    assert all(value < 300 for value in kept)
    assert any(value >= 300 for value in everything)


def test_the_truth_stops_at_the_moment_too() -> None:
    """An outcome after the moment had not happened, whichever vintage described it."""
    past = dataset().up_to(at(2))

    assert max(times(past.truth.history, "target_time")) == at(2)


def test_a_truncated_dataset_is_still_a_dataset() -> None:
    past = dataset().up_to(at(2))

    assert isinstance(past, of.ForecastDataset)
    assert past.at_origin(at(2)).origin_time == at(2)
    assert past.targets == ("price",)


def test_the_moment_need_not_be_one_of_the_origins() -> None:
    """It is a point in time; ``at_origin`` is the one that names a vintage."""
    between = dataset().up_to(at(2) + HOUR / 2)

    assert between.origins == (at(0), at(1), at(2))


def test_a_moment_before_any_vintage_is_an_error() -> None:
    with pytest.raises(DataError, match="no vintage was issued"):
        dataset().up_to(at(-1))
