from __future__ import annotations

import math
from datetime import datetime, timedelta

import pandas as pd
import pyarrow as pa
import pytest

from openforecast import DataError, TimeSeriesFrame
from tests import factories


def is_nan(value: object) -> bool:
    return isinstance(value, float) and math.isnan(value)


def test_single_univariate() -> None:
    frame = TimeSeriesFrame.from_pandas(
        factories.history(periods=4),
        time="timestamp",
        frequency="1h",
        targets=["load"],
    )
    assert not frame.schema.is_panel
    assert frame.schema.is_univariate
    assert frame.history.column_names == ["timestamp", "load"]
    assert frame.history.num_rows == 4
    assert frame.future is None
    assert frame.static is None
    assert frame.instances == ((),)


def test_single_multivariate() -> None:
    frame = TimeSeriesFrame.from_pandas(
        factories.history(periods=4, targets=("load", "price")),
        time="timestamp",
        frequency="1h",
        targets=["load", "price"],
    )
    assert frame.schema.is_multivariate
    assert frame.history.column_names == ["timestamp", "load", "price"]


def test_panel_univariate() -> None:
    frame = TimeSeriesFrame.from_pandas(
        factories.history(instances=("DE", "FR", "NL"), instance_key="country", periods=4),
        time="timestamp",
        frequency="1h",
        instance_keys=["country"],
        targets=["load"],
    )
    assert frame.schema.is_panel
    assert frame.schema.is_univariate
    assert frame.history.column_names == ["country", "timestamp", "load"]
    assert frame.history.num_rows == 12
    assert frame.instances == (("DE",), ("FR",), ("NL",))


def test_panel_multivariate() -> None:
    frame = TimeSeriesFrame.from_pandas(
        factories.history(
            instances=("DE", "FR"),
            instance_key="country",
            periods=4,
            targets=("load", "price"),
        ),
        time="timestamp",
        frequency="1h",
        instance_keys=["country"],
        targets=["load", "price"],
    )
    assert frame.schema.is_panel
    assert frame.schema.is_multivariate
    assert frame.history.column_names == ["country", "timestamp", "load", "price"]


def test_the_documented_public_api() -> None:
    """The example from the plan, end to end."""
    frame = TimeSeriesFrame.from_pandas(
        factories.history(
            instances=("DE", "FR"),
            instance_key="country",
            periods=6,
            observed=("temperature_actual",),
            known=("temperature_forecast",),
            static={"capacity": {"DE": 80.0, "FR": 60.0}},
        ),
        time="timestamp",
        frequency="1h",
        instance_keys=["country"],
        targets=["load"],
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
    assert frame.history.column_names == [
        "country",
        "timestamp",
        "load",
        "temperature_actual",
        "temperature_forecast",
    ]
    assert frame.future is not None
    assert frame.future.column_names == ["country", "timestamp", "temperature_forecast"]
    assert frame.static is not None
    assert frame.static.column_names == ["country", "capacity"]
    assert frame.static.to_pylist() == [
        {"country": "DE", "capacity": 80.0},
        {"country": "FR", "capacity": 60.0},
    ]


def test_static_features_may_come_from_their_own_table() -> None:
    frame = TimeSeriesFrame.from_pandas(
        factories.history(instances=("DE", "FR"), instance_key="country", periods=3),
        time="timestamp",
        frequency="1h",
        instance_keys=["country"],
        targets=["load"],
        static_features=["capacity"],
        static=pd.DataFrame({"country": ["DE", "FR"], "capacity": [80.0, 60.0]}),
    )
    assert frame.static is not None
    assert frame.static.to_pylist() == [
        {"country": "DE", "capacity": 80.0},
        {"country": "FR", "capacity": 60.0},
    ]


def test_a_static_feature_that_varies_within_an_instance_is_rejected() -> None:
    history = factories.history(instances=("DE",), instance_key="country", periods=3)
    history["capacity"] = [80.0, 80.0, 81.0]
    with pytest.raises(DataError, match="varies within an instance"):
        TimeSeriesFrame.from_pandas(
            history,
            time="timestamp",
            frequency="1h",
            instance_keys=["country"],
            targets=["load"],
            static_features=["capacity"],
        )


def test_a_static_feature_of_all_nans_is_still_static() -> None:
    """NaN is a value here, not a marker: it must not read as 'varies'."""
    history = pa.Table.from_pandas(
        factories.history(instances=("DE",), instance_key="country", periods=3),
        preserve_index=False,
    ).append_column("capacity", pa.array([float("nan")] * 3, type=pa.float64()))
    frame = TimeSeriesFrame.from_arrow(
        history,
        time="timestamp",
        frequency="1h",
        instance_keys=["country"],
        targets=["load"],
        static_features=["capacity"],
    )
    assert frame.static is not None
    assert is_nan(frame.static.column("capacity").to_pylist()[0])


def test_undeclared_columns_are_dropped() -> None:
    history = factories.history(periods=3)
    history["scratch"] = 1.0
    frame = TimeSeriesFrame.from_pandas(
        history,
        time="timestamp",
        frequency="1h",
        targets=["load"],
    )
    assert "scratch" not in frame.history.column_names


def test_missing_target_values_are_never_imputed() -> None:
    history = factories.history(periods=3)
    history.loc[1, "load"] = float("nan")
    frame = TimeSeriesFrame.from_pandas(
        history,
        time="timestamp",
        frequency="1h",
        targets=["load"],
    )
    # pandas maps its own NaN onto an Arrow null on conversion; either way the
    # value stays missing.
    assert frame.history.column("load").to_pylist()[1] is None
    assert frame.history.column("load").null_count == 1


def test_nan_is_carried_through_arrow_untouched() -> None:
    """A NaN that arrives as a NaN stays one; it is not folded into null."""
    table = pa.table(
        {
            "timestamp": pa.array(factories.timestamps(3), type=pa.timestamp("us")),
            "load": pa.array([1.0, float("nan"), None], type=pa.float64()),
        }
    )
    frame = TimeSeriesFrame.from_arrow(table, time="timestamp", frequency="1h", targets=["load"])
    values = frame.history.column("load").to_pylist()
    assert values[0] == 1.0
    assert is_nan(values[1])
    assert values[2] is None


def test_gaps_in_the_time_axis_are_allowed() -> None:
    """A missing observation is information; filling it in would be a repair."""
    history = factories.history(periods=6).drop(index=[2, 3]).reset_index(drop=True)
    frame = TimeSeriesFrame.from_pandas(
        history,
        time="timestamp",
        frequency="1h",
        targets=["load"],
    )
    assert frame.history.num_rows == 4


def test_a_declared_column_must_exist() -> None:
    with pytest.raises(DataError, match="missing declared columns"):
        TimeSeriesFrame.from_pandas(
            factories.history(periods=3),
            time="timestamp",
            frequency="1h",
            targets=["load", "price"],
        )


def test_duplicate_instance_time_rows_are_rejected() -> None:
    history = factories.history(instances=("DE",), instance_key="country", periods=3)
    duplicated = pd.concat([history, history.iloc[[1]]], ignore_index=True)
    with pytest.raises(DataError, match="duplicate instance/time rows"):
        TimeSeriesFrame.from_pandas(
            duplicated,
            time="timestamp",
            frequency="1h",
            instance_keys=["country"],
            targets=["load"],
        )


def test_the_same_event_time_may_appear_once_per_instance() -> None:
    frame = TimeSeriesFrame.from_pandas(
        factories.history(instances=("DE", "FR"), instance_key="country", periods=3),
        time="timestamp",
        frequency="1h",
        instance_keys=["country"],
        targets=["load"],
    )
    assert frame.history.num_rows == 6


def test_off_grid_timestamps_are_rejected() -> None:
    history = factories.history(periods=4)
    moments = factories.timestamps(4)
    moments[2] += timedelta(minutes=20)
    history["timestamp"] = moments
    with pytest.raises(DataError, match="do not sit on the 1h grid"):
        TimeSeriesFrame.from_pandas(
            history,
            time="timestamp",
            frequency="1h",
            targets=["load"],
        )


def test_a_declared_frequency_that_is_too_coarse_is_rejected() -> None:
    with pytest.raises(DataError, match="do not sit on the 1d grid"):
        TimeSeriesFrame.from_pandas(
            factories.history(periods=4),
            time="timestamp",
            frequency="1d",
            targets=["load"],
        )


def test_a_finer_declared_frequency_is_accepted_because_gaps_are_legal() -> None:
    """Hourly data is not 15-minute data, even though every hour is on the grid."""
    frame = TimeSeriesFrame.from_pandas(
        factories.history(periods=4),
        time="timestamp",
        frequency="15m",
        targets=["load"],
    )
    # 15m divides 1h, so this is accepted: the timestamps really do sit on the
    # declared grid. Gaps are legal, so nothing here is a contradiction.
    assert frame.history.num_rows == 4


def test_the_time_column_must_be_a_timestamp() -> None:
    table = pa.table({"timestamp": ["2026-01-01T00:00:00"], "load": [1.0]})
    with pytest.raises(DataError, match="must be a timestamp"):
        TimeSeriesFrame.from_arrow(table, time="timestamp", frequency="1h", targets=["load"])


def test_null_timestamps_are_rejected() -> None:
    table = pa.table(
        {
            "timestamp": pa.array([datetime(2026, 1, 1), None], type=pa.timestamp("us")),
            "load": [1.0, 2.0],
        }
    )
    with pytest.raises(DataError, match="null values in time column"):
        TimeSeriesFrame.from_arrow(table, time="timestamp", frequency="1h", targets=["load"])


def test_null_instance_keys_are_rejected() -> None:
    table = pa.table(
        {
            "country": pa.array(["DE", None], type=pa.string()),
            "timestamp": pa.array(
                [datetime(2026, 1, 1), datetime(2026, 1, 1, 1)], type=pa.timestamp("us")
            ),
            "load": [1.0, 2.0],
        }
    )
    with pytest.raises(DataError, match="null values in instance key"):
        TimeSeriesFrame.from_arrow(
            table,
            time="timestamp",
            frequency="1h",
            instance_keys=["country"],
            targets=["load"],
        )


def test_the_future_must_not_contain_targets() -> None:
    future = factories.future(periods=2, known=("temperature_forecast",))
    future["load"] = 1.0
    with pytest.raises(DataError, match="must not contain target columns"):
        TimeSeriesFrame.from_pandas(
            factories.history(periods=4, known=("temperature_forecast",)),
            time="timestamp",
            frequency="1h",
            targets=["load"],
            known_features=["temperature_forecast"],
            future=future,
        )


def test_the_future_must_not_contain_observed_features() -> None:
    future = factories.future(periods=2, known=("temperature_forecast",))
    future["temperature_actual"] = 1.0
    with pytest.raises(DataError, match="must not contain observed-only features"):
        TimeSeriesFrame.from_pandas(
            factories.history(
                periods=4,
                observed=("temperature_actual",),
                known=("temperature_forecast",),
            ),
            time="timestamp",
            frequency="1h",
            targets=["load"],
            observed_features=["temperature_actual"],
            known_features=["temperature_forecast"],
            future=future,
        )


def test_the_future_must_declare_its_known_features() -> None:
    with pytest.raises(DataError, match="future is missing declared columns"):
        TimeSeriesFrame.from_pandas(
            factories.history(periods=4, known=("temperature_forecast",)),
            time="timestamp",
            frequency="1h",
            targets=["load"],
            known_features=["temperature_forecast"],
            future=factories.future(periods=2),
        )


def test_the_future_shares_the_history_grid() -> None:
    future = factories.future(periods=2, known=("temperature_forecast",))
    future["timestamp"] = future["timestamp"] + timedelta(minutes=30)
    with pytest.raises(DataError, match="future has 2 timestamps"):
        TimeSeriesFrame.from_pandas(
            factories.history(periods=4, known=("temperature_forecast",)),
            time="timestamp",
            frequency="1h",
            targets=["load"],
            known_features=["temperature_forecast"],
            future=future,
        )


def test_the_future_must_not_introduce_new_instances() -> None:
    with pytest.raises(DataError, match="instances absent from history"):
        TimeSeriesFrame.from_pandas(
            factories.history(instances=("DE",), instance_key="country", periods=4),
            time="timestamp",
            frequency="1h",
            instance_keys=["country"],
            targets=["load"],
            future=factories.future(instances=("DE", "FR"), instance_key="country", periods=2),
        )


def test_static_needs_one_row_per_instance() -> None:
    with pytest.raises(DataError, match="missing rows for instances"):
        TimeSeriesFrame.from_pandas(
            factories.history(instances=("DE", "FR"), instance_key="country", periods=3),
            time="timestamp",
            frequency="1h",
            instance_keys=["country"],
            targets=["load"],
            static_features=["capacity"],
            static=pd.DataFrame({"country": ["DE"], "capacity": [80.0]}),
        )


def test_static_must_not_repeat_an_instance() -> None:
    with pytest.raises(DataError, match="exactly one row per instance"):
        TimeSeriesFrame.from_pandas(
            factories.history(instances=("DE",), instance_key="country", periods=3),
            time="timestamp",
            frequency="1h",
            instance_keys=["country"],
            targets=["load"],
            static_features=["capacity"],
            static=pd.DataFrame({"country": ["DE", "DE"], "capacity": [80.0, 80.0]}),
        )


def test_static_must_not_describe_unknown_instances() -> None:
    with pytest.raises(DataError, match="instances absent from history"):
        TimeSeriesFrame.from_pandas(
            factories.history(instances=("DE",), instance_key="country", periods=3),
            time="timestamp",
            frequency="1h",
            instance_keys=["country"],
            targets=["load"],
            static_features=["capacity"],
            static=pd.DataFrame({"country": ["DE", "FR"], "capacity": [80.0, 60.0]}),
        )


def test_null_instance_keys_in_static_are_rejected() -> None:
    frame = TimeSeriesFrame.from_pandas(
        factories.history(instances=("DE",), instance_key="country", periods=3),
        time="timestamp",
        frequency="1h",
        instance_keys=["country"],
        targets=["load"],
        static_features=["capacity"],
        static=pd.DataFrame({"country": ["DE"], "capacity": [80.0]}),
    )
    with pytest.raises(DataError, match="static has null values in instance key"):
        TimeSeriesFrame(
            history=frame.history,
            schema=frame.schema,
            static=pa.table(
                {
                    "country": pa.array([None], type=pa.string()),
                    "capacity": pa.array([80.0], type=pa.float64()),
                }
            ),
        )


def test_the_constructor_requires_the_static_table_it_declares() -> None:
    """``from_arrow`` may lift static features out of history; the constructor may not."""
    frame = TimeSeriesFrame.from_pandas(
        factories.history(instances=("DE",), instance_key="country", periods=3),
        time="timestamp",
        frequency="1h",
        instance_keys=["country"],
        targets=["load"],
        static_features=["capacity"],
        static=pd.DataFrame({"country": ["DE"], "capacity": [80.0]}),
    )
    with pytest.raises(DataError, match="no static table was provided"):
        TimeSeriesFrame(history=frame.history, schema=frame.schema)


def test_a_single_series_static_table_holds_exactly_one_row() -> None:
    frame = TimeSeriesFrame.from_pandas(
        factories.history(periods=3),
        time="timestamp",
        frequency="1h",
        targets=["load"],
        static_features=["capacity"],
        static=pd.DataFrame({"capacity": [80.0]}),
    )
    assert frame.static is not None
    assert frame.static.num_rows == 1


def test_declared_static_features_need_a_static_table() -> None:
    schema_columns = factories.history(periods=3)
    table = pa.Table.from_pandas(schema_columns, preserve_index=False)
    with pytest.raises(DataError, match="static features"):
        TimeSeriesFrame.from_arrow(
            table,
            time="timestamp",
            frequency="1h",
            targets=["load"],
            static_features=["capacity"],
        )


def test_a_static_table_without_static_features_is_rejected() -> None:
    frame = TimeSeriesFrame.from_pandas(
        factories.history(periods=3),
        time="timestamp",
        frequency="1h",
        targets=["load"],
    )
    with pytest.raises(DataError, match="declares no static features"):
        TimeSeriesFrame(
            history=frame.history,
            schema=frame.schema,
            static=pa.table({"capacity": [80.0]}),
        )


def test_history_must_be_an_arrow_table() -> None:
    with pytest.raises(DataError, match="must be a pyarrow.Table"):
        TimeSeriesFrame.from_arrow(
            factories.history(periods=3),  # type: ignore[arg-type]
            time="timestamp",
            frequency="1h",
            targets=["load"],
        )


def test_from_pandas_rejects_non_dataframes() -> None:
    with pytest.raises(DataError, match="not a pandas DataFrame"):
        TimeSeriesFrame.from_pandas(
            {"timestamp": [], "load": []},
            time="timestamp",
            frequency="1h",
            targets=["load"],
        )


def test_repr_describes_the_shape() -> None:
    frame = TimeSeriesFrame.from_pandas(
        factories.history(instances=("DE", "FR"), instance_key="country", periods=3),
        time="timestamp",
        frequency="1h",
        instance_keys=["country"],
        targets=["load"],
    )
    assert repr(frame) == (
        "TimeSeriesFrame(panel univariate, frequency=1h, instances=2, "
        "history_rows=6, future_rows=0)"
    )
