# Forecasts

*Generated from the code by `uv run generate-reference`. Do not edit by hand.*

What a forecast is: one long table, and the projections of it.

## `Forecast`

*Class — `openforecast.runtime.forecast`*

```python
Forecast(table: pa.Table, *, origin_time: datetime, horizon: int, targets: Sequence[str], instance_keys: Sequence[str] = (), model: str) -> None
```

The answer to one forecast request, at one origin.

| Member | Kind | Summary |
| --- | --- | --- |
| `event_times` | property | The event times forecast, in ascending order. |
| `horizon` | property |  |
| `instance_keys` | property |  |
| `kind` | property | Which form of answer this forecast holds. |
| `model` | property | The artifact reference that produced it: ``local/de-price@01K...``. |
| `num_rows` | property |  |
| `origin_time` | property | The moment everything in this forecast was known at. |
| `point(self) -> pa.Table` | method | Just the point forecasts, without the columns that describe none. |
| `quantile(self, level: float) -> pa.Table` | method | One quantile level, in the shape :meth:`point` returns. |
| `quantile_levels` | property | The levels this forecast holds, ascending; empty if it holds none. |
| `sample(self, draw: int) -> pa.Table` | method | One sample path, in the shape :meth:`point` returns. |
| `sample_indices` | property | The draw indices this forecast holds, ascending; empty if it holds none. |
| `table` | property | The long forecast, in canonical column order. |
| `targets` | property |  |
| `to_pandas(self) -> Any` | method | The long forecast as a pandas ``DataFrame``. |
| `to_quantiles(self, levels: Sequence[float]) -> Forecast` | method | The quantiles of the sample paths this forecast holds. |
| `to_wide(self) -> pa.Table` | method | One row per instance and event time, one column per thing forecast. |
