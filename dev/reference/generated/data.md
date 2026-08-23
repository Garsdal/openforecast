# Semantic data

*Generated from the code by `uv run generate-reference`. Do not edit by hand.*

What you hand OpenForecast: ordinary event-time data, real forecast vintages, and one inference origin cut out of them.

## `FeatureAvailability`

*Enumeration — `openforecast.data.features`*

| Member | Value |
| --- | --- |
| `OBSERVED` | `'observed'` |
| `KNOWN` | `'known'` |

## `FeatureKind`

*Enumeration — `openforecast.data.features`*

| Member | Value |
| --- | --- |
| `TEMPORAL` | `'temporal'` |
| `STATIC` | `'static'` |

## `FeatureSpec`

*Pydantic model — `openforecast.data.features`*

One non-target column, with its semantics.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | `str` | *required* |  |
| `kind` | `FeatureKind` | `FeatureKind.TEMPORAL` |  |
| `availability` | `FeatureAvailability \| None` | `None` |  |

## `ForecastContext`

*Class — `openforecast.data.forecast_context`*

```python
ForecastContext(origin_time: str | datetime, frame: TimeSeriesFrame) -> None
```

One :class:`TimeSeriesFrame` split at one origin time.

The split is validated, not assumed: every history event time must be at or
before the origin and every future event time strictly after it. A history
row past the origin is a value nobody had yet, which is the leakage the
whole point-in-time model exists to prevent.

| Member | Kind | Summary |
| --- | --- | --- |
| `frame` | property |  |
| `future` | property |  |
| `history` | property |  |
| `instances` | property |  |
| `origin_time` | property |  |
| `schema` | property |  |
| `static` | property |  |

## `ForecastDataset`

*Class — `openforecast.data.forecast_dataset`*

```python
ForecastDataset(information: PointInTimeFrame, truth: TimeSeriesFrame) -> None
```

Point-in-time information and the outcomes it was trying to predict.

| Member | Kind | Summary |
| --- | --- | --- |
| `at_origin(self, origin_time: str \| datetime) -> ForecastContext` | method | Everything, and only what, was knowable at ``origin_time``. |
| `information` | property |  |
| `instances` | property |  |
| `origins` | property | Every distinct origin time in the information, in ascending order. |
| `targets` | property |  |
| `truth` | property |  |
| `up_to(self, moment: str \| datetime) -> ForecastDataset` | method | Every vintage issued at or before ``moment``, and the truth known by then. |
| `write(self, path: str \| Path) -> Path` | method | Write the information and truth frames into subdirectories of ``path``. |

## `Frequency`

*Pydantic model — `openforecast.data.frequency`*

How far apart consecutive steps of a time axis are.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `unit` | `FrequencyUnit` | *required* |  |
| `step` | `int` | `1` |  |

## `FrequencyUnit`

*Enumeration — `openforecast.data.frequency`*

| Member | Value |
| --- | --- |
| `SECOND` | `'second'` |
| `MINUTE` | `'minute'` |
| `HOUR` | `'hour'` |
| `DAY` | `'day'` |
| `WEEK` | `'week'` |
| `MONTH` | `'month'` |

## `PointInTimeFrame`

*Class — `openforecast.data.point_in_time`*

```python
PointInTimeFrame(table: pa.Table, schema: PointInTimeSchema) -> None
```

One Arrow table keyed by ``(instance keys..., origin_time, event_time)``.

NaNs and nulls are preserved exactly. An availability that improves between
vintages — ``NaN``, ``NaN``, ``42`` — is information about the data feed,
and imputing it away would destroy the very thing point-in-time training is
for.

| Member | Kind | Summary |
| --- | --- | --- |
| `at_origin(self, origin_time: str \| datetime) -> PointInTimeFrame` | method | The single vintage issued at ``origin_time``. |
| `event_times` | property | Every distinct event time, in ascending order. |
| `instances` | property | The distinct instance keys present, in first-seen order. |
| `origins` | property | Every distinct origin time, in ascending order. |
| `schema` | property |  |
| `table` | property |  |
| `with_lead_time(self, unit: str \| Frequency = 'hour', *, name: str = 'lead_time') -> PointInTimeFrame` | method | A copy carrying ``event_time - origin_time`` as a known feature. |
| `write(self, path: str \| Path) -> Path` | method | Write ``schema.json`` and ``table.arrow`` into ``path``. |

## `PointInTimeSchema`

*Pydantic model — `openforecast.data.point_in_time`*

What the columns of a :class:`PointInTimeFrame` mean.

There are no targets here. A point-in-time frame describes information, not
outcomes; what actually happened lives in the ``truth`` side of a
:class:`~openforecast.data.forecast_dataset.ForecastDataset`.

``origin_frequency`` is optional because vintages are often irregular — a
day-ahead run at 10:00 and an intraday run at 14:00 sit on no single grid.
Declaring it opts into validating that they do.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `origin_time` | `str` | *required* |  |
| `event_time` | `str` | *required* |  |
| `event_frequency` | `Frequency` | *required* |  |
| `origin_frequency` | `Frequency \| None` | `None` |  |
| `instance_keys` | `tuple[str, ...]` | `()` |  |
| `features` | `tuple[FeatureSpec, ...]` | *required* |  |

## `TimeSeriesFrame`

*Class — `openforecast.data.frame`*

```python
TimeSeriesFrame(history: pa.Table, schema: TimeSeriesSchema, future: pa.Table | None = None, static: pa.Table | None = None) -> None
```

Event-time time-series data validated against a :class:`TimeSeriesSchema`.

The tables are stored in canonical column order. Columns of the input that
the schema does not declare are dropped; columns the schema does declare
must be present.

| Member | Kind | Summary |
| --- | --- | --- |
| `future` | property |  |
| `history` | property |  |
| `instances` | property | The distinct instance keys present in ``history``, in first-seen order. |
| `schema` | property |  |
| `static` | property |  |
| `up_to(self, moment: str \| datetime) -> TimeSeriesFrame` | method | This frame as it would have looked at ``moment``. |
| `write(self, path: str \| Path) -> Path` | method | Write ``schema.json`` and one Arrow IPC file per table into ``path``. |

## `TimeSeriesSchema`

*Pydantic model — `openforecast.data.schema`*

What the columns of a :class:`~openforecast.data.frame.TimeSeriesFrame` mean.

The semantic axes are orthogonal and the interesting categories are derived
from them: a panel of several targets is ``is_panel and is_multivariate``,
not a ``PANEL_MULTIVARIATE`` enum member.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `time` | `str` | *required* |  |
| `frequency` | `Frequency` | *required* |  |
| `instance_keys` | `tuple[str, ...]` | `()` |  |
| `targets` | `tuple[str, ...]` | *required* |  |
| `features` | `tuple[FeatureSpec, ...]` | `()` |  |
