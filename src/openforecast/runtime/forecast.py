"""``Forecast``: what a fitted model answered, in one shape.

```text
zone event_time target kind     quantile sample value

DE   12:00      price  point    null     null   80
DE   13:00      price  point    null     null   78
```

One long table rather than a wide one. A wide forecast's columns depend on what
was asked for — one per target, or per target and quantile, or per target and
sample path — so a wide representation changes shape with the request and cannot
be read by one reader. The long one is the same table however the question was
put, which is what lets it be an Arrow file, an HTTP response and a DataFrame
without a translation at each boundary.

The instance keys come first under the caller's own names, because a forecast
has to come back labeled with the instance it is about.

The conveniences read *out* of that one table rather than replacing it:

```text
forecast.table          the long forecast, as it is
forecast.point()        the point rows, without the columns describing none
forecast.quantile(0.5)  one level, in the same shape
forecast.to_wide()      one column per target, quantile level or sample path
forecast.to_pandas()    the long forecast as a DataFrame
```

``to_wide`` is the one that changes shape with the request, which is exactly why
it is a projection and not the representation.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

import pyarrow as pa

from openforecast.data._arrow import InstanceKey, column_values, key_rows
from openforecast.errors import DataError, ProviderError
from openforecast.protocol.vocabulary import ForecastColumn, forecast_columns

__all__ = ["Forecast"]

#: How a wide column names the part of the distribution it holds. A point
#: forecast is just the target, because there is only one of it.
QUANTILE_SUFFIX = "q"
SAMPLE_SUFFIX = "s"


class Forecast:
    """The answer to one forecast request, at one origin."""

    def __init__(
        self,
        table: pa.Table,
        *,
        origin_time: datetime,
        horizon: int,
        targets: Sequence[str],
        instance_keys: Sequence[str] = (),
        model: str,
    ) -> None:
        self._origin_time = origin_time
        self._horizon = horizon
        self._targets = tuple(targets)
        self._instance_keys = tuple(instance_keys)
        self._model = model
        self._table = _canonical(table, self._instance_keys)

    # -- accessors ---------------------------------------------------------

    @property
    def table(self) -> pa.Table:
        """The long forecast, in canonical column order."""
        return self._table

    @property
    def origin_time(self) -> datetime:
        """The moment everything in this forecast was known at."""
        return self._origin_time

    @property
    def horizon(self) -> int:
        return self._horizon

    @property
    def targets(self) -> tuple[str, ...]:
        return self._targets

    @property
    def instance_keys(self) -> tuple[str, ...]:
        return self._instance_keys

    @property
    def model(self) -> str:
        """The artifact reference that produced it: ``local/de-price@01K...``."""
        return self._model

    @property
    def num_rows(self) -> int:
        return self._table.num_rows

    @property
    def event_times(self) -> tuple[datetime, ...]:
        """The event times forecast, in ascending order."""
        values: list[Any] = self._table.column(ForecastColumn.EVENT_TIME.value).to_pylist()
        return tuple(sorted(set(values)))

    # -- conveniences ------------------------------------------------------

    def point(self) -> pa.Table:
        """Just the point forecasts, without the columns that describe none.

        ``kind``, ``quantile`` and ``sample`` say which part of a predictive
        distribution a row holds; when every row is a point forecast they say
        the same thing on every row, so they are dropped rather than carried.
        """
        return self._narrow([value == "point" for value in self._kinds()])

    def quantile(self, level: float) -> pa.Table:
        """One quantile level, in the shape :meth:`point` returns.

        A level that was never asked for is refused rather than interpolated
        between the ones that were: a 0.5 derived from a 0.1 and a 0.9 is a
        different number from the one the model would have produced.
        """
        levels = column_values(self._table, ForecastColumn.QUANTILE.value)
        mask = [
            kind == "quantile" and found == level
            for kind, found in zip(self._kinds(), levels, strict=True)
        ]
        if not any(mask):
            available = sorted({found for found in levels if found is not None})
            raise DataError(
                f"this forecast holds no quantile {level}; it holds {available or 'none'}. "
                f"Ask for the levels you need with of.OutputSpec.quantiles([...])"
            )
        return self._narrow(mask)

    def to_wide(self) -> pa.Table:
        """One row per instance and event time, one column per thing forecast.

        ```text
        zone event_time price_q0.1 price_q0.5 price_q0.9

        DE   12:00      65         78         95
        ```

        The column names are the target for a point forecast, and the target
        with the level or the draw index for a probabilistic one. Which columns
        exist therefore depends on what was asked for, which is why this is a
        projection of :attr:`table` and not what a forecast *is*.
        """
        index_columns = [*self._instance_keys, ForecastColumn.EVENT_TIME.value]
        index = key_rows(self._table, index_columns)
        labels = self._wide_labels()
        values = column_values(self._table, ForecastColumn.VALUE.value)

        rows = _ordered(index)
        columns = _ordered(labels)
        cells: dict[tuple[InstanceKey, str], Any] = {}
        for key, label, value in zip(index, labels, values, strict=True):
            if (key, label) in cells:
                raise ProviderError(
                    f"this forecast holds two values for {label} at {key}, so it cannot be "
                    f"widened; the long table in .table is what was answered"
                )
            cells[key, label] = value

        widened: dict[str, pa.Array[Any]] = {
            name: pa.array([key[position] for key in rows], type=self._table.column(name).type)
            for position, name in enumerate(index_columns)
        }
        for label in columns:
            widened[label] = pa.array([cells.get((key, label)) for key in rows], type=pa.float64())
        return pa.table(widened)

    def to_pandas(self) -> Any:
        """The long forecast as a pandas ``DataFrame``.

        pandas is not a dependency of OpenForecast — this converts through
        Arrow, which is where the data already is.
        """
        return self._table.to_pandas()  # pyright: ignore[reportUnknownMemberType]

    # -- internals ---------------------------------------------------------

    def _kinds(self) -> list[str]:
        return column_values(self._table, ForecastColumn.KIND.value)

    def _narrow(self, mask: Sequence[bool]) -> pa.Table:
        """The selected rows, without the columns that describe every one alike.

        ``kind``, ``quantile`` and ``sample`` say which part of a predictive
        distribution a row holds; once the rows have been narrowed to one of
        them they say the same thing on every row, so they are dropped rather
        than carried.
        """
        kept = self._table.filter(pa.array(list(mask)))
        return kept.select(
            [
                *self._instance_keys,
                ForecastColumn.EVENT_TIME.value,
                ForecastColumn.TARGET.value,
                ForecastColumn.VALUE.value,
            ]
        )

    def _wide_labels(self) -> list[str]:
        """The wide column each long row belongs in."""
        targets: list[str] = column_values(self._table, ForecastColumn.TARGET.value)
        levels = column_values(self._table, ForecastColumn.QUANTILE.value)
        draws = column_values(self._table, ForecastColumn.SAMPLE.value)
        return [
            _wide_label(target, kind, level, draw)
            for target, kind, level, draw in zip(targets, self._kinds(), levels, draws, strict=True)
        ]

    # -- dunder ------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Forecast):
            return NotImplemented
        return (
            self._origin_time == other._origin_time
            and self._horizon == other._horizon
            and self._targets == other._targets
            and self._instance_keys == other._instance_keys
            and self._model == other._model
            and bool(self._table.equals(other._table))
        )

    def __repr__(self) -> str:
        return (
            f"Forecast(model={self._model}, origin_time={self._origin_time.isoformat()}, "
            f"horizon={self._horizon}, targets={list(self._targets)}, "
            f"rows={self._table.num_rows})"
        )


def _wide_label(target: str, kind: str, level: float | None, draw: int | None) -> str:
    """What to call the wide column one long row belongs in.

    A point forecast is the target under its own name, because there is one of
    it. Anything else is the target and which part of the distribution it is —
    ``price_q0.9``, ``price_s7`` — since a wide table has to hold several
    numbers per target and event time and tell them apart by column name.
    """
    if kind == "quantile" and level is not None:
        return f"{target}_{QUANTILE_SUFFIX}{level:g}"
    if kind == "sample" and draw is not None:
        return f"{target}_{SAMPLE_SUFFIX}{draw}"
    return target


def _ordered(items: Sequence[Any]) -> list[Any]:
    """The distinct ``items``, in the order they first appear.

    First appearance rather than sorted: the long table already came back in the
    order the forecast was produced in, and re-sorting it here would silently
    disagree with ``.table`` about what row 0 is.
    """
    return list(dict.fromkeys(items))


def _canonical(table: pa.Table, instance_keys: Sequence[str]) -> pa.Table:
    """Require the canonical columns and put them in canonical order.

    Raised as a :class:`~openforecast.errors.ProviderError` because by the time a
    forecast is being constructed the request has already been validated: a table
    that is not a forecast came back from something that was asked for one.
    """
    expected = forecast_columns(instance_keys)
    missing = [name for name in expected if name not in table.column_names]
    if missing:
        raise ProviderError(
            f"a forecast is missing the columns {missing}; it holds {table.column_names}"
        )
    return table.select(list(expected))
