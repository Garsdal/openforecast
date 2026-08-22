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
has to come back labeled with the instance it is about. The wide conveniences —
``to_wide``, ``quantile`` — arrive with the public V1 surface in Step 15; what
is here is the representation they will be built on.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

import pyarrow as pa

from openforecast.errors import ProviderError
from openforecast.protocol.vocabulary import ForecastColumn, forecast_columns

__all__ = ["Forecast"]


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
        mask = [
            value == "point" for value in self._table.column(ForecastColumn.KIND.value).to_pylist()
        ]
        kept = self._table.filter(pa.array(mask))
        return kept.select(
            [
                *self._instance_keys,
                ForecastColumn.EVENT_TIME.value,
                ForecastColumn.TARGET.value,
                ForecastColumn.VALUE.value,
            ]
        )

    def to_pandas(self) -> Any:
        """The long forecast as a pandas ``DataFrame``.

        pandas is not a dependency of OpenForecast — this converts through
        Arrow, which is where the data already is.
        """
        return self._table.to_pandas()  # pyright: ignore[reportUnknownMemberType]

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
