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
forecast.sample(7)      one draw, in the same shape
forecast.to_wide()      one column per target, quantile level or sample path
forecast.to_pandas()    the long forecast as a DataFrame
```

``to_wide`` is the one that changes shape with the request, which is exactly why
it is a projection and not the representation.

``to_quantiles`` is the one conversion between output kinds, and it goes one
way:

```text
samples  ->  quantiles      the draws are the distribution; read it
quantiles -> samples        refused: the paths would have to be invented
point    ->  anything       refused: there is no distribution to read
```
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

import pyarrow as pa

from openforecast.data._arrow import InstanceKey, column_values, is_missing, key_rows
from openforecast.errors import DataError, ProviderError
from openforecast.protocol.quantiles import quantile_of_samples
from openforecast.protocol.vocabulary import ForecastColumn, forecast_columns
from openforecast.tasks.forecast import OutputKind, OutputSpec

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

    @property
    def kind(self) -> OutputKind:
        """Which form of answer this forecast holds.

        One kind, because a forecast answers one request: a table mixing point
        rows with quantile rows could not say what its ``value`` column means at
        a given event time, so it is refused rather than reported.
        """
        found = {OutputKind.of_row(kind) for kind in self._kinds()}
        if len(found) != 1:
            raise ProviderError(
                f"a forecast holds one kind of answer and this one holds "
                f"{sorted(kind.value for kind in found)}"
            )
        return found.pop()

    @property
    def quantile_levels(self) -> tuple[float, ...]:
        """The levels this forecast holds, ascending; empty if it holds none."""
        levels = column_values(self._table, ForecastColumn.QUANTILE.value)
        return tuple(sorted({level for level in levels if level is not None}))

    @property
    def sample_indices(self) -> tuple[int, ...]:
        """The draw indices this forecast holds, ascending; empty if it holds none."""
        draws = column_values(self._table, ForecastColumn.SAMPLE.value)
        return tuple(sorted({draw for draw in draws if draw is not None}))

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

    def sample(self, draw: int) -> pa.Table:
        """One sample path, in the shape :meth:`point` returns."""
        draws = column_values(self._table, ForecastColumn.SAMPLE.value)
        mask = [
            kind == "sample" and found == draw
            for kind, found in zip(self._kinds(), draws, strict=True)
        ]
        if not any(mask):
            raise DataError(
                f"this forecast holds no sample path {draw}; it holds "
                f"{list(self.sample_indices) or 'none'}"
            )
        return self._narrow(mask)

    def to_quantiles(self, levels: Sequence[float]) -> Forecast:
        """The quantiles of the sample paths this forecast holds.

        ```python
        forecast = of.forecast(model=model, data=context, horizon=24,
                               output=of.OutputSpec.samples(200))
        quantiles = forecast.to_quantiles([0.1, 0.5, 0.9])
        ```

        The draws *are* the predictive distribution, so reading levels out of
        them is a projection of what the model said — the same kind of thing
        :meth:`to_wide` is. The reverse is not: turning quantiles back into
        sample paths would require inventing the paths, and reading either out of
        a point forecast would require inventing the distribution, so both are
        refused rather than approximated.

        What comes back is a :class:`Forecast` like any other, which is the point:
        downstream code consuming quantiles does not learn whether the provider
        produced them natively or drew them.
        """
        if self.kind is not OutputKind.SAMPLES:
            raise DataError(
                f"only a sample forecast can be reduced to quantiles, and this one holds "
                f"{self.kind.value}; ask the model for quantiles with "
                f"of.OutputSpec.quantiles({list(levels)})"
            )
        requested = OutputSpec.quantiles(levels).levels  # the same validation a request gets
        return Forecast(
            _quantiles_of_samples(self._table, self._instance_keys, requested),
            origin_time=self._origin_time,
            horizon=self._horizon,
            targets=self._targets,
            instance_keys=self._instance_keys,
            model=self._model,
        )

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


def _quantiles_of_samples(
    table: pa.Table, instance_keys: Sequence[str], levels: Sequence[float]
) -> pa.Table:
    """The sample rows of ``table``, reduced to one row per level.

    The draws of one instance, event time and target are one predictive
    distribution, so they are gathered before anything is read out of them. A
    distribution holding a missing draw yields a missing quantile rather than a
    quantile of the draws that happen to be numbers: a model that answered a NaN
    said it did not know, and dropping that would report a narrower distribution
    than it gave.
    """
    index_columns = [
        *instance_keys,
        ForecastColumn.EVENT_TIME.value,
        ForecastColumn.TARGET.value,
    ]
    observations = key_rows(table, index_columns)
    values = column_values(table, ForecastColumn.VALUE.value)

    draws: dict[InstanceKey, list[Any]] = {}
    for observation, value in zip(observations, values, strict=True):
        draws.setdefault(observation, []).append(value)

    rows = [(observation, level) for observation in draws for level in levels]
    built: dict[str, pa.Array[Any]] = {
        name: pa.array(
            [observation[position] for observation, _ in rows], type=table.column(name).type
        )
        for position, name in enumerate(index_columns)
    }
    built[ForecastColumn.KIND.value] = pa.array(
        [OutputKind.QUANTILES.row_kind] * len(rows), type=pa.string()
    )
    built[ForecastColumn.QUANTILE.value] = pa.array([level for _, level in rows], type=pa.float64())
    built[ForecastColumn.SAMPLE.value] = pa.nulls(len(rows), type=pa.int64())
    built[ForecastColumn.VALUE.value] = pa.array(
        [_level_of(draws[observation], level) for observation, level in rows], type=pa.float64()
    )
    return pa.table(built)


def _level_of(draws: Sequence[Any], level: float) -> float | None:
    """One quantile of one predictive distribution, or null if a draw is missing."""
    if any(value is None or is_missing(value) for value in draws):
        return None
    return quantile_of_samples([float(value) for value in draws], level)


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
