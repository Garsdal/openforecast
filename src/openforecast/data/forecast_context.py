"""``ForecastContext``: exactly one inference origin.

A context is the answer to "what did we know at this moment": target history and
observed features up to the origin, known features beyond it, static features
throughout. It carries no vintages, because there is only one.

That single-origin shape is what inference always has in production — a live
pipeline holds today's inputs, not a history of what it once believed — so the
same type serves both a slice of a
:class:`~openforecast.data.forecast_dataset.ForecastDataset` and freshly built
live data.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any

import pyarrow as pa

from openforecast.data._arrow import (
    InstanceKey,
    column_values,
    summarize,
    table_from_pandas,
)
from openforecast.data.frame import TimeSeriesFrame
from openforecast.data.frequency import Frequency
from openforecast.data.point_in_time import parse_moment
from openforecast.data.schema import TimeSeriesSchema
from openforecast.errors import DataError

__all__ = ["ForecastContext"]


class ForecastContext:
    """One :class:`TimeSeriesFrame` split at one origin time.

    The split is validated, not assumed: every history event time must be at or
    before the origin and every future event time strictly after it. A history
    row past the origin is a value nobody had yet, which is the leakage the
    whole point-in-time model exists to prevent.
    """

    def __init__(self, origin_time: str | datetime, frame: TimeSeriesFrame) -> None:
        if not isinstance(frame, TimeSeriesFrame):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise DataError(f"frame must be a TimeSeriesFrame, got {type(frame).__name__}")
        self._origin_time = parse_moment(origin_time, "origin_time")
        self._frame = frame
        self._validate_split()

    def _validate_split(self) -> None:
        origin = self._origin_time
        time = self._frame.schema.time
        late = _out_of_order(self._frame.history, time, lambda moment: moment > origin)
        if late:
            raise DataError(
                f"history holds {len(late)} event times after the origin "
                f"{origin.isoformat()}: {summarize(late)}; a context describes what was known "
                f"at its origin, so later event times belong in future"
            )
        if self._frame.future is not None:
            early = _out_of_order(self._frame.future, time, lambda moment: moment <= origin)
            if early:
                raise DataError(
                    f"future holds {len(early)} event times at or before the origin "
                    f"{origin.isoformat()}: {summarize(early)}"
                )

    # -- accessors ---------------------------------------------------------

    @property
    def origin_time(self) -> datetime:
        return self._origin_time

    @property
    def frame(self) -> TimeSeriesFrame:
        return self._frame

    @property
    def schema(self) -> TimeSeriesSchema:
        return self._frame.schema

    @property
    def history(self) -> pa.Table:
        return self._frame.history

    @property
    def future(self) -> pa.Table | None:
        return self._frame.future

    @property
    def static(self) -> pa.Table | None:
        return self._frame.static

    @property
    def instances(self) -> tuple[InstanceKey, ...]:
        return self._frame.instances

    # -- construction ------------------------------------------------------

    @classmethod
    def from_arrow(
        cls,
        history: pa.Table,
        *,
        origin_time: str | datetime,
        event_time: str,
        frequency: str | Frequency,
        targets: Sequence[str],
        instance_keys: Sequence[str] = (),
        observed_features: Sequence[str] = (),
        known_features: Sequence[str] = (),
        static_features: Sequence[str] = (),
        future: pa.Table | None = None,
        static: pa.Table | None = None,
    ) -> ForecastContext:
        """Build a context directly, for live inference at ``origin_time``."""
        return cls(
            origin_time=origin_time,
            frame=TimeSeriesFrame.from_arrow(
                history,
                time=event_time,
                frequency=frequency,
                targets=targets,
                instance_keys=instance_keys,
                observed_features=observed_features,
                known_features=known_features,
                static_features=static_features,
                future=future,
                static=static,
            ),
        )

    @classmethod
    def from_pandas(
        cls,
        history: Any,
        *,
        origin_time: str | datetime,
        event_time: str,
        frequency: str | Frequency,
        targets: Sequence[str],
        instance_keys: Sequence[str] = (),
        observed_features: Sequence[str] = (),
        known_features: Sequence[str] = (),
        static_features: Sequence[str] = (),
        future: Any | None = None,
        static: Any | None = None,
    ) -> ForecastContext:
        """Same as :meth:`from_arrow`, for pandas ``DataFrame`` inputs."""
        return cls.from_arrow(
            table_from_pandas(history, "history"),
            origin_time=origin_time,
            event_time=event_time,
            frequency=frequency,
            targets=targets,
            instance_keys=instance_keys,
            observed_features=observed_features,
            known_features=known_features,
            static_features=static_features,
            future=None if future is None else table_from_pandas(future, "future"),
            static=None if static is None else table_from_pandas(static, "static"),
        )

    # -- dunder ------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ForecastContext):
            return NotImplemented
        return self._origin_time == other._origin_time and self._frame == other._frame

    def __repr__(self) -> str:
        return (
            f"ForecastContext(origin_time={self._origin_time.isoformat()}, "
            f"instances={len(self.instances)}, "
            f"history_rows={self.history.num_rows}, "
            f"future_rows={0 if self.future is None else self.future.num_rows})"
        )


def _out_of_order(
    table: pa.Table, time: str, offends: Callable[[datetime], bool]
) -> list[datetime]:
    moments: list[datetime] = column_values(table, time)
    return sorted({moment for moment in moments if offends(moment)})
