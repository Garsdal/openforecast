"""``ForecastView``: the standardized inference representation.

One origin, one horizon, three tables. Unlike the fit views, this one keeps the
caller's instance keys: a forecast has to come back labeled with the instance it
belongs to.

```text
history   instance keys, event_time, targets, observed features, known features
future    instance keys, event_time, known features
static    instance keys, static features
```

``future`` names exactly the event times being asked about — the horizon steps
after the origin — so a provider never has to derive them from a horizon count
and a frequency. Everything at or before the origin is in ``history`` and
everything after it is in ``future``, which is the only split a provider needs
to know about.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import ClassVar

import pyarrow as pa
from pydantic import Field

from openforecast.data._arrow import (
    InstanceKey,
    column_values,
    group_times,
    key_rows,
    require_unique,
    summarize,
    tables_equal,
)
from openforecast.errors import DataError
from openforecast.views.base import (
    EVENT_TIME,
    ViewKind,
    ViewSchema,
    prepare,
    require_rows,
)

__all__ = ["ForecastView", "ForecastViewMetadata"]


class ForecastViewMetadata(ViewSchema):
    """What the columns of a :class:`ForecastView` mean, and how far it looks.

    ``context`` is the number of history steps the view was trimmed to. It is
    ``None`` when the whole history was kept, which is what a local forecaster
    that fits at inference time needs.
    """

    expected_kind: ClassVar[ViewKind] = ViewKind.FORECAST
    reserved: ClassVar[tuple[str, ...]] = (EVENT_TIME,)

    kind: ViewKind = ViewKind.FORECAST
    horizon: int = Field(ge=1)
    context: int | None = Field(default=None, ge=1)

    @property
    def history_columns(self) -> tuple[str, ...]:
        """``instance keys, event_time, targets, observed features, known features``."""
        return (*self.instance_keys, EVENT_TIME, *self.targets, *self.temporal_feature_names)

    @property
    def future_columns(self) -> tuple[str, ...]:
        """``instance keys, event_time, known features``."""
        return (
            *self.instance_keys,
            EVENT_TIME,
            *(feature.name for feature in self.known_features),
        )

    @property
    def static_columns(self) -> tuple[str, ...]:
        """``instance keys, static features``."""
        return (*self.instance_keys, *self.static_feature_names)


class ForecastView:
    """Everything knowable at one origin, and the event times to predict."""

    def __init__(
        self,
        origin_time: datetime,
        history: pa.Table,
        future: pa.Table,
        metadata: ForecastViewMetadata,
        static: pa.Table | None = None,
    ) -> None:
        self._origin_time = origin_time
        self._metadata = metadata
        self._history = require_rows(
            prepare(history, metadata.history_columns, "history"), "history"
        )
        self._future = require_rows(prepare(future, metadata.future_columns, "future"), "future")
        self._static = _resolve_static(static, metadata, self.instances)
        _validate_split(self._history, self._future, origin_time, metadata)

    # -- accessors ---------------------------------------------------------

    @property
    def kind(self) -> ViewKind:
        return self._metadata.kind

    @property
    def origin_time(self) -> datetime:
        return self._origin_time

    @property
    def metadata(self) -> ForecastViewMetadata:
        return self._metadata

    @property
    def history(self) -> pa.Table:
        return self._history

    @property
    def future(self) -> pa.Table:
        """The event times being forecast, with the features known for them."""
        return self._future

    @property
    def static(self) -> pa.Table | None:
        return self._static

    @property
    def instances(self) -> tuple[InstanceKey, ...]:
        return tuple(dict.fromkeys(key_rows(self._history, self._metadata.instance_keys)))

    @property
    def event_times(self) -> tuple[datetime, ...]:
        """The horizon steps being forecast, in ascending order."""
        return tuple(sorted(set(column_values(self._future, EVENT_TIME))))

    # -- dunder ------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ForecastView):
            return NotImplemented
        return (
            self._origin_time == other._origin_time
            and self._metadata == other._metadata
            and bool(self._history.equals(other._history))
            and bool(self._future.equals(other._future))
            and tables_equal(self._static, other._static)
        )

    def __repr__(self) -> str:
        return (
            f"ForecastView(origin_time={self._origin_time.isoformat()}, "
            f"instances={len(self.instances)}, "
            f"horizon={self._metadata.horizon}, "
            f"history_rows={self._history.num_rows})"
        )


def _resolve_static(
    static: pa.Table | None, metadata: ForecastViewMetadata, instances: Sequence[InstanceKey]
) -> pa.Table | None:
    if static is None:
        if metadata.has_static_features:
            raise DataError(
                f"metadata declares static features {list(metadata.static_feature_names)} "
                f"but no static table was provided"
            )
        return None
    if not metadata.has_static_features:
        raise DataError("a static table was provided but the metadata declares no static features")
    table = prepare(static, metadata.static_columns, "static")
    keys = key_rows(table, metadata.instance_keys)
    if sorted(keys, key=repr) != sorted(instances, key=repr):
        raise DataError(
            f"static must hold exactly one row per instance; it holds {len(keys)} rows "
            f"for {len(instances)} instances"
        )
    return table


def _validate_split(
    history: pa.Table,
    future: pa.Table,
    origin: datetime,
    metadata: ForecastViewMetadata,
) -> None:
    """The origin splits the two tables, and the future is exactly the horizon."""
    frequency = metadata.frequency
    for table, label in ((history, "history"), (future, "future")):
        require_unique(
            table,
            (*metadata.instance_keys, EVENT_TIME),
            label,
            what="instance/event",
            hint="each event time may appear once per instance",
        )

    late = sorted({moment for moment in column_values(history, EVENT_TIME) if moment > origin})
    if late:
        raise DataError(
            f"history holds {len(late)} event times after the origin {origin.isoformat()}: "
            f"{summarize(late)}; a forecast view describes what was known at its origin"
        )

    expected = [frequency.shift(origin, step) for step in range(1, metadata.horizon + 1)]
    grouped = group_times(
        key_rows(future, metadata.instance_keys), column_values(future, EVENT_TIME)
    )
    for instance, times in grouped.items():
        if sorted(times) != expected:
            raise DataError(
                f"future must hold exactly the {metadata.horizon} horizon steps after "
                f"{origin.isoformat()}{_of(instance)}, expected {summarize(expected)}, "
                f"got {summarize(sorted(times))}"
            )

    if metadata.context is None:
        return
    history_times = group_times(
        key_rows(history, metadata.instance_keys), column_values(history, EVENT_TIME)
    )
    wanted = [frequency.shift(origin, -step) for step in reversed(range(metadata.context))]
    for instance, times in history_times.items():
        if sorted(times) != wanted:
            raise DataError(
                f"history must hold exactly the {metadata.context} context steps ending at "
                f"{origin.isoformat()}{_of(instance)}, expected {summarize(wanted)}, "
                f"got {summarize(sorted(times))}"
            )


def _of(instance: InstanceKey) -> str:
    return f" for instance {instance}" if instance else ""
