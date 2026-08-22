"""Vocabulary shared by layers that may not import each other.

A model's ``TrainingContract`` names the execution view it consumes, and the
view types are defined in :mod:`openforecast.views` — which sits *below*
``models/`` in the layering, so ``models/`` cannot import it. Defining a second
enum with the same members there would let the two drift apart silently, and
one of the two spellings would eventually reach the wire.

So the enum lives here instead, in the innermost layer, where both the contract
that requests a view and the view that satisfies it can name the same one.
:mod:`openforecast.views` re-exports it, so a provider's import surface is
unchanged.

The forecast columns are here for the same reason and one more: they are the
*answer* half of the provider boundary. A provider writes them and the engine
reads them, and the two are on opposite sides of the layering — in Step 9,
opposite sides of a subprocess.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

__all__ = ["ForecastColumn", "ViewKind", "forecast_columns"]


class ViewKind(StrEnum):
    """Which execution view a model consumes.

    Each names a training unit rather than a model family:

    ```text
    series      one complete time series           ARIMA, ETS, Theta
    sequences   many context -> horizon sequences  NHiTS, TFT, PatchTST
    tabular     individual supervised target rows  LightGBM, XGBoost, CatBoost
    ```

    ``forecast`` is the inference counterpart of all three.
    """

    SERIES = "series"
    SEQUENCES = "sequences"
    TABULAR = "tabular"
    FORECAST = "forecast"


class ForecastColumn(StrEnum):
    """The columns of a forecast, whoever produced it.

    One long table rather than a wide one, because the shape of a wide forecast
    depends on what was asked for — one column per target, or per target and
    quantile, or per target and sample path — and a representation that changes
    shape with the request cannot be read by one reader:

    ```text
    zone event_time target kind     quantile sample value

    DE   12:00      price  point    null     null   80
    DE   12:00      price  quantile 0.1      null   65
    DE   12:00      price  quantile 0.9      null   95
    ```

    The instance keys come first, under the names the caller gave them: a
    forecast has to come back labeled with the instance it belongs to.
    """

    EVENT_TIME = "event_time"
    TARGET = "target"
    #: ``point``, ``quantile`` or ``sample`` — the spellings of
    #: :class:`~openforecast.tasks.OutputKind`, singular because one row is one
    #: number rather than a set of them.
    KIND = "kind"
    #: The level, for a quantile row; null otherwise.
    QUANTILE = "quantile"
    #: The draw index, for a sample row; null otherwise.
    SAMPLE = "sample"
    VALUE = "value"


def forecast_columns(instance_keys: Sequence[str] = ()) -> tuple[str, ...]:
    """The full column order of a forecast over ``instance_keys``."""
    return (*instance_keys, *(column.value for column in ForecastColumn))
