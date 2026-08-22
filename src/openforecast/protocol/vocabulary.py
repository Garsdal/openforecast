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
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["ViewKind"]


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
