"""Fitted-artifact fixtures: a materialized view, and the artifact describing it.

Built through the real ``ViewPlanner`` rather than by hand, so that what the
manifest records is what a fit would actually have been handed — a training
record assembled from a view nobody could have materialized would test nothing.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

import openforecast as of
from openforecast.artifacts import ModelArtifact, new_artifact_id
from openforecast.recipes.nodes import Recipe
from openforecast.views import (
    SequenceView,
    SeriesView,
    TabularView,
    ViewKind,
    ViewPlanner,
    ViewRequest,
)
from tests import factories

START = datetime(2026, 1, 1, 0, 0, 0)
HOUR = timedelta(hours=1)

CONTEXT = 3
HORIZON = 2

planner = ViewPlanner()


def at(step: int) -> datetime:
    return START + HOUR * step


def frame(**overrides: Any) -> of.TimeSeriesFrame:
    """A two-zone panel with one target and one known feature."""
    options: dict[str, Any] = {
        "instances": ("DE", "FR"),
        "instance_key": "zone",
        "periods": 8,
        "known": ["temp_fc"],
    }
    options.update(overrides)
    return of.TimeSeriesFrame.from_pandas(
        history=factories.history(**options),
        time="timestamp",
        frequency="1h",
        instance_keys=["zone"],
        targets=["load"],
        known_features=["temp_fc"],
    )


def dataset() -> of.ForecastDataset:
    """The same shape, from real vintages, so origin fidelity is ``observed``."""
    rows = [
        {
            "zone": "DE",
            "ref_time": at(origin),
            "target_time": at(event),
            "price": float(event),
            "wind_fc": float(event),
        }
        for origin in range(8)
        for event in range(8)
    ]
    return of.ForecastDataset.from_pandas(
        pd.DataFrame(rows),
        origin_time="ref_time",
        event_time="target_time",
        instance_keys=["zone"],
        targets=["price"],
        known_features=["wind_fc"],
        event_frequency="1h",
        origin_frequency="1h",
    )


def sequence_view(data: object | None = None, **request: Any) -> SequenceView:
    options: dict[str, Any] = {"context": CONTEXT, "horizon": HORIZON}
    options.update(request)
    view = planner.fit_view(
        frame() if data is None else data, ViewRequest(kind=ViewKind.SEQUENCES, **options)
    )
    assert isinstance(view, SequenceView)
    return view


def series_view(data: object | None = None, **request: Any) -> SeriesView:
    view = planner.fit_view(
        frame() if data is None else data, ViewRequest(kind=ViewKind.SERIES, **request)
    )
    assert isinstance(view, SeriesView)
    return view


def tabular_view(data: object | None = None, **request: Any) -> TabularView:
    options: dict[str, Any] = {"horizon": HORIZON}
    options.update(request)
    view = planner.fit_view(
        frame() if data is None else data, ViewRequest(kind=ViewKind.TABULAR, **options)
    )
    assert isinstance(view, TabularView)
    return view


def artifact_id(second: int) -> str:
    """A deterministic id whose ordering is the ordering of ``second``."""
    return new_artifact_id(
        moment=datetime(2026, 8, 22, tzinfo=UTC) + timedelta(seconds=second),
        entropy=bytes(10),
    )


def artifact(
    *,
    name: str = "de-price",
    recipe: Recipe | None = None,
    view: SequenceView | SeriesView | TabularView | None = None,
    source_model: str = "nixtla/nhits",
    plan: of.FitPlan | None = None,
    artifact_id: str | None = None,
) -> ModelArtifact:
    """The artifact one fit of ``recipe`` on ``view`` would produce."""
    return ModelArtifact.of_fit(
        name=name,
        source_model=source_model,
        recipe=of.Model(source_model) if recipe is None else recipe,
        view=sequence_view() if view is None else view,
        provider="nixtla",
        provider_version="0.1.0",
        openforecast_version=of.__version__,
        plan=of.FitPlan(window=of.WindowPlan(context=CONTEXT), seed=42) if plan is None else plan,
        artifact_id=artifact_id,
    )
