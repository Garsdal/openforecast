"""A model with no hand-written entry executes through the shared adapter."""

from __future__ import annotations

import math
from pathlib import Path

import golden

import openforecast as of


def test_fft_is_discovered_fitted_and_forecast(tmp_path: Path) -> None:
    frame = golden.event_time_frame(periods=24)
    client = golden.client(tmp_path)

    handle = client.fit("darts/fft", frame)
    forecast = client.forecast(handle, frame, horizon=3)

    assert forecast.num_rows == 3
    assert all(math.isfinite(value) for value in golden.values(forecast))


def test_global_naive_is_discovered_fitted_and_forecast(tmp_path: Path) -> None:
    frame = golden.event_time_frame(periods=24, future_periods=3)
    client = golden.client(tmp_path)

    handle = client.fit(
        "darts/global-naive-aggregate",
        frame,
        horizon=3,
        plan=of.FitPlan(window=of.WindowPlan(context=6)),
    )
    forecast = client.forecast(handle, frame, horizon=3)

    assert forecast.num_rows == 3
    assert all(math.isfinite(value) for value in golden.values(forecast))
