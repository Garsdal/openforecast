"""A model with no hand-written entry executes through the shared adapter."""

from __future__ import annotations

import math
from pathlib import Path

import golden


def test_naive_is_discovered_fitted_and_forecast(tmp_path: Path) -> None:
    frame = golden.event_time_frame(periods=24)
    client = golden.client(tmp_path)

    handle = client.fit("sktime/naive", frame)
    forecast = client.forecast(handle, frame, horizon=3)

    assert forecast.num_rows == 3
    assert all(math.isfinite(value) for value in golden.values(forecast))
