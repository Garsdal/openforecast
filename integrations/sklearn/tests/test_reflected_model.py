"""A model with no hand-written entry executes through the shared adapter."""

from __future__ import annotations

import math
from pathlib import Path

import golden


def test_ridge_is_discovered_fitted_and_forecast(tmp_path: Path) -> None:
    data = golden.point_in_time_dataset(origins=8, horizon=3)
    client = golden.client(tmp_path)

    handle = client.fit("sklearn/ridge", data, horizon=3)
    forecast = client.forecast(handle, data.at_origin(golden.at(9)), horizon=3)

    assert forecast.num_rows == 3
    assert all(math.isfinite(value) for value in golden.values(forecast))
