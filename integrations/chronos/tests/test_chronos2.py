"""The real checkpoint, end to end. Deselected unless ``-m weights`` is given.

Everything else in this suite runs against a stand-in, because a checkpoint is a
download and none of the structural assertions are about the numbers. This file
is the one that would notice if the library's own API moved underneath the
adapter — a renamed keyword, a changed tensor layout, a pipeline class that no
longer answers ``predict_quantiles``.

```bash
uv run pytest -m weights
```

What it asserts is deliberately weak about accuracy and strong about semantics.
"Chronos forecasts electricity prices well" is not a property this repository
can test; "a zero-shot forecast of 24 steps comes back as 24 finite numbers
labeled with the event times that were asked about, without anything having been
fitted" is, and that is the claim Step 23 makes.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from openforecast_chronos import ChronosProvider

import openforecast as of
from openforecast.errors import ModelDoesNotSupportFit
from openforecast.models.catalog import ModelCatalog
from openforecast.runtime.provider import ProviderRegistry

pytestmark = pytest.mark.weights

START = datetime(2026, 1, 1)
HOUR = timedelta(hours=1)
PERIODS = 96
HORIZON = 24
LEVELS = (0.1, 0.5, 0.9)

PROVIDER = ChronosProvider()


@pytest.fixture
def client(tmp_path: Path) -> of.OpenForecast:
    return of.OpenForecast(
        store=tmp_path,
        catalog=ModelCatalog(PROVIDER.descriptors()),
        providers=ProviderRegistry([PROVIDER]),
    )


def frame() -> of.TimeSeriesFrame:
    """A daily-seasonal hourly series, which is what the model is shown."""
    import pandas as pd

    moments = [START + HOUR * step for step in range(PERIODS)]
    values = [100.0 + 20.0 * math.sin(2 * math.pi * step / 24) for step in range(PERIODS)]
    return of.TimeSeriesFrame.from_pandas(
        pd.DataFrame({"timestamp": moments, "load": values}),
        time="timestamp",
        frequency="1h",
        targets=["load"],
    )


def test_a_pretrained_model_forecasts_without_being_fitted(client: of.OpenForecast) -> None:
    forecast = client.forecast("amazon/chronos-2", frame(), horizon=HORIZON)

    assert forecast.model == "amazon/chronos-2"
    assert len(forecast.event_times) == HORIZON
    assert forecast.event_times[0] == START + HOUR * PERIODS
    assert forecast.table.num_rows == HORIZON
    values = forecast.table.column("value").to_pylist()
    assert all(value is not None and math.isfinite(value) for value in values)


def test_the_same_model_answers_a_quantile_request(client: of.OpenForecast) -> None:
    forecast = client.forecast(
        "amazon/chronos-2",
        frame(),
        horizon=HORIZON,
        output=of.OutputSpec.quantiles(list(LEVELS)),
    )

    assert forecast.kind is of.OutputKind.QUANTILES
    assert forecast.quantile_levels == LEVELS
    assert forecast.table.num_rows == HORIZON * len(LEVELS)


def test_fitting_it_is_refused(client: of.OpenForecast) -> None:
    with pytest.raises(ModelDoesNotSupportFit, match="cannot be fitted"):
        client.fit("amazon/chronos-2", frame(), horizon=HORIZON)


def test_it_is_backtested_beside_a_fitted_model_at_the_same_origins(
    client: of.OpenForecast,
) -> None:
    """The done-when of Step 23, with one of the two candidates in this environment."""
    result = of.backtest(
        models=["amazon/chronos-2"],
        data=frame(),
        validation=of.RollingOrigin(horizon=HORIZON, windows=2),
        metrics=[of.MAE()],
        client=client,
    )

    table = result.leaderboard("mae")
    assert table.num_rows == 1
    # Nothing was fitted at any origin, and the result says so rather than
    # reporting a fit of zero seconds.
    assert set(result.metrics.column("fit_seconds").to_pylist()) == {None}
    assert set(result.metrics.column("origin_fidelity").to_pylist()) == {"pretrained"}
    assert set(result.metrics.column("artifact").to_pylist()) == {"amazon/chronos-2"}
