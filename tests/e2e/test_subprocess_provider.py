"""Step 9's "done when", end to end.

> ``Engine`` can swap in a subprocess provider without knowing it is a
> subprocess.

So the test is a comparison rather than an assertion about plumbing: the same
model is fitted and forecast twice — once in this process, once by a child
process reached over JSON Lines and Arrow bundles — and the two forecasts have
to be the same numbers. Anything the transport quietly changed about the view,
the parameters or the answer would show up here.

The provider on the far side is the built-in one, served by the harness an
integration's ``__main__`` will run. Nothing about it knows it is remote.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import openforecast as of
from openforecast.models.catalog import ModelCatalog
from openforecast.providers import BUILTIN_PROVIDER
from openforecast.runtime import Engine, ProviderClient, ProviderRegistry, SubprocessProvider
from openforecast.runtime.providers import register_descriptors
from tests import wire

MODEL = "builtin/seasonal-naive"
PARAMS = {"season_length": 4}
START = datetime(2026, 1, 1)
HOUR = timedelta(hours=1)


def at(step: int) -> datetime:
    return START + HOUR * step


def frame(periods: int = 16) -> of.TimeSeriesFrame:
    rows: list[dict[str, Any]] = [
        {"zone": zone, "timestamp": at(step), "load": float(step + offset * 100)}
        for offset, zone in enumerate(("DE", "FR"))
        for step in range(periods)
    ]
    return of.TimeSeriesFrame.from_pandas(
        history=pd.DataFrame(rows),
        time="timestamp",
        frequency="1h",
        instance_keys=["zone"],
        targets=["load"],
    )


def engine(store: Path, provider: ProviderClient) -> Engine:
    """An engine that knows one provider and nothing about how it runs."""
    catalog = ModelCatalog()
    register_descriptors([provider], catalog)
    return Engine(
        store=of.OpenForecast(store=store).engine.store,
        catalog=catalog,
        providers=ProviderRegistry([provider]),
    )


@pytest.fixture
def remote() -> Iterator[SubprocessProvider]:
    """The built-in provider, in a child process speaking the protocol."""
    provider = SubprocessProvider(wire.REFERENCE_PROVIDER, timeout=60)
    yield provider
    provider.close()


def test_a_subprocess_provider_is_discovered_exactly_like_a_local_one(
    remote: SubprocessProvider,
) -> None:
    catalog = ModelCatalog()

    register_descriptors([remote], catalog)

    assert [str(ref) for ref in catalog.list()] == [MODEL]
    assert catalog.get(MODEL) == BUILTIN_PROVIDER.descriptors()[0]


def test_the_same_model_forecasts_the_same_numbers_over_a_process_boundary(
    tmp_path: Path, remote: SubprocessProvider
) -> None:
    data = frame()
    context = of.ForecastContext(origin_time=at(15), frame=data)

    local = engine(tmp_path / "local-store", BUILTIN_PROVIDER)
    remote_engine = engine(tmp_path / "remote-store", remote)

    here = local.forecast(
        local.fit(MODEL, data, params=PARAMS, name="here"), context, horizon=6
    ).table
    there = remote_engine.forecast(
        remote_engine.fit(MODEL, data, params=PARAMS, name="there"), context, horizon=6
    ).table

    assert there.equals(here)


def test_an_artifact_fitted_over_the_wire_reloads_and_forecasts_again(
    tmp_path: Path, remote: SubprocessProvider
) -> None:
    """A fit through the transport is an ordinary artifact, not a session."""
    data = frame()
    store = tmp_path / "store"
    fitting = engine(store, remote)

    handle = fitting.fit(MODEL, data, params=PARAMS, name="de-load")
    manifest = handle.manifest

    reloaded = engine(store, remote)
    forecast = reloaded.forecast(
        "local/de-load", of.ForecastContext(origin_time=at(15), frame=data), horizon=4
    )

    assert manifest.provider == "builtin"
    assert manifest.provider_version == BUILTIN_PROVIDER.version
    assert forecast.num_rows == 8  # two zones, four steps


def test_a_failure_inside_the_provider_arrives_as_the_error_it_is(
    tmp_path: Path, remote: SubprocessProvider
) -> None:
    """A model rejecting its own parameters is a recipe error, wherever it ran."""
    fitting = engine(tmp_path / "store", remote)

    with pytest.raises(of.RecipeError, match=r"season_length"):
        fitting.fit(MODEL, frame(), params={"season_length": -1}, name="broken")

    assert not list((tmp_path / "store" / "models").glob("*")), "a failed fit left an artifact"
