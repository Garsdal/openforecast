"""Transports, and the client that does not know which one it is holding.

Step 16's "done when" is a claim about *sameness*, so most of it belongs in
``tests/e2e/test_remote_transport.py``, where a real service answers over a real
socket. What is here is the part that can be checked without one: that a client
given a transport uses it, that a fitted model is named by reference on both
sides, and that a failure projects to a status code and comes back as the
exception it was.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import openforecast as of
from openforecast.errors import (
    DataError,
    ModelRequiresFit,
    OpenForecastError,
    ProviderError,
    RecipeError,
    UnknownModelError,
)
from openforecast.server.transport import HttpTransport, LocalTransport, status_for
from openforecast.server.wire import ForecastBody, ForecastPayload, encode_data

MODEL = "builtin/seasonal-naive"
START = datetime(2026, 1, 1)
HOUR = timedelta(hours=1)


def at(step: int) -> datetime:
    return START + HOUR * step


def frame(periods: int = 12) -> of.TimeSeriesFrame:
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


@pytest.fixture
def transport(tmp_path: Path) -> LocalTransport:
    return LocalTransport(store=tmp_path / "openforecast")


def test_a_client_defaults_to_executing_here(tmp_path: Path) -> None:
    client = of.OpenForecast(store=tmp_path / "openforecast")

    assert isinstance(client.transport, LocalTransport)
    assert client.engine.store.root == tmp_path / "openforecast"


def test_a_client_given_a_transport_uses_it(transport: LocalTransport) -> None:
    client = of.OpenForecast(transport=transport)

    assert client.transport is transport
    assert client.engine is transport.engine


def test_configuring_local_machinery_on_a_client_with_a_transport_is_refused(
    transport: LocalTransport, tmp_path: Path
) -> None:
    """Two answers to "which store" is one of them being silently ignored."""
    with pytest.raises(RecipeError, match="configure them on the transport instead"):
        of.OpenForecast(store=tmp_path / "elsewhere", transport=transport)


def test_a_remote_client_has_no_engine_here() -> None:
    """Reaching for the store through a remote client is assuming local execution."""
    client = of.OpenForecast(transport=HttpTransport("http://localhost:8321"))

    with pytest.raises(RecipeError, match="executes elsewhere"):
        _ = client.engine


def test_the_models_namespace_answers_the_same_two_questions(transport: LocalTransport) -> None:
    client = of.OpenForecast(transport=transport)

    assert MODEL in {str(ref) for ref in client.models.refs()}
    assert client.models.get(MODEL).provider == "builtin"
    assert client.models.list(provider="nowhere") == ()
    assert MODEL in client.models


def test_a_fit_and_a_forecast_go_through_the_transport(transport: LocalTransport) -> None:
    client = of.OpenForecast(transport=transport)
    handle = client.fit(MODEL, frame(), params={"season_length": 4}, name="de-load")

    assert str(handle.ref).startswith("local/de-load@")

    answer = client.forecast(handle, frame(), horizon=4)

    assert answer.horizon == 4
    assert answer.model == str(handle.ref)
    assert answer.num_rows == 8  # two zones, four steps


def test_a_fitted_model_is_named_by_reference_across_the_boundary(
    transport: LocalTransport,
) -> None:
    """A handle is a pinned reference plus a manifest the service already has.

    Sending the reference is therefore not a narrowing of the local API: the
    alias, the pinned revision and the handle all name the same artifact, and a
    remote call can only send the name.
    """
    client = of.OpenForecast(transport=transport)
    handle = client.fit(MODEL, frame(), params={"season_length": 4}, name="de-load")

    by_handle = client.forecast(handle, frame(), horizon=4)
    by_revision = client.forecast(str(handle.ref), frame(), horizon=4)
    by_alias = client.forecast("local/de-load", frame(), horizon=4)

    assert by_handle == by_revision == by_alias


def test_an_artifact_can_be_described_without_loading_the_model(
    transport: LocalTransport,
) -> None:
    client = of.OpenForecast(transport=transport)
    handle = client.fit(MODEL, frame(), params={"season_length": 4}, name="de-load")

    assert client.artifact("local/de-load").ref == handle.ref
    assert client.artifact(handle.ref).manifest.provider == "builtin"


def test_forecasting_with_a_recipe_is_refused_rather_than_fitted(
    transport: LocalTransport,
) -> None:
    """A number from a model the caller never trained is worse than an error."""
    client = of.OpenForecast(transport=transport)

    with pytest.raises(RecipeError, match="a forecast is made with a fitted model"):
        client.forecast(
            of.Ensemble(models=(of.Model(MODEL), of.Model(MODEL, params={"season_length": 4}))),
            frame(),
            horizon=4,
        )


def test_fitting_a_fitted_artifact_is_refused(transport: LocalTransport) -> None:
    client = of.OpenForecast(transport=transport)
    handle = client.fit(MODEL, frame(), params={"season_length": 4})

    with pytest.raises(RecipeError, match="is a fitted artifact, not a model to fit"):
        client.fit(handle, frame())


def test_an_unfitted_reference_still_refuses_to_forecast(transport: LocalTransport) -> None:
    client = of.OpenForecast(transport=transport)

    with pytest.raises(ModelRequiresFit):
        client.forecast(MODEL, frame(), horizon=4)


def test_the_transport_carries_a_recipe_and_a_reference_alike(
    transport: LocalTransport,
) -> None:
    """``of.fit("builtin/...")`` and ``of.fit(of.Model("builtin/..."))`` are one call."""
    client = of.OpenForecast(transport=transport)

    from_string = client.fit(MODEL, frame(), params={"season_length": 4}, name="a")
    from_recipe = client.fit(of.Model(MODEL, params={"season_length": 4}), frame(), name="b")

    assert from_string.manifest.recipe_hash == from_recipe.manifest.recipe_hash


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (UnknownModelError("no such model"), 404),
        (DataError("that is not the data it was fitted on"), 422),
        (RecipeError("that pipeline forecasts nothing"), 422),
        (ModelRequiresFit("fit it first"), 422),
        (ProviderError("the provider died"), 502),
        (OpenForecastError("something else"), 422),
    ],
)
def test_a_failure_projects_to_the_status_that_says_who_has_to_act(
    error: OpenForecastError, expected: int
) -> None:
    """A name that resolves to nothing, a request that is refused, or a provider."""
    assert status_for(error) == expected


def test_an_unreachable_service_is_a_provider_failure_naming_the_url() -> None:
    """Nothing was asked of an engine, so it is not the request that is wrong."""
    transport = HttpTransport("http://127.0.0.1:9", timeout=1.0)

    with pytest.raises(ProviderError, match="cannot reach the forecasting service"):
        transport.models()


def test_a_local_transport_answers_a_forecast_as_the_wire_payload(
    transport: LocalTransport,
) -> None:
    """The metadata is control and the table is bulk, on both transports."""
    client = of.OpenForecast(transport=transport)
    handle = client.fit(MODEL, frame(), params={"season_length": 4}, name="de-load")

    payload = transport.forecast(
        ForecastBody(model=str(handle.ref), data=encode_data(frame()), horizon=4)
    )

    assert isinstance(payload, ForecastPayload)
    assert payload.targets == ("load",)
    assert payload.instance_keys == ("zone",)
    assert payload.origin_time == at(11)
