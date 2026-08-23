"""Step 15's "done when", against the real integrations.

> The local OpenForecast V1 experience is coherent and provider-independent.

The other end-to-end modules prove that a mechanism works. This one proves that
the *experience* does: everything below goes through ``of.fit``, ``of.forecast``
and ``of.models``, over models executed by three different forecasting libraries
in three different interpreters, and nothing in it names a library.

```text
discover              nixtla, darts and sktime models in one catalog
fit AutoARIMA         one vintage of a point-in-time dataset
reload and forecast   through a second client, sharing only a directory
fit NHiTS             across every historical origin
fit darts/tide        the same dataset, the same plan, one string changed
fit sktime            the reduction, whose horizon is not bound at fit
aliases               local/de-price follows the latest fit; @01K... does not
terminology           nothing a provider calls its own reaches a public object
```

The fits are deliberately tiny — two optimization steps, one epoch, five
boosting iterations. What is being asserted is that the same request means the
same thing to every provider, not how well a neural network fits a six-step
window.

Running this needs the integrations installed as provider environments, which
is what ``openforecast providers install nixtla`` does. Nothing is installed
here: a test suite that builds several gigabytes of PyTorch as a side effect of
being collected is not one anybody runs twice. Without them the module skips —
unless ``OPENFORECAST_E2E`` is set, which is how the CI job that does install
them says that a skip would be a failure.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pytest

import openforecast as of
from openforecast.models.catalog import ModelCatalog
from openforecast.providers import BUILTIN_PROVIDER
from openforecast.runtime import ProviderClient, ProviderRegistry, SubprocessProvider
from openforecast.runtime.environments import ProviderEnvironment, ProviderEnvironments
from openforecast.runtime.providers import register_descriptors
from tests.conformance import datasets
from tests.unit.test_architecture import PROVIDER_TERMS

#: Where to look for installed provider environments. The default is the one
#: ``openforecast providers install`` writes to; a checkout that would rather not
#: build several gigabytes into a user cache directory can point this elsewhere.
PROVIDER_ROOT_VARIABLE = "OPENFORECAST_PROVIDER_ROOT"

#: Set where the environments were installed on purpose — the CI job that
#: installs them, for one. A missing provider is then a failure rather than a
#: skip, because a suite that quietly skipped all of itself is indistinguishable
#: from one that passed.
REQUIRED_VARIABLE = "OPENFORECAST_E2E"

AUTOARIMA = "nixtla/autoarima"
NHITS = "nixtla/nhits"
TIDE = "darts/tide"
POOLED_TREES = "sktime/pooled-trees"

#: The shape of the golden point-in-time datasets below, which is the shape the
#: conformance suite uses: origin ``k`` sits at event step ``context - 1 + k``.
CONTEXT = 3
HORIZON = 3
ORIGINS = 6
#: The freshest origin the vintages carry, and the step it sits at.
LATEST_ORIGIN_STEP = CONTEXT - 1 + ORIGINS - 1
LATEST_ORIGIN = datasets.at(LATEST_ORIGIN_STEP)

#: Enough optimization to prove the request arrived; see the module docstring.
FAST: Mapping[str, Mapping[str, Any]] = {
    NHITS: {"max_steps": 2},
    TIDE: {"n_epochs": 1},
    POOLED_TREES: {"max_iter": 5},
}

BuildClient = Callable[..., of.OpenForecast]


# -- the data ---------------------------------------------------------------


def windows() -> of.ForecastDataset:
    """Real vintages, each carrying the window around its own origin.

    What every global model here is fitted on: three zones, six historical
    origins, one target and a known feature whose values name the origin that
    published them, so a vintage leaking into another is identifiable rather
    than merely suspicious.
    """
    return datasets.point_in_time(
        instances=3, origins=ORIGINS, context=CONTEXT, horizon=HORIZON, static=True
    )


def complete_series() -> of.ForecastDataset:
    """The same vintages, each reaching back to the first event time.

    A series model trains on one *complete* time series at one origin, so its
    vintages have to describe everything behind that origin rather than a
    window. The static feature is dropped because AutoARIMA takes exogenous
    regressors that reach into the future and nothing else — which is a
    statement the descriptor makes and the engine enforces, not something this
    test knows about ARIMA.
    """
    return datasets.point_in_time(
        instances=3, origins=ORIGINS, context=CONTEXT, horizon=HORIZON, cumulative=True
    )


def sequences_plan() -> of.FitPlan:
    """Learn from every historical origin, over a window of ``CONTEXT`` steps."""
    return of.FitPlan(origins=of.AllOrigins(), window=of.WindowPlan(context=CONTEXT), seed=11)


def values(forecast: of.Forecast) -> list[Any]:
    column: list[Any] = forecast.point().column("value").to_pylist()
    return column


# -- the installed providers ------------------------------------------------


@pytest.fixture(scope="session")
def installed() -> Mapping[str, ProviderEnvironment]:
    root = os.environ.get(PROVIDER_ROOT_VARIABLE)
    environments = ProviderEnvironments(root) if root else ProviderEnvironments()
    return {environment.name: environment for environment in environments.list()}


@pytest.fixture
def build(tmp_path: Path, installed: Mapping[str, ProviderEnvironment]) -> Iterator[BuildClient]:
    """A client over the named providers, or a skip saying how to get them.

    Called twice with the same ``store`` it hands back two clients sharing only
    a directory on disk, which is what reloading an artifact has to survive.
    """
    opened: list[SubprocessProvider] = []

    def make(*names: str, store: str = "store") -> of.OpenForecast:
        missing = [name for name in names if name not in installed]
        if missing:
            how = " && ".join(f"openforecast providers install {name}" for name in missing)
            message = f"needs the {missing} provider environments: {how}"
            if os.environ.get(REQUIRED_VARIABLE):
                pytest.fail(message)
            pytest.skip(message)
        clients: list[ProviderClient] = [BUILTIN_PROVIDER]
        for name in names:
            client = installed[name].client()
            opened.append(client)
            clients.append(client)
        catalog = ModelCatalog()
        register_descriptors(clients, catalog)
        return of.OpenForecast(
            store=tmp_path / store, catalog=catalog, providers=ProviderRegistry(clients)
        )

    yield make
    for provider in opened:
        provider.close()


# -- discovery ---------------------------------------------------------------


def test_every_installed_model_is_discoverable_by_reference_alone(build: BuildClient) -> None:
    """``of.models`` over three libraries, and nothing in the answer names one."""
    models = build("nixtla", "darts", "sktime").models

    assert {str(ref) for ref in models.refs()} >= {AUTOARIMA, NHITS, TIDE, POOLED_TREES}
    assert models.providers() == ("builtin", "darts", "nixtla", "sktime")
    # A listing is for the references; the fields are what ``get`` is for.
    assert repr(models.get(NHITS)) == f"ModelDescriptor({NHITS})"


@pytest.mark.parametrize(
    ("ref", "provider", "view", "horizon_bound"),
    [
        (AUTOARIMA, "nixtla", "series", False),
        (NHITS, "nixtla", "sequences", True),
        (TIDE, "darts", "sequences", True),
        (POOLED_TREES, "sktime", "sequences", False),
    ],
)
def test_a_descriptor_answers_what_a_model_needs_before_it_is_fitted(
    build: BuildClient, ref: str, provider: str, view: str, horizon_bound: bool
) -> None:
    """What ``of.models.get`` is for: planning against a model without running it."""
    descriptor = build(provider).models.get(ref)

    assert descriptor.lifecycle.requires_fit
    assert descriptor.training.view == view
    assert descriptor.training.horizon_bound_at_fit is horizon_bound
    assert descriptor.capabilities.outputs.point


# -- fitting, reloading and forecasting --------------------------------------


def test_a_series_model_fits_one_vintage_reloads_and_forecasts(build: BuildClient) -> None:
    """AutoARIMA at one origin, and a forecast made by a client that never fitted it."""
    data = complete_series()

    handle = build("nixtla").fit(
        AUTOARIMA,
        data,
        plan=of.FitPlan(origins=of.AtOrigin(LATEST_ORIGIN)),
        name="de-price",
    )
    forecast = build("nixtla").forecast(
        "local/de-price", data.at_origin(LATEST_ORIGIN), horizon=HORIZON
    )

    assert handle.training.view == "series"
    assert handle.training.origin_fidelity == "observed"
    assert forecast.model == str(handle.ref)
    assert forecast.origin_time == LATEST_ORIGIN
    assert forecast.num_rows == 3 * HORIZON
    assert all(value == value for value in values(forecast)), "the answer holds NaNs"


@pytest.mark.parametrize(
    ("ref", "provider", "samples"),
    [
        (NHITS, "nixtla", 3 * ORIGINS),
        (TIDE, "darts", 3 * ORIGINS),
        (POOLED_TREES, "sktime", 3 * ORIGINS),
    ],
)
def test_a_global_model_learns_from_every_historical_origin(
    build: BuildClient, ref: str, provider: str, samples: int
) -> None:
    """The Step 15 claim, three times: only the reference changed.

    The same dataset, the same plan, the same horizon and the same two calls —
    fitted by PyTorch through Nixtla, by PyTorch through Darts, and by a pooled
    scikit-learn regressor through sktime. Each learns one training sample per
    real historical origin, which is what makes the fit point-in-time rather
    than windows cut out of one freshest series.
    """
    data = windows()

    handle = build(provider).fit(
        ref, data, horizon=HORIZON, plan=sequences_plan(), params=dict(FAST[ref]), name="de-price"
    )
    forecast = build(provider).forecast(
        "local/de-price", data.at_origin(LATEST_ORIGIN), horizon=HORIZON
    )

    assert handle.training.source == "forecast_dataset"
    assert handle.training.origin_fidelity == "observed"
    assert handle.training.context == CONTEXT
    assert handle.training.samples == samples
    assert forecast.origin_time == LATEST_ORIGIN
    assert forecast.event_times == tuple(
        datasets.at(LATEST_ORIGIN_STEP + step) for step in range(1, HORIZON + 1)
    )
    assert all(value == value for value in values(forecast)), "the answer holds NaNs"


def test_switching_library_changes_the_reference_and_nothing_else(build: BuildClient) -> None:
    """Two libraries, one request, two forecasts that are the same shape.

    Not the same numbers — they are different models. The same instances, the
    same event times, the same targets, labeled the caller's way by both.
    """
    data = windows()
    client = build("nixtla", "darts")
    at_origin = data.at_origin(LATEST_ORIGIN)

    forecasts = [
        client.forecast(
            client.fit(
                ref,
                data,
                horizon=HORIZON,
                plan=sequences_plan(),
                params=dict(FAST[ref]),
                name=ref.replace("/", "-"),
            ),
            at_origin,
            horizon=HORIZON,
        )
        for ref in (NHITS, TIDE)
    ]

    nixtla, darts = forecasts
    assert nixtla.table.drop_columns(["value"]).equals(darts.table.drop_columns(["value"]))
    assert nixtla.instance_keys == darts.instance_keys == ("zone",)


def test_a_pipeline_and_an_ensemble_span_two_libraries(build: BuildClient) -> None:
    """Composition is OpenForecast's, so its members need not agree on anything.

    An ensemble of a Nixtla model and a Darts model, one of them wrapped in a
    pipeline that scales the targets — fitted, combined and unscaled by
    OpenForecast, which is why no part of it is a library's feature.
    """
    data = windows()
    client = build("nixtla", "darts")
    recipe = of.Ensemble(
        models=(
            of.Pipeline(
                steps=(
                    of.StandardScaler(columns=of.ColumnSet.TARGETS),
                    of.Model(NHITS, params=dict(FAST[NHITS])),
                )
            ),
            of.Model(TIDE, params=dict(FAST[TIDE])),
        ),
        combine=of.WeightedMean(weights=(3, 1)),
    )

    handle = client.fit(recipe, data, horizon=HORIZON, plan=sequences_plan(), name="blend")
    forecast = client.forecast(handle, data.at_origin(LATEST_ORIGIN), horizon=HORIZON)

    assert handle.is_composite
    assert handle.manifest.provider == "openforecast"
    assert len(handle.training_records) == 2
    assert forecast.num_rows == 3 * HORIZON


def test_a_horizon_the_reduction_never_bound_is_still_servable(build: BuildClient) -> None:
    """The one contract difference between the three, visible where it belongs.

    ``sktime/pooled-trees`` is a reduction: the horizon is a property of the
    prediction rather than of the fitted parameters. That is a declaration on
    the descriptor, so a longer forecast simply works, where the same request of
    ``nixtla/nhits`` is refused before a provider is started.
    """
    data = windows()
    client = build("sktime")

    handle = client.fit(
        POOLED_TREES,
        data,
        horizon=HORIZON,
        plan=sequences_plan(),
        params=dict(FAST[POOLED_TREES]),
    )
    forecast = client.forecast(handle, data.at_origin(LATEST_ORIGIN), horizon=HORIZON + 2)

    assert handle.serves_horizon(HORIZON + 2)
    assert forecast.num_rows == 3 * (HORIZON + 2)


def test_a_horizon_a_global_model_bound_at_fit_is_refused(build: BuildClient) -> None:
    data = windows()
    client = build("nixtla")

    handle = client.fit(
        NHITS, data, horizon=HORIZON, plan=sequences_plan(), params=dict(FAST[NHITS])
    )

    with pytest.raises(of.IncompatibleForecastTask):
        client.forecast(handle, data.at_origin(LATEST_ORIGIN), horizon=HORIZON + 2)


# -- artifacts ---------------------------------------------------------------


def test_an_alias_follows_the_latest_fit_and_a_revision_does_not(build: BuildClient) -> None:
    """What a scheduled job relies on: name the model once, retrain underneath it."""
    data = complete_series()
    client = build("nixtla")
    plan = of.FitPlan(origins=of.AtOrigin(LATEST_ORIGIN))

    first = client.fit(AUTOARIMA, data, plan=plan, name="de-price")
    second = client.fit(AUTOARIMA, data, plan=plan, name="de-price")

    reader = build("nixtla")
    latest = reader.forecast("local/de-price", data.at_origin(LATEST_ORIGIN), horizon=HORIZON)
    pinned = reader.forecast(str(first.ref), data.at_origin(LATEST_ORIGIN), horizon=HORIZON)

    assert first.ref != second.ref
    assert latest.model == str(second.ref)
    assert pinned.model == str(first.ref)


# -- terminology (rule 6) -----------------------------------------------------


def _strings(node: Any) -> Iterator[str]:
    """Every string anywhere in a serialized object, keys included."""
    if isinstance(node, dict):
        for key, value in cast("dict[str, Any]", node).items():
            yield str(key)
            yield from _strings(value)
    elif isinstance(node, list):
        for item in cast("list[Any]", node):
            yield from _strings(item)
    elif isinstance(node, str):
        yield node


def test_no_provider_spelling_escapes_into_a_public_object(build: BuildClient) -> None:
    """Rule 6 against the real integrations, which is where the spellings live.

    ``tests/unit/test_architecture.py`` checks the shape of the public types.
    This checks the values that actually travel: the descriptors three
    integrations advertise — including the parameter schemas they publish, which
    OpenForecast never looks inside — and the manifest a fit writes down.
    """
    client = build("nixtla", "darts", "sktime")
    handle = client.fit(
        POOLED_TREES,
        windows(),
        horizon=HORIZON,
        plan=sequences_plan(),
        params=dict(FAST[POOLED_TREES]),
        name="de-price",
    )

    published = [
        (str(descriptor.ref), descriptor.model_dump(mode="json"))
        for descriptor in client.models.list()
    ]
    published.append((str(handle.ref), handle.manifest.model_dump(mode="json")))

    offenders = {
        origin: sorted(set(_strings(payload)) & PROVIDER_TERMS) for origin, payload in published
    }
    assert not {origin: found for origin, found in offenders.items() if found}


def test_a_forecast_is_labeled_the_way_the_caller_named_things(build: BuildClient) -> None:
    """No ``unique_id``, no ``ds``, no ``y`` — the caller's zone and the caller's target."""
    data = windows()
    client = build("nixtla")

    handle = client.fit(
        NHITS, data, horizon=HORIZON, plan=sequences_plan(), params=dict(FAST[NHITS])
    )
    forecast = client.forecast(handle, data.at_origin(LATEST_ORIGIN), horizon=HORIZON)

    assert forecast.table.column_names == [
        "zone",
        "event_time",
        "target",
        "kind",
        "quantile",
        "sample",
        "value",
    ]
    assert set(forecast.point().column("target").to_pylist()) == {"price"}
    assert forecast.to_wide().column_names == ["zone", "event_time", "price"]
    assert isinstance(forecast.origin_time, datetime)
