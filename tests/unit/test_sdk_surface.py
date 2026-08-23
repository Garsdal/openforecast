"""Step 24: the public SDK, frozen.

``tests/unit/test_public_api.py`` asserts *which names* are exported. This file
asserts the shape of what they are, which is the part Step 24 adds:

1. One name per intent. There is a ``fit``, a ``forecast`` and a ``backtest``,
   and no ``train``, ``predict``, ``infer``, ``evaluate`` or
   ``historical_forecasts`` anywhere on the surface.
2. Every operation is reachable both ways, with the same signature. A
   module-level function is its method on the default client and nothing else,
   so ``of.backtest`` and ``client.backtest`` cannot drift apart.
3. The implementation classes stay implementation classes. A ``ViewPlanner`` or
   a ``SubprocessProvider`` is real, documented and importable from the module
   that owns it — and is not something a user of the library has to name.
4. The simple case stays short: a model reference, some data, a horizon.
"""

from __future__ import annotations

import inspect
from importlib import import_module
from types import ModuleType
from typing import Any

import pytest

import openforecast as of
from openforecast.client import Models

# The four operations, each of which is a function in ``openforecast`` and a
# method on ``of.OpenForecast``. Nothing else in ``of.__all__`` is a verb.
OPERATIONS = ("backtest", "eligible_models", "fit", "forecast")

# Second doors onto rooms that already have one. Every one of these is a real
# spelling in some forecasting library, which is exactly why the check exists.
ALIASES = frozenset(
    {
        "train",
        "predict",
        "infer",
        "inference",
        "evaluate",
        "historical_forecasts",
        "backtesting",
        "score",
        "run",
        "execute",
    }
)


def _functions_of(module: ModuleType) -> set[str]:
    return {name for name in module.__all__ if inspect.isfunction(getattr(module, name))}


def test_the_module_level_operations_are_exactly_the_four() -> None:
    """``parse_recipe`` is the fifth function and not a fifth operation.

    It reads a recipe out of the dict a manifest or a request body carries,
    which is a constructor for ``of.Pipeline`` and friends rather than something
    that fits, forecasts or scores anything.
    """
    assert _functions_of(of) == {*OPERATIONS, "parse_recipe"}


def test_every_operation_is_also_a_method_on_a_client() -> None:
    """``of.backtest`` is ``client.backtest`` on the default client, and so on.

    Step 19 left the question open — whether a client mirrors the module-level
    function — for Step 24 to answer once for every operation rather than
    separately for each. This is that answer, in the form that fails if a fifth
    operation lands on one side only.
    """
    for name in OPERATIONS:
        assert callable(getattr(of.OpenForecast, name)), f"of.{name} has no client method"


@pytest.mark.parametrize("name", OPERATIONS)
def test_a_function_and_its_method_take_the_same_arguments(name: str) -> None:
    """Apart from ``client=``, which is *which* client — the thing a method is.

    A backtest against a service is ``client.backtest(...)`` or
    ``of.backtest(..., client=client)``; those have to mean the same call, so
    everything else about the two signatures is identical.
    """
    function = inspect.signature(getattr(of, name))
    method = inspect.signature(getattr(of.OpenForecast, name))

    on_the_function = dict(function.parameters)
    on_the_method = {
        key: value for key, value in method.parameters.items() if key not in {"self", "client"}
    }
    on_the_function.pop("client", None)

    assert list(on_the_function) == list(on_the_method)
    for key, parameter in on_the_function.items():
        mirrored = on_the_method[key]
        assert parameter.kind == mirrored.kind, f"{name}({key}) is passed differently"
        assert parameter.default == mirrored.default, f"{name}({key}) defaults differently"


def _public_names(value: Any) -> set[str]:
    return {name for name in dir(value) if not name.startswith("_")}


def test_one_name_per_intent() -> None:
    """No aliases on the package, on a client, or on what either hands back."""
    surfaces = {
        "openforecast": set(of.__all__),
        "OpenForecast": _public_names(of.OpenForecast),
        "Models": _public_names(Models),
        "Forecast": _public_names(of.Forecast),
        "BacktestResult": _public_names(of.BacktestResult),
    }
    offenders = {
        where: sorted(names & ALIASES) for where, names in surfaces.items() if names & ALIASES
    }

    assert not offenders, f"one intent reachable under two names: {offenders}"


def test_the_client_surface_is_exactly_the_operations_and_what_they_need() -> None:
    """A client is four operations, the models it can run them with, and where.

    ``artifact`` is the fifth thing and not a fifth operation: a fit hands back a
    reference, so looking one up again is how you get from the string to what it
    records — the same lookup ``models.get`` is for an unfitted reference.
    """
    assert _public_names(of.OpenForecast) == {
        *OPERATIONS,
        "artifact",
        "engine",
        "models",
        "transport",
    }


def test_a_catalog_is_the_two_methods_step_twenty_four_names_plus_convenience() -> None:
    assert {"list", "get"} <= _public_names(Models)
    assert _public_names(Models) == {"list", "get", "refs", "providers"}


# The machinery every one of those operations is made of. Each is real, has a
# docstring, and is imported from the module that owns it — by an integration, a
# test or the engine itself — and none of it is vocabulary a user writes.
INTERNAL = {
    "openforecast.views": ("ViewPlanner", "SequenceView", "SeriesView", "TabularView"),
    "openforecast.runtime": ("Engine", "SubprocessProvider", "ProviderRegistry"),
    "openforecast.artifacts": ("ArtifactStore", "ModelHandle", "ModelManifest"),
    "openforecast.registry": ("ModelRegistry",),
}


@pytest.mark.parametrize(("module_path", "names"), sorted(INTERNAL.items()))
def test_implementation_classes_stay_out_of_the_public_surface(
    module_path: str, names: tuple[str, ...]
) -> None:
    module = import_module(module_path)
    for name in names:
        implementation = getattr(module, name)
        assert implementation.__doc__, f"{name} is internal, not undocumented"
        assert name not in of.__all__, f"{name} is implementation, not public vocabulary"
        assert not hasattr(of, name)


def test_the_public_surface_names_no_provider_and_no_framework() -> None:
    """``of.`` is provider-independent vocabulary, so nothing in it is a library."""
    forbidden = ("nixtla", "darts", "sktime", "sklearn", "chronos", "arima", "torch")
    offenders = [name for name in of.__all__ for term in forbidden if term in name.lower()]

    assert not offenders


@pytest.mark.parametrize(
    ("name", "required"),
    [("fit", ["model", "data"]), ("forecast", ["model", "data", "horizon"])],
)
def test_the_simple_case_needs_a_model_some_data_and_a_horizon(
    name: str, required: list[str]
) -> None:
    """``client.fit("sklearn/hist-gradient-boosting", data=data, horizon=24)``.

    Nothing else may become mandatory: the explicit forms — a ``Pipeline``, a
    ``FitPlan``, an ``OutputSpec`` — stay available and stay optional.
    """
    parameters = inspect.signature(getattr(of.OpenForecast, name)).parameters
    without_defaults = [
        key
        for key, parameter in parameters.items()
        if key != "self" and parameter.default is inspect.Parameter.empty
    ]

    assert without_defaults == required
    assert [key for key in parameters if key in {"model", "data"}] == ["model", "data"]
    assert all(
        parameters[key].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for key in ("model", "data")
    )


def test_a_backtest_is_asked_for_models_plural_and_metrics() -> None:
    """A one-model backtest passes a list of one; there is no ``model=`` alias."""
    parameters = inspect.signature(of.OpenForecast.backtest).parameters

    assert "model" not in parameters
    assert [
        key
        for key, parameter in parameters.items()
        if key != "self" and parameter.default is inspect.Parameter.empty
    ] == ["models", "data", "validation", "metrics"]


def test_data_is_constructed_the_one_documented_way() -> None:
    """``of.TimeSeriesFrame.from_pandas`` and ``of.ForecastDataset.from_pandas``."""
    for kind in (of.TimeSeriesFrame, of.PointInTimeFrame, of.ForecastDataset):
        assert callable(kind.from_pandas)
    assert not _public_names(of.TimeSeriesFrame) & {"from_dataframe", "from_df", "read_pandas"}
