"""What the provider says about itself, before anything is fitted.

A handshake is the only thing that happens at installation time, so what it
reports has to be right and it has to be cheap: the descriptors below are what
the engine plans every fit against, and answering reflects the installed
estimator registry without constructing or fitting models.

The last section is the boundary claim of Step 18, and it is checked by parsing
this distribution's own imports rather than by trusting them: a provider that
imported a ``ForecastDataset`` would have been handed something the view
abstraction was supposed to absorb, and one that reached for a forecasting
framework would be reducing a forecasting problem OpenForecast had already
reduced. The final test says the same thing about what actually *runs*, since an
import that never executes and one that executes are both worth forbidding.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest
from openforecast_sklearn import PROVIDER_NAME, PROVIDER_VERSION, SklearnProvider, catalog
from openforecast_sklearn.adapter import HIST_GRADIENT_BOOSTING

from openforecast.errors import UnknownModelError
from openforecast.models import ModelDescriptor
from openforecast.models.capabilities import MissingValueSupport
from openforecast.models.contract import OriginScope
from openforecast.protocol.vocabulary import ViewKind

PROVIDER = SklearnProvider()

#: ``integrations/sklearn/tests`` -> the code that ships in the distribution.
SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"

#: Semantic source datasets. A provider that names one has reached past the view.
SOURCE_TYPES = (
    "ForecastContext",
    "ForecastDataset",
    "PointInTimeFrame",
    "PointInTimeSchema",
    "TimeSeriesFrame",
    "TimeSeriesSchema",
)

#: A forecasting framework that already knows how to turn forecasting into
#: regression is the one thing this execution path must not reach for: that
#: transformation is OpenForecast's, and doing it twice means the library's
#: version wins silently.
FRAMEWORKS = ("sktime", "darts", "neuralforecast", "statsforecast", "mlforecast", "lightgbm")


def descriptor_for(name: str) -> ModelDescriptor:
    (found,) = [candidate for candidate in PROVIDER.descriptors() if candidate.ref.name == name]
    return found


def test_the_provider_is_the_namespace_of_the_models_it_advertises() -> None:
    assert PROVIDER.name == PROVIDER_NAME == "sklearn"
    assert PROVIDER.version == PROVIDER_VERSION
    refs = {str(descriptor.ref) for descriptor in PROVIDER.descriptors()}
    assert {
        "sklearn/hist-gradient-boosting",
        "sklearn/random-forest",
        "sklearn/ridge",
    } <= refs
    assert len(refs) > 3
    assert all(descriptor.provider == "sklearn" for descriptor in PROVIDER.descriptors())


def test_the_estimator_declares_what_a_tabular_model_can_do() -> None:
    """The first ``TabularView`` consumer, and its contract says why it is one.

    It learns across origins, because a tabular view of several vintages *is*
    that; it requires no context length, because a row is not a window; it does
    not bind its horizon, because a lead is not a feature; and it takes an
    instance it never saw, because the parameters are shared across every row.
    """
    descriptor = descriptor_for("hist-gradient-boosting")
    contract = descriptor.training
    capabilities = descriptor.capabilities

    assert contract.view is ViewKind.TABULAR
    assert contract.origin_scope is OriginScope.MULTIPLE
    assert contract.learns_across_origins
    assert not contract.context_required
    assert not contract.horizon_bound_at_fit
    assert contract.supports_unseen_instances

    assert (capabilities.instances.single, capabilities.instances.panel) == (True, True)
    assert (capabilities.targets.univariate, capabilities.targets.multivariate) == (True, False)
    assert capabilities.features.known
    assert capabilities.features.static
    # Declared, and not a column: a tabular row describes an event time after
    # its origin, so a measurement has no value there. Refusing the data would
    # refuse most real datasets; the view simply does not offer it as a feature.
    assert capabilities.features.observed
    assert capabilities.missing_values is MissingValueSupport.NATIVE
    assert capabilities.tolerates_missing_values
    assert descriptor.lifecycle.requires_fit


def test_the_seed_and_the_shape_are_never_parameters() -> None:
    """What OpenForecast owns has nowhere to be written down twice.

    A fit plan states the seed and a forecast task states the horizon, so
    ``random_state`` is absent from what this model advertises — and the
    estimator's own reduction-flavoured knobs are absent because there is no
    reduction happening on this side of the boundary.
    """
    schema = descriptor_for("hist-gradient-boosting").parameters_schema

    for owned in ("random_state", "lags", "window_length", "horizon", "n_jobs"):
        assert owned not in schema["properties"], f"{owned} is OpenForecast's, not a parameter"


def test_the_declared_parameters_are_the_ones_that_are_accepted() -> None:
    """The schema a caller reads and the check a caller hits are one table."""
    schema = descriptor_for("hist-gradient-boosting").parameters_schema

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["properties"]["learning_rate"] == {
        "type": "number",
        "description": "Shrinkage applied to each tree.",
        "minimum": 0,
    }
    assert schema["properties"]["max_bins"] == {
        "type": "integer",
        "description": "Histogram bins per feature.",
        "minimum": 2,
        "maximum": 255,
    }


def test_a_model_this_provider_does_not_have_is_named_as_such() -> None:
    with pytest.raises(UnknownModelError, match=r"sklearn/hist-gradient-boosting"):
        catalog.adapter_for("sklearn/not-a-regressor", "sklearn")

    with pytest.raises(UnknownModelError, match=r"not a model of the 'sklearn' provider"):
        catalog.adapter_for("sktime/theta", "sklearn")


def test_the_handshake_discovers_from_the_installed_estimator_library() -> None:
    """The installed sklearn version, rather than a copied list, is the catalog."""
    probe = (
        "import sys\n"
        "from openforecast_sklearn import SklearnProvider\n"
        "SklearnProvider().descriptors()\n"
        "print([name for name in ('sklearn', 'scipy', 'pandas') if name in sys.modules])\n"
    )
    completed = subprocess.run(  # noqa: S603 - the command is this interpreter
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert "sklearn" in completed.stdout


def imported_names() -> list[tuple[str, str, str]]:
    """``(file, module, name)`` for everything this distribution imports.

    Parsed rather than grepped, and imports rather than text: the modules below
    discuss `ForecastDataset` and sktime's reduction API at length, because
    explaining what this integration deliberately does *not* reach for is
    documentation of the boundary rather than a breach of it.
    """
    found: list[tuple[str, str, str]] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        where = str(path.relative_to(SOURCE_ROOT))
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.extend((where, alias.name, alias.name) for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                found.extend((where, module, alias.name) for alias in node.names)
    return found


def test_the_scan_of_this_distribution_finds_something_to_scan() -> None:
    """The two checks below would pass vacuously against an empty source tree."""
    imports = imported_names()

    assert {where for where, _, _ in imports} >= {
        "openforecast_sklearn/adapter.py",
        "openforecast_sklearn/conversion.py",
        "openforecast_sklearn/provider.py",
    }
    assert (
        "openforecast_sklearn/adapter.py",
        "sklearn.ensemble",
        "HistGradientBoostingRegressor",
    ) in imports


@pytest.mark.parametrize("name", SOURCE_TYPES)
def test_this_provider_never_imports_a_semantic_source_dataset(name: str) -> None:
    """Rule 3, asserted against the distribution rather than assumed of it.

    ``ForecastDataset`` and ``PointInTimeFrame`` are where the point-in-time
    semantics live, and the reason this integration is as thin as it is that it
    has never heard of either: what it is handed is a ``TabularView``, whose rows
    were already placed at their origins by the ``ViewPlanner``.
    """
    offenders = [
        f"{where}: {module}.{imported}"
        for where, module, imported in imported_names()
        if imported == name or module.endswith(f".{name}")
    ]

    assert not offenders, f"{name} is imported by {offenders}"


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_no_forecasting_framework_is_on_this_execution_path(framework: str) -> None:
    """Nothing reduces the problem twice, and no import can quietly start doing so.

    The import scan is the half that holds even for code that never runs; the
    module check below is the half that proves a real fit and a real forecast
    executed without one being loaded.
    """
    offenders = [
        f"{where}: {module}"
        for where, module, _ in imported_names()
        if module == framework or module.startswith(f"{framework}.")
    ]

    assert not offenders, f"{framework} is imported by {offenders}"


def test_a_whole_fit_and_forecast_loads_no_forecasting_framework(tmp_path: Path) -> None:
    """The subprocess is the point: a library already imported cannot be unimported."""
    probe = "\n".join(
        [
            "import sys",
            f"sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})",
            "import golden",
            "from golden import HIST_GRADIENT_BOOSTING, FAST, at",
            "data = golden.point_in_time_dataset(instances=2, origins=4, static=True)",
            f"client = golden.client({str(tmp_path)!r})",
            "handle = client.fit(HIST_GRADIENT_BOOSTING, data, horizon=3, params=FAST)",
            "client.forecast(handle, data.at_origin(at(5)), horizon=3)",
            f"print([name for name in {FRAMEWORKS!r} if name in sys.modules])",
        ]
    )
    completed = subprocess.run(  # noqa: S603 - the command is this interpreter
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert completed.stdout.strip().endswith("[]"), completed.stdout


def test_the_adapter_says_which_model_it_is() -> None:
    assert HIST_GRADIENT_BOOSTING.name == "hist-gradient-boosting"
    assert "hist-gradient-boosting" in repr(HIST_GRADIENT_BOOSTING)
    assert "hist-gradient-boosting" in catalog.model_names()
    assert "random-forest" in catalog.model_names()
    assert repr(PROVIDER) == (
        f"SklearnProvider(version={PROVIDER_VERSION}, models={len(catalog.model_names())})"
    )


def test_a_discovered_estimator_gets_native_parameters_and_conservative_capabilities() -> None:
    ridge = descriptor_for("ridge")

    assert "alpha" in ridge.parameters_schema["properties"]
    assert "random_state" not in ridge.parameters_schema["properties"]
    assert ridge.training.view is ViewKind.TABULAR
    assert ridge.capabilities.missing_values is MissingValueSupport.REQUIRES_TRANSFORM
