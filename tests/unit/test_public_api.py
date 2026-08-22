from __future__ import annotations

import builtins
from importlib import import_module
from types import ModuleType

import pytest

import openforecast as of

# The semantic data layer of Steps 2 and 3, the errors of Steps 4 and 5, and the
# ``of.models`` namespace. Recipes and the engine join this list in later steps;
# it is asserted exactly so that nothing reaches the public surface by accident.
# The execution views are not here on purpose: they are a provider boundary,
# imported from ``openforecast.views``.
EXPECTED_PUBLIC_SURFACE = {
    "DataError",
    "DuplicateModelError",
    "FeatureAvailability",
    "FeatureKind",
    "FeatureSpec",
    "ForecastContext",
    "ForecastDataset",
    "Frequency",
    "FrequencyError",
    "FrequencyUnit",
    "InconsistentTruthError",
    "ModelError",
    "ModelRefError",
    "OpenForecastError",
    "OriginScopeError",
    "PointInTimeFrame",
    "PointInTimeSchema",
    "SchemaError",
    "TimeSeriesFrame",
    "TimeSeriesSchema",
    "UnknownModelError",
    "__version__",
    "models",
}

# Exported but excluded from the attribute comparison below: they are modules,
# and importing any submodule anywhere binds it on the package regardless.
EXPORTED_MODULES = {"models"}


def test_version_is_exported() -> None:
    assert isinstance(of.__version__, str)
    assert of.__version__


def test_public_surface_is_exactly_what_is_implemented() -> None:
    """Submodules are excluded: importing one anywhere binds it on the package."""
    assert set(of.__all__) == EXPECTED_PUBLIC_SURFACE
    public = {
        name
        for name in dir(of)
        if not name.startswith("_") and not isinstance(getattr(of, name), ModuleType)
    }
    assert public == EXPECTED_PUBLIC_SURFACE - {"__version__"} - EXPORTED_MODULES


def test_the_models_namespace_is_reachable_without_importing_it() -> None:
    """``of.models.get(...)`` has to work off a bare ``import openforecast``."""
    assert isinstance(of.models, ModuleType)
    assert of.models.list() == ()


def test_all_is_sorted() -> None:
    assert list(of.__all__) == sorted(of.__all__)


# Everything a provider may import. Nothing here names a semantic source
# dataset: a provider consumes views, never a TimeSeriesFrame, PointInTimeFrame
# or ForecastDataset.
EXPECTED_VIEW_SURFACE = {
    "FeatureAvailability",
    "FeatureKind",
    "FeatureSpec",
    "FitView",
    "ForecastView",
    "ForecastViewMetadata",
    "Frequency",
    "FrequencyUnit",
    "MATERIALIZER_VERSION",
    "OriginFidelity",
    "OriginMode",
    "OriginSelection",
    "SequenceView",
    "SequenceViewSchema",
    "SeriesView",
    "SeriesViewSchema",
    "SourceKind",
    "TabularView",
    "TabularViewSchema",
    "ViewKind",
    "ViewPlanner",
    "ViewProvenance",
    "ViewRequest",
}


def test_the_view_surface_is_exactly_the_provider_boundary() -> None:
    from openforecast import views

    assert set(views.__all__) == EXPECTED_VIEW_SURFACE
    assert list(views.__all__) == sorted(views.__all__)
    exported = {
        name
        for name in dir(views)
        if not name.startswith("_") and not isinstance(getattr(views, name), ModuleType)
    }
    assert exported == EXPECTED_VIEW_SURFACE


# What a model reference resolves to, and everything needed to declare one.
EXPECTED_MODEL_SURFACE = {
    "DEFAULT_CATALOG",
    "FeatureCapabilities",
    "InstanceCapabilities",
    "MissingValueSupport",
    "ModelCapabilities",
    "ModelCatalog",
    "ModelDescriptor",
    "ModelLifecycle",
    "ModelRef",
    "OriginScope",
    "OutputCapabilities",
    "TargetCapabilities",
    "TrainingContract",
    "ViewKind",
    "get",
    "list",
    "register",
}


def test_the_model_surface_is_exactly_what_step_five_defines() -> None:
    assert set(of.models.__all__) == EXPECTED_MODEL_SURFACE
    assert builtins.list(of.models.__all__) == sorted(of.models.__all__)


def test_the_protocol_layer_exports_only_shared_vocabulary() -> None:
    """``ViewKind`` lives here so that ``models/`` and ``views/`` can both name it.

    The wire messages themselves arrive in Step 9. Until then this layer holds
    exactly the vocabulary that would otherwise have to be spelled twice.
    """
    from openforecast import protocol
    from openforecast.views import ViewKind

    assert protocol.__all__ == ["ViewKind"]
    assert protocol.ViewKind is ViewKind is of.models.ViewKind


@pytest.mark.parametrize(
    "name",
    [
        "artifacts",
        "commands",
        "recipes",
        "registry",
        "runtime",
        "server",
        "tasks",
    ],
)
def test_unimplemented_subpackages_are_importable_but_empty(name: str) -> None:
    """The skeleton is real packages, not stub APIs."""
    module = import_module(f"openforecast.{name}")

    assert module.__doc__
    assert [attribute for attribute in dir(module) if not attribute.startswith("_")] == []
