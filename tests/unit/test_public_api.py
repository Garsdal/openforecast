from __future__ import annotations

import builtins
from importlib import import_module
from types import ModuleType

import pytest

import openforecast as of

# The semantic data layer of Steps 2 and 3, the ``of.models`` namespace of
# Step 5, and the recipes, plans and tasks of Step 6. The engine joins this list
# in Step 8; it is asserted exactly so that nothing reaches the public surface by
# accident. The execution views are not here on purpose: they are a provider
# boundary, imported from ``openforecast.views``.
EXPECTED_PUBLIC_SURFACE = {
    "Accelerator",
    "AllOrigins",
    "ArtifactError",
    "AtOrigin",
    "ColumnSet",
    "DataError",
    "DuplicateModelError",
    "Ensemble",
    "FeatureAvailability",
    "FeatureKind",
    "FeatureSpec",
    "FitPlan",
    "ForecastContext",
    "ForecastDataset",
    "ForecastTask",
    "Frequency",
    "FrequencyError",
    "FrequencyUnit",
    "Impute",
    "ImputeMethod",
    "InconsistentTruthError",
    "LatestOrigin",
    "LeadTimeFeature",
    "Mean",
    "MissingIndicator",
    "Model",
    "ModelError",
    "ModelRefError",
    "ModelRequiresFit",
    "OpenForecastError",
    "OriginCalendarFeatures",
    "OriginScopeError",
    "OriginSelection",
    "OriginsBetween",
    "OutputKind",
    "OutputSpec",
    "Pipeline",
    "PointInTimeFrame",
    "PointInTimeSchema",
    "Recipe",
    "RecipeError",
    "Reduction",
    "ReductionStrategy",
    "Resources",
    "SchemaError",
    "StandardScaler",
    "TimeSeriesFrame",
    "TimeSeriesSchema",
    "UnknownModelError",
    "UnsupportedPlanError",
    "WeightedMean",
    "WindowPlan",
    "__version__",
    "models",
    "parse_recipe",
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
    "AllOrigins",
    "AtOrigin",
    "FeatureAvailability",
    "FeatureKind",
    "FeatureSpec",
    "FitView",
    "ForecastView",
    "ForecastViewMetadata",
    "Frequency",
    "FrequencyUnit",
    "LatestOrigin",
    "MATERIALIZER_VERSION",
    "OriginFidelity",
    "OriginMode",
    "OriginSelection",
    "OriginsBetween",
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

    assert protocol.__all__ == ["PROTOCOL_VERSION", "ViewKind"]
    assert protocol.ViewKind is ViewKind is of.models.ViewKind
    assert isinstance(protocol.PROTOCOL_VERSION, int)


# What ``of.fit(model=...)`` accepts, and everything a recipe can be built from.
EXPECTED_RECIPE_SURFACE = {
    "ColumnSelector",
    "ColumnSet",
    "ColumnTransform",
    "Combiner",
    "CombinerKind",
    "Ensemble",
    "Impute",
    "ImputeMethod",
    "LeadTimeFeature",
    "Mean",
    "MissingIndicator",
    "Model",
    "OriginCalendarFeatures",
    "Pipeline",
    "PipelineStep",
    "Recipe",
    "RecipeKind",
    "Reduction",
    "ReductionStrategy",
    "StandardScaler",
    "Transform",
    "WeightedMean",
    "declared_transforms",
    "estimator_refs",
    "parse_recipe",
}

# How to fit, and what to predict. ``SearchPlan`` is reachable here but not from
# the top level: it is reserved, and attaching one to a plan is refused.
EXPECTED_TASK_SURFACE = {
    "Accelerator",
    "AllOrigins",
    "AtOrigin",
    "FitPlan",
    "ForecastTask",
    "LatestOrigin",
    "OriginMode",
    "OriginSelection",
    "OriginsBetween",
    "OutputKind",
    "OutputSpec",
    "Resources",
    "SearchPlan",
    "SearchStrategy",
    "WindowPlan",
}


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("recipes", EXPECTED_RECIPE_SURFACE),
        ("tasks", EXPECTED_TASK_SURFACE),
    ],
)
def test_the_step_six_surfaces_are_exactly_what_is_defined(name: str, expected: set[str]) -> None:
    module = import_module(f"openforecast.{name}")

    assert set(module.__all__) == expected
    assert builtins.list(module.__all__) == sorted(module.__all__)
    exported = {
        attribute
        for attribute in dir(module)
        if not attribute.startswith("_") and not isinstance(getattr(module, attribute), ModuleType)
    }
    assert exported == expected


# The artifact lifecycle of Step 7: immutable revisions, their manifests and the
# aliases that point at them. ``ModelRegistry`` is the reference lookup built on
# top of it and on Step 5's catalog.
EXPECTED_ARTIFACT_SURFACE = {
    "ARTIFACT_ID_LENGTH",
    "LOCAL_NAMESPACE",
    "ArtifactStaging",
    "ArtifactStore",
    "MissingValueTransform",
    "ModelArtifact",
    "ModelHandle",
    "ModelManifest",
    "TrainedSchema",
    "TrainingRecord",
    "artifact_time",
    "content_hash",
    "default_root",
    "is_artifact_id",
    "new_artifact_id",
}

EXPECTED_REGISTRY_SURFACE = {"ModelRegistry", "Resolution"}


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("artifacts", EXPECTED_ARTIFACT_SURFACE),
        ("registry", EXPECTED_REGISTRY_SURFACE),
    ],
)
def test_the_step_seven_surfaces_are_exactly_what_is_defined(name: str, expected: set[str]) -> None:
    module = import_module(f"openforecast.{name}")

    assert set(module.__all__) == expected
    assert builtins.list(module.__all__) == sorted(module.__all__)


def test_a_fitted_artifact_is_not_part_of_the_top_level_surface() -> None:
    """Fitting returns a reference; the store behind it is not user vocabulary.

    ``of.fit`` in Step 8 hands back ``local/de-price@01K...``, and a forecast
    takes that string. Nothing about manifests, staging directories or aliases
    has to be named to use the library, so none of it is exported here.
    """
    assert not {name for name in of.__all__ if "Artifact" in name} - {"ArtifactError"}


@pytest.mark.parametrize(
    "name",
    [
        "commands",
        "runtime",
        "server",
    ],
)
def test_unimplemented_subpackages_are_importable_but_empty(name: str) -> None:
    """The skeleton is real packages, not stub APIs."""
    module = import_module(f"openforecast.{name}")

    assert module.__doc__
    assert [attribute for attribute in dir(module) if not attribute.startswith("_")] == []
