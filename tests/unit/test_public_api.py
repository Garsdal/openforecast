from __future__ import annotations

import builtins
from importlib import import_module
from types import ModuleType

import pytest

import openforecast as of

# The semantic data layer of Steps 2 and 3, the ``of.models`` namespace of
# Step 5, the recipes, plans and tasks of Step 6, the engine of Step 8 —
# ``of.fit``, ``of.forecast``, the client behind them and what they hand back —
# and the backtesting of Step 17, which is built on those two calls and adds
# the vocabulary for comparing what comes out of them.
# Asserted exactly so that nothing reaches the public surface by accident. The
# execution views are not here on purpose: they are a provider boundary,
# imported from ``openforecast.views``.
EXPECTED_PUBLIC_SURFACE = {
    "Accelerator",
    "AllOrigins",
    "ArtifactError",
    "AtOrigin",
    "BacktestResult",
    "Bias",
    "Candidate",
    "ColumnSet",
    "DataError",
    "DuplicateModelError",
    "Eligibility",
    "Ensemble",
    "FeatureAvailability",
    "FeatureKind",
    "FeatureSpec",
    "FitPlan",
    "Forecast",
    "ForecastContext",
    "ForecastDataset",
    "ForecastOriginValidation",
    "ForecastTask",
    "Frequency",
    "FrequencyError",
    "FrequencyUnit",
    "HttpTransport",
    "Impute",
    "ImputeMethod",
    "IncompatibleForecastTask",
    "InconsistentTruthError",
    "LatestOrigin",
    "LeadTimeFeature",
    "LocalTransport",
    "MAE",
    "MAPE",
    "Mean",
    "Metric",
    "MissingIndicator",
    "Model",
    "ModelError",
    "ModelRefError",
    "ModelRequiresFit",
    "OpenForecast",
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
    "ProviderError",
    "RMSE",
    "Recipe",
    "RecipeError",
    "Reduction",
    "ReductionStrategy",
    "Resources",
    "RollingOrigin",
    "SchemaError",
    "StandardScaler",
    "TimeSeriesFrame",
    "TimeSeriesSchema",
    "Transport",
    "UnknownModelError",
    "UnsupportedPlanError",
    "Validation",
    "WeightedMean",
    "WindowPlan",
    "__version__",
    "backtest",
    "eligible_models",
    "fit",
    "forecast",
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
    assert of.models.get("builtin/seasonal-naive").provider == "builtin"


def test_all_is_sorted() -> None:
    assert list(of.__all__) == sorted(of.__all__)


# Everything a provider may import. Nothing here names a semantic source
# dataset: a provider consumes views, never a TimeSeriesFrame, PointInTimeFrame
# or ForecastDataset.
EXPECTED_VIEW_SURFACE = {
    "AllOrigins",
    "AtOrigin",
    "CONTEXT_END",
    "CONTEXT_START",
    "EVENT_TIME",
    "FORECAST_END",
    "FORECAST_START",
    "FeatureAvailability",
    "FeatureKind",
    "FeatureSpec",
    "FitView",
    "ForecastColumn",
    "ForecastView",
    "ForecastViewMetadata",
    "Frequency",
    "FrequencyUnit",
    "HORIZON_STEP",
    "LatestOrigin",
    "MATERIALIZER_VERSION",
    "ORIGIN_TIME",
    "OriginFidelity",
    "OriginMode",
    "OriginSelection",
    "OriginsBetween",
    "ROW_ID",
    "SAMPLE_ID",
    "SERIES_ID",
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
    "forecast_columns",
    "read_answer",
    "read_fit_view",
    "read_forecast_view",
    "read_view",
    "write_answer",
    "write_view",
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


# The vocabulary both sides of a boundary have to spell the same way, and the
# messages they exchange over one. Nothing here names a view type, a descriptor
# or a semantic dataset: this is the innermost layer.
EXPECTED_PROTOCOL_SURFACE = {
    "ErrorCode",
    "ErrorPayload",
    "FitRequest",
    "FitResult",
    "ForecastColumn",
    "ForecastRequest",
    "ForecastResult",
    "HandshakeRequest",
    "HandshakeResult",
    "Operation",
    "PROTOCOL_VERSION",
    "Request",
    "Response",
    "Status",
    "ViewKind",
    "ViewRef",
    "forecast_columns",
    "parse_request",
    "parse_response",
}


def test_the_protocol_layer_exports_only_shared_vocabulary() -> None:
    """``ViewKind`` lives here so that ``models/`` and ``views/`` can both name it.

    The wire messages of Step 9 are here for the same reason: the engine writes
    them and a provider in another process reads them, and neither may import
    the other.
    """
    from openforecast import protocol
    from openforecast.views import ViewKind

    assert set(protocol.__all__) == EXPECTED_PROTOCOL_SURFACE
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
    "COMPOSITE_PROVIDER",
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


# The engine and the providers it executes models with. ``ProviderClient`` is
# the shape Step 9's subprocess client will have; the rest is what ``of.fit``
# and ``of.forecast`` are made of.
EXPECTED_RUNTIME_SURFACE = {
    "Engine",
    "Forecast",
    "Leaf",
    "ProviderClient",
    "ProviderEnvironment",
    "ProviderEnvironments",
    "ProviderRecord",
    "ProviderRegistry",
    "SubprocessProvider",
    "TransformState",
    "default_providers",
    "install_default_providers",
    "installed_providers",
    "leaves",
    "normalize_forecast_context",
    "normalize_recipe",
    "validate_view",
}

# The provider half of the boundary: the contract, the serving harness an
# integration runs, and the reference provider itself.
EXPECTED_PROVIDER_SURFACE = {
    "BUILTIN_PROVIDER",
    "BuiltinProvider",
    "ProviderClient",
    "ProviderServer",
    "serve",
}


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("runtime", EXPECTED_RUNTIME_SURFACE),
        ("providers", EXPECTED_PROVIDER_SURFACE),
        ("client", {"Models", "OpenForecast", "fit", "forecast"}),
    ],
)
def test_the_step_eight_surfaces_are_exactly_what_is_defined(name: str, expected: set[str]) -> None:
    module = import_module(f"openforecast.{name}")

    assert set(module.__all__) == expected
    assert builtins.list(module.__all__) == sorted(module.__all__)


# The remote surface of Step 16: the request and response models the HTTP
# projection is generated from, and the two transports a client is configured
# with. The FastAPI application and the OpenAPI generator are deliberately not
# here — they need the ``openforecast[server]`` extra, and importing this
# package must not require a web framework.
EXPECTED_SERVER_SURFACE = {
    "DEFAULT_PORT",
    "DataKind",
    "DataPayload",
    "ErrorBody",
    "ErrorInfo",
    "FitBody",
    "ForecastBody",
    "ForecastContextPayload",
    "ForecastDatasetPayload",
    "ForecastPayload",
    "HttpTransport",
    "LocalTransport",
    "ModelListing",
    "PointInTimePayload",
    "TimeSeriesPayload",
    "Transport",
    "decode_data",
    "encode_data",
    "status_for",
}


def test_the_step_sixteen_surface_is_exactly_what_is_defined() -> None:
    from openforecast import server

    assert set(server.__all__) == EXPECTED_SERVER_SURFACE
    assert builtins.list(server.__all__) == sorted(server.__all__)


# Backtesting and point-in-time evaluation, Step 17. ``Fold``, ``plan_for`` and
# the result-table vocabulary are reachable here but not from the top level: they
# are how a backtest is built rather than what a caller writes.
EXPECTED_EVALUATION_SURFACE = {
    "BACKTEST_COLUMNS",
    "BacktestColumn",
    "BacktestResult",
    "Bias",
    "Candidate",
    "Eligibility",
    "Fold",
    "ForecastOriginValidation",
    "MAE",
    "MAPE",
    "Metric",
    "MetricKind",
    "RMSE",
    "RollingOrigin",
    "Validation",
    "ValidationMode",
    "backtest",
    "eligible_models",
    "plan_for",
}


def test_the_step_seventeen_surface_is_exactly_what_is_defined() -> None:
    from openforecast import evaluation

    assert set(evaluation.__all__) == EXPECTED_EVALUATION_SURFACE
    assert builtins.list(evaluation.__all__) == sorted(evaluation.__all__)


def test_backtesting_adds_no_dependency_and_no_provider_vocabulary() -> None:
    """It is a loop over the public API, so it needs nothing the API does not."""
    import openforecast.evaluation as evaluation

    assert evaluation.backtest is of.backtest
    assert evaluation.eligible_models is of.eligible_models


def test_importing_openforecast_does_not_import_a_web_framework() -> None:
    """The core install is three libraries, and calling a service adds none.

    ``openforecast.server`` is the *semantics* of the remote surface — Pydantic
    models and a urllib client. The framework belongs to whoever serves, so a
    remote-only user installs neither it nor an ASGI server.
    """
    import subprocess
    import sys

    probe = (
        "import sys, openforecast, openforecast.server;"
        "print(sorted(name for name in sys.modules if name in {'fastapi', 'starlette', 'uvicorn'}))"
    )
    answer = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert answer.stdout.strip() == "[]"


def test_the_cli_exposes_a_parser_and_an_entry_point() -> None:
    """The CLI is a projection, so its surface is the two functions that run it."""
    from openforecast import commands

    assert set(commands.__all__) == {"build_parser", "main"}
