from __future__ import annotations

from importlib import import_module
from types import ModuleType

import pytest

import openforecast as of

# The semantic data layer of Steps 2 and 3, plus the errors Step 4 adds. Models,
# recipes and the engine join this list in later steps; it is asserted exactly so
# that nothing reaches the public surface by accident. The execution views are
# not here on purpose: they are a provider boundary, imported from
# ``openforecast.views``.
EXPECTED_PUBLIC_SURFACE = {
    "DataError",
    "FeatureAvailability",
    "FeatureKind",
    "FeatureSpec",
    "ForecastContext",
    "ForecastDataset",
    "Frequency",
    "FrequencyError",
    "FrequencyUnit",
    "InconsistentTruthError",
    "OpenForecastError",
    "OriginScopeError",
    "PointInTimeFrame",
    "PointInTimeSchema",
    "SchemaError",
    "TimeSeriesFrame",
    "TimeSeriesSchema",
    "__version__",
}


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
    assert public == EXPECTED_PUBLIC_SURFACE - {"__version__"}


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


@pytest.mark.parametrize(
    "name",
    [
        "artifacts",
        "commands",
        "models",
        "protocol",
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
