from __future__ import annotations

from types import ModuleType

import openforecast as of

# The semantic data layer of Steps 2 and 3. Views, models, recipes and the
# engine join this list in later steps; it is asserted exactly so that nothing
# reaches the public surface by accident.
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


def test_unimplemented_subpackages_are_importable_but_empty() -> None:
    """The skeleton is real packages, not stub APIs."""
    from openforecast import views

    assert views.__doc__
    assert [name for name in dir(views) if not name.startswith("_")] == []
