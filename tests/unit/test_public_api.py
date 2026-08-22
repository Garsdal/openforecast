from __future__ import annotations

from types import ModuleType

import openforecast as of


def test_version_is_exported() -> None:
    assert isinstance(of.__version__, str)
    assert of.__version__


def test_public_surface_is_only_the_version() -> None:
    """Step 1 exposes no semantic types; they are added when implemented.

    Submodules are excluded because merely importing ``openforecast.data``
    anywhere in the session binds it as an attribute of the package.
    """
    assert of.__all__ == ["__version__"]
    public = {
        name
        for name in dir(of)
        if not name.startswith("_") and not isinstance(getattr(of, name), ModuleType)
    }
    assert public == set()


def test_subpackages_are_importable_but_empty() -> None:
    """The skeleton is real packages, not stub APIs."""
    from openforecast import views

    assert views.__doc__
    assert [name for name in dir(views) if not name.startswith("_")] == []
