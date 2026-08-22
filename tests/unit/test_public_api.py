from __future__ import annotations

import openforecast as of


def test_version_is_exported() -> None:
    assert isinstance(of.__version__, str)
    assert of.__version__


def test_public_surface_is_only_the_version() -> None:
    """Stage 1 deliberately exposes nothing else; stubs are added when implemented."""
    assert of.__all__ == ["__version__"]
    public = {name for name in dir(of) if not name.startswith("_")}
    assert public == set()
