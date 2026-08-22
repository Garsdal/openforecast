"""Architecture tests for the rules in ARCHITECTURE.md.

Both invariants are checked by scanning source rather than by importing
anything, so a violation fails the suite even if the offending module is never
executed:

1. ``openforecast`` never depends on a forecasting framework (rule 1).
2. Imports only ever flow one way down the layer stack (rules 1, 2 and 7).

The provider boundary test (rules 2 and 3) lands with the views package in
Step 4; the forbidden-terminology scan (rule 6) lands in Step 15.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from tests._imports import (
    PACKAGE_ROOT,
    ImportSite,
    iter_imports,
    iter_source_files,
    module_name,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

FORECASTING_FRAMEWORKS = frozenset(
    {
        "neuralforecast",
        "statsforecast",
        "mlforecast",
        "hierarchicalforecast",
        "utilsforecast",
        "darts",
        "sktime",
        "lightgbm",
        "xgboost",
        "torch",
        "pytorch_lightning",
        "lightning",
        "jax",
        "tensorflow",
        "keras",
        "prophet",
        "gluonts",
    }
)

# Lower index == inner layer. A module may import its own layer and any layer
# above it in this list, never one below. Mirrors the diagram in ARCHITECTURE.md.
LAYERS: tuple[tuple[str, ...], ...] = (
    # errors/ is importable from anywhere and imports nothing itself.
    ("openforecast.errors", "openforecast.protocol"),
    ("openforecast.data", "openforecast.models", "openforecast.recipes", "openforecast.tasks"),
    # views/ materializes from semantic datasets, so it sits below them and
    # above everything that executes against them.
    ("openforecast.views",),
    ("openforecast.runtime", "openforecast.registry", "openforecast.artifacts"),
    ("openforecast.client", "openforecast.commands", "openforecast.server"),
)


def _layer_of(module: str) -> int | None:
    for index, layer in enumerate(LAYERS):
        if any(module == name or module.startswith(f"{name}.") for name in layer):
            return index
    return None


def _all_imports() -> list[ImportSite]:
    return [site for path in iter_source_files() for site in iter_imports(path)]


def _module_path(module: str) -> Path:
    relative = Path(*module.split(".")[1:])
    return PACKAGE_ROOT / relative


def test_every_layer_maps_to_a_real_module() -> None:
    """The layer map is only meaningful if every name in it actually exists."""
    missing = [
        module
        for layer in LAYERS
        for module in layer
        if not (_module_path(module) / "__init__.py").is_file()
        and not _module_path(module).with_suffix(".py").is_file()
    ]
    assert not missing, f"layer map references modules that do not exist: {missing}"


def test_package_root_exists() -> None:
    assert (PACKAGE_ROOT / "__init__.py").is_file()


def test_no_forecasting_framework_is_imported() -> None:
    offenders = [str(site) for site in _all_imports() if site.top_level in FORECASTING_FRAMEWORKS]
    assert not offenders, (
        "openforecast must not import a forecasting framework; "
        "integrations depend on openforecast, never the reverse:\n" + "\n".join(offenders)
    )


def test_the_runtime_dependency_set_stays_lightweight() -> None:
    """The core install is the three libraries the semantics are built on.

    A fourth runtime dependency is an architectural decision, not a convenience,
    so it should require changing this list deliberately.
    """
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = {_requirement_name(item) for item in pyproject["project"]["dependencies"]}
    assert declared == {"pydantic", "pyarrow", "platformdirs"}


def test_pandas_is_never_imported() -> None:
    """``from_pandas`` converts through Arrow rather than depending on pandas.

    pandas is a test dependency: OpenForecast accepts a DataFrame at its edge,
    hands it to ``pyarrow``, and stores Arrow from then on.
    """
    offenders = [str(site) for site in _all_imports() if site.top_level == "pandas"]
    assert not offenders, "openforecast must not import pandas:\n" + "\n".join(offenders)


def test_no_forecasting_framework_is_declared_as_a_dependency() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    declared: list[str] = list(project["dependencies"])
    for extra in project.get("optional-dependencies", {}).values():
        declared.extend(extra)

    offenders = [
        requirement
        for requirement in declared
        if _requirement_name(requirement) in FORECASTING_FRAMEWORKS
    ]
    assert not offenders, f"forecasting frameworks declared in pyproject.toml: {offenders}"


def _requirement_name(requirement: str) -> str:
    name = requirement.strip()
    for separator in ("[", "=", "<", ">", "!", "~", ";", " "):
        name = name.split(separator, 1)[0]
    return name.strip().replace("-", "_").lower()


def test_inner_layers_do_not_import_outer_layers() -> None:
    violations: list[str] = []
    for path in iter_source_files():
        importer = module_name(path)
        importer_layer = _layer_of(importer)
        if importer_layer is None:
            continue
        for site in iter_imports(path):
            imported_layer = _layer_of(site.module)
            if imported_layer is None or imported_layer <= importer_layer:
                continue
            violations.append(
                f"{site} -- {importer} (layer {importer_layer}) "
                f"may not import layer {imported_layer}"
            )
    assert not violations, "layering violations:\n" + "\n".join(violations)
