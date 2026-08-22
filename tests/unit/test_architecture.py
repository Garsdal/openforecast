"""Architecture tests for the rules in ARCHITECTURE.md.

Every invariant is checked by scanning source rather than by importing
anything, so a violation fails the suite even if the offending module is never
executed:

1. ``openforecast`` never depends on a forecasting framework (rule 1).
2. Imports only ever flow one way down the layer stack (rules 1, 2 and 7).
3. Providers — the built-in one as much as an external integration — import
   execution views, never semantic source datasets (rules 2 and 3).

The forbidden-terminology scan (rule 6) lands in Step 15.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

from tests._imports import (
    PACKAGE_ROOT,
    ImportSite,
    imports_in_source,
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
    (
        "openforecast.runtime",
        "openforecast.registry",
        "openforecast.artifacts",
        "openforecast.providers",
    ),
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


# -- the provider boundary (rules 2 and 3) ---------------------------------

INTEGRATIONS_ROOT = REPO_ROOT / "integrations"
#: The built-in reference provider is held to the boundary it is the reference
#: for, so this check is no longer only tested against a fixture.
PROVIDERS_ROOT = PACKAGE_ROOT / "providers"

# What a provider may import from OpenForecast. ``views`` re-exports the
# vocabulary its schemas are built from, so a provider never needs ``data``;
# ``models`` is how it declares what it provides, which is a descriptor and
# never a dataset.
PROVIDER_MODULES = frozenset(
    {
        "openforecast.views",
        "openforecast.errors",
        "openforecast.protocol",
        "openforecast.models",
    }
)

# Semantic source datasets. A provider that names one of these has been handed
# something the view abstraction was supposed to absorb.
SOURCE_TYPES = frozenset(
    {
        "ForecastContext",
        "ForecastDataset",
        "PointInTimeFrame",
        "PointInTimeSchema",
        "TimeSeriesFrame",
        "TimeSeriesSchema",
    }
)


def _provider_violations(source: str, path: Path, own: str | None = None) -> list[str]:
    """Every way ``source`` could reach past the view boundary.

    ``own`` is the provider's own package, which it may of course import from;
    for an external integration that is a package OpenForecast never sees.
    """
    allowed_modules: set[str] = set(PROVIDER_MODULES)
    if own is not None:
        allowed_modules.add(own)
    violations: list[str] = []
    for site in imports_in_source(source, path):
        if site.top_level != "openforecast":
            continue
        if not any(
            site.module == allowed or site.module.startswith(f"{allowed}.")
            for allowed in allowed_modules
        ):
            violations.append(f"{site} -- providers may only import {sorted(allowed_modules)}")

    tree = ast.parse(source, filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not (node.module or "").startswith(
            "openforecast"
        ):
            continue
        named = sorted({alias.name for alias in node.names} & SOURCE_TYPES)
        if named:
            violations.append(
                f"{path}:{node.lineno}: {named} -- a provider consumes execution views, "
                f"never a semantic source dataset"
            )
    return violations


@pytest.mark.parametrize(
    ("root", "own"),
    [(INTEGRATIONS_ROOT, None), (PROVIDERS_ROOT, "openforecast.providers")],
)
def test_providers_do_not_import_semantic_source_datasets(root: Path, own: str | None) -> None:
    """Every provider, shipped or external, consumes views and nothing else."""
    violations = [
        violation
        for path in sorted(root.rglob("*.py"))
        for violation in _provider_violations(path.read_text(encoding="utf-8"), path, own)
    ]
    assert not violations, "provider boundary violations:\n" + "\n".join(violations)


def test_the_built_in_provider_is_a_real_provider() -> None:
    """The check above is worth something only if it has something to check."""
    assert sorted(path.name for path in PROVIDERS_ROOT.rglob("*.py"))


def test_the_provider_boundary_check_actually_catches_a_violation() -> None:
    """``integrations/`` holds no Python yet, so the check is tested on a fixture."""
    offending = (
        "from openforecast import ForecastDataset\n"
        "from openforecast.data.frame import TimeSeriesFrame\n"
    )
    reported = "\n".join(
        _provider_violations(offending, INTEGRATIONS_ROOT / "example" / "provider.py")
    )
    assert "ForecastDataset" in reported
    assert "TimeSeriesFrame" in reported
    assert "openforecast.data.frame" in reported

    allowed = "from openforecast.views import SequenceView, ViewKind\n"
    assert not _provider_violations(allowed, INTEGRATIONS_ROOT / "example" / "provider.py")


def test_the_view_vocabulary_is_defined_exactly_once() -> None:
    """A model's training contract names the same ``ViewKind`` the views are keyed by.

    ``models/`` sits above ``views/`` and so cannot import it. The tempting fix
    is a second enum with the same members, which would drift apart silently and
    eventually put two spellings of one concept on the wire. The enum lives in
    the innermost layer instead, where both can reach it.
    """
    definitions = [
        path
        for path in iter_source_files()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        if isinstance(node, ast.ClassDef) and node.name == "ViewKind"
    ]
    assert definitions == [PACKAGE_ROOT / "protocol" / "vocabulary.py"], (
        f"ViewKind must be defined once, in the shared vocabulary: {definitions}"
    )


def test_the_origin_selections_are_defined_exactly_once() -> None:
    """The four selections a user writes are the four the planner resolves.

    ``of.AllOrigins()`` and the origins a ``ViewPlanner`` materializes have to be
    the same objects. ``tasks/`` sits above ``views/`` in the layering, so the
    planner imports them from there and ``openforecast.views`` re-exports them,
    leaving a provider's import surface unchanged — the same arrangement
    ``ViewKind`` uses, and for the same reason.
    """
    selections = {"AllOrigins", "LatestOrigin", "AtOrigin", "OriginsBetween", "OriginMode"}
    definitions = {
        node.name: path
        for path in iter_source_files()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        if isinstance(node, ast.ClassDef) and node.name in selections
    }
    assert set(definitions) == selections
    assert set(definitions.values()) == {PACKAGE_ROOT / "tasks" / "origins.py"}


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
