"""AST helpers for the architecture tests.

Kept dependency-free on purpose: the architecture tests must be able to reason
about the package by reading source, not by importing it.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "openforecast"


@dataclass(frozen=True)
class ImportSite:
    """A single imported module name, with enough context to report it."""

    module: str
    path: Path
    lineno: int

    @property
    def top_level(self) -> str:
        return self.module.split(".", 1)[0]

    def __str__(self) -> str:
        return f"{self.path}:{self.lineno}: {self.module}"


def iter_source_files(root: Path = PACKAGE_ROOT) -> Iterator[Path]:
    yield from sorted(root.rglob("*.py"))


def iter_imports(path: Path) -> Iterator[ImportSite]:
    """Yield every module name imported by the file at ``path``."""
    yield from imports_in_source(path.read_text(encoding="utf-8"), path)


def imports_in_source(source: str, path: Path) -> Iterator[ImportSite]:
    """Yield every module name imported by ``source``.

    Relative imports are resolved against the file's own package so that the
    layering check sees canonical ``openforecast.*`` names. A file outside the
    package cannot reach it relatively, so there its relative imports are
    reported as written.
    """
    tree = ast.parse(source, filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield ImportSite(alias.name, path, node.lineno)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level and _inside_package(path):
                module = _resolve_relative(path, node.level, module)
            if not module:
                continue
            yield ImportSite(module, path, node.lineno)
            # ``from openforecast import runtime`` imports a module, but the
            # node only records the package. Emit the dotted candidates too so
            # the layering check sees them; for names that are not modules the
            # extra candidate simply matches nothing new.
            for alias in node.names:
                if alias.name != "*":
                    yield ImportSite(f"{module}.{alias.name}", path, node.lineno)


def _inside_package(path: Path) -> bool:
    return path.resolve().is_relative_to(PACKAGE_ROOT)


def _resolve_relative(path: Path, level: int, module: str) -> str:
    """Turn a ``from ..x import y`` into ``openforecast.<...>.x``."""
    parts = list(module_name(path).split("."))
    if path.name != "__init__.py":
        parts.pop()
    for _ in range(level - 1):
        if parts:
            parts.pop()
    if module:
        parts.append(module)
    return ".".join(parts)


def module_name(path: Path) -> str:
    """The dotted module name of a file inside the package."""
    relative = path.resolve().relative_to(PACKAGE_ROOT.parent)
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)
