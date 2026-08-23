"""``docs/reference/generated``, generated from the objects it describes.

```bash
uv run generate-reference
git diff --exit-code docs/reference/generated
```

Step 16's rule for OpenAPI, applied to the reference documentation: nothing here
is written by hand. Every signature, field, enum member, default and description
is read off the public surface — ``openforecast.__all__`` and
``openforecast.models.__all__`` — so a renamed parameter or a rewritten docstring
shows up as a diff in these pages rather than as documentation that quietly
disagrees with the library.

The pages are a pure function of the *types*, never of the environment. No
provider process is started, no provider environment is read and no model
catalog is listed: ``of.models.list()`` answers what happens to be installed, so
a page built from it would differ between two machines and could not be diffed.
What is generated instead is the vocabulary a catalog is expressed in —
``ModelDescriptor``, ``TrainingContract``, ``ModelCapabilities`` — which is the
same everywhere.

The guides and concept pages beside these are written by hand, and they never
retype a signature: that is what makes documentation drift a diff rather than a
discovery.
"""

from __future__ import annotations

import enum
import inspect
import re
import types
import typing
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel
from pydantic.fields import FieldInfo

__all__ = ["REFERENCE_ROOT", "Page", "PAGES", "main", "pages", "write"]

#: Where the committed pages live, relative to the repository root.
REFERENCE_ROOT = Path("docs") / "reference" / "generated"


@dataclass(frozen=True)
class Page:
    """One generated page: a slug, a title, and what belongs on it.

    ``modules`` are import-path prefixes. An exported name is documented on the
    page whose prefix its defining module starts with, so a name that moves
    between modules moves between pages rather than being documented twice.
    """

    slug: str
    title: str
    blurb: str
    modules: tuple[str, ...]


PAGES: tuple[Page, ...] = (
    Page(
        slug="client",
        title="Client and operations",
        blurb=(
            "The four operations, and the client every one of them is a method on. "
            "The module-level `of.fit`, `of.forecast`, `of.backtest` and "
            "`of.eligible_models` are these methods on a default client, so their "
            "signatures differ only by `client=`."
        ),
        modules=("openforecast.client",),
    ),
    Page(
        slug="data",
        title="Semantic data",
        blurb=(
            "What you hand OpenForecast: ordinary event-time data, real forecast "
            "vintages, and one inference origin cut out of them."
        ),
        modules=("openforecast.data",),
    ),
    Page(
        slug="models",
        title="Models and descriptors",
        blurb=(
            "A model reference, and what one resolves to. The catalog itself is not "
            "generated here — it holds whatever providers are installed, which is a "
            "property of a machine rather than of the library."
        ),
        modules=("openforecast.models", "openforecast.protocol"),
    ),
    Page(
        slug="recipes",
        title="Recipes",
        blurb="What to fit: models, pipelines, ensembles, reductions and transforms.",
        modules=("openforecast.recipes",),
    ),
    Page(
        slug="tasks",
        title="Plans, tasks and outputs",
        blurb="How to fit it, what to predict, and what kind of answer to produce.",
        modules=("openforecast.tasks",),
    ),
    Page(
        slug="forecasts",
        title="Forecasts",
        blurb="What a forecast is: one long table, and the projections of it.",
        modules=("openforecast.runtime",),
    ),
    Page(
        slug="evaluation",
        title="Backtesting and metrics",
        blurb="Comparing models over origins, and scoring what comes back.",
        modules=("openforecast.evaluation",),
    ),
    Page(
        slug="transports",
        title="Transports",
        blurb="Where a client executes, which is the only thing a transport decides.",
        modules=("openforecast.server",),
    ),
    Page(
        slug="errors",
        title="Errors",
        blurb=(
            "Every failure OpenForecast raises deliberately, with the `error.code` a "
            "caller branches on instead of on the prose."
        ),
        modules=("openforecast.errors",),
    ),
)

_KIND_LABELS = {
    "model": "Pydantic model",
    "enum": "Enumeration",
    "exception": "Exception",
    "class": "Class",
    "function": "Function",
    "alias": "Type alias",
    "value": "Value",
}

_QUALIFIER = re.compile(r"\b(?:openforecast|pyarrow|datetime|pathlib|typing)(?:\.\w+)*\.(\w+)")
#: ``<OriginMode.ALL: 'all'>`` is how Python reprs an enum member. The name is
#: what a reader writes, so that is what a default or a ``Literal`` renders as.
_ENUM_REPR = re.compile(r"<(\w+\.\w+): [^>]*>")
#: ``from __future__ import annotations`` makes every annotation a string, and
#: ``inspect.signature`` renders it quoted. A default that is a string keeps its
#: quotes: only what follows ``:`` or ``->`` is an annotation.
_QUOTED_ANNOTATION = re.compile(r"(?<=: )'([^']*)'|(?<=-> )'([^']*)'")
#: The replacement for :data:`_ENUM_REPR`, spelled out rather than inlined
#: because a backslash may not appear inside an f-string on Python 3.11.
_MEMBER_NAME = r"\1"


def pages() -> dict[str, str]:
    """Every generated page, keyed by file name, in the order they are written."""
    import openforecast as of

    exported = _exports()
    rendered = {"index.md": _index(exported, of.__version__)}
    for page in PAGES:
        names = sorted(name for name, home in exported.items() if _page_of(home) is page)
        rendered[f"{page.slug}.md"] = _page(page, names, exported)
    return rendered


def write(root: Path | None = None) -> tuple[Path, ...]:
    """Write every page under ``root``, and return where they went.

    A page whose name is no longer generated is removed rather than left behind:
    a stale reference page is worse than a missing one, because it looks current.
    """
    directory = (Path.cwd() if root is None else root) / REFERENCE_ROOT
    directory.mkdir(parents=True, exist_ok=True)
    rendered = pages()
    for stale in sorted(directory.glob("*.md")):
        if stale.name not in rendered:
            stale.unlink()
    written: list[Path] = []
    for name, text in rendered.items():
        path = directory / name
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return tuple(written)


def main() -> int:
    """The ``generate-reference`` console script."""
    for path in write():
        print(path)
    return 0


# -- what gets documented ---------------------------------------------------


def _exports() -> dict[str, str]:
    """Every public name, mapped to the module that defines it.

    ``openforecast.__all__`` is the user surface and ``openforecast.models``'s is
    the vocabulary a descriptor is written in; a provider author reads the second
    and a caller reads the first. ``of.models`` itself and ``__version__`` are
    not entries — the first is a module and the second is on the index page.
    """
    import openforecast as of

    found: dict[str, str] = {}
    for name in of.__all__:
        if name in {"models", "__version__"}:
            continue
        found[name] = _home_module(getattr(of, name))
    for name in of.models.__all__:
        if name in found:
            continue
        found[name] = _home_module(getattr(of.models, name))
    return found


def _home_module(obj: object) -> str:
    """Which module an exported object belongs to.

    A discriminated-union alias — ``of.Recipe``, ``of.Metric`` — reports
    ``typing`` as its module, so it is placed where its members live instead. A
    union whose members disagreed about that would be a union that should not
    have been one.
    """
    if _is_alias(obj):
        modules = {_home_module(member) for member in _alias_members(obj)}
        if len(modules) != 1:
            raise AssertionError(f"a type alias spanning {sorted(modules)} has no home page")
        return modules.pop()
    module = getattr(obj, "__module__", None)
    return module if isinstance(module, str) else "openforecast"


def _page_of(module: str) -> Page:
    for page in PAGES:
        if any(module == name or module.startswith(f"{name}.") for name in page.modules):
            return page
    raise AssertionError(f"nothing documents {module}; add it to a page in PAGES")


# -- rendering --------------------------------------------------------------


def _index(exported: dict[str, str], version: str) -> str:
    """The surface, whole: every public name and the page that documents it."""
    lines = [
        "# API reference",
        "",
        "*Generated from the code by `uv run generate-reference`. Do not edit by hand.*",
        "",
        f"OpenForecast {version}. `openforecast.__all__` is the whole public",
        "surface of the library and is asserted exactly, so this table is that",
        "assertion rendered: a name that is not here is not public.",
        "",
        "| Name | Kind | Documented in |",
        "| --- | --- | --- |",
    ]
    for name in sorted(exported):
        page = _page_of(exported[name])
        kind = _KIND_LABELS[_kind(_resolve(name))]
        lines.append(f"| `{name}` | {kind} | [{page.title}]({page.slug}.md) |")
    lines += ["", "## Pages", ""]
    lines += [f"- [{page.title}]({page.slug}.md) — {page.blurb}" for page in PAGES]
    return "\n".join(lines) + "\n"


def _page(page: Page, names: list[str], exported: dict[str, str]) -> str:
    lines = [
        f"# {page.title}",
        "",
        "*Generated from the code by `uv run generate-reference`. Do not edit by hand.*",
        "",
        page.blurb,
        "",
    ]
    for name in names:
        lines += _entry(name, exported[name])
    return "\n".join(lines).rstrip("\n") + "\n"


def _entry(name: str, module: str) -> list[str]:
    obj = _resolve(name)
    kind = _kind(obj)
    lines = [f"## `{name}`", "", f"*{_KIND_LABELS[kind]} — `{module}`*", ""]
    signature = _signature(name, obj)
    if signature is not None:
        lines += ["```python", signature, "```", ""]
    # A value's docstring is its *type's* docstring, which describes the class
    # rather than this instance of it; the type is the useful fact instead.
    doc = None if kind == "value" else _own_doc(obj)
    if doc:
        lines += [doc, ""]
    lines += _body(obj, kind)
    return lines


def _resolve(name: str) -> object:
    import openforecast as of

    return getattr(of, name) if name in of.__all__ else getattr(of.models, name)


def _kind(obj: object) -> str:
    if _is_alias(obj):
        return "alias"
    if isinstance(obj, type):
        if issubclass(obj, BaseException):
            return "exception"
        if issubclass(obj, BaseModel):
            return "model"
        if issubclass(obj, enum.Enum):
            return "enum"
        return "class"
    if inspect.isfunction(obj) or inspect.isbuiltin(obj):
        return "function"
    return "value"


def _body(obj: object, kind: str) -> list[str]:
    if kind == "alias":
        members = ", ".join(f"`{_name_of(member)}`" for member in _alias_members(obj))
        return [f"One of: {members}.", ""]
    if kind == "exception" and isinstance(obj, type):
        return [f"`error.code` is `{_error_code(obj)}`.", ""]
    if kind == "model" and isinstance(obj, type) and issubclass(obj, BaseModel):
        return _fields(obj)
    if kind == "enum" and isinstance(obj, type) and issubclass(obj, enum.Enum):
        return _members(obj)
    if kind == "class" and isinstance(obj, type):
        return _methods(obj)
    if kind == "value":
        instance_of = type(cast("object", obj)).__name__
        return [f"A `{instance_of}`, documented as the type it is an instance of.", ""]
    return []


def _fields(model: type[BaseModel]) -> list[str]:
    if not model.model_fields:
        return []
    lines = ["| Field | Type | Default | Description |", "| --- | --- | --- | --- |"]
    for name, field in model.model_fields.items():
        lines.append(
            f"| `{name}` | `{_annotation(field)}` | {_default(field)} | {_summary(field)} |"
        )
    return [*lines, ""]


def _members(members: type[enum.Enum]) -> list[str]:
    lines = ["| Member | Value |", "| --- | --- |"]
    for member in members:
        lines.append(f"| `{member.name}` | `{member.value!r}` |")
    return [*lines, ""]


def _methods(cls: type) -> list[str]:
    """The methods and properties a caller can reach, with their signatures.

    Only what this class defines: an inherited method is documented where it is
    written, and a private one is not documented at all.
    """
    rows: list[str] = []
    for name, attribute in sorted(vars(cls).items()):
        if name.startswith("_"):
            continue
        summary = _first_line(inspect.getdoc(attribute))
        if isinstance(attribute, property):
            rows.append(f"| `{name}` | property | {summary} |")
            continue
        signature = _signature(name, attribute)
        if signature is None:
            continue
        rows.append(f"| `{_cell(signature)}` | {_method_kind(cls, name)} | {summary} |")
    if not rows:
        return []
    return ["| Member | Kind | Summary |", "| --- | --- | --- |", *rows, ""]


def _method_kind(cls: type, name: str) -> str:
    raw = inspect.getattr_static(cls, name)
    if isinstance(raw, classmethod):
        return "classmethod"
    if isinstance(raw, staticmethod):
        return "staticmethod"
    return "method"


def _signature(name: str, obj: object) -> str | None:
    """``name(...) -> ...``, or nothing for an object that is not callable.

    Pydantic models are described by their field table rather than by a
    generated ``__init__``, which would name every field twice.
    """
    if isinstance(obj, type) and issubclass(obj, (BaseModel, enum.Enum, BaseException)):
        return None
    if not callable(cast("object", obj)):
        return None
    try:
        signature = inspect.signature(cast("Callable[..., object]", obj))
    except (TypeError, ValueError):  # pragma: no cover - unreachable for the public surface
        return None
    unquoted = _QUOTED_ANNOTATION.sub(
        lambda match: match.group(1) or match.group(2), str(signature)
    )
    return f"{name}{_clean(unquoted)}"


def _annotation(field: FieldInfo) -> str:
    return _cell(_clean(_name_of(field.annotation)))


def _default(field: FieldInfo) -> str:
    if field.is_required():
        return "*required*"
    if field.default_factory is not None:
        return "*computed*"
    if isinstance(field.default, enum.Enum):
        return f"`{type(field.default).__name__}.{field.default.name}`"
    rendered = _ENUM_REPR.sub(_MEMBER_NAME, repr(field.default))
    return f"`{_cell(rendered)}`"


def _summary(field: FieldInfo) -> str:
    return _first_line(field.description)


def _first_line(doc: str | None) -> str:
    if not doc:
        return ""
    return _cell(doc.strip().split("\n", 1)[0])


def _own_doc(obj: object) -> str | None:
    """An object's own docstring, dedented, or nothing if it has none of its own.

    A class is asked for ``vars(cls)["__doc__"]`` rather than for ``cls.__doc__``,
    which walks the MRO: an undocumented model would otherwise be documented with
    ``BaseModel``'s prose and an undocumented error with ``Exception``'s.

    Reading the raw attribute is also what makes the pages the same on every
    interpreter. Python 3.13 dedents docstrings at compile time and earlier
    versions do not, so a generator that compared ``__doc__`` against
    ``inspect.getdoc`` — one indented, the other not — decided that almost every
    docstring was inherited on 3.11 and 3.12, and produced pages that could not be
    diffed against the committed ones. ``cleandoc`` is what both versions agree
    on, and is a no-op on the already-dedented form.
    """
    if isinstance(obj, type):
        own = vars(obj).get("__doc__")
        raw = own if isinstance(own, str) else None
    else:
        raw = inspect.getdoc(obj)
    if raw is None or not raw.strip():
        return None
    return inspect.cleandoc(raw).strip()


def _name_of(annotation: object) -> str:
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation)


def _clean(text: str) -> str:
    """Strip import paths from a rendered annotation or signature.

    ``openforecast.tasks.plan.FitPlan | None`` reads as ``FitPlan | None``: the
    module the object comes from is on the page already, and the qualified form
    is noise everywhere else.
    """
    return _ENUM_REPR.sub(_MEMBER_NAME, _QUALIFIER.sub(_MEMBER_NAME, text))


def _cell(text: str) -> str:
    """A table cell cannot hold a bare pipe, and a union annotation is full of them."""
    return text.replace("|", "\\|")


def _error_code(error: type) -> str:
    """The code the exception itself reports, read off an instance of it.

    Derived from the class name by ``OpenForecastError.code``, so reading it here
    rather than recomputing it is what keeps the documented code and the raised
    one the same fact.
    """
    code = getattr(error(), "code", None)
    return code if isinstance(code, str) else ""


def _is_alias(obj: object) -> bool:
    return typing.get_origin(obj) in {typing.Annotated, types.UnionType, typing.Union}


def _alias_members(obj: object) -> tuple[Any, ...]:
    """The union an ``Annotated[A | B, Field(discriminator=...)]`` alias holds."""
    inner = typing.get_args(obj)[0] if typing.get_origin(obj) is typing.Annotated else obj
    return typing.get_args(inner)


if __name__ == "__main__":  # pragma: no cover - exercised as a console script
    raise SystemExit(main())
