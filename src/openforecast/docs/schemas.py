"""``spec/schemas``, generated from the objects the schemas describe.

```bash
uv run generate-schemas
git diff --exit-code spec/schemas
```

Step 27.1. An agent should not have to reverse-engineer a Python signature or
read prose to find out what a fit request is, so every executable request and
every object one is built out of has a JSON Schema — generated from the Pydantic
type, committed, and diffed in CI like the OpenAPI document and the reference
pages beside it. Never maintained by hand: a schema written next to the code is
a schema that disagrees with it eventually, and this one is *derived*, so a
renamed field is a diff rather than a surprise at runtime.

```text
spec/schemas/
    fit-request.json         openforecast fit --config
    forecast-request.json    openforecast forecast --config
    backtest-request.json    openforecast backtest --config
    model-recipe.json        the recipe a request names
    fit-plan.json            how a view is materialized for it
    output-spec.json         what kind of forecast to answer with
    time-series-schema.json  what a dataset's columns mean
    model-descriptor.json    what a model advertises
    error.json               the one error envelope
```

The three requests are the *config* models of :mod:`openforecast.commands.config`
rather than the HTTP bodies of :mod:`openforecast.server.wire`. That is
deliberate and not a second protocol: the two differ only in how bulk data
travels — a path on a command line, base64 Arrow over a socket — and the nested
objects are literally the same Pydantic types. The HTTP bodies are already
described, in full, by ``spec/openapi/openapi.json``, which is generated from the
same models; what is here is the surface that document cannot describe, which
includes a backtest, because HTTP has no backtest endpoint yet.

The documents are a pure function of the types. No catalog is listed, no provider
is started and no artifact store is read, so regenerating on a machine with
different providers installed produces the same bytes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from openforecast.commands.config import BacktestConfig, FitConfig, ForecastConfig
from openforecast.data.schema import TimeSeriesSchema
from openforecast.models.descriptor import ModelDescriptor
from openforecast.recipes.nodes import Recipe
from openforecast.server.wire import ErrorBody
from openforecast.tasks.forecast import OutputSpec
from openforecast.tasks.plan import FitPlan

__all__ = ["SCHEMAS", "SCHEMA_ROOT", "Schema", "document", "find", "main", "render", "write"]

#: Where the committed documents live, relative to the repository root.
SCHEMA_ROOT = Path("spec") / "schemas"

#: The dialect the documents declare. Pinned rather than read off Pydantic, so
#: that a library upgrade that changed dialects would be a diff to review.
DIALECT = "https://json-schema.org/draft/2020-12/schema"


@dataclass(frozen=True)
class Schema:
    """One generated document: what it describes, and what to call it.

    ``name`` is what the CLI answers to — ``openforecast schema fit`` — and
    ``slug`` is the file it is committed as. Two names rather than one because
    they have different jobs: a file is read next to its neighbours and says
    ``fit-request``, and a command is typed.
    """

    name: str
    slug: str
    title: str
    summary: str
    #: The Pydantic model, or any type a ``TypeAdapter`` accepts — ``Recipe`` is
    #: a discriminated union rather than a class, and it is as much a part of the
    #: protocol as the models that hold one.
    describes: Any

    @property
    def filename(self) -> str:
        return f"{self.slug}.json"


SCHEMAS: tuple[Schema, ...] = (
    Schema(
        name="fit",
        slug="fit-request",
        title="Fit request",
        summary=(
            "The arguments of `of.fit`, as `openforecast fit --config` takes them. "
            "`data` is the directory a written dataset lives in."
        ),
        describes=FitConfig,
    ),
    Schema(
        name="forecast",
        slug="forecast-request",
        title="Forecast request",
        summary=(
            "The arguments of `of.forecast`. `model` is the reference a fit "
            "produced, so a recipe is not accepted here."
        ),
        describes=ForecastConfig,
    ),
    Schema(
        name="backtest",
        slug="backtest-request",
        title="Backtest request",
        summary=(
            "The arguments of `of.backtest`: the models to compare, the validation "
            "strategy that selects the origins, and the metrics to score."
        ),
        describes=BacktestConfig,
    ),
    Schema(
        name="recipe",
        slug="model-recipe",
        title="Model recipe",
        summary=(
            "What is fitted: a model, a pipeline ending in one, an ensemble of "
            "them, or a reduction. Discriminated on `kind`."
        ),
        describes=Recipe,
    ),
    Schema(
        name="plan",
        slug="fit-plan",
        title="Fit plan",
        summary=(
            "How the view a model trains on is materialized — the context window, "
            "the origins, the lags. Never what the model is."
        ),
        describes=FitPlan,
    ),
    Schema(
        name="output",
        slug="output-spec",
        title="Output specification",
        summary="What kind of forecast to answer with: a point, quantiles, or sample paths.",
        describes=OutputSpec,
    ),
    Schema(
        name="series-schema",
        slug="time-series-schema",
        title="Time series schema",
        summary=(
            "What the columns of an event-time dataset mean: the timestamp, the "
            "targets, the features and their availability."
        ),
        describes=TimeSeriesSchema,
    ),
    Schema(
        name="model",
        slug="model-descriptor",
        title="Model descriptor",
        summary=(
            "What a model advertises: its lifecycle, its training contract, the "
            "shapes it accepts and the outputs it can produce."
        ),
        describes=ModelDescriptor,
    ),
    Schema(
        name="error",
        slug="error",
        title="Error envelope",
        summary=(
            "How every failure is reported: a stable `code` to branch on, a "
            "`message` for a person, and the `details` behind it."
        ),
        describes=ErrorBody,
    ),
)


def find(name: str) -> Schema:
    """The schema ``name`` selects.

    Raises :class:`KeyError` rather than answering with something close: the CLI
    constrains the argument to :data:`SCHEMAS`, so a miss here is a caller
    holding a name this build does not have.
    """
    for schema in SCHEMAS:
        if schema.name == name:
            return schema
    raise KeyError(name)


def document(schema: Schema) -> dict[str, Any]:
    """One JSON Schema, with the dialect and the identity a consumer expects.

    ``$id`` is the committed filename rather than a URL that has to stay
    resolvable: the documents are read from a checkout or from
    ``openforecast schema``, and a dead hostname in a ``$ref`` is worse than no
    hostname at all.
    """
    generated = TypeAdapter(schema.describes).json_schema(
        by_alias=True, ref_template="#/$defs/{model}"
    )
    # The title and the summary are stated here rather than taken from the model,
    # and they win: what the root of the document names is the *request*, and
    # ``FitConfig`` is the class that happens to implement one. Every nested
    # object keeps its own title and docstring, which is where a reader following
    # a ``$ref`` expects to find them.
    return {
        "$schema": DIALECT,
        "$id": schema.filename,
        **generated,
        "title": schema.title,
        "description": schema.summary,
    }


def render(schema: Schema) -> str:
    """The exact bytes that are committed.

    Sorted keys and a trailing newline, so that regenerating on another machine
    or another Python produces the same file and ``git diff --exit-code`` means
    what it says.
    """
    return json.dumps(document(schema), indent=2, sort_keys=True) + "\n"


def write(root: Path | None = None) -> tuple[Path, ...]:
    """Write every document under ``root``, and return where they went.

    A document that is no longer generated is removed rather than left behind: a
    stale schema is worse than a missing one, because an agent that reads it has
    no way to tell it is stale.
    """
    directory = (Path.cwd() if root is None else root) / SCHEMA_ROOT
    directory.mkdir(parents=True, exist_ok=True)
    generated = {schema.filename for schema in SCHEMAS}
    for stale in sorted(directory.glob("*.json")):
        if stale.name not in generated:
            stale.unlink()
    written: list[Path] = []
    for schema in SCHEMAS:
        path = directory / schema.filename
        path.write_text(render(schema), encoding="utf-8")
        written.append(path)
    return tuple(written)


def main() -> int:
    """The ``generate-schemas`` console script."""
    for path in write():
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a console script
    raise SystemExit(main())
