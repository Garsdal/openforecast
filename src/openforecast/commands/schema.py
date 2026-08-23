"""``openforecast schema`` — what a request has to look like, from the build itself.

```bash
openforecast schema fit --json
openforecast schema forecast --json
openforecast schema backtest --json

openforecast schema fit
```

Step 27.2, and the loop it closes is the whole point:

```text
inspect schema  ->  construct request  ->  execute
```

An agent asks this build what a fit request is, writes one, and runs
``openforecast fit --config`` with it — without guessing at a Python signature,
without a doc page, and without a schema pinned to a version that is not the one
installed. ``--json`` prints the JSON Schema itself, which is byte-for-byte the
document committed under ``spec/schemas``; without it, the fields are listed as a
table, because a person reading a request shape wants the field names and not the
``$defs``.

Nothing here generates anything: the documents come from
:mod:`openforecast.docs.schemas`, which is what ``uv run generate-schemas``
writes and CI diffs. A command that derived its own would be a second answer to
the same question.
"""

from __future__ import annotations

import argparse
from typing import IO, TYPE_CHECKING, Any

from openforecast.commands import output
from openforecast.commands.exit_codes import EXIT_OK

if TYPE_CHECKING:
    from openforecast.docs.schemas import Schema

__all__ = ["add_parser", "run"]


def _schemas() -> Any:
    """The generator module, imported when a command runs rather than at import.

    The cycle is real and this is the direction it is broken in: the generator
    reads *this* package's config models, because what `openforecast fit
    --config` accepts is what a fit request is. So the definition lives beside
    the requests, and the command that prints it reaches for the generator when
    it is asked to.
    """
    from openforecast.docs import schemas

    return schemas


def add_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Register ``openforecast schema``."""
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "schema",
        help="the JSON Schema of a request or a protocol object",
        description=(
            "Print the schema of one request or protocol object, so that a request "
            "can be constructed from what this build accepts rather than guessed at."
        ),
    )
    # A positional with choices rather than a verb per schema: the help output
    # then lists everything there is to ask for, which is the discovery an agent
    # does first. `list` is deliberately not among them — `--help` already is.
    parser.add_argument(
        "name",
        choices=[schema.name for schema in _schemas().SCHEMAS],
        help="what to describe",
    )
    parser.add_argument("--json", action="store_true", help="print the JSON Schema itself")
    parser.set_defaults(handler=run)
    return parser


def run(args: argparse.Namespace, out: IO[str]) -> int:
    """Print one schema, as the document or as its fields."""
    schema = _schemas().find(args.name)
    if args.json:
        output.dump(_schemas().document(schema), out)
        return EXIT_OK
    return _summarize(schema, out)


def _summarize(schema: Schema, out: IO[str]) -> int:
    """The fields a caller has to fill in, and which of them are required.

    The top level only. A request is nested — a plan inside a fit, a validation
    strategy inside a backtest — and following it here would reprint a large part
    of the protocol as a tree nobody reads; ``--json`` is the complete answer,
    and each nested object has a schema of its own to ask for.
    """
    body: dict[str, Any] = _schemas().document(schema)
    print(f"{schema.name}  {schema.title}", file=out)
    print(f"  {schema.summary}", file=out)
    print(f"  spec/schemas/{schema.filename}", file=out)
    properties: dict[str, Any] = body.get("properties", {})
    if not properties:
        # A discriminated union has no properties of its own: the alternatives
        # are what there is to say about it.
        print(f"\n  one of: {', '.join(_alternatives(body)) or output.MISSING}", file=out)
        return EXIT_OK
    required = set(body.get("required", []))
    output.table(
        ("FIELD", "REQUIRED", "TYPE"),
        [
            (name, output.cell(name in required), _type(field))
            for name, field in sorted(properties.items())
        ],
        out,
    )
    return EXIT_OK


def _alternatives(body: dict[str, Any]) -> list[str]:
    """The named alternatives of a union, as the ``$defs`` they point at."""
    branches: list[dict[str, Any]] = body.get("oneOf", body.get("anyOf", []))
    return [_reference(branch) for branch in branches if _reference(branch)]


def _type(field: dict[str, Any]) -> str:
    """One field's type, as much of it as a line can hold.

    A union prints as its alternatives and a reference as the object it names, so
    that ``model`` reads as ``Model | Pipeline | Ensemble | Reduction | string``
    rather than as ``anyOf``. Anything else is JSON's own word for it.
    """
    named = _reference(field)
    if named:
        return named
    branches: list[dict[str, Any]] = field.get("anyOf", field.get("oneOf", []))
    if branches:
        return " | ".join(_type(branch) for branch in branches)
    kind = field.get("type")
    if kind == "array":
        return f"array of {_type(field.get('items', {}))}"
    if isinstance(kind, str):
        return kind
    if "enum" in field:
        return " | ".join(str(value) for value in field["enum"])
    return "object"


def _reference(field: dict[str, Any]) -> str:
    """The name a ``$ref`` points at, or empty if this is not one."""
    pointer = field.get("$ref")
    return str(pointer).rsplit("/", 1)[-1] if isinstance(pointer, str) else ""
