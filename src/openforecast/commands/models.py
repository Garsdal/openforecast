"""``openforecast models`` — what this build can fit, and what one model is.

```bash
openforecast models list
openforecast models list --provider nixtla
openforecast models get nixtla/nhits
openforecast models get nixtla/nhits --json
```

The catalog, read. Two verbs, because there are two questions: which models are
there, and what is this one. Both are ``client.models.list()`` and
``client.models.get(...)`` with the answer printed — no provider process is
started to answer either, since a descriptor is what the provider already
advertised when it was installed.

``--json`` prints the ``ModelDescriptor`` itself, which is the same document the
HTTP projection returns for ``GET /v1/models``. That is deliberate: an agent
reading a model over the CLI and one reading it over the network are reading one
schema, not two spellings of it.
"""

from __future__ import annotations

import argparse
from typing import IO, Any

from openforecast.commands import output
from openforecast.commands.exit_codes import EXIT_OK
from openforecast.commands.session import add_store_argument, client_for
from openforecast.models.descriptor import ModelDescriptor

__all__ = ["add_parser", "run"]


def add_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Register ``openforecast models`` and its two verbs."""
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "models",
        help="what this build can fit",
        description="Read the catalog: every model this build can execute, and what one is.",
    )
    add_store_argument(parser)
    verbs = parser.add_subparsers(dest="verb", required=True, metavar="VERB")

    listing = verbs.add_parser("list", help="every model, or one provider's")
    listing.add_argument("--provider", default=None, help="only this provider's models")
    listing.add_argument("--json", action="store_true", help="print JSON instead of a table")

    get = verbs.add_parser("get", help="what one reference resolves to")
    get.add_argument("ref", help="the model to describe, such as 'builtin/seasonal-naive'")
    get.add_argument("--json", action="store_true", help="print JSON instead of a summary")

    parser.set_defaults(handler=run)
    return parser


def run(args: argparse.Namespace, out: IO[str]) -> int:
    """Execute the verb ``args`` names. Errors are raised, not printed."""
    models = client_for(args).models
    if args.verb == "list":
        return _list(models.list(provider=args.provider), out, as_json=args.json)
    return _get(models.get(args.ref), out, as_json=args.json)


def _list(found: tuple[ModelDescriptor, ...], out: IO[str], *, as_json: bool) -> int:
    if as_json:
        output.dump({"models": [_summary(item) for item in found]}, out)
        return EXIT_OK
    if not found:
        # Impossible with the built-in provider present, and reported rather
        # than printed as an empty table if a catalog ever is empty.
        print(
            "no models are available in this build\n"
            "install a provider with: openforecast providers install nixtla",
            file=out,
        )
        return EXIT_OK
    rows = [
        [
            str(item.ref),
            item.display_name,
            output.cell(item.is_fittable),
            _view(item),
            _outputs(item),
        ]
        for item in found
    ]
    output.table(("MODEL", "NAME", "FIT", "VIEW", "OUTPUTS"), rows, out)
    return EXIT_OK


def _get(descriptor: ModelDescriptor, out: IO[str], *, as_json: bool) -> int:
    if as_json:
        output.dump(_summary(descriptor), out)
        return EXIT_OK
    print(f"{descriptor.ref}  {descriptor.display_name}", file=out)
    print(f"  provider     {descriptor.provider}", file=out)
    print(f"  fit          {output.cell(descriptor.is_fittable)}", file=out)
    print(f"  requires fit {output.cell(descriptor.lifecycle.requires_fit)}", file=out)
    print(f"  view         {_view(descriptor)}", file=out)
    print(f"  outputs      {_outputs(descriptor)}", file=out)
    if descriptor.training is not None:
        training = descriptor.training
        print(f"  origins      {training.origin_scope}", file=out)
        print(f"  context      {'required' if training.context_required else 'optional'}", file=out)
    if descriptor.parameters_schema:
        parameters = sorted(descriptor.parameters_schema.get("properties", {}))
        print(f"  params       {', '.join(parameters) or output.MISSING}", file=out)
    return EXIT_OK


def _summary(descriptor: ModelDescriptor) -> dict[str, Any]:
    """The descriptor as JSON, which is what the HTTP projection returns too."""
    return descriptor.model_dump(mode="json")


def _view(descriptor: ModelDescriptor) -> str:
    """What the model trains on.

    A pretrained model trains on no view at all and says so rather than
    borrowing the spelling of one it does not have, which is the same sentence
    ``openforecast providers inspect`` prints.
    """
    return "zero-shot" if descriptor.training is None else str(descriptor.training.view)


def _outputs(descriptor: ModelDescriptor) -> str:
    outputs = descriptor.capabilities.outputs
    kinds = [
        name
        for name, supported in (
            ("point", outputs.point),
            ("quantiles", outputs.quantiles),
            ("samples", outputs.samples),
        )
        if supported
    ]
    return ",".join(kinds) or output.MISSING
