"""``openforecast providers`` — install and inspect provider environments.

```bash
openforecast providers list
openforecast providers install nixtla
openforecast providers inspect nixtla
openforecast providers remove nixtla
```

The four verbs of a package manager, over the uv environments of
:mod:`openforecast.runtime.environments`. Nothing here decides anything: an
install is a build followed by a handshake, and a listing is the record that
handshake produced. That is deliberate — a CLI that computed something the
Python API did not would be a second implementation of the same idea.

Human output goes to stdout as aligned columns; ``--json`` prints the same facts
as one JSON document, for anything that has to parse them. Errors go to stderr
and exit non-zero. A provider is not a thing to be silently absent: asking to
inspect one that is not installed is a failure, not an empty table.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import IO, Any

from openforecast.runtime.environments import ProviderEnvironment, ProviderEnvironments

__all__ = ["add_parser", "run"]


def add_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Register ``openforecast providers`` and its four verbs."""
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "providers",
        help="install and inspect provider environments",
        description="Manage the isolated uv environments integrations run in.",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="the provider cache to operate on (default: the user cache directory)",
    )
    verbs = parser.add_subparsers(dest="verb", required=True, metavar="VERB")

    listing = verbs.add_parser("list", help="every installed provider")
    listing.add_argument("--json", action="store_true", help="print JSON instead of a table")

    install = verbs.add_parser("install", help="build an environment for a provider")
    install.add_argument("name", help="the provider to install, such as 'nixtla'")
    install.add_argument(
        "--source",
        default=None,
        help="what to install: a path or a requirement (default: the published distribution)",
    )
    install.add_argument(
        "--module",
        default=None,
        help="the module 'python -m' runs (default: openforecast_<name>)",
    )
    install.add_argument("--json", action="store_true", help="print JSON instead of a summary")

    inspect = verbs.add_parser("inspect", help="what one provider is and what it advertises")
    inspect.add_argument("name")
    inspect.add_argument("--json", action="store_true", help="print JSON instead of a summary")

    remove = verbs.add_parser("remove", help="delete a provider's environment")
    remove.add_argument("name")
    parser.set_defaults(handler=run)
    return parser


def run(args: argparse.Namespace, out: IO[str]) -> int:
    """Execute the verb ``args`` names. Errors are raised, not printed."""
    environments = ProviderEnvironments(args.root)
    if args.verb == "list":
        return _list(environments, out, as_json=args.json)
    if args.verb == "install":
        environment = environments.install(args.name, source=args.source, module=args.module)
        return _describe(environment, out, as_json=args.json, verb="installed")
    if args.verb == "inspect":
        return _describe(environments.get(args.name), out, as_json=args.json, verb="installed")
    removed = environments.remove(args.name)
    print(f"removed {args.name} ({removed})", file=out)
    return 0


def _list(environments: ProviderEnvironments, out: IO[str], *, as_json: bool) -> int:
    found = environments.list()
    if as_json:
        _dump(
            {
                "root": str(environments.root),
                "providers": [_summary(environment) for environment in found],
            },
            out,
        )
        return 0
    if not found:
        print(
            f"no providers are installed in {environments.root}\n"
            f"install one with: openforecast providers install nixtla",
            file=out,
        )
        return 0
    rows = [
        (environment.name, environment.version, str(len(environment.record.models)))
        for environment in found
    ]
    _table(("PROVIDER", "VERSION", "MODELS"), rows, out)
    return 0


def _describe(environment: ProviderEnvironment, out: IO[str], *, as_json: bool, verb: str) -> int:
    if as_json:
        _dump(_summary(environment), out)
        return 0
    record = environment.record
    print(f"{record.provider} {record.provider_version} ({verb})", file=out)
    print(f"  source     {record.source}", file=out)
    print(f"  command    {' '.join(environment.command)}", file=out)
    print(f"  protocol   {record.protocol_version}", file=out)
    print(f"  path       {environment.path}", file=out)
    print(f"  models     {len(record.models)}", file=out)
    for descriptor in record.descriptors:
        # A pretrained model trains on no view at all, and says so rather than
        # borrowing the spelling of one it does not have.
        view = "zero-shot" if descriptor.training is None else str(descriptor.training.view)
        print(f"    {descriptor.ref}  view={view}", file=out)
    return 0


def _summary(environment: ProviderEnvironment) -> dict[str, Any]:
    return {
        **environment.record.model_dump(mode="json"),
        "path": str(environment.path),
        "command": list(environment.command),
    }


def _dump(payload: object, out: IO[str]) -> None:
    print(json.dumps(payload, indent=2), file=out)


def _table(header: Sequence[str], rows: Sequence[Sequence[str]], out: IO[str]) -> None:
    widths = [max(len(row[index]) for row in (header, *rows)) for index in range(len(header))]
    for row in (header, *rows):
        cells = (cell.ljust(width) for cell, width in zip(row, widths, strict=True))
        print("  ".join(cells).rstrip(), file=out)
