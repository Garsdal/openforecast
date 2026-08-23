"""``openforecast doctor`` — is this installation able to forecast?

```bash
openforecast doctor
openforecast doctor --json
```

One command that answers the questions a failing setup raises, before the failure
does: which Python this is, where artifacts are written and whether that is
writable, whether ``uv`` is on the PATH so a provider can be installed at all,
which provider environments exist and whether their interpreters are still
there, how many models the catalog holds, and whether the HTTP extra is present.

Every check is a fact read off this machine, and no forecast is run: a doctor
that fitted a model would be slow enough not to be run, and would fail for
reasons that are not about the installation. A check that finds something wrong
is a ``fail`` and the command exits non-zero — Step 26.5, so that a container's
health check is an exit status rather than a grep over prose. Something that is
merely absent is a ``warn``, which does not fail: an installation with no
providers is a working installation of the built-in models.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import IO, Any

from openforecast.artifacts.store import ArtifactStore
from openforecast.commands import output
from openforecast.commands.exit_codes import EXIT_ERROR, EXIT_OK
from openforecast.commands.session import add_store_argument
from openforecast.errors import OpenForecastError
from openforecast.protocol.version import PROTOCOL_VERSION
from openforecast.runtime.environments import ProviderEnvironments

__all__ = ["Check", "Status", "add_parser", "checks", "run"]


class Status(StrEnum):
    """What one check found."""

    #: As it should be.
    OK = "ok"
    #: Absent rather than broken. Does not fail the command.
    WARN = "warn"
    #: Broken: something that is supposed to work here does not.
    FAIL = "fail"


@dataclass(frozen=True)
class Check:
    """One question, its answer, and the sentence a person reads."""

    name: str
    status: Status
    detail: str

    def as_json(self) -> dict[str, Any]:
        return {"check": self.name, "status": str(self.status), "detail": self.detail}


def add_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Register ``openforecast doctor``."""
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "doctor",
        help="check this installation",
        description="Report what this installation can do, and what is stopping it.",
    )
    add_store_argument(parser)
    parser.add_argument(
        "--root",
        default=None,
        help="the provider cache to inspect (default: the user cache directory)",
    )
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    parser.set_defaults(handler=run)
    return parser


def run(args: argparse.Namespace, out: IO[str]) -> int:
    """Print every check, and fail if any of them did."""
    found = checks(store=args.store, root=args.root)
    failed = [check for check in found if check.status is Status.FAIL]
    if args.json:
        output.dump(
            {"ok": not failed, "checks": [check.as_json() for check in found]},
            out,
        )
    else:
        output.table(
            ("STATUS", "CHECK", "DETAIL"),
            [(str(check.status), check.name, check.detail) for check in found],
            out,
        )
    return EXIT_ERROR if failed else EXIT_OK


def checks(*, store: str | None = None, root: str | None = None) -> tuple[Check, ...]:
    """Every check, in the order a reader wants them: what this is, then what it can do."""
    return (
        _python(),
        _openforecast(),
        _store(store),
        _uv(),
        *_providers(root),
        _models(),
        _server_extra(),
    )


def _python() -> Check:
    version = platform.python_version()
    return Check("python", Status.OK, f"{version} at {sys.executable}")


def _openforecast() -> Check:
    from openforecast import __version__

    location = Path(__file__).resolve().parents[1]
    return Check(
        "openforecast",
        Status.OK,
        f"{__version__} at {location}, protocol {PROTOCOL_VERSION}",
    )


def _store(store: str | None) -> Check:
    """Where fits land, and whether one could be written there.

    Checked by creating the directory and writing a file into it, because a path
    that exists and is not writable is exactly the case a fit discovers at the
    end of training rather than at the start.
    """
    root = ArtifactStore(store).root
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".doctor"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
    except OSError as error:
        return Check("artifact store", Status.FAIL, f"{root} is not writable: {error.strerror}")
    artifacts = len(ArtifactStore(root).list())
    return Check("artifact store", Status.OK, f"{root}, {artifacts} artifacts")


def _uv() -> Check:
    """Provider environments are uv-managed, so a missing uv is a missing install path."""
    found = shutil.which("uv")
    if found is None:
        return Check(
            "uv",
            Status.WARN,
            "not on the PATH; installing a provider needs it (https://docs.astral.sh/uv/)",
        )
    return Check("uv", Status.OK, found)


def _providers(root: str | None) -> tuple[Check, ...]:
    """The installed environments, and whether each is still usable.

    An environment is a recorded handshake plus an interpreter. The record can
    outlive the interpreter — a cleared cache, a deleted virtualenv — and that is
    a failure rather than an absence: the provider is advertised, and the models
    it advertised cannot run.
    """
    environments = ProviderEnvironments(root)
    try:
        installed = environments.list()
    except OpenForecastError as error:
        return (Check("providers", Status.FAIL, f"{environments.root}: {error}"),)
    if not installed:
        return (
            Check(
                "providers",
                Status.WARN,
                f"none installed in {environments.root}; "
                f"install one with: openforecast providers install nixtla",
            ),
        )
    found = [
        Check(
            "providers",
            Status.OK,
            f"{len(installed)} installed in {environments.root}: "
            f"{', '.join(item.name for item in installed)}",
        )
    ]
    for environment in installed:
        record = environment.record
        if not environment.python.exists():
            found.append(
                Check(
                    f"provider {environment.name}",
                    Status.FAIL,
                    f"the environment's interpreter is missing ({environment.python}); "
                    f"reinstall with: openforecast providers install {environment.name}",
                )
            )
        elif record.protocol_version != PROTOCOL_VERSION:
            found.append(
                Check(
                    f"provider {environment.name}",
                    Status.FAIL,
                    f"speaks protocol {record.protocol_version} and this build speaks "
                    f"{PROTOCOL_VERSION}; reinstall with: "
                    f"openforecast providers install {environment.name}",
                )
            )
    return tuple(found)


def _models() -> Check:
    """What the catalog holds, which is the built-in models plus every provider's.

    The catalog is installed when the package is imported, from the default
    provider cache — so this counts what *this process* can fit, which is not
    ``--root``'s business. ``--root`` is a question about environments on disk;
    this is a question about the build that is answering.
    """
    from openforecast import models

    try:
        available = models.list()
    except OpenForecastError as error:
        return Check("models", Status.FAIL, f"the catalog could not be read: {error}")
    if not available:
        return Check("models", Status.FAIL, "the catalog is empty, so nothing can be fitted")
    providers = sorted({descriptor.provider for descriptor in available})
    return Check("models", Status.OK, f"{len(available)} from {', '.join(providers)}")


def _server_extra() -> Check:
    """Serving needs ``openforecast[server]``; calling a service needs nothing."""
    from importlib.util import find_spec

    missing = [name for name in ("fastapi", "uvicorn") if find_spec(name) is None]
    if missing:
        return Check(
            "server extra",
            Status.WARN,
            f"{', '.join(missing)} not installed, so 'openforecast serve' cannot run; "
            f"pip install 'openforecast[server]'",
        )
    return Check("server extra", Status.OK, "installed, so 'openforecast serve' can run")
