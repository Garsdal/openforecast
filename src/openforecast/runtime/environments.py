"""Provider environments: one isolated interpreter per integration.

```text
~/.cache/openforecast/providers/
    nixtla/
        0.1.0/
            environment.json
            .venv/
```

The reason this exists is rule 1. Nixtla wants one version of `torch`, Darts
wants another, and OpenForecast itself wants neither — so an integration is not
installed into the OpenForecast environment at all. It gets its own, built with
`uv`, and it is reached over the subprocess protocol.

An environment is only published once it has answered a handshake. What that
handshake said is written into ``environment.json``, which is what makes
discovery cheap: listing providers and registering the models they advertise
reads recorded JSON and starts no process. A process starts when a model is
actually fitted or forecast with, and the handshake is repeated then — an
environment whose contents changed underneath the record is refused rather than
executed as something it no longer is.

Installation is staged and published by rename, for the same reason a fitted
artifact is: a half-built environment that is nevertheless discoverable would
advertise models it cannot run.

One version of a provider is installed at a time. The version is in the path so
that an upgrade is visible on disk and so that the layout has room for more, but
``nixtla`` has to resolve to one provider, and a directory listing is not a place
to hide a choice between two.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from platformdirs import user_cache_path
from pydantic import BaseModel, ConfigDict

from openforecast.errors import ProviderError, ProviderNotInstalled
from openforecast.models.descriptor import ModelDescriptor
from openforecast.protocol.version import PROTOCOL_VERSION
from openforecast.providers.builtin import PROVIDER_NAME as BUILTIN_PROVIDER_NAME
from openforecast.runtime.subprocess import DEFAULT_TIMEOUT, SubprocessProvider

__all__ = [
    "ENVIRONMENT_FILENAME",
    "EnvironmentBuilder",
    "ProviderEnvironment",
    "ProviderEnvironments",
    "ProviderRecord",
    "UvBuilder",
    "default_cache_root",
    "integration_for",
    "known_source",
    "module_for",
    "shipped_provider_names",
]

ENVIRONMENT_FILENAME = "environment.json"
VENV_DIRNAME = ".venv"
STAGING_DIRNAME = ".tmp"

#: How an integration is installed and imported, from its provider name alone.
#: ``nixtla`` is the distribution ``openforecast-nixtla`` and the module
#: ``openforecast_nixtla``, which is the layout every integration follows.
DISTRIBUTION_PREFIX = "openforecast-"
MODULE_PREFIX = "openforecast_"

#: The integrations whose distribution is not named after the provider they
#: advertise, because the two answer different questions. A provider name is the
#: *namespace of the models*, and Chronos-2 is published as ``amazon/chronos-2``
#: — the reference a user already knows. A distribution is named after the
#: library it wraps, which is ``chronos``. Amazon publishes more than one
#: forecasting model, so naming the distribution ``openforecast-amazon`` would
#: claim a vendor rather than a library.
#:
#: A table rather than a rule, and a short one on purpose: the convention is that
#: the two names agree, and every entry here is a place it had to be written down
#: that they do not.
INTEGRATION_NAMES: Mapping[str, str] = {"amazon": "chronos"}


def default_cache_root() -> Path:
    """``~/.cache/openforecast/providers`` and its equivalent on each platform.

    A cache rather than data: everything in it can be rebuilt by installing
    again, and nothing a user created is lost if it is deleted. Fitted
    artifacts, which cannot be rebuilt, live in the data directory instead.
    """
    return user_cache_path("openforecast", appauthor=False) / "providers"


def shipped_provider_names() -> frozenset[str]:
    """The provider names this build already answers to, in this process.

    A name may not be installed over: it is the namespace of the models the
    provider advertises, so two providers under one name would make a model
    reference mean two different things depending on load order.
    """
    return frozenset({BUILTIN_PROVIDER_NAME})


def integration_for(name: str) -> str:
    """The distribution that provides the models namespaced ``name``."""
    return INTEGRATION_NAMES.get(name, name)


def module_for(name: str) -> str:
    """The module ``python -m`` runs to serve the provider called ``name``."""
    return f"{MODULE_PREFIX}{integration_for(name).replace('-', '_')}"


def known_source(name: str, repository: Path | None = None) -> str:
    """What to install for the provider called ``name``.

    A checkout of OpenForecast has the integrations beside it, and installing
    the one in the working tree is what a contributor means. Anywhere else it is
    the published distribution.
    """
    integration = integration_for(name)
    root = repository if repository is not None else _repository_root()
    if root is not None:
        candidate = root / "integrations" / integration
        if (candidate / "pyproject.toml").is_file():
            return str(candidate)
    return f"{DISTRIBUTION_PREFIX}{integration}"


class ProviderRecord(BaseModel):
    """``environment.json``: what the provider said when it was installed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    provider_version: str
    protocol_version: int = PROTOCOL_VERSION
    #: What ``python -m`` runs.
    module: str
    #: What was installed into the environment: a path, or a requirement.
    source: str
    #: Serialized descriptors, exactly as the handshake reported them.
    models: tuple[dict[str, Any], ...] = ()

    @property
    def descriptors(self) -> tuple[ModelDescriptor, ...]:
        return tuple(ModelDescriptor.model_validate(payload) for payload in self.models)


class ProviderEnvironment:
    """One installed integration, and the command that runs it."""

    def __init__(self, path: str | Path, record: ProviderRecord) -> None:
        self._path = Path(path)
        self._record = record

    @property
    def path(self) -> Path:
        """The version directory holding the environment and its record."""
        return self._path

    @property
    def record(self) -> ProviderRecord:
        return self._record

    @property
    def name(self) -> str:
        return self._record.provider

    @property
    def version(self) -> str:
        return self._record.provider_version

    @property
    def venv(self) -> Path:
        return self._path / VENV_DIRNAME

    @property
    def python(self) -> Path:
        """The interpreter inside the environment, whatever the platform calls it."""
        return venv_python(self.venv)

    @property
    def command(self) -> tuple[str, ...]:
        return (str(self.python), "-m", self._record.module)

    def client(self, *, timeout: float = DEFAULT_TIMEOUT) -> SubprocessProvider:
        """A provider client for this environment, with nothing started yet."""
        return SubprocessProvider(
            self.command,
            name=self.name,
            version=self.version,
            descriptors=self._record.descriptors,
            timeout=timeout,
        )

    @classmethod
    def read(cls, path: str | Path) -> ProviderEnvironment:
        directory = Path(path)
        record = directory / ENVIRONMENT_FILENAME
        if not record.is_file():
            raise ProviderError(f"{directory} is not a provider environment")
        return cls(directory, ProviderRecord.model_validate(_read_json(record)))

    def __repr__(self) -> str:
        return f"ProviderEnvironment({self.name}=={self.version}, path={self._path})"


class EnvironmentBuilder(Protocol):
    """How an environment is created and populated.

    A protocol so that the mechanism is replaceable: ``uv`` is the tool
    OpenForecast uses, and the installer does not need to be the only way to put
    a provider on disk.
    """

    def create(self, venv: Path) -> None:
        """Create an empty virtual environment at ``venv``."""
        ...

    def install(self, venv: Path, source: str) -> None:
        """Install ``source`` into the environment at ``venv``."""
        ...


class UvBuilder:
    """Builds environments with `uv`, which is how OpenForecast manages them."""

    def __init__(self, uv: str | None = None) -> None:
        self._uv = uv

    @property
    def uv(self) -> str:
        found = self._uv if self._uv is not None else shutil.which("uv")
        if found is None:
            raise ProviderError(
                "uv is not installed, and provider environments are uv-managed; install it "
                "from https://docs.astral.sh/uv/ and try again"
            )
        return found

    def create(self, venv: Path) -> None:
        self._run([self.uv, "venv", str(venv)])

    def install(self, venv: Path, source: str) -> None:
        self._run([self.uv, "pip", "install", "--python", str(venv_python(venv)), source])

    def _run(self, command: Sequence[str]) -> None:
        completed = subprocess.run(  # noqa: S603 - the command is uv and its arguments
            list(command),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise ProviderError(
                f"{' '.join(command)} failed with exit code {completed.returncode}:\n"
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )

    def __repr__(self) -> str:
        return f"UvBuilder(uv={self._uv!r})"


class ProviderEnvironments:
    """The installed provider environments: list, install, inspect, remove."""

    def __init__(
        self, root: str | Path | None = None, *, builder: EnvironmentBuilder | None = None
    ) -> None:
        self._root = Path(root) if root is not None else default_cache_root()
        self._builder = builder if builder is not None else UvBuilder()

    @property
    def root(self) -> Path:
        return self._root

    @property
    def staging_root(self) -> Path:
        return self._root / STAGING_DIRNAME

    # -- discovery ----------------------------------------------------------

    def list(self) -> tuple[ProviderEnvironment, ...]:
        """Every installed environment, by provider name.

        A directory without a readable record is not an environment and is
        skipped: an interrupted install should not stop the ones that worked
        from being discovered.
        """
        return tuple(sorted(self._discover(), key=lambda found: found.name))

    def _discover(self) -> Iterator[ProviderEnvironment]:
        if not self._root.is_dir():
            return
        for provider in sorted(self._root.iterdir()):
            if provider.name.startswith(".") or not provider.is_dir():
                continue
            for version in sorted(provider.iterdir()):
                if (version / ENVIRONMENT_FILENAME).is_file():
                    yield ProviderEnvironment.read(version)

    def get(self, name: str) -> ProviderEnvironment:
        """The environment installed for ``name``."""
        for environment in self.list():
            if environment.name == name:
                return environment
        installed = [environment.name for environment in self.list()]
        raise ProviderNotInstalled(
            f"no provider named {name!r} is installed"
            + (f"; installed: {installed}" if installed else "; nothing is installed yet")
            + f". Install it with: openforecast providers install {name}",
            provider=name,
            installed=installed,
        )

    def __contains__(self, name: str) -> bool:
        return any(environment.name == name for environment in self.list())

    def clients(self, *, timeout: float = DEFAULT_TIMEOUT) -> tuple[SubprocessProvider, ...]:
        """A client per installed environment. Starts nothing."""
        return tuple(environment.client(timeout=timeout) for environment in self.list())

    # -- installation -------------------------------------------------------

    def install(
        self,
        name: str,
        *,
        source: str | None = None,
        module: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> ProviderEnvironment:
        """Build an environment for ``name`` and publish it once it handshakes."""
        if name in shipped_provider_names():
            raise ProviderError(
                f"{name!r} is a provider OpenForecast ships with, and a provider name is "
                f"the namespace of the models it advertises, so installing a second one "
                f"under that name would make a model reference mean two things"
            )
        resolved = source if source is not None else known_source(name)
        entry = module if module is not None else module_for(name)
        staging = self.staging_root / name
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        try:
            venv = staging / VENV_DIRNAME
            self._builder.create(venv)
            self._builder.install(venv, resolved)
            record = self._probe(name, venv, entry, resolved, timeout=timeout)
            (staging / ENVIRONMENT_FILENAME).write_text(
                record.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
            return self._publish(staging, record)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def _probe(
        self, name: str, venv: Path, module: str, source: str, *, timeout: float
    ) -> ProviderRecord:
        """Ask the freshly built environment who it is.

        The one moment a provider is executed during installation, and it is
        what makes the recorded models trustworthy: nothing is written down that
        the provider did not itself say.
        """
        command = (str(venv_python(venv)), "-m", module)
        with SubprocessProvider(command, timeout=timeout) as provider:
            if provider.name != name:
                raise ProviderError(
                    f"{source} was installed as provider {name!r} and identifies itself as "
                    f"{provider.name!r}; a provider name is the namespace of the models it "
                    f"advertises, so the two have to agree"
                )
            return ProviderRecord(
                provider=provider.name,
                provider_version=provider.version,
                module=module,
                source=source,
                models=tuple(
                    descriptor.model_dump(mode="json") for descriptor in provider.descriptors()
                ),
            )

    def _publish(self, staged: Path, record: ProviderRecord) -> ProviderEnvironment:
        """Move a complete environment into place and retire any older one."""
        destination = self._root / record.provider / record.provider_version
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(destination, ignore_errors=True)
        os.rename(staged, destination)
        for sibling in sorted(destination.parent.iterdir()):
            if sibling != destination and sibling.is_dir():
                shutil.rmtree(sibling, ignore_errors=True)
        return ProviderEnvironment(destination, record)

    def remove(self, name: str) -> Path:
        """Delete the environment installed for ``name``, and return where it was."""
        environment = self.get(name)
        shutil.rmtree(environment.path.parent)
        return environment.path

    def __repr__(self) -> str:
        return f"ProviderEnvironments({self._root})"


def venv_python(venv: Path) -> Path:
    """The interpreter of a virtual environment, on this platform."""
    if sys.platform == "win32":  # pragma: no cover - exercised on Windows only
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _repository_root() -> Path | None:
    """The OpenForecast checkout this package was imported from, if it is one."""
    root = Path(__file__).resolve().parents[3]
    looks_like_one = (root / "pyproject.toml").is_file() and (root / "integrations").is_dir()
    return root if looks_like_one else None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ProviderError(f"{path} is not valid JSON: {error}") from error
