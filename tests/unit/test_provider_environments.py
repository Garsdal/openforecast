"""Provider environments: install, discover, inspect, remove.

`uv` itself is not exercised here — building a real environment would make the
suite depend on a network. What is exercised is everything around it: an
environment is published only after the provider it contains has answered a
handshake, discovery reads that recorded handshake and starts nothing, and a
build that fails leaves nothing behind that could be discovered.

The builder is replaced by one that makes the environment's interpreter this
one, so the handshake at the end of an install is a real provider in a real
subprocess answering the real protocol.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from openforecast.errors import ProviderError, UnknownModelError
from openforecast.models.catalog import ModelCatalog
from openforecast.runtime.environments import (
    ENVIRONMENT_FILENAME,
    INTEGRATION_NAMES,
    ProviderEnvironment,
    ProviderEnvironments,
    UvBuilder,
    default_cache_root,
    integration_for,
    known_source,
    module_for,
)
from openforecast.runtime.providers import install_default_providers
from openforecast.views import SeriesView
from tests import stub_provider

#: An installable provider that is not the one OpenForecast ships. Standing in
#: for ``openforecast_nixtla``, which arrives with the Nixtla integration.
PROVIDER = stub_provider.PROVIDER_NAME
VERSION = stub_provider.PROVIDER_VERSION
MODULE = "tests.stub_provider"


class FakeBuilder:
    """Points the "environment" at the one running the tests, which already has
    a provider in it.

    Everything an installation does other than resolving and downloading
    packages is real: the venv layout, the staging directory, the handshake with
    a live subprocess, the record, the publish.
    """

    def __init__(self) -> None:
        self.created: list[Path] = []
        self.installed: list[tuple[Path, str]] = []

    def create(self, venv: Path) -> None:
        self.created.append(venv)
        venv.parent.mkdir(parents=True, exist_ok=True)
        venv.symlink_to(Path(sys.prefix), target_is_directory=True)

    def install(self, venv: Path, source: str) -> None:
        self.installed.append((venv, source))


class FailingBuilder(FakeBuilder):
    def install(self, venv: Path, source: str) -> None:
        raise ProviderError("resolution failed: no version of torch satisfies both")


@pytest.fixture
def environments(tmp_path: Path) -> ProviderEnvironments:
    return ProviderEnvironments(tmp_path / "providers", builder=FakeBuilder())


def install(environments: ProviderEnvironments, name: str = PROVIDER) -> ProviderEnvironment:
    return environments.install(name, source="openforecast-example", module=MODULE)


# -- installation -----------------------------------------------------------


def test_an_install_records_what_the_provider_said_about_itself(
    environments: ProviderEnvironments,
) -> None:
    environment = install(environments)

    assert environment.name == PROVIDER
    assert environment.version == VERSION
    assert environment.record.descriptors == stub_provider.PROVIDER.descriptors()
    assert environment.record.source == "openforecast-example"
    assert environment.path == environments.root / PROVIDER / VERSION
    assert (environment.path / ENVIRONMENT_FILENAME).is_file()


def test_an_installed_environment_runs_its_own_interpreter(
    environments: ProviderEnvironments,
) -> None:
    environment = install(environments)

    assert environment.command == (str(environment.python), "-m", MODULE)
    assert environment.python.is_relative_to(environment.venv)


def test_an_install_that_fails_leaves_nothing_behind(tmp_path: Path) -> None:
    """A half-built environment that is discoverable advertises what it cannot run."""
    environments = ProviderEnvironments(tmp_path / "providers", builder=FailingBuilder())

    with pytest.raises(ProviderError, match=r"resolution failed"):
        install(environments)

    assert environments.list() == ()
    assert not list(environments.staging_root.glob("*"))


def test_a_provider_that_is_not_the_one_being_installed_is_refused(
    environments: ProviderEnvironments,
) -> None:
    with pytest.raises(ProviderError, match=r"identifies itself as 'example'"):
        environments.install("nixtla", source="openforecast-example", module=MODULE)

    assert environments.list() == ()


def test_a_provider_openforecast_ships_cannot_be_installed_over(
    environments: ProviderEnvironments,
) -> None:
    """One namespace is one provider, whoever it is that provides it."""
    with pytest.raises(ProviderError, match=r"'builtin' is a provider OpenForecast ships"):
        environments.install("builtin", source="openforecast", module=MODULE)

    assert environments.list() == ()


def test_reinstalling_replaces_the_version_that_was_there(
    environments: ProviderEnvironments,
) -> None:
    """One name resolves to one provider, so one version of it is installed."""
    first = install(environments)
    (first.path / "marker").write_text("old", encoding="utf-8")

    second = install(environments)

    assert [found.name for found in environments.list()] == [PROVIDER]
    assert not (second.path / "marker").exists()


# -- discovery --------------------------------------------------------------


def test_discovery_reads_the_record_and_starts_nothing(
    environments: ProviderEnvironments,
) -> None:
    install(environments)

    clients = environments.clients()

    assert [client.name for client in clients] == [PROVIDER]
    assert clients[0].descriptors() == stub_provider.PROVIDER.descriptors()
    assert not clients[0].is_running


def test_an_installed_provider_makes_its_models_discoverable(tmp_path: Path) -> None:
    """The catalog cannot tell a subprocess provider from a shipped one."""
    environments = ProviderEnvironments(tmp_path / "providers", builder=FakeBuilder())
    install(environments)
    catalog = ModelCatalog()

    providers = install_default_providers(catalog, environments)

    assert [str(ref) for ref in catalog.list()] == ["builtin/seasonal-naive", "example/echo"]
    assert sorted(provider.name for provider in providers) == ["builtin", "example"]


def test_nothing_installed_is_an_empty_listing_and_a_named_miss_is_an_error(
    environments: ProviderEnvironments,
) -> None:
    assert environments.list() == ()
    assert "nixtla" not in environments

    with pytest.raises(UnknownModelError, match=r"nothing is installed yet"):
        environments.get("nixtla")


def test_an_interrupted_install_does_not_hide_the_environments_that_worked(
    environments: ProviderEnvironments,
) -> None:
    install(environments)
    (environments.root / "half-built" / "0.1.0").mkdir(parents=True)

    assert [found.name for found in environments.list()] == [PROVIDER]


def test_a_record_that_is_not_json_says_so(environments: ProviderEnvironments) -> None:
    environment = install(environments)
    (environment.path / ENVIRONMENT_FILENAME).write_text("{", encoding="utf-8")

    with pytest.raises(ProviderError, match=r"is not valid JSON"):
        environments.list()


def test_a_provider_whose_environment_changed_underneath_the_record_is_refused(
    environments: ProviderEnvironments,
) -> None:
    """The recorded handshake is checked against the real one before anything runs."""
    environment = install(environments)
    record = json.loads((environment.path / ENVIRONMENT_FILENAME).read_text(encoding="utf-8"))
    record["models"] = []
    (environment.path / ENVIRONMENT_FILENAME).write_text(json.dumps(record), encoding="utf-8")

    client = environments.get(PROVIDER).client()

    with pytest.raises(ProviderError, match=r"advertises different models"):
        client.fit(
            model="example/echo",
            params={},
            view=_series_view(),
            seed=None,
            into=environment.path,
        )


# -- removal ----------------------------------------------------------------


def test_removing_a_provider_removes_every_version_of_it(
    environments: ProviderEnvironments,
) -> None:
    environment = install(environments)

    environments.remove(PROVIDER)

    assert environments.list() == ()
    assert not environment.path.exists()


def test_removing_something_that_is_not_installed_is_an_error(
    environments: ProviderEnvironments,
) -> None:
    with pytest.raises(UnknownModelError, match=r"no provider named 'nixtla'"):
        environments.remove("nixtla")


# -- conventions ------------------------------------------------------------


def test_a_provider_name_says_what_to_install_and_what_to_run() -> None:
    assert module_for("nixtla") == "openforecast_nixtla"
    assert known_source("nixtla", repository=Path("/nowhere")) == "openforecast-nixtla"


def test_a_checkout_installs_the_integration_beside_it(tmp_path: Path) -> None:
    """What a contributor means by ``providers install nixtla`` in a checkout."""
    integration = tmp_path / "integrations" / "nixtla"
    integration.mkdir(parents=True)
    (integration / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

    assert known_source("nixtla", repository=tmp_path) == str(integration)


def test_a_provider_named_after_its_publisher_still_finds_its_distribution() -> None:
    """The one pair in the repository where the two names disagree.

    A provider name is the namespace of the models it advertises, so Chronos-2 is
    ``amazon/chronos-2``; the distribution is named after the library it wraps.
    ``INTEGRATION_NAMES`` is where that is written down, and it is what makes
    ``openforecast providers install amazon`` find ``integrations/chronos``.
    """
    assert integration_for("amazon") == "chronos"
    assert integration_for("nixtla") == "nixtla"
    assert module_for("amazon") == "openforecast_chronos"
    assert known_source("amazon", repository=Path("/nowhere")) == "openforecast-chronos"


def test_the_checkout_holds_an_integration_for_every_renamed_provider() -> None:
    """A table entry pointing at nothing would be a name that installs nothing."""
    root = Path(__file__).resolve().parents[2]
    for provider, integration in INTEGRATION_NAMES.items():
        assert (root / "integrations" / integration / "pyproject.toml").is_file(), provider


def test_environments_live_in_the_cache_and_artifacts_do_not() -> None:
    """Everything here can be rebuilt; a fitted artifact cannot."""
    root = default_cache_root()

    assert root.name == "providers"
    assert "openforecast" in str(root)


def test_uv_is_how_environments_are_built_and_its_absence_is_explained() -> None:
    builder = UvBuilder(uv=None)
    assert builder.uv  # uv is what runs this suite

    with pytest.raises(ProviderError, match=r"failed with exit code"):
        UvBuilder(uv=sys.executable).create(Path("-c-not-a-venv-path"))


def _series_view() -> SeriesView:
    from tests.unit.test_view_bundle import series_view

    return series_view()
