"""``openforecast providers`` — the four verbs, and what they print.

The CLI is a projection, so the assertions are about the projection: what goes
to stdout, what goes to stderr, and what the exit code is. That contract is what
a script wrapping OpenForecast depends on, and it is the same one the provider
protocol keeps — stdout is the answer, stderr is everything else.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from openforecast.commands import build_parser
from tests.cli import Run, run
from tests.unit.test_provider_environments import MODULE, PROVIDER, VERSION, FakeBuilder


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    return tmp_path / "providers"


@pytest.fixture
def uv_free(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every install in this module builds the environment the fake way."""

    def builder(*_args: object, **_kwargs: object) -> FakeBuilder:
        return FakeBuilder()

    monkeypatch.setattr("openforecast.runtime.environments.UvBuilder", builder)


def install(cache: Path) -> Run:
    return run(
        "providers",
        "--root",
        str(cache),
        "install",
        PROVIDER,
        "--source",
        "openforecast-example",
        "--module",
        MODULE,
    )


def test_the_cli_parses_its_own_help() -> None:
    parser = build_parser()

    assert parser.prog == "openforecast"
    assert parser.parse_args(["providers", "list"]).verb == "list"


def test_nothing_installed_says_how_to_install_something(cache: Path) -> None:
    result = run("providers", "--root", str(cache), "list")

    assert result.code == 0
    assert "no providers are installed" in result.out
    assert "providers install" in result.out


def test_installing_a_provider_reports_what_it_advertises(cache: Path, uv_free: None) -> None:
    result = install(cache)

    assert result.code == 0
    assert f"{PROVIDER} {VERSION}" in result.out
    assert "example/echo" in result.out
    assert result.err == ""


def test_a_listing_names_every_installed_provider(cache: Path, uv_free: None) -> None:
    install(cache)

    result = run("providers", "--root", str(cache), "list")

    assert result.code == 0
    assert result.out.splitlines()[0].split() == ["PROVIDER", "VERSION", "MODELS"]
    assert result.out.splitlines()[1].split() == [PROVIDER, VERSION, "1"]


def test_json_output_is_the_same_facts_for_something_that_parses(
    cache: Path, uv_free: None
) -> None:
    install(cache)

    listed = run("providers", "--root", str(cache), "list", "--json")
    inspected = run("providers", "--root", str(cache), "inspect", PROVIDER, "--json")

    assert listed.json["root"] == str(cache)
    assert listed.json["providers"] == [inspected.json]
    assert inspected.json["provider"] == PROVIDER
    assert inspected.json["module"] == MODULE
    assert inspected.json["command"][-1] == MODULE


def test_removing_a_provider_leaves_nothing_installed(cache: Path, uv_free: None) -> None:
    install(cache)

    removed = run("providers", "--root", str(cache), "remove", PROVIDER)

    assert removed.code == 0
    assert f"removed {PROVIDER}" in removed.out
    assert run("providers", "--root", str(cache), "list", "--json").json["providers"] == []


def test_inspecting_something_that_is_not_installed_fails_loudly(cache: Path) -> None:
    """Not an empty table: a provider that is absent is a thing to be told about."""
    result = run("providers", "--root", str(cache), "inspect", "nixtla")

    assert result.code == 1
    assert result.out == ""
    assert "no provider named 'nixtla'" in result.err


def test_an_unknown_command_is_refused_by_the_parser() -> None:
    with pytest.raises(SystemExit):
        run("providers", "levitate")


# -- openforecast serve -----------------------------------------------------


def test_serve_binds_to_loopback_unless_told_otherwise() -> None:
    """A forecasting service has no authentication yet.

    So the default has to be the one that does not publish an unauthenticated
    service to a network by accident; ``--host 0.0.0.0`` is a decision the
    operator makes out loud.
    """
    args = build_parser().parse_args(["serve"])

    assert args.host == "127.0.0.1"
    assert args.port == 8321
    assert args.store is None


def test_serve_runs_the_application_over_a_local_transport(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The command is a projection: it builds the app and hands it to uvicorn.

    Asserted without binding a socket, because what the command decides is which
    engine and which address — the serving itself is uvicorn's.
    """
    from openforecast.commands import serve as serve_command
    from openforecast.server.transport import LocalTransport

    served: dict[str, Any] = {}

    class FakeUvicorn:
        @staticmethod
        def run(app: object, *, host: str, port: int, log_level: str) -> None:
            served.update(app=app, host=host, port=port)

    captured: dict[str, Any] = {}

    def application(transport: LocalTransport) -> object:
        captured["store"] = transport.engine.store.root
        return "the app"

    monkeypatch.setattr(serve_command, "_uvicorn", lambda: FakeUvicorn)
    monkeypatch.setattr(serve_command, "_application", application)

    result = run("serve", "--store", str(tmp_path / "store"), "--port", "9999")

    assert result.code == 0
    assert served == {"app": "the app", "host": "127.0.0.1", "port": 9999}
    assert captured["store"] == tmp_path / "store"
    assert "http://127.0.0.1:9999/v1" in result.out
