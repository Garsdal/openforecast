"""``openforecast providers`` — the four verbs, and what they print.

The CLI is a projection, so the assertions are about the projection: what goes
to stdout, what goes to stderr, and what the exit code is. That contract is what
a script wrapping OpenForecast depends on, and it is the same one the provider
protocol keeps — stdout is the answer, stderr is everything else.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

from openforecast.commands import build_parser, main
from tests.unit.test_provider_environments import MODULE, PROVIDER, VERSION, FakeBuilder


class Run:
    """One CLI invocation, and everything it produced."""

    def __init__(self, code: int, out: str, err: str) -> None:
        self.code = code
        self.out = out
        self.err = err

    @property
    def json(self) -> Any:
        return json.loads(self.out)


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    return tmp_path / "providers"


@pytest.fixture
def uv_free(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every install in this module builds the environment the fake way."""

    def builder(*_args: object, **_kwargs: object) -> FakeBuilder:
        return FakeBuilder()

    monkeypatch.setattr("openforecast.runtime.environments.UvBuilder", builder)


def run(*argv: str) -> Run:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, out=out, err=err)
    return Run(code, out.getvalue(), err.getvalue())


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
