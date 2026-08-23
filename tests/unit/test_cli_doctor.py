"""``openforecast doctor`` — what it checks, and when it fails.

The point of the command is the exit code: a container's health check and an
agent's first call both want "is this installation able to forecast" answered
without parsing prose. So what is asserted here is which status each situation
produces, and that a ``fail`` really does exit non-zero.

Every test names its own store and provider cache. A doctor that wrote to the
real ones would be a test that changed the machine it ran on — and the store
check writes, deliberately, because a path that exists and is not writable is
exactly what a fit discovers at the end of training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from openforecast.commands.doctor import Status
from openforecast.protocol.version import PROTOCOL_VERSION
from openforecast.runtime.environments import ENVIRONMENT_FILENAME, VENV_DIRNAME
from tests.cli import run


@pytest.fixture
def paths(tmp_path: Path) -> tuple[str, str]:
    """A store and a provider cache of this test's own."""
    return str(tmp_path / "store"), str(tmp_path / "providers")


def doctor(paths: tuple[str, str], *extra: str) -> Any:
    store, root = paths
    result = run("doctor", "--store", store, "--root", root, *extra)
    return result


def statuses(payload: Any) -> dict[str, str]:
    return {check["check"]: check["status"] for check in payload["checks"]}


def test_a_healthy_installation_passes(paths: tuple[str, str]) -> None:
    result = doctor(paths, "--json")

    assert result.code == 0
    assert result.json["ok"] is True
    found = statuses(result.json)
    assert found["python"] == "ok"
    assert found["openforecast"] == "ok"
    assert found["artifact store"] == "ok"
    assert found["models"] == "ok"


def test_no_providers_installed_is_a_warning_rather_than_a_failure(
    paths: tuple[str, str],
) -> None:
    """An installation with only the built-in models is a working installation."""
    result = doctor(paths, "--json")

    assert result.code == 0
    assert statuses(result.json)["providers"] == Status.WARN


def test_the_table_and_the_document_are_the_same_checks(paths: tuple[str, str]) -> None:
    table = doctor(paths)
    document = doctor(paths, "--json")

    assert table.out.splitlines()[0].split() == ["STATUS", "CHECK", "DETAIL"]
    assert len(table.out.splitlines()) == len(document.json["checks"]) + 1


def test_a_store_that_cannot_be_written_fails(tmp_path: Path) -> None:
    """Not writable is the failure a fit would otherwise discover after training."""
    blocked = tmp_path / "file"
    blocked.write_text("not a directory", encoding="utf-8")

    result = run(
        "doctor", "--store", str(blocked / "store"), "--root", str(tmp_path / "p"), "--json"
    )

    assert result.code == 1
    assert result.json["ok"] is False
    assert statuses(result.json)["artifact store"] == Status.FAIL


def test_an_environment_whose_interpreter_is_gone_fails(tmp_path: Path) -> None:
    """The record can outlive the interpreter, and then the models it advertises cannot run."""
    _record(tmp_path / "providers" / "nixtla" / "1.0.0", protocol=PROTOCOL_VERSION)

    result = run(
        "doctor",
        "--store",
        str(tmp_path / "store"),
        "--root",
        str(tmp_path / "providers"),
        "--json",
    )

    assert result.code == 1
    assert statuses(result.json)["provider nixtla"] == Status.FAIL
    detail = _detail(result.json, "provider nixtla")
    assert "interpreter is missing" in detail
    assert "openforecast providers install nixtla" in detail


def test_an_environment_speaking_another_protocol_fails(tmp_path: Path) -> None:
    version = tmp_path / "providers" / "nixtla" / "1.0.0"
    _record(version, protocol=PROTOCOL_VERSION + 1)
    interpreter = version / VENV_DIRNAME / "bin"
    interpreter.mkdir(parents=True)
    (interpreter / "python").write_text("", encoding="utf-8")
    (interpreter / "python.exe").write_text("", encoding="utf-8")

    result = run(
        "doctor",
        "--store",
        str(tmp_path / "store"),
        "--root",
        str(tmp_path / "providers"),
        "--json",
    )

    assert result.code == 1
    assert statuses(result.json)["provider nixtla"] == Status.FAIL
    assert f"speaks protocol {PROTOCOL_VERSION + 1}" in _detail(result.json, "provider nixtla")


def test_an_installed_provider_is_reported_as_installed(tmp_path: Path) -> None:
    version = tmp_path / "providers" / "nixtla" / "1.0.0"
    _record(version, protocol=PROTOCOL_VERSION)
    interpreter = version / VENV_DIRNAME / "bin"
    interpreter.mkdir(parents=True)
    (interpreter / "python").write_text("", encoding="utf-8")
    (interpreter / "python.exe").write_text("", encoding="utf-8")

    result = run(
        "doctor",
        "--store",
        str(tmp_path / "store"),
        "--root",
        str(tmp_path / "providers"),
        "--json",
    )

    assert result.code == 0
    assert statuses(result.json)["providers"] == Status.OK
    assert "nixtla" in _detail(result.json, "providers")


def _record(path: Path, *, protocol: int) -> None:
    path.mkdir(parents=True)
    (path / ENVIRONMENT_FILENAME).write_text(
        json.dumps(
            {
                "provider": "nixtla",
                "provider_version": "1.0.0",
                "protocol_version": protocol,
                "module": "openforecast_nixtla",
                "source": "openforecast-nixtla",
                "models": [],
            }
        ),
        encoding="utf-8",
    )


def _detail(payload: Any, name: str) -> str:
    return next(check["detail"] for check in payload["checks"] if check["check"] == name)
