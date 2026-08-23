"""Step 26's "done when", from the shell and nothing else.

> An agent can perform the normal OpenForecast workflow entirely through shell
> commands and structured JSON.

So this is that workflow, with ``--json`` on every step and no Python object
crossing between them: discover a model, fit it, forecast with the reference the
fit printed, backtest two candidates, and check the installation. Each step reads
the previous step's stdout as JSON — which is also what makes the stream contract
an assertion rather than a claim, since a log line on stdout would break the
parse.

The data is written to disk first, because a command line cannot hand over a
frame. That is the one thing the CLI has that the SDK does not, and it is
``TimeSeriesFrame.write`` — the same file layout ``of.TimeSeriesFrame.read``
loads, not a CLI format.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

import openforecast as of
from tests.cli import Run, run, write_config

MODEL = "builtin/seasonal-naive"
START = datetime(2026, 1, 1)
HOUR = timedelta(hours=1)
SEASON = 24
HORIZON = 6


@pytest.fixture
def store(tmp_path: Path) -> str:
    return str(tmp_path / "store")


@pytest.fixture
def data(tmp_path: Path) -> str:
    """A written dataset, which is how a shell hands data to OpenForecast."""
    rows = [{"timestamp": START + HOUR * step, "load": float(step % SEASON)} for step in range(96)]
    frame = of.TimeSeriesFrame.from_pandas(
        history=pd.DataFrame(rows),
        time="timestamp",
        frequency="1h",
        targets=["load"],
    )
    frame.write(tmp_path / "dataset")
    return str(tmp_path / "dataset")


def fit(store: str, data: str, *extra: str) -> Run:
    return run("fit", "--store", store, "--model", MODEL, "--data", data, *extra)


def test_the_whole_workflow_is_json_on_stdout(store: str, data: str, tmp_path: Path) -> None:
    """Discover, fit, forecast, backtest — each step reading the last one's stdout."""
    listed = run("models", "--store", store, "list", "--json").json
    assert MODEL in {
        f"{item['ref']['namespace']}/{item['ref']['name']}" for item in listed["models"]
    }

    fitted = fit(store, data, "--name", "de-price", "--json").json
    reference = f"local/{fitted['name']}@{fitted['artifact_id']}"
    assert fitted["provider"] == "builtin"

    forecast = run(
        "forecast",
        "--store",
        store,
        "--model",
        reference,
        "--data",
        data,
        "--horizon",
        str(HORIZON),
        "--json",
    ).json
    assert forecast["horizon"] == HORIZON
    assert forecast["kind"] == "point"
    assert forecast["targets"] == ["load"]
    assert len(forecast["rows"]) == HORIZON
    assert {row["value"] for row in forecast["rows"]} <= {float(step) for step in range(SEASON)}

    config = write_config(
        tmp_path / "backtest.json",
        {
            "models": [MODEL, {"model": MODEL, "name": "again"}],
            "data": data,
            "validation": {"mode": "rolling", "horizon": HORIZON, "windows": 2},
            "metrics": [{"metric": "mae"}],
        },
    )
    backtested = run("backtest", "--store", store, "--config", config, "--json").json
    assert sorted(backtested["models"]) == ["again", MODEL]
    assert backtested["metrics"] == ["mae"]
    assert len(backtested["origins"]) == 2
    assert [entry["model"] for entry in backtested["leaderboards"]["mae"]]

    checked = run("doctor", "--store", store, "--json").json
    assert checked["ok"] is True


def test_the_unpinned_alias_follows_the_latest_fit(store: str, data: str) -> None:
    """``local/de-price`` is what a script forecasts with between fits."""
    first = fit(store, data, "--name", "de-price", "--json").json
    second = fit(store, data, "--name", "de-price", "--json").json

    assert first["artifact_id"] != second["artifact_id"]

    forecast = run(
        "forecast",
        "--store",
        store,
        "--model",
        "local/de-price",
        "--data",
        data,
        "--horizon",
        "3",
        "--json",
    ).json

    assert forecast["model"].endswith(second["artifact_id"])


def test_a_quantile_request_a_model_cannot_answer_fails_before_anything_is_printed(
    store: str, data: str
) -> None:
    """The built-in model is deterministic, and nothing invents a distribution for it."""
    fitted = fit(store, data, "--name", "de-price", "--json").json

    result = run(
        "forecast",
        "--store",
        store,
        "--model",
        f"local/de-price@{fitted['artifact_id']}",
        "--data",
        data,
        "--horizon",
        "3",
        "--quantiles",
        "0.1,0.9",
    )

    assert result.code == 1
    assert result.out == ""
    assert "quantile" in result.err


def test_forecasting_with_a_model_nobody_fitted_fails(store: str, data: str) -> None:
    result = run(
        "forecast",
        "--store",
        store,
        "--model",
        "local/never-fitted",
        "--data",
        data,
        "--horizon",
        "3",
        "--json",
    )

    assert result.code == 1
    assert result.out == ""
    assert result.err.startswith("error: ")


def test_the_human_rendering_says_how_much_it_is_not_showing(store: str, data: str) -> None:
    """A preview, never a silent truncation: --json is the complete answer."""
    fit(store, data, "--name", "de-price")

    result = run(
        "forecast",
        "--store",
        store,
        "--model",
        "local/de-price",
        "--data",
        data,
        "--horizon",
        "48",
    )

    assert result.code == 0
    assert "... 28 more rows; --json prints all 48" in result.out


# -- the real process -------------------------------------------------------


def test_the_entry_point_is_the_same_command_in_a_real_process(store: str) -> None:
    """Everything above runs ``main`` in process. This runs it as a process.

    Which is what makes the stream contract a fact about the installed command
    rather than about a test harness: stdout is parsed as JSON, and stderr is
    read separately.
    """
    completed = subprocess.run(  # noqa: S603 - the command is this interpreter
        [sys.executable, "-m", "openforecast.commands.main", "models", "--store", store, "list"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert MODEL in completed.stdout


def test_a_failure_in_a_real_process_is_a_non_zero_exit_and_a_clean_stdout(store: str) -> None:
    completed = subprocess.run(  # noqa: S603 - the command is this interpreter
        [
            sys.executable,
            "-m",
            "openforecast.commands.main",
            "models",
            "--store",
            store,
            "get",
            "nixtla/nhits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert completed.stdout == ""
    assert "error: " in completed.stderr
