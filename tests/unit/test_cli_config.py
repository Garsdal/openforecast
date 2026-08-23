"""The config files the three operations take, and the flags that stand in for them.

What is asserted here is that a config file is the SDK's own types and nothing
more: a recipe deserializes into a ``Recipe``, a plan into a ``FitPlan``, a
metric into a ``Metric``, and a key nobody recognizes is refused rather than
ignored. A typo that silently changed a fit would be the worst failure a
config-driven CLI can have, since the run succeeds and answers the wrong
question.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

import openforecast as of
from openforecast.commands import config as configs
from openforecast.errors import DataError, OpenForecastError, RecipeError
from tests import factories
from tests.cli import run, write_config

MODEL = "builtin/seasonal-naive"


def frame(periods: int = 12) -> of.TimeSeriesFrame:
    rows = [
        {"timestamp": datetime(2026, 1, 1) + pd.Timedelta(hours=step), "load": float(step)}
        for step in range(periods)
    ]
    return of.TimeSeriesFrame.from_pandas(
        history=pd.DataFrame(rows),
        time="timestamp",
        frequency="1h",
        targets=["load"],
    )


# -- what a config file is --------------------------------------------------


def test_a_fit_config_is_the_arguments_of_of_fit(tmp_path: Path) -> None:
    """Every nested field is the type the library already has for it."""
    path = write_config(
        tmp_path / "fit.json",
        {
            "model": {"kind": "model", "ref": MODEL, "params": {"season_length": 24}},
            "data": "./dataset",
            "horizon": 24,
            "name": "de-price",
            "plan": {"window": {"context": 168}, "seed": 7},
        },
    )

    settings = configs.load(path, configs.FitConfig)

    assert isinstance(settings.model, of.Model)
    assert settings.model.params == {"season_length": 24}
    assert settings.plan == of.FitPlan(window=of.WindowPlan(context=168), seed=7)
    assert settings.data == Path("./dataset")


def test_a_model_reference_is_the_short_spelling_of_a_recipe(tmp_path: Path) -> None:
    path = write_config(tmp_path / "fit.json", {"model": MODEL, "data": "./dataset"})

    assert configs.load(path, configs.FitConfig).model == MODEL


def test_a_backtest_config_holds_a_validation_and_its_metrics(tmp_path: Path) -> None:
    path = write_config(
        tmp_path / "backtest.json",
        {
            "models": [MODEL, {"model": MODEL, "name": "again"}],
            "data": "./dataset",
            "validation": {"mode": "rolling", "horizon": 3, "windows": 2},
            "metrics": [{"metric": "mae"}, {"metric": "pinball", "level": 0.9}],
        },
    )

    settings = configs.load(path, configs.BacktestConfig)

    assert settings.validation == of.RollingOrigin(horizon=3, windows=2)
    assert settings.metrics == (of.MAE(), of.PinballLoss(0.9))
    assert isinstance(settings.models[1], of.Candidate)


def test_a_key_nobody_recognizes_is_refused_by_name(tmp_path: Path) -> None:
    """Ignored would mean a typo runs a different fit and says nothing about it."""
    path = write_config(tmp_path / "fit.json", {"model": MODEL, "data": "./dataset", "horzon": 24})

    with pytest.raises(RecipeError, match="horzon"):
        configs.load(path, configs.FitConfig)


def test_a_field_that_is_out_of_range_is_refused_by_the_librarys_own_rule(tmp_path: Path) -> None:
    path = write_config(tmp_path / "fit.json", {"model": MODEL, "data": "./d", "horizon": 0})

    with pytest.raises(RecipeError, match="horizon"):
        configs.load(path, configs.FitConfig)


def test_a_config_file_that_is_not_there_says_so(tmp_path: Path) -> None:
    with pytest.raises(OpenForecastError, match="cannot read the config file"):
        configs.load(tmp_path / "absent.json", configs.FitConfig)


def test_a_config_file_that_is_not_json_says_so(tmp_path: Path) -> None:
    path = tmp_path / "fit.json"
    path.write_text("model: builtin/seasonal-naive\n", encoding="utf-8")

    with pytest.raises(OpenForecastError, match="is not valid JSON"):
        configs.load(path, configs.FitConfig)


# -- what a data directory is ----------------------------------------------


def test_each_written_dataset_is_recognized_by_what_is_in_it(tmp_path: Path) -> None:
    """Read off the directory rather than declared: a written dataset already says."""
    frame().write(tmp_path / "frame")
    dataset = of.ForecastDataset.from_pandas(
        factories.point_in_time(targets=("price",), known=("wind_fc",)),
        origin_time="ref_time",
        event_time="target_time",
        event_frequency="1h",
        origin_frequency="1h",
        targets=["price"],
        known_features=["wind_fc"],
    )
    dataset.write(tmp_path / "dataset")
    dataset.information.write(tmp_path / "vintages")

    assert isinstance(configs.read_data(tmp_path / "frame"), of.TimeSeriesFrame)
    assert isinstance(configs.read_data(tmp_path / "dataset"), of.ForecastDataset)
    assert isinstance(configs.read_data(tmp_path / "vintages"), of.PointInTimeFrame)


def test_a_directory_that_is_not_a_dataset_says_what_it_found(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")

    with pytest.raises(DataError, match="notes.txt"):
        configs.read_data(tmp_path)


def test_a_path_that_is_not_a_directory_at_all_says_how_to_write_one(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="frame.write"):
        configs.read_data(tmp_path / "absent")


# -- the flags that stand in for a config ----------------------------------


def test_flags_and_a_config_are_refused_together(tmp_path: Path) -> None:
    """A precedence rule between the two is a rule somebody has to remember."""
    path = write_config(tmp_path / "fit.json", {"model": MODEL, "data": "./dataset"})

    result = run("fit", "--config", path, "--name", "de-price")

    assert result.code == 1
    assert result.out == ""
    assert "--config and --name configure the same fields" in result.err


def test_the_flags_a_command_cannot_do_without_are_named(tmp_path: Path) -> None:
    result = run("fit", "--model", MODEL)

    assert result.code == 1
    assert "--data is required without --config" in result.err


def test_a_backtest_has_no_flag_only_form() -> None:
    """Its validation and metrics are nested objects, so --config is required."""
    with pytest.raises(SystemExit):
        run("backtest", "--store", "anywhere")


def test_quantile_levels_on_the_command_line_are_the_librarys_own_rule(tmp_path: Path) -> None:
    frame().write(tmp_path / "frame")

    descending = run(
        "forecast",
        "--store",
        str(tmp_path / "store"),
        "--model",
        "local/absent",
        "--data",
        str(tmp_path / "frame"),
        "--horizon",
        "3",
        "--quantiles",
        "0.9,0.1",
    )

    assert descending.code == 1
    assert "ascending" in descending.err


def test_quantile_levels_that_are_not_numbers_say_what_the_flag_takes() -> None:
    result = run(
        "forecast",
        "--model",
        "local/absent",
        "--data",
        "./dataset",
        "--horizon",
        "3",
        "--quantiles",
        "0.1,middle",
    )

    assert result.code == 1
    assert "--quantiles takes a comma-separated list" in result.err
