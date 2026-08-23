"""The config files ``fit``, ``forecast`` and ``backtest`` take.

```json
{
  "model": "builtin/seasonal-naive",
  "horizon": 24,
  "data": "./dataset",
  "plan": {"window": {"context": 168}}
}
```

Step 26.2: a simple command is flags, and a command with a nested recipe in it is
a file. The alternative is dozens of flags spelling out a pipeline, which is a
second syntax for something the library already has one for.

So these models are deliberately thin. Every nested field is the *same Pydantic
type the SDK uses* — ``Recipe``, ``FitPlan``, ``OutputSpec``, ``Validation``,
``Metric``, ``Candidate`` — so a config file is the arguments of ``of.fit`` as
JSON and nothing is validated twice or interpreted differently here. What these
add is the one thing a command line has that a Python call does not: ``data`` is
a *path* rather than a loaded dataset, since a shell cannot hand over a frame.

Two consequences worth stating. Paths resolve against the working directory,
like every other path on a command line, rather than against the config file —
so ``--config`` does not change what ``./dataset`` means. And a key nobody
recognizes is refused rather than ignored: ``extra="forbid"`` is what turns a
typo into a message instead of a silently different fit.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from openforecast.data.forecast_dataset import INFORMATION_DIRNAME, TRUTH_DIRNAME, ForecastDataset
from openforecast.data.frame import HISTORY_FILENAME, TimeSeriesFrame
from openforecast.data.point_in_time import TABLE_FILENAME, PointInTimeFrame
from openforecast.errors import DataError, OpenForecastError, RecipeError
from openforecast.evaluation.backtest import Candidate
from openforecast.evaluation.metrics import Metric
from openforecast.evaluation.validation import Validation
from openforecast.recipes.nodes import Recipe
from openforecast.tasks.forecast import OutputSpec
from openforecast.tasks.plan import FitPlan

__all__ = [
    "BacktestConfig",
    "Config",
    "FitConfig",
    "ForecastConfig",
    "load",
    "read_data",
    "validate",
]


class Config(BaseModel):
    """Frozen and closed, like every other declaration in the package."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The dataset directory, as written by ``frame.write(...)`` or
    #: ``dataset.write(...)``. Which of the three it holds is read off what is
    #: in it rather than declared, because the directory already says.
    data: Path


class FitConfig(Config):
    """``openforecast fit --config`` — the arguments of ``of.fit``.

    ``model`` is a reference or a whole recipe, which is the same union the
    library and the HTTP projection accept: ``"builtin/seasonal-naive"`` is the
    short spelling of ``{"kind": "model", "ref": "builtin/seasonal-naive"}``.
    """

    model: Recipe | str
    horizon: int | None = Field(default=None, ge=1)
    plan: FitPlan | None = None
    name: str | None = None
    params: dict[str, Any] | None = None


class ForecastConfig(Config):
    """``openforecast forecast --config`` — the arguments of ``of.forecast``.

    ``model`` is the reference a fit produced: ``local/de-price@01K...`` for one
    revision, or ``local/de-price`` for whichever is latest. A recipe is refused
    by the client, because forecasting from one would mean fitting a model the
    caller never asked to fit.
    """

    model: str
    horizon: int = Field(ge=1)
    output: OutputSpec | None = None
    origin_time: datetime | None = None


class BacktestConfig(Config):
    """``openforecast backtest --config`` — the arguments of ``of.backtest``.

    The one command that has no flag-only form. A backtest needs a validation
    strategy and a set of metrics, both of which are nested objects, and
    inventing a flag syntax for them is exactly what 26.2 says not to do.
    """

    models: tuple[str | Candidate, ...] = Field(min_length=1)
    validation: Validation
    metrics: tuple[Metric, ...] = Field(min_length=1)
    output: OutputSpec | None = None
    plan: FitPlan | None = None


ConfigType = TypeVar("ConfigType", bound=Config)


def load(path: str | Path, kind: type[ConfigType]) -> ConfigType:
    """Read one config file as ``kind``.

    Every failure is reported as the thing to fix: a file that is not there, a
    document that is not JSON, a document that is JSON and does not describe an
    executable operation. None of them is a traceback, because none of them is a
    bug in OpenForecast.
    """
    file = Path(path)
    try:
        text = file.read_text(encoding="utf-8")
    except OSError as error:
        raise OpenForecastError(f"cannot read the config file {file}: {error.strerror}") from error
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise OpenForecastError(f"{file} is not valid JSON: {error}") from error
    return validate(payload, kind, source=str(file))


def validate(payload: object, kind: type[ConfigType], *, source: str) -> ConfigType:
    """The same validation for a file and for the flags that stand in for one.

    A flag form is turned into the mapping a config file would have held and
    validated here, so ``--horizon 0`` is refused by the same rule that refuses
    ``"horizon": 0`` and there is only one place either can be wrong.
    """
    try:
        return kind.model_validate(payload)
    except ValidationError as error:
        raise RecipeError(
            f"{source} does not describe {_article(kind)}:\n{_problems(error)}"
        ) from error


def read_data(path: str | Path) -> TimeSeriesFrame | PointInTimeFrame | ForecastDataset:
    """The dataset in ``path``, whichever of the three it holds.

    Read off the directory rather than declared in the config, because a written
    dataset already says what it is: a ``ForecastDataset`` is an ``information/``
    and a ``truth/``, a ``PointInTimeFrame`` is a ``table.arrow``, and a
    ``TimeSeriesFrame`` is a ``history.arrow``. Every one of them is loaded
    through the ordinary constructor, so a truncated table fails to load here
    exactly as it would in Python.
    """
    directory = Path(path)
    if not directory.is_dir():
        raise DataError(
            f"{directory} is not a dataset directory; write one with frame.write(path) "
            f"or dataset.write(path) and point the config at it"
        )
    if (directory / INFORMATION_DIRNAME).is_dir() and (directory / TRUTH_DIRNAME).is_dir():
        return ForecastDataset.read(directory)
    if (directory / TABLE_FILENAME).is_file():
        return PointInTimeFrame.read(directory)
    if (directory / HISTORY_FILENAME).is_file():
        return TimeSeriesFrame.read(directory)
    raise DataError(
        f"{directory} holds none of the three written datasets: expected "
        f"{HISTORY_FILENAME} (a TimeSeriesFrame), {TABLE_FILENAME} (a PointInTimeFrame), "
        f"or {INFORMATION_DIRNAME}/ and {TRUTH_DIRNAME}/ (a ForecastDataset); found "
        f"{sorted(item.name for item in directory.iterdir())}"
    )


def _problems(error: ValidationError) -> str:
    """Pydantic's report, one line per field, without the URLs."""
    lines: list[str] = []
    for problem in error.errors():
        location = ".".join(str(part) for part in problem["loc"]) or "(document)"
        lines.append(f"  {location}: {problem['msg']}")
    return "\n".join(lines)


def _article(kind: type[Config]) -> str:
    named: dict[type[Config], str] = {
        FitConfig: "a fit",
        ForecastConfig: "a forecast",
        BacktestConfig: "a backtest",
    }
    return named.get(kind, kind.__name__)
