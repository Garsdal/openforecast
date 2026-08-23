"""``openforecast fit``, ``forecast`` and ``backtest`` — the three operations.

```bash
openforecast fit --model builtin/seasonal-naive --data ./dataset --name de-price
openforecast fit --config fit.json

openforecast forecast --model local/de-price --data ./dataset --horizon 24 --json
openforecast backtest --config backtest.json --json
```

The same three operations the SDK has, under the same three names — there is no
``train``, no ``predict`` and no ``evaluate`` here either, because Step 24's one
name per intent is a property of OpenForecast rather than of Python. Each command
loads a config, reads the dataset it points at, calls the client method, and
prints what came back. Nothing in this module decides anything a Python caller
would not have decided.

Two shapes per command, which is Step 26.2. The flat flags are the top-level
scalars — a model, a dataset, a horizon — and ``--config`` is everything,
including the nested recipes and plans a flag syntax has no business spelling.
Mixing them is refused rather than merged: a rule for which one wins is a rule
somebody has to remember, and the point of a boring CLI is that there is nothing
to remember.
"""

from __future__ import annotations

import argparse
from typing import IO, Any

from openforecast.artifacts.handle import ModelHandle
from openforecast.commands import config as configs
from openforecast.commands import output
from openforecast.commands.config import ConfigType
from openforecast.commands.exit_codes import EXIT_OK
from openforecast.commands.session import add_store_argument, client_for
from openforecast.errors import OpenForecastError
from openforecast.evaluation.result import BacktestColumn, BacktestResult
from openforecast.runtime.forecast import Forecast

__all__ = ["add_parsers", "run_backtest", "run_fit", "run_forecast"]

#: How many rows of a forecast the human rendering shows. A forecast is often
#: thousands of rows long, and ``--json`` is the complete answer — so this is a
#: preview that says how much it is not showing, never a silent truncation.
PREVIEW_ROWS = 20


def add_parsers(subparsers: Any) -> None:
    """Register the three operations."""
    _add_fit(subparsers)
    _add_forecast(subparsers)
    _add_backtest(subparsers)


# -- openforecast fit -------------------------------------------------------


def _add_fit(subparsers: Any) -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "fit",
        help="fit a model and publish the artifact",
        description="Fit a model on a written dataset, the way of.fit does.",
    )
    _add_common(parser, "fit.json")
    parser.add_argument("--model", default=None, help="the model to fit, such as 'nixtla/nhits'")
    parser.add_argument("--data", default=None, help="the dataset directory to fit on")
    parser.add_argument("--horizon", type=int, default=None, help="the horizon to fit for")
    parser.add_argument("--name", default=None, help="what to call the artifact")
    parser.set_defaults(handler=run_fit)
    return parser


def run_fit(args: argparse.Namespace, out: IO[str]) -> int:
    """Fit, and print the reference the fit produced."""
    settings = _configuration(
        args,
        configs.FitConfig,
        flags={
            "model": args.model,
            "data": args.data,
            "horizon": args.horizon,
            "name": args.name,
        },
        required=("model", "data"),
    )
    handle = client_for(args).fit(
        settings.model,
        configs.read_data(settings.data),
        horizon=settings.horizon,
        plan=settings.plan,
        name=settings.name,
        params=settings.params,
    )
    return _report_fit(handle, out, as_json=args.json)


def _report_fit(handle: ModelHandle, out: IO[str], *, as_json: bool) -> int:
    if as_json:
        output.dump(handle.manifest.model_dump(mode="json"), out)
        return EXIT_OK
    manifest = handle.manifest
    print(str(handle.ref), file=out)
    print(f"  model      {manifest.source_model or 'composite'}", file=out)
    print(f"  provider   {manifest.provider}", file=out)
    print(f"  path       {handle.path}", file=out)
    for record in manifest.training_records:
        print(f"  trained    {record.view} on {record.samples} samples", file=out)
    return EXIT_OK


# -- openforecast forecast --------------------------------------------------


def _add_forecast(subparsers: Any) -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "forecast",
        help="forecast with a fitted model",
        description="Forecast ahead of what a written dataset knows, the way of.forecast does.",
    )
    _add_common(parser, "forecast.json")
    parser.add_argument(
        "--model", default=None, help="the fitted reference, such as 'local/de-price'"
    )
    parser.add_argument("--data", default=None, help="the dataset directory to forecast from")
    parser.add_argument("--horizon", type=int, default=None, help="how many steps ahead")
    parser.add_argument(
        "--origin-time",
        default=None,
        help="the origin to forecast from (default: the last moment the data knows)",
    )
    distribution = parser.add_mutually_exclusive_group()
    distribution.add_argument(
        "--quantiles",
        default=None,
        help="ask for quantiles instead of a point forecast, as in '0.1,0.5,0.9'",
    )
    distribution.add_argument(
        "--samples", type=int, default=None, help="ask for this many sample paths"
    )
    parser.set_defaults(handler=run_forecast)
    return parser


def run_forecast(args: argparse.Namespace, out: IO[str]) -> int:
    """Forecast, and print what came back."""
    settings = _configuration(
        args,
        configs.ForecastConfig,
        flags={
            "model": args.model,
            "data": args.data,
            "horizon": args.horizon,
            "origin_time": args.origin_time,
            "output": _output_spec(args),
        },
        required=("model", "data", "horizon"),
    )
    forecast = client_for(args).forecast(
        settings.model,
        configs.read_data(settings.data),
        horizon=settings.horizon,
        output=settings.output,
        origin_time=settings.origin_time,
    )
    return _report_forecast(forecast, out, as_json=args.json)


def _output_spec(args: argparse.Namespace) -> dict[str, Any] | None:
    """``--quantiles`` and ``--samples`` as the ``OutputSpec`` they mean.

    Built as the mapping a config file would have held rather than as an object,
    so the levels are validated by ``OutputSpec`` itself: ascending, distinct and
    strictly between 0 and 1 is one rule, stated once, in the library.
    """
    if args.quantiles is not None:
        return {"kind": "quantiles", "levels": _levels(args.quantiles)}
    if args.samples is not None:
        return {"kind": "samples", "draws": args.samples}
    return None


def _levels(value: str) -> list[float]:
    try:
        return [float(item) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise OpenForecastError(
            f"--quantiles takes a comma-separated list of levels, as in '0.1,0.5,0.9': {error}"
        ) from error


def _report_forecast(forecast: Forecast, out: IO[str], *, as_json: bool) -> int:
    if as_json:
        output.dump(
            {
                "model": forecast.model,
                "origin_time": forecast.origin_time,
                "horizon": forecast.horizon,
                "kind": str(forecast.kind),
                "targets": list(forecast.targets),
                "instance_keys": list(forecast.instance_keys),
                "rows": output.rows_of(forecast.table),
            },
            out,
        )
        return EXIT_OK
    print(
        f"{forecast.model}  origin {forecast.origin_time.isoformat()}  "
        f"horizon {forecast.horizon}  {forecast.kind}",
        file=out,
    )
    output.rows_as_table(output.rows_of(forecast.table, limit=PREVIEW_ROWS), out)
    hidden = forecast.num_rows - PREVIEW_ROWS
    if hidden > 0:
        print(f"... {hidden} more rows; --json prints all {forecast.num_rows}", file=out)
    return EXIT_OK


# -- openforecast backtest --------------------------------------------------


def _add_backtest(subparsers: Any) -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "backtest",
        help="compare models over historical origins",
        description="Evaluate models at every origin a validation strategy selects.",
    )
    # No flag-only form: a backtest names a validation strategy and a set of
    # metrics, both of which are nested objects. Spelling those as flags is the
    # syntax 26.2 exists to avoid.
    _add_common(parser, "backtest.json", config_required=True)
    parser.set_defaults(handler=run_backtest)
    return parser


def run_backtest(args: argparse.Namespace, out: IO[str]) -> int:
    """Backtest, and print the ranking it produced."""
    settings = configs.load(args.config, configs.BacktestConfig)
    result = client_for(args).backtest(
        list(settings.models),
        configs.read_data(settings.data),
        validation=settings.validation,
        metrics=list(settings.metrics),
        output=settings.output,
        plan=settings.plan,
    )
    return _report_backtest(result, out, as_json=args.json)


def _report_backtest(result: BacktestResult, out: IO[str], *, as_json: bool) -> int:
    boards = {name: output.rows_of(result.leaderboard(name)) for name in result.metric_names}
    if as_json:
        output.dump(
            {
                "models": list(result.models),
                "metrics": list(result.metric_names),
                "origins": list(result.origins),
                "leaderboards": boards,
                "folds": output.rows_of(result.metrics),
            },
            out,
        )
        return EXIT_OK
    print(
        f"{len(result.models)} models over {len(result.origins)} origins, "
        f"scored on {', '.join(result.metric_names)}",
        file=out,
    )
    for name, rows in boards.items():
        print(f"\n{name}", file=out)
        output.rows_as_table(
            [
                {
                    "model": row[BacktestColumn.MODEL.value],
                    "value": row[BacktestColumn.VALUE.value],
                    "folds": row["folds"],
                }
                for row in rows
            ],
            out,
        )
    return EXIT_OK


# -- shared -----------------------------------------------------------------


def _add_common(
    parser: argparse.ArgumentParser, example: str, *, config_required: bool = False
) -> None:
    add_store_argument(parser)
    parser.add_argument(
        "--config",
        default=None,
        required=config_required,
        help=f"a JSON config file, as in --config {example}",
    )
    parser.add_argument("--json", action="store_true", help="print JSON instead of a summary")


def _configuration(
    args: argparse.Namespace,
    kind: type[ConfigType],
    *,
    flags: dict[str, Any],
    required: tuple[str, ...],
) -> ConfigType:
    """One config, from ``--config`` or from the flags that stand in for it.

    Refused rather than merged when both are given. A precedence rule between a
    file and a flag is a thing to remember, and a command whose behaviour has to
    be remembered is not the boring one this step asks for.
    """
    given = {name: value for name, value in flags.items() if value is not None}
    if args.config is not None:
        if given:
            raise OpenForecastError(
                f"--config and {_spelled(given)} configure the same fields; use one or the "
                f"other, and put what the flags cannot say in the file"
            )
        return configs.load(args.config, kind)
    missing = [name for name in required if name not in given]
    if missing:
        raise OpenForecastError(
            f"{_spelled(dict.fromkeys(missing))} "
            f"{'is' if len(missing) == 1 else 'are'} required without --config"
        )
    return configs.validate(given, kind, source="the command line")


def _spelled(names: dict[str, Any]) -> str:
    """Field names as the flags a caller typed, so a message names what they wrote."""
    return ", ".join(f"--{name.replace('_', '-')}" for name in names)
