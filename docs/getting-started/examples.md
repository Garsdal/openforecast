# Executable examples

Seven scripts live in [`examples/`](https://github.com/Garsdal/openforecast/tree/main/examples)
in the repository. They are complete programs rather than fragments: clone it and
run any of them.

```bash
uv run examples/01_quickstart.py
```

| Example | Teaches | Guide |
| --- | --- | --- |
| [`01_quickstart.py`](https://github.com/Garsdal/openforecast/blob/main/examples/01_quickstart.py) | fit, forecast, and what an artifact is | [Quickstart](quickstart.md) |
| [`02_panel.py`](https://github.com/Garsdal/openforecast/blob/main/examples/02_panel.py) | many series in one frame, and per-series scores | [Event-time data](../guides/event-time.md) |
| [`03_point_in_time.py`](https://github.com/Garsdal/openforecast/blob/main/examples/03_point_in_time.py) | real forecast vintages, and what they change | [Point-in-time data](../guides/point-in-time.md) |
| [`04_backtest.py`](https://github.com/Garsdal/openforecast/blob/main/examples/04_backtest.py) | comparing candidates, and slicing the result | [Backtesting](../guides/backtesting.md) |
| [`05_probabilistic.py`](https://github.com/Garsdal/openforecast/blob/main/examples/05_probabilistic.py) | requesting a distribution, and being refused one | [Probabilistic forecasts](../guides/probabilistic.md) |
| [`06_ensemble.py`](https://github.com/Garsdal/openforecast/blob/main/examples/06_ensemble.py) | recipes: pipelines, transforms, ensembles | [Pipelines and ensembles](../guides/ensembles.md) |
| [`07_zero_shot.py`](https://github.com/Garsdal/openforecast/blob/main/examples/07_zero_shot.py) | pretrained models, and reading a lifecycle | [Model lifecycles](../concepts/model-lifecycle.md) |

Seven, and deliberately not fifty. An example nobody runs is documentation that
has stopped being true, so every one of these is executed as a script by
`tests/examples/test_examples.py` on every CI run: an API change that breaks an
example breaks the build, the same way a changed signature breaks the diff of the
[generated reference](../reference/generated/index.md).

## What they assume

Nothing but a core install and `builtin/seasonal-naive`, the model every build
ships with. Each script generates its own tiny deterministic table — no
downloads, no provider environments, no external datasets, and the same numbers
on the second run as on the first.

Two of them are about capabilities no built-in model has. `05_probabilistic.py`
needs a model that declares quantile output, and `07_zero_shot.py` needs a
pretrained one. Both read the catalog rather than hard-coding a reference, so
they run either way: with nothing installed they demonstrate the refusal — a
`UNSUPPORTED_OUTPUT` and a `MODEL_REQUIRES_FIT`, which are as much of the
protocol as the quantiles are — and print what to install. With

```bash
openforecast providers install nixtla
openforecast providers install amazon
```

they run the real thing against the same data.

## Examples and pages

The two are checked in the same way and neither is the other's copy. Every
Python block on a page under `docs/` is executed by `tests/docs/`, in page order
and in one namespace, so a page is a program too; the scripts under `examples/`
are the version you can run without a test harness, and the pages above link to
the script that goes furthest on their subject rather than restating it.

## Adapting them

The reference is the part to change. Swapping `builtin/seasonal-naive` for
`nixtla/nhits`, `darts/tide`, `sktime/pooled-trees` or
`sklearn/hist-gradient-boosting` changes that string and nothing else about the
surrounding code — which is the claim the whole library exists to make, and the
reason these are worth copying. See [Installation](installation.md) for the
provider environments.
