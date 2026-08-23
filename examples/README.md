# Examples

Seven scripts, each one complete. Clone the repository and run any of them:

```bash
uv run examples/01_quickstart.py
```

| Example | Teaches |
| --- | --- |
| [`01_quickstart.py`](01_quickstart.py) | fit, forecast, and what an artifact is |
| [`02_panel.py`](02_panel.py) | many series in one frame, and per-series scores |
| [`03_point_in_time.py`](03_point_in_time.py) | real forecast vintages, and what they change |
| [`04_backtest.py`](04_backtest.py) | comparing candidates, and slicing the result |
| [`05_probabilistic.py`](05_probabilistic.py) | requesting a distribution, and being refused one |
| [`06_ensemble.py`](06_ensemble.py) | recipes: pipelines, transforms, ensembles |
| [`07_zero_shot.py`](07_zero_shot.py) | pretrained models, and reading a lifecycle |

Seven, and deliberately not fifty: an example nobody runs is documentation that
has stopped being true. `tests/examples/test_examples.py` executes every one of
these as a script on every CI run, so an API change that breaks an example breaks
the build.

## What they assume

A core install and the model every build ships with,
`builtin/seasonal-naive` — no downloads, no provider environments, no external
datasets. Each script generates its own tiny deterministic table, so two runs
print the same numbers and a run with no network prints them too.

Two of the examples are about capabilities no built-in model has:
`05_probabilistic.py` needs a model that declares quantile output and
`07_zero_shot.py` needs a pretrained one. Both read the catalog rather than
hard-coding a reference, so they run either way: with nothing installed they
demonstrate the refusal and print what to install, and with

```bash
openforecast providers install nixtla   # nixtla/autoarima: quantiles
openforecast providers install amazon   # amazon/chronos-2: zero-shot
```

they run the real thing against the same data.

## Adapting them

The reference is the part to change. Swapping `builtin/seasonal-naive` for
`nixtla/nhits`, `darts/tide`, `sktime/pooled-trees` or
`sklearn/hist-gradient-boosting` changes that string and nothing else about the
surrounding code — which is the claim the whole library is for, and the reason
these examples are worth copying.

See [the documentation](https://garsdal.github.io/openforecast/) for the guides
and the generated reference.
