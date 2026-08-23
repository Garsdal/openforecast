"""Backtesting, point-in-time evaluation, and the foundation of `openforecast/auto`.

```python
result = of.backtest(
    models=["builtin/seasonal-naive", "nixtla/autoarima", "nixtla/nhits"],
    data=data,
    validation=of.RollingOrigin(horizon=24, windows=5),
    metrics=[of.MAE(), of.Bias()],
)

result.leaderboard("mae")
result.best("mae")
```

This is where the universal abstraction stops being a way to call other people's
libraries and starts being worth something on its own: the same models, over the
same origins, scored the same way, with no provider-specific backtesting code
anywhere. Everything here is built on ``ModelRecipe``, ``ForecastDataset`` /
``TimeSeriesFrame``, the ``ViewPlanner``, ``FitPlan``, ``ForecastTask`` and
``Forecast`` — and on ``of.fit`` and ``of.forecast``, which is why it lives in
the outermost layer beside the client rather than inside the engine.

Four modules, in the order the concepts arrive:

```text
metrics.py       what a forecast is scored by
validation.py    which historical origins, and what "correct" means there
backtest.py     the loop over of.fit and of.forecast, and one Arrow result
eligibility.py   which models could be fitted at all — the auto foundation
result.py        the result table and the projections people read it as
```

The claim worth checking is in ``validation.py``: for a ``ForecastDataset``,
evaluating at a historical origin uses the vintage that actually existed, and the
later ones are not merely unused but absent from the object the model is handed.
"""

from openforecast.evaluation.backtest import Candidate, backtest, plan_for
from openforecast.evaluation.eligibility import Eligibility, eligible_models
from openforecast.evaluation.metrics import MAE, MAPE, RMSE, Bias, Metric, MetricKind
from openforecast.evaluation.result import BACKTEST_COLUMNS, BacktestColumn, BacktestResult
from openforecast.evaluation.validation import (
    Fold,
    ForecastOriginValidation,
    RollingOrigin,
    Validation,
    ValidationMode,
)

__all__ = [
    "BACKTEST_COLUMNS",
    "BacktestColumn",
    "BacktestResult",
    "Bias",
    "Candidate",
    "Eligibility",
    "Fold",
    "ForecastOriginValidation",
    "MAE",
    "MAPE",
    "Metric",
    "MetricKind",
    "RMSE",
    "RollingOrigin",
    "Validation",
    "ValidationMode",
    "backtest",
    "eligible_models",
    "plan_for",
]
