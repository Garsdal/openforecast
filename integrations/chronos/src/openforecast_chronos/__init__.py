"""The Chronos integration: the first model that forecasts without being fitted.

```bash
openforecast providers install amazon
```

```python
forecast = of.forecast(
    model="amazon/chronos-2", data=context, horizon=72
)
```

Every integration before this one advertises models with a training contract,
and every one of them is used the same way: fit, then forecast with the
``local/...`` reference that comes back. Chronos-2 is pretrained. There is
nothing to fit, so ``of.fit`` on it raises ``ModelDoesNotSupportFit`` and the
reference itself is what a forecast names.

That is the whole of what this integration is here to prove, and the interesting
part is how little of OpenForecast it changes. A pretrained model is not a new
kind of data primitive, a new call or a new result object:

```text
ForecastContext          the same one a fitted model is forecast from
      ↓
ViewPlanner              the same materialization
      ↓
ForecastView             the same inference view
      ↓
Chronos-2                a provider call, with no fitted state behind it
      ↓
canonical forecast       the same Arrow table, point or quantile
```

So the same point-in-time dataset backtests ``sklearn/hist-gradient-boosting``,
``nixtla/nhits``, ``darts/tide`` and ``amazon/chronos-2`` in one call, and the
zero-shot model is scored at every origin on exactly the information the fitted
ones were given at that origin. OpenForecast owns the information vintage;
Chronos receives a context and answers it.

The provider is called ``amazon`` because a provider name is the namespace of
the models it advertises, and ``amazon/chronos-2`` is the reference the model is
published under. The distribution is called ``openforecast-chronos`` after the
library it wraps. The two disagree here and nowhere else, which is why
``INTEGRATION_NAMES`` in ``openforecast.runtime.environments`` has exactly one
entry.
"""

from openforecast_chronos._version import __version__
from openforecast_chronos.provider import PROVIDER_NAME, PROVIDER_VERSION, ChronosProvider

__all__ = ["PROVIDER_NAME", "PROVIDER_VERSION", "ChronosProvider", "__version__"]
