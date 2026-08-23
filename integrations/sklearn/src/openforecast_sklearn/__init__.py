"""The scikit-learn integration: a ``TabularView``, and ``fit(X, y)``.

```bash
openforecast providers install sklearn
```

```python
model = of.fit(
    model="sklearn/hist-gradient-boosting", data=forecast_dataset, horizon=72
)
forecast = of.forecast(model=model, data=forecast_dataset.at_origin(now), horizon=72)
```

The fourth ecosystem, and the first one that is not a forecasting framework at
all. scikit-learn knows nothing about a forecast origin, an event time, a lead
or an information vintage — it knows a design matrix and a label vector. That is
exactly why it is the integration that proves the ``TabularView`` boundary: the
transformation

```text
forecasting problem  ->  supervised rows  ->  X / y
```

is OpenForecast's, and everything past it is ``estimator.fit(X, y)``.

The alternative would have been to reduce through a forecasting framework that
already knows how to turn forecasting into regression. That puts the same
semantics in two places, and the library's version wins silently: OpenForecast
already knows the origin, the target time, the lead, the vintage and the truth
alignment, and a ``TabularView`` is what it materializes them into. So this
integration reduces nothing. It receives rows.

What crosses the boundary is an execution view and an Arrow answer, so nothing
in here has an opinion about where the data came from — a ``TimeSeriesFrame`` and
a ``ForecastDataset`` are indistinguishable from inside, and the duplicated
labels of a point-in-time fit are simply four rows rather than a problem to
reconcile.
"""

from openforecast_sklearn._version import __version__
from openforecast_sklearn.provider import PROVIDER_NAME, PROVIDER_VERSION, SklearnProvider

__all__ = ["PROVIDER_NAME", "PROVIDER_VERSION", "SklearnProvider", "__version__"]
