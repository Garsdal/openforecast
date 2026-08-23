"""The sktime integration: sktime's forecasters, behind the OpenForecast boundary.

```bash
openforecast providers install sktime
```

```python
model = of.fit(model="sktime/theta", data=timeseries)
model = of.fit(
    model="sktime/pooled-trees", data=forecast_dataset, horizon=72, plan=plan
)
```

The distribution lives in its own environment because sktime brings its own
dependency graph — scikit-learn and statsmodels among it — and OpenForecast
depends on none of it. What crosses the boundary in either direction is an
execution view and an Arrow answer, so nothing in here has an opinion about
where the data came from.

This is the third ecosystem, and it is the one with explicit panel and global
semantics: sktime says out loud that a forecaster handed a panel is *vectorized*
over its instances unless it is told to pool across them. That makes it the
sharpest available test of the view abstraction — a ``SeriesView`` is one
vectorized unit, a ``SequenceView`` is a pooled panel, and neither of those
words appears anywhere in OpenForecast. The translation lives in
:mod:`openforecast_sktime.conversion`: sktime's ``MultiIndex`` panel, its single
exogenous ``X`` frame and its ``window_length`` are constructed there from a
view and never travel back out.
"""

from openforecast_sktime._version import __version__
from openforecast_sktime.provider import PROVIDER_NAME, PROVIDER_VERSION, SktimeProvider

__all__ = ["PROVIDER_NAME", "PROVIDER_VERSION", "SktimeProvider", "__version__"]
