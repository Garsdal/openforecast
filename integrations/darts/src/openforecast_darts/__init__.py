"""The Darts integration: Darts' models, behind the OpenForecast boundary.

```bash
openforecast providers install darts
```

```python
model = of.fit(model="darts/theta", data=timeseries)
model = of.fit(model="darts/tide", data=forecast_dataset, horizon=72, plan=plan)
```

The distribution lives in its own environment because Darts brings its own
dependency graph — PyTorch and Lightning among it — and OpenForecast depends on
none of it. What crosses the boundary in either direction is an execution view
and an Arrow answer, so nothing in here has an opinion about where the data came
from.

This is the second global-model implementation, and it exists to answer a
question the first one could not: whether the abstraction is Nixtla-shaped.
Darts identifies a series by a ``TimeSeries`` object rather than by a
``unique_id`` string, splits covariates into ``past``, ``future`` and ``static``
rather than into three ``*_exog_list`` arguments, and calls the context window
``input_chunk_length``. None of that reaches OpenForecast: it is constructed in
:mod:`openforecast_darts.conversion` from a ``SeriesView`` or a ``SequenceView``
and never travels back out.
"""

from openforecast_darts._version import __version__
from openforecast_darts.provider import PROVIDER_NAME, PROVIDER_VERSION, DartsProvider

__all__ = ["PROVIDER_NAME", "PROVIDER_VERSION", "DartsProvider", "__version__"]
