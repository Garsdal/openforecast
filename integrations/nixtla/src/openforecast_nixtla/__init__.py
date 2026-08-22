"""The Nixtla integration: Nixtla's models, behind the OpenForecast boundary.

```bash
openforecast providers install nixtla
```

```python
model = of.fit(model="nixtla/autoarima", data=timeseries)
model = of.fit(model="nixtla/nhits", data=forecast_dataset, horizon=72, plan=plan)
```

The distribution lives in its own environment because Nixtla's libraries bring
their own dependency graph — PyTorch among it — and OpenForecast depends on none
of it. What crosses the boundary in either direction is an execution view and an
Arrow answer, so nothing in here has an opinion about where the data came from.

``unique_id``, ``ds``, ``y`` and the three ``*_exog_list`` roles are legal inside
this distribution and nowhere else. They are constructed in
:mod:`openforecast_nixtla.conversion` from a ``SeriesView`` or a
``SequenceView`` and never travel back out.
"""

from openforecast_nixtla._version import __version__
from openforecast_nixtla.provider import PROVIDER_NAME, PROVIDER_VERSION, NixtlaProvider

__all__ = ["PROVIDER_NAME", "PROVIDER_VERSION", "NixtlaProvider", "__version__"]
