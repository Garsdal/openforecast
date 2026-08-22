# openforecast-nixtla

Nixtla's models as an OpenForecast provider, in their own environment.

```bash
openforecast providers install nixtla
```

```python
import openforecast as of

model = of.fit(model="nixtla/autoarima", data=timeseries, params={"season_length": 24})
forecast = of.forecast(model=model, data=context, horizon=48)
```

## What it provides

```text
nixtla/autoarima    order selection over ARIMA models, one model per series
```

`autoarima` is a *local* model: every series is fitted on its own. Its
descriptor says so, and the engine reads that rather than asking:

```yaml
training:
  view: series
  origin_scope: single
  horizon_bound_at_fit: false

capabilities:
  instances:  single, panel
  targets:    univariate
  features:   known (as exogenous regressors)
  outputs:    point
  missing:    unsupported
```

Which means point-in-time data is usable at one origin and not across origins:

```python
of.fit(model="nixtla/autoarima", data=forecast_dataset,
       plan=of.FitPlan(origins=of.AtOrigin(ref_time)))   # a SeriesView, so fine

of.fit(model="nixtla/autoarima", data=forecast_dataset,
       plan=of.FitPlan(origins=of.AllOrigins()))         # OriginScopeError
```

AutoARIMA does not learn jointly across historical forecast origins, so the
second request is refused by the engine before this integration is started.

## Layout

```text
src/openforecast_nixtla/
    __main__.py     the serving harness, two lines
    provider.py     the three provider calls, dispatched
    catalog.py      which models exist, and which adapter runs each
    conversion.py   SeriesView/ForecastView <-> Nixtla's long frame
    adapters/
        statsforecast.py    local statistical models
```

`unique_id`, `ds` and `y` are legal inside this distribution and nowhere else in
OpenForecast. They are constructed in `conversion.py` on the way into a Nixtla
library and taken off again on the way out; what crosses the provider boundary
is an execution view and an Arrow table in the canonical forecast columns.

## Development

```bash
uv sync
uv run pytest
```

The tests include the OpenForecast conformance suite, which is generated from
what the descriptors above declare: every capability becomes a fit that must
succeed over both semantic sources, and everything withheld becomes a request
that must be refused.
