# Backtesting and metrics

*Generated from the code by `uv run generate-reference`. Do not edit by hand.*

Comparing models over origins, and scoring what comes back.

## `BacktestResult`

*Class — `openforecast.evaluation.result`*

```python
BacktestResult(metrics: pa.Table, predictions: pa.Table, *, scored_by: Sequence[Metric]) -> None
```

The predictions of one backtest, the metrics over them, and the readings.

The predictions are where the memory goes: one row per model, fold,
instance, event time and target — origins × horizon × instances × targets
per model. A year of hourly origins over a wide panel is a large table, and
it is retained by default anyway, because the metrics are derivable from it
and not the reverse.

| Member | Kind | Summary |
| --- | --- | --- |
| `best(self, metric: str \| Metric \| None = None) -> str` | method | The label of the model that ranked first — the winner, as a string. |
| `instance_keys` | property | The caller's own instance key columns, as ``predictions`` carries them. |
| `leaderboard(self, metric: str \| Metric \| None = None) -> pa.Table` | method | The models ranked by one metric, averaged over the folds. |
| `metric_names` | property | What was measured, spelled as the ``metric`` column spells it. |
| `metrics` | property | The long metric table, in canonical column order. |
| `metrics_by(self, keys: str \| Sequence[str]) -> pa.Table` | method | Every metric again, grouped by columns of :attr:`predictions`. |
| `models` | property | The candidates, in the order they were backtested. |
| `origins` | property | The evaluation origins, in ascending order. |
| `predictions` | property | Every point prediction the metrics were computed from. |
| `to_pandas(self) -> Any` | method | The long metric table as a pandas ``DataFrame``. |

## `Bias`

*Pydantic model — `openforecast.evaluation.metrics`*

Mean signed error: positive means the model forecast too high.

Not an accuracy metric and not ranked as one. Zero is the best value a bias
can have, so :meth:`rank` compares magnitudes — a model biased by -3 and one
biased by +3 are equally biased, and a leaderboard that put one above the
other would be reporting the sign as a virtue.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `metric` | `Literal[MetricKind.BIAS]` | `MetricKind.BIAS` |  |

## `Candidate`

*Pydantic model — `openforecast.evaluation.backtest`*

One entry of a backtest, when the model reference alone is not enough.

```python
of.Candidate("nixtla/nhits", plan=of.FitPlan(window=of.WindowPlan(context=336)))
of.Candidate(recipe, name="scaled-ensemble")
```

Only ever needed for the two things a bare reference cannot say: the plan
this model in particular should be fitted with, and what to call it in the
result. Parameters belong on the ``of.Model`` inside, where every other part
of OpenForecast reads them from.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `model` | `Model \| Pipeline \| Ensemble \| Reduction` | *required* |  |
| `name` | `str \| None` | `None` |  |
| `plan` | `FitPlan \| None` | `None` |  |

## `Coverage`

*Pydantic model — `openforecast.evaluation.metrics`*

How often the outcome fell inside the interval, as a fraction.

```python
of.Coverage()        # of the 0.1 to 0.9 interval
of.Coverage(0.5)     # of the 0.25 to 0.75 interval
```

The calibration question, and the one a point metric cannot ask: an 80%
interval that contains the outcome 55% of the time is overconfident, and one
that contains it 99% of the time is useless in the other direction. So the
best value is the nominal level rather than the highest, and :meth:`rank`
compares the distance to it — a leaderboard that ranked coverage upwards
would be ranking width.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `level` | `float` | `0.8` |  |
| `metric` | `Literal[MetricKind.COVERAGE]` | `MetricKind.COVERAGE` |  |

## `Eligibility`

*Pydantic model — `openforecast.evaluation.eligibility`*

Whether one model could be fitted on the data at hand, and why not.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `model` | `ModelRef` | *required* |  |
| `eligible` | `bool` | *required* |  |
| `reason` | `str \| None` | `None` |  |

## `ForecastOriginValidation`

*Pydantic model — `openforecast.evaluation.validation`*

The origins a point-in-time dataset actually holds, as evaluation origins.

This is the strategy the semantic model exists for. At each selected vintage
the model is fitted on
:meth:`~openforecast.data.forecast_dataset.ForecastDataset.up_to` that
origin, forecast from
:meth:`~openforecast.data.forecast_dataset.ForecastDataset.at_origin` it, and
scored against the truth frame — so what it is given is what was on the wire
that day, revisions included, and the artifact records
``origin_fidelity: observed`` to say so.

Only a ``ForecastDataset``: an event-time frame has no vintages to select,
and :class:`RollingOrigin` is how its origins are simulated instead.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `horizon` | `int` | *required* |  |
| `mode` | `Literal[ValidationMode.ORIGINS]` | `ValidationMode.ORIGINS` |  |
| `origins` | `AllOrigins \| LatestOrigin \| AtOrigin \| OriginsBetween` | `AllOrigins(mode=OriginMode.ALL, stride=1)` |  |

## `IntervalWidth`

*Pydantic model — `openforecast.evaluation.metrics`*

How wide the interval was, in the units of the target.

```python
of.IntervalWidth()   # mean width of the 0.1 to 0.9 interval
```

Sharpness, and only half a question on its own: the narrowest interval any
model can produce is a degenerate one, which scores perfectly here and fails
a :class:`Coverage` completely. Ranked lower-is-better anyway, because that
is what it means among models whose coverage is comparable — reading it
without one beside it is the mistake, and no ranking rule prevents that.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `level` | `float` | `0.8` |  |
| `metric` | `Literal[MetricKind.INTERVAL_WIDTH]` | `MetricKind.INTERVAL_WIDTH` |  |

## `MAE`

*Pydantic model — `openforecast.evaluation.metrics`*

Mean absolute error. The default choice, and the hardest to misread.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `metric` | `Literal[MetricKind.MAE]` | `MetricKind.MAE` |  |

## `MAPE`

*Pydantic model — `openforecast.evaluation.metrics`*

Mean absolute percentage error, in percent.

Refused rather than approximated where an outcome is zero: the percentage
error of a zero outcome is not a large number, it is not a number. Skipping
those rows would silently score a different subset of the horizon for one
model than for another, which is exactly the comparison a backtest exists to
make sound.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `metric` | `Literal[MetricKind.MAPE]` | `MetricKind.MAPE` |  |

## `Metric`

*Type alias — `openforecast.evaluation.metrics`*

```python
Metric(*args, **kwargs)
```

Runtime representation of an annotated type.

At its core 'Annotated[t, dec1, dec2, ...]' is an alias for the type 't'
with extra annotations. The alias behaves like a normal typing alias.
Instantiating is the same as instantiating the underlying type; binding
it to types is also the same.

The metadata itself is stored in a '__metadata__' attribute as a tuple.

One of: `MAE`, `RMSE`, `MAPE`, `Bias`, `PinballLoss`, `Coverage`, `IntervalWidth`.

## `PinballLoss`

*Pydantic model — `openforecast.evaluation.metrics`*

The loss a quantile forecast is the optimal answer to.

```python
of.PinballLoss(0.9)
```

```text
outcome above the forecast    (outcome - forecast) * level
outcome below the forecast    (forecast - outcome) * (1 - level)
```

Asymmetric on purpose, and that asymmetry is the whole content of the metric:
a 0.9 quantile is meant to be exceeded one time in ten, so being under the
outcome is charged nine times what being over it is. Minimized in expectation
by exactly the quantile it names, which is what makes it the metric that says
whether a *quantile* was any good rather than whether a number was close.

One level per metric, because one loss is one level. Scoring three levels is
three metrics, and their sum is not a thing this returns: which levels to
weight and how is a decision about the result, and
:attr:`~openforecast.evaluation.result.BacktestResult.metrics` holds them all
separately so it can be made there.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `metric` | `Literal[MetricKind.PINBALL]` | `MetricKind.PINBALL` |  |
| `level` | `float` | *required* |  |

## `RMSE`

*Pydantic model — `openforecast.evaluation.metrics`*

Root mean squared error — the same units as the target, weighted to the tails.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `metric` | `Literal[MetricKind.RMSE]` | `MetricKind.RMSE` |  |

## `RollingOrigin`

*Pydantic model — `openforecast.evaluation.validation`*

``windows`` origins stepping back from the end of an event-time history.

The last fold's forecast window ends at the last event time the data holds,
and each earlier fold steps back by ``stride`` — the horizon by default,
which makes the windows consecutive and non-overlapping. A shorter stride
evaluates more often over the same history and is honest about the folds
then sharing outcomes.

Only a ``TimeSeriesFrame``. Point-in-time data has origins of its own and
:class:`ForecastOriginValidation` is how they are used; choosing one vintage
per rolling origin here would be this module inventing the very thing the
data already records.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `horizon` | `int` | *required* |  |
| `mode` | `Literal[ValidationMode.ROLLING]` | `ValidationMode.ROLLING` |  |
| `windows` | `int` | *required* |  |
| `stride` | `int \| None` | `None` |  |

## `Validation`

*Type alias — `openforecast.evaluation.validation`*

```python
Validation(*args, **kwargs)
```

Runtime representation of an annotated type.

At its core 'Annotated[t, dec1, dec2, ...]' is an alias for the type 't'
with extra annotations. The alias behaves like a normal typing alias.
Instantiating is the same as instantiating the underlying type; binding
it to types is also the same.

The metadata itself is stored in a '__metadata__' attribute as a tuple.

One of: `RollingOrigin`, `ForecastOriginValidation`.

## `backtest`

*Function — `openforecast.evaluation.backtest`*

```python
backtest(models: Sequence[ModelInput | Candidate], data: object, *, validation: Validation, metrics: Sequence[Metric], output: OutputSpec | None = None, plan: FitPlan | None = None, client: OpenForecast | None = None) -> BacktestResult
```

Evaluate every model at every origin ``validation`` selects.

A trainable candidate is fitted on the data of each origin and forecasts
from it; a pinned revision skips the fit and forecasts as it stands. Both
end up in one result, and one caveat comes with mixing them: a frozen
artifact was fitted on data that may postdate the early origins, so its
numbers are optimistic beside a candidate fitted per fold. That is reported
— ``fit_seconds`` is null and ``artifact`` names the revision — rather than
refused, the same way ``origin_fidelity`` is.

Leaving ``client`` out uses the same default client ``of.fit`` and
``of.forecast`` do, so a backtest writes its artifacts where everything else
does. Passing one pointed at a service backtests there.

``output`` is what every candidate is asked for, and it defaults to a point
forecast. Asking for quantiles is what makes ``of.PinballLoss``,
``of.Coverage`` and ``of.IntervalWidth`` computable, and the metrics are
checked against it before anything is fitted: a coverage of a point forecast
is refused in the first line of the run rather than after an hour of fits.

The result holds every prediction as well as the metrics over them, which is
what makes ``result.metrics_by("horizon_step")`` a projection rather than a
second run. It is also the larger of the two tables by far: origins ×
horizon × instances × targets rows per model, once per quantile level or
sample draw.

## `eligible_models`

*Function — `openforecast.evaluation.eligibility`*

```python
eligible_models(data: object, *, horizon: int | None = None, plan: FitPlan | None = None, models: Sequence[ModelRef | str] | None = None, client: OpenForecast | None = None) -> tuple[Eligibility, ...]
```

Every model the client can reach, and whether this data could fit it.

``models`` narrows the question to the references named; leaving it out asks
it of the whole catalog. ``horizon`` is required by every model that does not
train on complete series, since the horizon is what bounds their samples.

``plan`` is adapted per model the way a backtest's is — a context window
reaches the models that size samples with one and no others — so one plan can
be asked of a whole catalog.
