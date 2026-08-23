# Client and operations

*Generated from the code by `uv run generate-reference`. Do not edit by hand.*

The four operations, and the client every one of them is a method on. The module-level `of.fit`, `of.forecast`, `of.backtest` and `of.eligible_models` are these methods on a default client, so their signatures differ only by `client=`.

## `OpenForecast`

*Class — `openforecast.client`*

```python
OpenForecast(*, store: str | Path | ArtifactStore | None = None, catalog: ModelCatalog | None = None, providers: ProviderRegistry | None = None, transport: Transport | None = None) -> None
```

Fits and forecasts, wherever its transport executes.

``OpenForecast()`` is local and owns a store and the providers this build
ships with. ``OpenForecast(transport=HttpTransport(...))`` is the same
object over a service, and the arguments naming local machinery — a store,
a catalog, a provider registry — belong to the local transport, so passing
both is refused rather than silently ignoring one.

| Member | Kind | Summary |
| --- | --- | --- |
| `artifact(self, ref: ModelRef \| str) -> ModelHandle` | method | One fitted artifact, described without loading the model behind it. |
| `backtest(self, models: Sequence[ModelInput \| Candidate], data: object, *, validation: Validation, metrics: Sequence[Metric], output: OutputSpec \| None = None, plan: FitPlan \| None = None) -> BacktestResult` | method | Evaluate every model at every origin ``validation`` selects. |
| `eligible_models(self, data: object, *, horizon: int \| None = None, plan: FitPlan \| None = None, models: Sequence[ModelRef \| str] \| None = None) -> tuple[Eligibility, ...]` | method | Which of this client's models this data could fit at all. |
| `engine` | property | The engine behind a local client. |
| `fit(self, model: ModelInput, data: object, *, horizon: int \| None = None, plan: FitPlan \| None = None, name: str \| None = None, params: dict[str, Any] \| None = None) -> ModelHandle` | method | Fit ``model`` on ``data``, returning the artifact it produced. |
| `forecast(self, model: ModelInput, data: object, *, horizon: int, output: OutputSpec \| None = None, origin_time: str \| datetime \| None = None) -> Forecast` | method | Forecast ``horizon`` steps ahead of what ``data`` knows. |
| `models` | property | The models this client can fit. |
| `transport` | property | Where this client executes. |

## `fit`

*Function — `openforecast.client`*

```python
fit(model: ModelInput, data: object, *, horizon: int | None = None, plan: FitPlan | None = None, name: str | None = None, params: dict[str, Any] | None = None) -> ModelHandle
```

Fit a model on data and publish the artifact it produced.

```python
model = of.fit(
    model="builtin/seasonal-naive",
    data=train,
    params={"season_length": 24},
)
```

## `forecast`

*Function — `openforecast.client`*

```python
forecast(model: ModelInput, data: object, *, horizon: int, output: OutputSpec | None = None, origin_time: str | datetime | None = None) -> Forecast
```

Forecast with a fitted model.

```python
forecast = of.forecast(model="local/de-price", data=context, horizon=24)
```

``model`` may be the handle a fit returned, a pinned revision, or the alias
that follows the latest one. A reference naming a model that was never
fitted raises ``ModelRequiresFit`` rather than quietly fitting one on the
data the forecast was handed.
