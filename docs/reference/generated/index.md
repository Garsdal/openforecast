# API reference

*Generated from the code by `uv run generate-reference`. Do not edit by hand.*

OpenForecast 0.1.0. `openforecast.__all__` is the whole public
surface of the library and is asserted exactly, so this table is that
assertion rendered: a name that is not here is not public.

| Name | Kind | Documented in |
| --- | --- | --- |
| `Accelerator` | Enumeration | [Plans, tasks and outputs](tasks.md) |
| `AllOrigins` | Pydantic model | [Plans, tasks and outputs](tasks.md) |
| `ArtifactError` | Exception | [Errors](errors.md) |
| `AtOrigin` | Pydantic model | [Plans, tasks and outputs](tasks.md) |
| `BacktestResult` | Class | [Backtesting and metrics](evaluation.md) |
| `Bias` | Pydantic model | [Backtesting and metrics](evaluation.md) |
| `Candidate` | Pydantic model | [Backtesting and metrics](evaluation.md) |
| `ColumnSet` | Enumeration | [Recipes](recipes.md) |
| `Coverage` | Pydantic model | [Backtesting and metrics](evaluation.md) |
| `DEFAULT_CATALOG` | Value | [Models and descriptors](models.md) |
| `DataError` | Exception | [Errors](errors.md) |
| `DuplicateModelError` | Exception | [Errors](errors.md) |
| `Eligibility` | Pydantic model | [Backtesting and metrics](evaluation.md) |
| `Ensemble` | Pydantic model | [Recipes](recipes.md) |
| `FeatureAvailability` | Enumeration | [Semantic data](data.md) |
| `FeatureCapabilities` | Pydantic model | [Models and descriptors](models.md) |
| `FeatureKind` | Enumeration | [Semantic data](data.md) |
| `FeatureSpec` | Pydantic model | [Semantic data](data.md) |
| `FitPlan` | Pydantic model | [Plans, tasks and outputs](tasks.md) |
| `Forecast` | Class | [Forecasts](forecasts.md) |
| `ForecastContext` | Class | [Semantic data](data.md) |
| `ForecastDataset` | Class | [Semantic data](data.md) |
| `ForecastOriginValidation` | Pydantic model | [Backtesting and metrics](evaluation.md) |
| `ForecastTask` | Pydantic model | [Plans, tasks and outputs](tasks.md) |
| `Frequency` | Pydantic model | [Semantic data](data.md) |
| `FrequencyError` | Exception | [Errors](errors.md) |
| `FrequencyUnit` | Enumeration | [Semantic data](data.md) |
| `HttpTransport` | Class | [Transports](transports.md) |
| `Impute` | Pydantic model | [Recipes](recipes.md) |
| `ImputeMethod` | Enumeration | [Recipes](recipes.md) |
| `IncompatibleForecastTask` | Exception | [Errors](errors.md) |
| `InconsistentTruthError` | Exception | [Errors](errors.md) |
| `InstanceCapabilities` | Pydantic model | [Models and descriptors](models.md) |
| `IntervalWidth` | Pydantic model | [Backtesting and metrics](evaluation.md) |
| `InvalidModelParameters` | Exception | [Errors](errors.md) |
| `LatestOrigin` | Pydantic model | [Plans, tasks and outputs](tasks.md) |
| `LeadTimeFeature` | Pydantic model | [Recipes](recipes.md) |
| `LocalTransport` | Class | [Transports](transports.md) |
| `MAE` | Pydantic model | [Backtesting and metrics](evaluation.md) |
| `MAPE` | Pydantic model | [Backtesting and metrics](evaluation.md) |
| `Metric` | Type alias | [Backtesting and metrics](evaluation.md) |
| `MissingIndicator` | Pydantic model | [Recipes](recipes.md) |
| `MissingValueSupport` | Enumeration | [Models and descriptors](models.md) |
| `Model` | Pydantic model | [Recipes](recipes.md) |
| `ModelCapabilities` | Pydantic model | [Models and descriptors](models.md) |
| `ModelCatalog` | Class | [Models and descriptors](models.md) |
| `ModelDescriptor` | Pydantic model | [Models and descriptors](models.md) |
| `ModelDoesNotSupportFit` | Exception | [Errors](errors.md) |
| `ModelError` | Exception | [Errors](errors.md) |
| `ModelLifecycle` | Pydantic model | [Models and descriptors](models.md) |
| `ModelRef` | Pydantic model | [Models and descriptors](models.md) |
| `ModelRefError` | Exception | [Errors](errors.md) |
| `ModelRequiresFit` | Exception | [Errors](errors.md) |
| `OpenForecast` | Class | [Client and operations](client.md) |
| `OpenForecastError` | Exception | [Errors](errors.md) |
| `OriginCalendarFeatures` | Pydantic model | [Recipes](recipes.md) |
| `OriginScope` | Enumeration | [Models and descriptors](models.md) |
| `OriginScopeError` | Exception | [Errors](errors.md) |
| `OriginSelection` | Type alias | [Plans, tasks and outputs](tasks.md) |
| `OriginsBetween` | Pydantic model | [Plans, tasks and outputs](tasks.md) |
| `OutputCapabilities` | Pydantic model | [Models and descriptors](models.md) |
| `OutputKind` | Enumeration | [Plans, tasks and outputs](tasks.md) |
| `OutputSpec` | Pydantic model | [Plans, tasks and outputs](tasks.md) |
| `PinballLoss` | Pydantic model | [Backtesting and metrics](evaluation.md) |
| `Pipeline` | Pydantic model | [Recipes](recipes.md) |
| `PointInTimeFrame` | Class | [Semantic data](data.md) |
| `PointInTimeSchema` | Pydantic model | [Semantic data](data.md) |
| `ProviderError` | Exception | [Errors](errors.md) |
| `ProviderNotInstalled` | Exception | [Errors](errors.md) |
| `RMSE` | Pydantic model | [Backtesting and metrics](evaluation.md) |
| `Recipe` | Type alias | [Recipes](recipes.md) |
| `RecipeError` | Exception | [Errors](errors.md) |
| `Reduction` | Pydantic model | [Recipes](recipes.md) |
| `ReductionStrategy` | Enumeration | [Recipes](recipes.md) |
| `Resources` | Pydantic model | [Plans, tasks and outputs](tasks.md) |
| `RollingOrigin` | Pydantic model | [Backtesting and metrics](evaluation.md) |
| `SchemaError` | Exception | [Errors](errors.md) |
| `StandardScaler` | Pydantic model | [Recipes](recipes.md) |
| `TargetCapabilities` | Pydantic model | [Models and descriptors](models.md) |
| `TimeSeriesFrame` | Class | [Semantic data](data.md) |
| `TimeSeriesSchema` | Pydantic model | [Semantic data](data.md) |
| `TrainingContract` | Pydantic model | [Models and descriptors](models.md) |
| `Transport` | Class | [Transports](transports.md) |
| `UnknownModelError` | Exception | [Errors](errors.md) |
| `UnsupportedDataShape` | Exception | [Errors](errors.md) |
| `UnsupportedFeature` | Exception | [Errors](errors.md) |
| `UnsupportedOutput` | Exception | [Errors](errors.md) |
| `UnsupportedPlanError` | Exception | [Errors](errors.md) |
| `Validation` | Type alias | [Backtesting and metrics](evaluation.md) |
| `ViewKind` | Enumeration | [Models and descriptors](models.md) |
| `WindowPlan` | Pydantic model | [Plans, tasks and outputs](tasks.md) |
| `backtest` | Function | [Backtesting and metrics](evaluation.md) |
| `eligible_models` | Function | [Backtesting and metrics](evaluation.md) |
| `fit` | Function | [Client and operations](client.md) |
| `forecast` | Function | [Client and operations](client.md) |
| `get` | Function | [Models and descriptors](models.md) |
| `list` | Function | [Models and descriptors](models.md) |
| `parse_recipe` | Function | [Recipes](recipes.md) |
| `register` | Function | [Models and descriptors](models.md) |

## Pages

- [Client and operations](client.md) — The four operations, and the client every one of them is a method on. The module-level `of.fit`, `of.forecast`, `of.backtest` and `of.eligible_models` are these methods on a default client, so their signatures differ only by `client=`.
- [Semantic data](data.md) — What you hand OpenForecast: ordinary event-time data, real forecast vintages, and one inference origin cut out of them.
- [Models and descriptors](models.md) — A model reference, and what one resolves to. The catalog itself is not generated here — it holds whatever providers are installed, which is a property of a machine rather than of the library.
- [Recipes](recipes.md) — What to fit: models, pipelines, ensembles, reductions and transforms.
- [Plans, tasks and outputs](tasks.md) — How to fit it, what to predict, and what kind of answer to produce.
- [Forecasts](forecasts.md) — What a forecast is: one long table, and the projections of it.
- [Backtesting and metrics](evaluation.md) — Comparing models over origins, and scoring what comes back.
- [Transports](transports.md) — Where a client executes, which is the only thing a transport decides.
- [Errors](errors.md) — Every failure OpenForecast raises deliberately, with the `error.code` a caller branches on instead of on the prose.
