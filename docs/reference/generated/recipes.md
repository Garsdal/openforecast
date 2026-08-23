# Recipes

*Generated from the code by `uv run generate-reference`. Do not edit by hand.*

What to fit: models, pipelines, ensembles, reductions and transforms.

## `ColumnSet`

*Enumeration — `openforecast.recipes.base`*

A role, rather than a list of names.

``columns="targets"`` keeps a recipe portable across datasets: the same
pipeline can be fitted on German prices and on Danish load without being
rewritten, because the role is resolved against whatever schema it meets.

| Member | Value |
| --- | --- |
| `TARGETS` | `'targets'` |
| `FEATURES` | `'features'` |

## `Ensemble`

*Pydantic model — `openforecast.recipes.nodes`*

Several recipes, averaged into one forecast.

```python
of.Ensemble(models=[of.Model("nixtla/nhits"), of.Model("sklearn/hist-gradient-boosting")])

of.Ensemble(models=[...], weights=[0.7, 0.3])
```

Members are recipes rather than models, so an ensemble of pipelines — or of
ensembles — needs no extra vocabulary. Whether a given combination can
actually be fitted on the data at hand is a question about the members'
training contracts, and the engine answers it before the first of them is
trained; a member's contract is not knowable here, where nothing has been
resolved yet.

Combination is a weighted mean and nothing else. ``weights`` left out is an
equal average, which is what an ensemble means when nobody said otherwise;
given, the weights are relative and normalized by their sum, so ``[7, 3]``
means the same thing as ``[0.7, 0.3]``. Every weight must be positive — a
zero-weighted member is fitted and then ignored, which is better said by
leaving it out. Weights are fixed rather than learned: a weight fitted on
data is a second model, and stacking is not what this node is.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `kind` | `Literal[RecipeKind.ENSEMBLE]` | `RecipeKind.ENSEMBLE` |  |
| `models` | `tuple[Annotated[Model \| Pipeline \| Ensemble \| Reduction, FieldInfo(annotation=NoneType, required=True, discriminator='kind')], ...]` | *required* |  |
| `weights` | `tuple[float, ...] \| None` | `None` |  |

## `Impute`

*Pydantic model — `openforecast.recipes.transforms`*

Fill missing values with a stated statistic.

``method`` has no default on purpose. Which fill is right depends on what
the column means, and a default would be OpenForecast quietly choosing for
every dataset that forgot to say.

The statistic is computed on the fitted data and recorded in the artifact,
so inference fills the same way training did rather than leaking the context
it happens to be given.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `columns` | `ColumnSet \| tuple[str, ...]` | *required* |  |
| `kind` | `Literal[RecipeKind.IMPUTE]` | `RecipeKind.IMPUTE` |  |
| `method` | `ImputeMethod` | *required* |  |

## `ImputeMethod`

*Enumeration — `openforecast.recipes.transforms`*

| Member | Value |
| --- | --- |
| `MEAN` | `'mean'` |
| `MEDIAN` | `'median'` |
| `ZERO` | `'zero'` |

## `LeadTimeFeature`

*Pydantic model — `openforecast.recipes.transforms`*

``event_time - origin_time``, as a feature the model can condition on.

Genuinely useful on point-in-time data: a wind forecast issued six hours
ahead is not the same quality of information as one issued sixty, and a
model given the lead time can learn that. Derived rather than stored, which
is why asking for it is a recipe step instead of a column in the source data.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `kind` | `Literal[RecipeKind.LEAD_TIME_FEATURE]` | `RecipeKind.LEAD_TIME_FEATURE` |  |
| `name` | `str` | `'lead_time'` |  |
| `unit` | `FrequencyUnit` | `FrequencyUnit.HOUR` |  |

## `MissingIndicator`

*Pydantic model — `openforecast.recipes.transforms`*

Add a boolean column recording where a value was missing.

Put it *before* an imputation, never after: an indicator computed on imputed
data is constant, and the fact that a feature was not published at an origin
is exactly what would have been lost.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `columns` | `ColumnSet \| tuple[str, ...]` | *required* |  |
| `kind` | `Literal[RecipeKind.MISSING_INDICATOR]` | `RecipeKind.MISSING_INDICATOR` |  |
| `suffix` | `str` | `'_is_missing'` |  |

## `Model`

*Pydantic model — `openforecast.recipes.nodes`*

One model, by reference, with the parameters only its provider understands.

```python
of.Model("nixtla/nhits", params={"max_steps": 500})
```

``params`` is opaque to OpenForecast and reaches the provider unchanged —
which is exactly why it may not carry anything OpenForecast owns. A context
length passed as ``input_size`` would be a second copy of
``WindowPlan(context=...)``, invisible to the planner that has to materialize
the samples, so it is rejected with the field to use instead.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `kind` | `Literal[RecipeKind.MODEL]` | `RecipeKind.MODEL` |  |
| `ref` | `ModelRef` | *required* |  |
| `params` | `dict[str, Any]` | `{}` |  |

## `OriginCalendarFeatures`

*Pydantic model — `openforecast.recipes.transforms`*

Calendar features of the *origin*, not of the event time.

The distinction matters on real vintages: a forecast issued at 06:00 for
tomorrow noon is built from a different information set than one issued at
18:00 for the same noon, and when a model is learning across origins that is
a systematic effect rather than noise.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `kind` | `Literal[RecipeKind.ORIGIN_CALENDAR_FEATURES]` | `RecipeKind.ORIGIN_CALENDAR_FEATURES` |  |
| `hour` | `bool` | `False` |  |
| `weekday` | `bool` | `False` |  |
| `month` | `bool` | `False` |  |
| `prefix` | `str` | `'origin'` |  |

## `Pipeline`

*Pydantic model — `openforecast.recipes.nodes`*

Transforms, then exactly one estimator, in order.

```python
of.Pipeline(steps=[
    of.MissingIndicator(columns="features"),
    of.Impute(columns="features", method="median"),
    of.Model("nixtla/nhits"),
])
```

The estimator has to come last, and there has to be exactly one: a pipeline
is a recipe for a forecast, and steps after the thing that forecasts have
nothing to act on. Transform order is otherwise the caller's, with one
exception — an imputation before a missing indicator over the same columns
would make the indicator constant, silently discarding the very fact it was
added to record.

Pipelines do not nest. Flattening one would change nothing about what it
means, so a nested one is a recipe written two ways.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `kind` | `Literal[RecipeKind.PIPELINE]` | `RecipeKind.PIPELINE` |  |
| `steps` | `tuple[Annotated[StandardScaler \| MissingIndicator \| Impute \| LeadTimeFeature \| OriginCalendarFeatures \| Model \| Ensemble \| Reduction, FieldInfo(annotation=NoneType, required=True, discriminator='kind')], ...]` | *required* |  |

## `Recipe`

*Type alias — `openforecast.recipes.nodes`*

```python
Recipe(*args, **kwargs)
```

Runtime representation of an annotated type.

At its core 'Annotated[t, dec1, dec2, ...]' is an alias for the type 't'
with extra annotations. The alias behaves like a normal typing alias.
Instantiating is the same as instantiating the underlying type; binding
it to types is also the same.

The metadata itself is stored in a '__metadata__' attribute as a tuple.

One of: `Model`, `Pipeline`, `Ensemble`, `Reduction`.

## `Reduction`

*Pydantic model — `openforecast.recipes.nodes`*

Forecasting reduced to tabular regression.

```python
of.Reduction(estimator="lightgbm/regressor", strategy="direct", lags=[1, 24, 168])
```

This is the recipe behind the point-in-time LightGBM setups that motivated
the whole design: the ``ViewPlanner`` materializes a ``TabularView`` holding
one row per instance, origin and horizon step, and the estimator never learns
that some of those rows came from different vintages of the same event time.

The protocol is defined here; execution is still to come. A recipe that
cannot be executed yet is still worth being able to write down and store,
which is why constructing one is not an error.

What a reduction adds is the ``lags``: generating supervised features from an
ordinary event-time series. A ``ForecastDataset`` already carries the
features a supervised row needs — that is what a forecast vintage *is* — so a
tabular model is fitted on one directly, without a ``Reduction`` anywhere in
the recipe.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `kind` | `Literal[RecipeKind.REDUCTION]` | `RecipeKind.REDUCTION` |  |
| `estimator` | `ModelRef` | *required* |  |
| `strategy` | `ReductionStrategy` | *required* |  |
| `lags` | `tuple[int, ...]` | *required* |  |

## `ReductionStrategy`

*Enumeration — `openforecast.recipes.nodes`*

How a horizon is turned into supervised regression problems.

| Member | Value |
| --- | --- |
| `RECURSIVE` | `'recursive'` |
| `DIRECT` | `'direct'` |
| `MULTIOUTPUT` | `'multioutput'` |

## `StandardScaler`

*Pydantic model — `openforecast.recipes.transforms`*

Center and scale columns, undoing it on the way back out.

``per_instance`` scales every series by its own history, which is what makes
a panel of a 40 GW zone and a 2 GW zone learnable by one global model. Turn
it off when the levels are comparable and the differences between them are
the signal.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `columns` | `ColumnSet \| tuple[str, ...]` | *required* |  |
| `kind` | `Literal[RecipeKind.STANDARD_SCALER]` | `RecipeKind.STANDARD_SCALER` |  |
| `per_instance` | `bool` | `True` |  |

## `parse_recipe`

*Function — `openforecast.recipes.nodes`*

```python
parse_recipe(value: object) -> Recipe
```

Read a recipe back from what ``recipe.model_dump()`` produced.

The entry point for every place a recipe crosses a boundary: an artifact
manifest, a provider request, an HTTP body. Nothing about it is
provider-specific, which is the point — a recipe written by one client is
readable by any other.
