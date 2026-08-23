# Models and descriptors

*Generated from the code by `uv run generate-reference`. Do not edit by hand.*

A model reference, and what one resolves to. The catalog itself is not generated here — it holds whatever providers are installed, which is a property of a machine rather than of the library.

## `DEFAULT_CATALOG`

*Value — `openforecast.models.catalog`*

A `ModelCatalog`, documented as the type it is an instance of.

## `FeatureCapabilities`

*Pydantic model — `openforecast.models.capabilities`*

Which feature roles the model can consume.

Named after the roles OpenForecast declares, not after any provider's
covariate vocabulary; ``hist_exog_list`` and friends are a translation that
happens inside an integration.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `observed` | `bool` | `False` |  |
| `known` | `bool` | `False` |  |
| `static` | `bool` | `False` |  |

## `InstanceCapabilities`

*Pydantic model — `openforecast.models.capabilities`*

How many series at once.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `single` | `bool` | `True` |  |
| `panel` | `bool` | `False` |  |

## `MissingValueSupport`

*Enumeration — `openforecast.models.capabilities`*

| Member | Value |
| --- | --- |
| `NATIVE` | `'native'` |
| `REQUIRES_TRANSFORM` | `'requires_transform'` |
| `UNSUPPORTED` | `'unsupported'` |

## `ModelCapabilities`

*Pydantic model — `openforecast.models.capabilities`*

The full capability declaration of one model.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `instances` | `InstanceCapabilities` | `InstanceCapabilities(single=True, panel=False)` |  |
| `targets` | `TargetCapabilities` | `TargetCapabilities(univariate=True, multivariate=False)` |  |
| `features` | `FeatureCapabilities` | `FeatureCapabilities(observed=False, known=False, static=False)` |  |
| `outputs` | `OutputCapabilities` | `OutputCapabilities(point=True, quantiles=False, samples=False)` |  |
| `missing_values` | `MissingValueSupport` | `MissingValueSupport.UNSUPPORTED` |  |

## `ModelCatalog`

*Class — `openforecast.models.catalog`*

```python
ModelCatalog(descriptors: Iterable[ModelDescriptor] = ()) -> None
```

A set of model descriptors, keyed by reference.

| Member | Kind | Summary |
| --- | --- | --- |
| `get(self, ref: ModelRef \| str) -> ModelDescriptor` | method | The descriptor for ``ref``, which may be a plain string. |
| `list(self, *, provider: str \| None = None) -> tuple[ModelDescriptor, ...]` | method | Every registered descriptor, in reference order. |
| `providers(self) -> tuple[str, ...]` | method |  |
| `refs(self) -> tuple[ModelRef, ...]` | method |  |
| `register(self, descriptor: ModelDescriptor) -> ModelDescriptor` | method | Add ``descriptor``, refusing to shadow one already registered. |

## `ModelDescriptor`

*Pydantic model — `openforecast.models.descriptor`*

One model, as the catalog and the engine see it.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `ref` | `ModelRef` | *required* |  |
| `provider` | `str` | *required* |  |
| `display_name` | `str` | *required* |  |
| `lifecycle` | `ModelLifecycle` | *required* |  |
| `training` | `TrainingContract \| None` | `None` |  |
| `capabilities` | `ModelCapabilities` | `ModelCapabilities(instances=InstanceCapabilities(single=True, panel=False), targets=TargetCapabilities(univariate=True, multivariate=False), features=FeatureCapabilities(observed=False, known=False, static=False), outputs=OutputCapabilities(point=True, quantiles=False, samples=False), missing_values=MissingValueSupport.UNSUPPORTED)` |  |
| `parameters_schema` | `dict[str, Any]` | `{}` |  |

## `ModelLifecycle`

*Pydantic model — `openforecast.models.lifecycle`*

What has to happen to a model before it can forecast.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `requires_fit` | `bool` | *required* |  |
| `supports_fit` | `bool` | *required* |  |
| `supports_update` | `bool` | `False` |  |

## `ModelRef`

*Pydantic model — `openforecast.models.ref`*

A parsed model reference.

Frozen and hashable, so it can key a catalog: two references that print the
same are the same reference.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `namespace` | `str` | *required* |  |
| `name` | `str` | *required* |  |
| `revision` | `str \| None` | `None` |  |

## `OriginScope`

*Enumeration — `openforecast.models.contract`*

| Member | Value |
| --- | --- |
| `SINGLE` | `'single'` |
| `MULTIPLE` | `'multiple'` |

## `OutputCapabilities`

*Pydantic model — `openforecast.models.capabilities`*

What the model can produce.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `point` | `bool` | `True` |  |
| `quantiles` | `bool` | `False` |  |
| `samples` | `bool` | `False` |  |

## `TargetCapabilities`

*Pydantic model — `openforecast.models.capabilities`*

How many targets at once.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `univariate` | `bool` | `True` |  |
| `multivariate` | `bool` | `False` |  |

## `TrainingContract`

*Pydantic model — `openforecast.models.contract`*

How OpenForecast must materialize data before this model executes.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `view` | `ViewKind` | *required* |  |
| `origin_scope` | `OriginScope` | *required* |  |
| `context_required` | `bool` | `False` |  |
| `horizon_bound_at_fit` | `bool` | `False` |  |
| `supports_unseen_instances` | `bool` | `False` |  |

## `ViewKind`

*Enumeration — `openforecast.protocol.vocabulary`*

Which execution view a model consumes.

Each names a training unit rather than a model family:

```text
series      one complete time series           ARIMA, ETS, Theta
sequences   many context -> horizon sequences  NHiTS, TFT, PatchTST
tabular     individual supervised target rows  LightGBM, XGBoost, CatBoost
```

``forecast`` is the inference counterpart of all three.

| Member | Value |
| --- | --- |
| `SERIES` | `'series'` |
| `SEQUENCES` | `'sequences'` |
| `TABULAR` | `'tabular'` |
| `FORECAST` | `'forecast'` |

## `get`

*Function — `openforecast.models`*

```python
get(ref: ModelRef | str) -> ModelDescriptor
```

The descriptor named by ``ref``, from the default catalog.

## `list`

*Function — `openforecast.models`*

```python
list(*, provider: str | None = None) -> tuple[ModelDescriptor, ...]
```

Every model the default catalog can name, in reference order.

## `register`

*Function — `openforecast.models`*

```python
register(descriptor: ModelDescriptor) -> ModelDescriptor
```

Add ``descriptor`` to the default catalog.

How a provider makes itself discoverable. Kept out of the read path on
purpose: a caller listing models should not be able to change what is
listed by accident.
