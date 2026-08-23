# Errors

*Generated from the code by `uv run generate-reference`. Do not edit by hand.*

Every failure OpenForecast raises deliberately, with the `error.code` a caller branches on instead of on the prose.

## `ArtifactError`

*Exception — `openforecast.errors`*

A fitted artifact is not what it claims to be, or cannot be written.

About the stored representation rather than about the model: a manifest whose
recipe no longer hashes to what it did, an artifact written by a protocol
version this build does not speak, a revision an alias points at that is no
longer there. An artifact is immutable, so any of these means something
outside OpenForecast changed it, and reading on would produce a forecast from
a model nobody can describe.

A reference that simply names no artifact raises
:class:`UnknownModelError` instead: the store is intact, the name is not in
it.

`error.code` is `INVALID_ARTIFACT`.

## `DataError`

*Exception — `openforecast.errors`*

Data does not satisfy the schema it was declared against.

Raised instead of repairing the data. Duplicate rows are not deduplicated,
off-grid timestamps are not snapped, missing values are not imputed.

`error.code` is `INVALID_DATA`.

## `DuplicateModelError`

*Exception — `openforecast.errors`*

Two descriptors claim the same reference.

A reference has to identify one model. Letting the second registration win
would make which model you get depend on provider load order.

`error.code` is `DUPLICATE_MODEL`.

## `FrequencyError`

*Exception — `openforecast.errors`*

A frequency cannot be parsed, or has no fixed duration.

`error.code` is `INVALID_FREQUENCY`.

## `IncompatibleForecastTask`

*Exception — `openforecast.errors`*

The artifact cannot answer the forecast that was asked of it.

A model that binds its horizon during training answers exactly that horizon;
one that binds none answers any. Truncating a 72-step forecast to the 48
steps the caller asked for would be a different question answered silently,
so the mismatch is raised instead.

`error.code` is `INCOMPATIBLE_FORECAST_TASK`.

## `InconsistentTruthError`

*Exception — `openforecast.errors`*

Vintages of the same event time disagree about what happened.

A point-in-time source table repeats its labels on every origin. If two of
those copies hold different realizations, only one can be the outcome, and
OpenForecast will not pick for you.

`error.code` is `INCONSISTENT_TRUTH`.

## `InvalidModelParameters`

*Exception — `openforecast.errors`*

A model's own parameters were rejected by the model.

The half of a recipe OpenForecast does not interpret: ``params`` is passed to
the provider as it was written, so a level the model has never heard of is
refused by the model rather than by the catalog. A :class:`RecipeError`
because the thing to fix is the recipe, and the code is its own because the
fix is a parameter rather than the shape of the recipe around it.

`error.code` is `INVALID_MODEL_PARAMETERS`.

## `ModelDoesNotSupportFit`

*Exception — `openforecast.errors`*

The reference names a model that cannot be fitted at all.

The other half of :class:`ModelRequiresFit`. A pretrained foundation model
forecasts from the reference itself — ``of.forecast(model="amazon/chronos-2",
...)`` — and ``of.fit`` on it has nothing to do: there is no training
contract behind it, so there is no view to materialize and no artifact to
publish. Accepting the call and returning something would hand back an
artifact that records a fit that never happened.

`error.code` is `MODEL_DOES_NOT_SUPPORT_FIT`.

## `ModelError`

*Exception — `openforecast.errors`*

A model could not be named or resolved.

About the identifier and the catalog, not about the data: a model whose
contract the data cannot satisfy raises a data or scope error instead.

`error.code` is `MODEL_ERROR`.

## `ModelRefError`

*Exception — `openforecast.errors`*

A model reference is not ``<namespace>/<name>[@revision]``.

Inherits from :class:`SchemaError` as well, because a malformed reference is
a declaration that is wrong before any data is looked at, and a caller
catching declaration errors should see this one too.

`error.code` is `INVALID_MODEL_REF`.

## `ModelRequiresFit`

*Exception — `openforecast.errors`*

The reference names a model that has to be fitted before it can forecast.

``of.forecast(model="nixtla/autoarima", ...)`` names a model, not a fitted
one. The alternative — quietly fitting on whatever data the forecast call was
given — would return a number that looks like a forecast from a model the
caller never trained, so the string lifecycle is explicit instead: fit it,
and forecast with the artifact reference that comes back.

`error.code` is `MODEL_REQUIRES_FIT`.

## `OpenForecastError`

*Exception — `openforecast.errors`*

Base class for every error OpenForecast raises deliberately.

Raised directly only where a failure is genuinely nothing more specific — a
config file that cannot be read, two flags that configure the same field.
Everything a caller might reasonably branch on has a subclass, and therefore
a code of its own.

``details`` are keyword arguments, so a raise site states the specifics
beside the sentence that reports them:

```python
raise ModelRequiresFit(f"{ref} has to be fitted first", model=str(ref))
```

They hold JSON-ready scalars and lists of them — a model reference as text,
a horizon as a number, the features a model cannot be given — because the
envelope they end up in crosses a process, a socket and a pipe. What they are
*not* is a second copy of the message: the message is a sentence about them.

`error.code` is `ERROR`.

## `OriginScopeError`

*Exception — `openforecast.errors`*

More forecast origins were selected than the requested view can express.

A ``SeriesView`` is one complete time series, so it has room for exactly one
origin. Asking for every historical vintage of a point-in-time dataset in
that shape has no meaning, and picking one of them silently would train a
model on data the caller never asked for.

`error.code` is `ORIGIN_SCOPE_ERROR`.

## `ProviderError`

*Exception — `openforecast.errors`*

A provider failed, or answered with something that is not a forecast.

About the execution boundary rather than about the request: a provider that
was asked for a horizon and returned half of it, or one that is named by a
descriptor but is not installed. The request was well-formed — what came
back was not.

`error.code` is `PROVIDER_EXECUTION_FAILED`.

## `ProviderNotInstalled`

*Exception — `openforecast.errors`*

A model was advertised by a provider that is not here to execute it.

Its own code because it is the one provider failure with an obvious remedy,
and the remedy is a command: ``openforecast providers install <name>``. An
agent that can read that off ``error.code`` and ``error.details['provider']``
can install what it needs and retry, which is the whole point of Step 27.4.

`error.code` is `PROVIDER_NOT_INSTALLED`.

## `RecipeError`

*Exception — `openforecast.errors`*

A recipe or a plan does not describe something executable.

A :class:`SchemaError` because a recipe is a declaration: a pipeline whose
last step forecasts nothing, an ensemble whose weights do not match its
members, a model parameter naming something OpenForecast owns. All of it is
wrong before any data is looked at.

`error.code` is `INVALID_RECIPE`.

## `SchemaError`

*Exception — `openforecast.errors`*

Declared semantics are internally inconsistent.

Raised before any data is looked at: a target that is also a feature, a
static feature carrying an availability, duplicate instance keys.

`error.code` is `INVALID_SCHEMA`.

## `UnknownModelError`

*Exception — `openforecast.errors`*

Nothing is registered under that reference.

Distinct from a malformed reference: the name is well-formed, and no
provider advertises it.

`error.code` is `MODEL_NOT_FOUND`.

## `UnsupportedDataShape`

*Exception — `openforecast.errors`*

The data is well-formed, and this model does not take that shape.

A panel handed to a model that declares a single series, or four targets
handed to a univariate one. Nothing is wrong with the data — the three
subclasses of :class:`DataError` here are the cases where the fix is to
choose a different model rather than to change the data, which is a different
recovery and therefore a different code.

`error.code` is `UNSUPPORTED_DATA_SHAPE`.

## `UnsupportedFeature`

*Exception — `openforecast.errors`*

The data carries a feature role the model declares it cannot consume.

Named separately from :class:`UnsupportedDataShape` because the two remedies
differ: a feature can be dropped from the data, where a shape cannot.

`error.code` is `UNSUPPORTED_FEATURE`.

## `UnsupportedOutput`

*Exception — `openforecast.errors`*

The model cannot produce the kind of forecast that was asked for.

A point-only model asked for quantiles. Refused from the declaration, before
a provider is started, so this is never discovered from a stack trace after a
fit has already run.

`error.code` is `UNSUPPORTED_OUTPUT`.

## `UnsupportedPlanError`

*Exception — `openforecast.errors`*

The configuration is expressible in the protocol but not yet executable.

Reserved fields exist so that the wire format does not have to change when
the capability lands. Until it does, a plan that uses one is refused loudly
rather than accepted and quietly ignored, which would look to the caller
like the search they asked for had run.

`error.code` is `UNSUPPORTED_PLAN`.
