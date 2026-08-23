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

`error.code` is `ARTIFACT_ERROR`.

## `DataError`

*Exception — `openforecast.errors`*

Data does not satisfy the schema it was declared against.

Raised instead of repairing the data. Duplicate rows are not deduplicated,
off-grid timestamps are not snapped, missing values are not imputed.

`error.code` is `DATA_ERROR`.

## `DuplicateModelError`

*Exception — `openforecast.errors`*

Two descriptors claim the same reference.

A reference has to identify one model. Letting the second registration win
would make which model you get depend on provider load order.

`error.code` is `DUPLICATE_MODEL_ERROR`.

## `FrequencyError`

*Exception — `openforecast.errors`*

A frequency cannot be parsed, or has no fixed duration.

`error.code` is `FREQUENCY_ERROR`.

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

`error.code` is `INCONSISTENT_TRUTH_ERROR`.

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

`error.code` is `MODEL_REF_ERROR`.

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

Every one of them carries a :attr:`code`: the class name in
``SCREAMING_SNAKE_CASE``, so ``ModelDoesNotSupportFit`` is
``MODEL_DOES_NOT_SUPPORT_FIT``. Derived rather than declared per class,
because a code that has to be written down beside the name is a code that
can disagree with it — and a caller branching on ``error.code`` should be
reading the same fact ``except`` reads.

The code is what a caller *acts* on and the message is what a person reads.
Step 27 is where the rest of the envelope lands — a structured ``details``
mapping, the CLI's ``--json`` failures, the HTTP body — and it hangs off
this property rather than replacing it.

`error.code` is `OPEN_FORECAST_ERROR`.

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

`error.code` is `PROVIDER_ERROR`.

## `RecipeError`

*Exception — `openforecast.errors`*

A recipe or a plan does not describe something executable.

A :class:`SchemaError` because a recipe is a declaration: a pipeline whose
last step forecasts nothing, an ensemble whose weights do not match its
members, a model parameter naming something OpenForecast owns. All of it is
wrong before any data is looked at.

`error.code` is `RECIPE_ERROR`.

## `SchemaError`

*Exception — `openforecast.errors`*

Declared semantics are internally inconsistent.

Raised before any data is looked at: a target that is also a feature, a
static feature carrying an availability, duplicate instance keys.

`error.code` is `SCHEMA_ERROR`.

## `UnknownModelError`

*Exception — `openforecast.errors`*

Nothing is registered under that reference.

Distinct from a malformed reference: the name is well-formed, and no
provider advertises it.

`error.code` is `UNKNOWN_MODEL_ERROR`.

## `UnsupportedPlanError`

*Exception — `openforecast.errors`*

The configuration is expressible in the protocol but not yet executable.

Reserved fields exist so that the wire format does not have to change when
the capability lands. Until it does, a plan that uses one is refused loudly
rather than accepted and quietly ignored, which would look to the caller
like the search they asked for had run.

`error.code` is `UNSUPPORTED_PLAN_ERROR`.
