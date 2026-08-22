# Architecture

The invariant the whole implementation is built around:

> **OpenForecast owns forecasting semantics. Providers only consume
> provider-neutral execution views. Point-in-time and ordinary event-time data
> are materialized into those views before crossing the provider boundary.**

And its immediate corollary:

> **Providers must never branch on whether source data came from a
> `TimeSeriesFrame` or a `ForecastDataset`.**

## The rules

1. **OpenForecast semantic types must never import provider libraries.**
   The core install is `pydantic`, `pyarrow`, `platformdirs`. Integrations
   depend on OpenForecast, never the reverse.
2. **Providers consume execution views, not source semantic datasets.**
   A provider receives a `SeriesView`, `SequenceView`, `TabularView` or
   `ForecastView` — never a `TimeSeriesFrame`, `PointInTimeFrame` or
   `ForecastDataset`. Its import surface is `openforecast.views`,
   `openforecast.errors`, `openforecast.protocol` and `openforecast.models` —
   the last of which is how it declares what it provides.
3. **Providers must not branch on `TimeSeriesFrame` versus `ForecastDataset`.**
   If a provider needs to know which one it came from, the view abstraction has
   failed and the fix belongs in the `ViewPlanner`, not the provider.
4. **Point-in-time vintages must never be silently replaced by newer
   information.** A value known at origin time stays the value known at origin
   time. Leakage is a correctness bug, not a convenience.
5. **Missing values must never be silently imputed.** A model declares its
   `MissingValueSupport`; OpenForecast either satisfies it with an explicit,
   recorded transform or rejects the request.
6. **Provider-specific terminology must not leak into the public OpenForecast
   protocol.** `unique_id`, `ds`, `y`, `hist_exog_list`, `futr_exog_list` and
   `stat_exog_list` are legal inside `integrations/`, nowhere else.
7. **OpenAPI is a projection of OpenForecast semantics, not their source.**
   The dependency direction is semantics → engine → HTTP → OpenAPI → remote
   SDKs, never the reverse.

## Layering

Imports flow in one direction only. A module may import its own layer and any
layer above it, never one below.

```text
              errors.py   protocol/
                        ↓
      data/  models/  recipes/  tasks/
                        ↓
                     views/
                        ↓
      runtime/  registry/  artifacts/  providers/
                        ↓
        client.py  commands/  server/
```

`protocol/` is the innermost layer and knows nothing about any provider.
`views/` sits below the semantic datasets it materializes from and above
everything that executes against them, which is what makes rule 2 mechanically
checkable rather than aspirational.

That layering has one consequence worth naming. A model's `TrainingContract` in
`models/` has to say which execution view it consumes, and `views/` is below it.
Rather than declare a second enum with the same members — two spellings of one
concept, free to drift, one of them eventually reaching the wire — `ViewKind` is
defined in `protocol/vocabulary.py` and re-exported by `openforecast.views`, so
a provider's import surface is unchanged. A test asserts it is defined exactly
once.

## Enforcement

These rules are tests, not documentation. `tests/unit/test_architecture.py`
AST-scans the package and fails on:

- any import of a forecasting framework, and any forecasting framework in the
  declared dependencies of `pyproject.toml` (rule 1);
- any runtime dependency beyond `pydantic`, `pyarrow` and `platformdirs`, and
  any import of `pandas` — a DataFrame is accepted at the edge and converted by
  `pyarrow`, never stored or depended on (rule 1);
- any import that points down the layer stack (rules 1, 2 and 7);
- provider terminology appearing in semantic protocol types (rule 6);
- any second definition of `ViewKind`, so the contract that requests a view and
  the view that satisfies it cannot drift apart, and any second definition of
  the four origin selections, for the same reason;
- any import in `integrations/` **or in `providers/`** that reaches past the view
  boundary: a provider may import `openforecast.views`, `openforecast.errors`,
  `openforecast.protocol` and `openforecast.models` — the last of which is how it
  declares what it provides, which is a descriptor and never a dataset — and may
  not name `TimeSeriesFrame`, `PointInTimeFrame`, `ForecastDataset` or
  `ForecastContext` (rules 2 and 3). `integrations/` holds no Python yet, so that
  check is also tested against a violating fixture; the built-in provider in
  `providers/` is real code held to the same rule.

CI additionally greps `uv tree --no-dev` so that a framework cannot arrive as
somebody else's transitive dependency.

One check is named here but lands with the code it constrains: the
forbidden-terminology scan over serialized public objects arrives in Step 15
(rule 6).

Rule 4 is enforced by the point-in-time semantic model: `at_origin` matches an
origin exactly rather than approximately, a vintage is filtered before anything
downstream sees it, and an observed feature carrying a value for an event time
after its own origin is rejected. The property tests generate datasets whose
feature values name the origin that produced them, so a leaked vintage is
detectable rather than plausible.

Rule 5 is enforced by the validation layers of Steps 3 and 6 and by the
conformance suite in Step 10, since it is a property of behavior rather than of
imports. In Step 6 that means imputation is only ever something a recipe asks
for: `of.Impute` is a step the caller writes and the manifest records, and a
`MissingIndicator` placed after an imputation of the same columns is refused,
because it would come out constant and discard the missingness it was added to
preserve.

Rule 6 has one deliberate exception in the source: `recipes/nodes.py` contains
the provider spellings `input_size`, `input_chunk_length`, `hist_exog_list` and
friends in a *rejection* list. They appear there so that they cannot appear
anywhere else — passing one as a provider parameter raises an error naming the
OpenForecast field to use instead. Nothing constructs them, and no public object
serializes them.

## The execution views

The three fit views are named after the training unit they hold rather than
after a model family, because that is the only thing a provider needs to know
about the data it is handed:

| View           | Training unit                        | Typical models                |
| -------------- | ------------------------------------ | ----------------------------- |
| `SeriesView`   | one complete time series             | ARIMA, ETS, Theta             |
| `SequenceView` | many context → horizon sequences     | NHiTS, TFT, PatchTST          |
| `TabularView`  | individual supervised target rows    | LightGBM, XGBoost, CatBoost   |

`ForecastView` is the inference counterpart of all three: one origin, one
horizon.

Two properties make rule 3 hold rather than merely being stated. First, both
semantic sources materialize into the *same* view types, with
`OriginFidelity` — `simulated` for windows cut out of one freshest series,
`observed` for real vintages — as the only difference. Second, the views are
keyed by opaque, deterministic identifiers (`series_id`, `sample_id`, `row_id`)
with the instance keys and origins held in a separate key table, so a provider
cannot condition on them even by accident.

## Model references and descriptors

A model is named by a string with a shape — `<namespace>/<name>[@revision]` —
and that string is a name, not a state. Whether anything has been fitted is a
question for the registry, which is what lets `nixtla/nhits` and
`local/de-price` both appear in the same argument position and mean the right
thing in each.

What the string resolves to is a `ModelDescriptor`, and the descriptor is
deliberately complete enough to plan against on its own:

| Declaration           | What the engine does with it                        |
| --------------------- | --------------------------------------------------- |
| `lifecycle`           | whether a bare reference can forecast at all         |
| `training.view`       | which execution view to materialize                  |
| `training.origin_scope` | whether several forecast origins may be learned from jointly |
| `capabilities`        | whether the materialized view is data this model accepts |
| `capabilities.missing_values` | whether an explicit transform is required, or the request refused |

No provider is started to answer any of these. That is what keeps rule 3 true in
the engine as well as in the providers: `fit()` reads a descriptor, asks the
`ViewPlanner` for the view it names, and hands that over — there is no place for
`if provider == "nixtla"` because there is nothing left for it to decide.

Contract invariants are enforced where the contract is declared rather than
where a user first trips over them. A `SeriesView` is one complete time series,
so a series model cannot claim to learn across origins, to bind a horizon at fit
time, or to generalize to an instance it never saw — it has no shared parameters
to generalize with. Those declarations are rejected at construction, which is
the same rule the user meets later as `OriginScopeError` when point-in-time data
reaches AutoARIMA with `AllOrigins()`.

## Recipes, plans and tasks

What to fit, how to fit it, and what to predict are three separate objects — a
`Recipe`, a `FitPlan` and a `ForecastTask` — so that one recipe can be fitted at
a single origin and across every origin, or asked for a different horizon,
without being rewritten.

`ViewRequest.for_contract(contract, plan=..., task=...)` is where the three
meet: the contract says which view, the plan says which origins and how much
context, the task says how far ahead. That translation is the only thing between
a model descriptor and a materialized view, which is what leaves the engine of
Step 8 with nothing to decide. A field the requested view does not bind — a
`WindowPlan` handed to a series model — is refused rather than dropped, since it
was written by someone expecting it to have an effect.

Two properties keep provider vocabulary out of what the user writes:

- **A concept OpenForecast owns is stated once.** `WindowPlan(context=168)`
  compiles to `input_size` or `input_chunk_length`; a horizon, a seed, a
  frequency and the feature roles work the same way. Passing any of them through
  `of.Model(params=...)` raises an error naming the field to use instead
  (rule 6), because two copies of one number are free to disagree and the
  provider's spelling would win silently.
- **The origin selections are source-agnostic.** `AllOrigins`, `LatestOrigin`,
  `AtOrigin` and `OriginsBetween` mean the same thing whether the origins are
  simulated from one freshest series or observed as real vintages, so the same
  plan works on both and only `OriginFidelity` differs. They live in `tasks/`,
  above `views/`, and `openforecast.views` re-exports them, so the four a user
  writes are the four the planner resolves — the same arrangement `ViewKind`
  uses, and asserted by the same kind of test.

Recipes are a serializable AST discriminated on `kind`, and `parse_recipe` reads
one back. Nothing in it is provider-specific, which is what lets the same JSON
be an artifact manifest field in Step 7, a provider request in Step 9 and an
HTTP body in Step 16.

## Fitted models

A fitted model is a resource with an identity, not a value a caller holds. Three
properties make the rest of the design work:

**Immutability.** A revision — `local/de-price@01K...` — is written once and
never rewritten, so the same pinned reference forecasts the same way forever.
What moves is the alias: `local/de-price` means the latest selected revision, so
a scheduled job names a model once and a rollback is a pointer move.

**Atomicity.** A provider trains into `.tmp/<artifact-id>` and the directory is
renamed into `models/` only after the fit succeeded. The failure mode being
avoided is not a lost artifact but a *resolvable* one: half-written provider
state that would forecast rather than fail.

**Provider ignorance in the registry.** The `provider/` subdirectory is created,
handed over and never opened. Everything needed to decide whether an artifact can
answer a request lives in the manifest, which is why resolving, listing, aliasing
and deleting need no provider process — the same reason `fit()` needs none in
order to plan.

The manifest is therefore held to one rule: **every training fact is read off the
materialized view rather than reported by whoever fitted it.** The sample count,
the context and horizon, and the `OriginFidelity` all come from the
`SeriesView`, `SequenceView` or `TabularView` that was handed over, so a manifest
cannot describe a fit that did not happen. The recipe and the training view's
schema live in their own files beside it and are hashed into it, so an artifact
edited on disk fails to load instead of forecasting as something it no longer is.

`PROTOCOL_VERSION` lives in `protocol/` rather than in the transport that will
negotiate it in Step 9, because the manifest needs it first and the two have to
be one number. An artifact written for another version is refused rather than
read optimistically: the provider directory is opaque, so guessing at a layout
that may have changed is exactly the mistake worth making impossible.

One consequence reaches the public API. `ModelRegistry` resolves a reference
against the catalog and the artifact store together, and forecasting with a
reference that names an unfitted model raises `ModelRequiresFit` rather than
fitting one on whatever data the forecast call happened to be given — a number
that looks like a forecast from a model the caller never trained is worse than an
error. A model declaring `requires_fit=False` resolves to its descriptor instead,
because zero-shot use is a declaration, not an assumption.

## The engine and the providers

`fit()` is a sequence, and its defining property is that no step of it branches
on who provides the model:

```text
normalize the recipe        a string, a Model, a Pipeline, an Ensemble
resolve every model         the registry answers what each reference means
materialize each view       the ViewPlanner, from the model's own contract
check it against the model  capabilities meeting data, before anything starts
hand it to the provider     into a staging directory, published on success
```

There is no place for `if provider == "nixtla"` because there is nothing left
for it to decide: the descriptor says which view to build and what the model can
be given, and the provider registry says who executes it. Point-in-time is
equally invisible — a `ForecastDataset` and a `TimeSeriesFrame` both go to
`ViewPlanner.fit_view`, and the only difference that survives is the
`OriginFidelity` in the manifest.

The provider interface is a structural `Protocol`, not a base class. Step 9's
provider runs in another process and another environment; what it shares with an
in-process one is the shape of three calls — `descriptors`, `fit`, `forecast` —
and not an inheritance chain it would have to import across a subprocess
boundary. Everything crossing those calls is either bulk data in an execution
view or a plain mapping (`params` as the user wrote them, `output` as it
serializes), which is exactly what becomes a JSON control message and an Arrow
bundle in Step 9.

`builtin/seasonal-naive` exists so the engine can be proved end to end before any
external library is involved. It is a real model with a real contract, and it is
held to the provider boundary as strictly as an integration will be.

**Checks happen before a provider starts.** A materialized view is validated
against the model's declared capabilities — instances, targets, feature roles,
missing values — so an unsupported request fails as a declaration mismatch
naming the model and the data, rather than as a stack trace from inside somebody
else's library. **Checks also happen after one answers.** A provider that
returns a shorter horizon, or a target it invented, has produced something that
looks exactly like a correct forecast, so the answer is matched against the
question.

**Composite recipes are OpenForecast's own execution.** A pipeline or an
ensemble is fitted leaf by leaf into one artifact — each leaf materialized,
transformed and handed to its own provider directory — and the forecasts are
combined on the way back out. Such an artifact records one `TrainingRecord` per
leaf rather than one for itself, because an ensemble's members may consume
different views and there is no single materialization it could honestly
describe. Its `provider` is `openforecast`: no library produced a weighted mean.

Transform statistics are fitted once and persisted. A scaler that recomputed its
mean from the forecast context would leak whatever that context happens to
contain into the answer, and nothing in the output would show it; the same
statistics are applied at inference and inverted out of the forecast, so what
comes back is on the scale the caller's data was on.

A forecast is one long Arrow table — instance keys, `event_time`, `target`,
`kind`, `quantile`, `sample`, `value`. A wide forecast changes shape with the
request (one column per target, or per target and quantile, or per sample path)
and so cannot be read by one reader. The column vocabulary lives in `protocol/`
for the same reason `ViewKind` does: a provider writes it and the engine reads
it, and in Step 9 they are on opposite sides of a process.
