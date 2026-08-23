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
   `openforecast.errors`, `openforecast.protocol`, `openforecast.models` and
   `openforecast.providers` — the last two being how it declares what it
   provides and how it is served. Nothing in that surface names a semantic
   source dataset, which is what makes the rule mechanically checkable.
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
   `stat_exog_list` are legal inside `integrations/`, nowhere else — and so are
   Darts' `past_covariates`, `future_covariates`, `input_chunk_length` and
   `output_chunk_length`, and sktime's `window_length`, `pooling` and
   `ForecastingHorizon`, which name the same concepts in other words. The
   scikit-learn integration adds nothing to this list, which is itself the point:
   `X` and `y` are what a `TabularView` already calls its own tables.
7. **OpenAPI is a projection of OpenForecast semantics, not their source.**
   The dependency direction is semantics → engine → HTTP → OpenAPI → remote
   SDKs, never the reverse. `spec/openapi/openapi.json` is *generated* from the
   Pydantic request and response models, committed, and diffed in CI, so it
   cannot drift from the code the generated SDKs are built against.

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
   client.py  commands/  server/  evaluation/
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
  `pyarrow`, never stored or depended on (rule 1). FastAPI is an *optional*
  dependency, `openforecast[server]`, so the declared runtime set is unchanged:
  a test starts a subprocess that imports `openforecast` and
  `openforecast.server` and asserts that no web framework was loaded, and CI has
  a job that installs neither the dev group nor the extra and builds a remote
  client in it;
- any import that points down the layer stack (rules 1, 2 and 7);
- provider terminology appearing in semantic protocol types (rule 6);
- any second definition of `ViewKind`, so the contract that requests a view and
  the view that satisfies it cannot drift apart, and any second definition of
  the four origin selections, for the same reason;
- any import in `integrations/` **or in `providers/`** that reaches past the view
  boundary: a provider may import `openforecast.views`, `openforecast.errors`,
  `openforecast.protocol`, `openforecast.models` and `openforecast.providers` —
  the last two being how it declares what it provides, which is a descriptor and
  never a dataset, and how it is served — and may not name `TimeSeriesFrame`,
  `PointInTimeFrame`, `ForecastDataset` or `ForecastContext` (rules 2 and 3).
  The built-in provider in `providers/` and the integrations under
  `integrations/*/src/` are real code held to this rule, and the check is also
  run against a violating fixture — including one that reaches for
  `openforecast.runtime` — so that it cannot pass vacuously. An integration's
  own tests are excluded, because they drive the public client from outside the
  provider; a separate check asserts that an integration keeps no other Python
  outside `src/`, so nothing escapes the scan by living beside the tests.

CI additionally greps `uv tree --no-dev` so that a framework cannot arrive as
somebody else's transitive dependency.

The forbidden-terminology scan of rule 6 is the one check that reads objects
rather than source, because what it has to constrain is what a public object
*serializes*. `openforecast.server` and `openforecast.evaluation` are in its list
for exactly that reason: an HTTP body and a backtest result are public objects, and a caller reading one should no more have to
know which library executed the model than a caller reading a manifest does. It imports every public module, walks the JSON Schema of every
exported model plus the members of every exported enum and the canonical
forecast columns, and fails if any field name, enum value or column is spelled
the way a provider spells it. Prose is skipped on purpose: a docstring saying
that `input_size` is rejected documents the rule rather than breaking it. A
second test asserts that everything the scan forbids is also refused on the way
*in* — `of.Model(params=...)` rejects a provider parameter naming something
OpenForecast owns — since those parameters travel to the provider unchanged and
are recorded in the manifest. `tests/e2e/test_v1_experience.py` then runs the
same scan over the values that actually travel: the descriptors the three
integrations advertise, the parameter schemas they publish, and the manifest a
fit writes down.

Rule 4 is enforced by the point-in-time semantic model: `at_origin` matches an
origin exactly rather than approximately, a vintage is filtered before anything
downstream sees it, and an observed feature carrying a value for an event time
after its own origin is rejected. The property tests generate datasets whose
feature values name the origin that produced them, so a leaked vintage is
detectable rather than plausible.

Rule 5 is enforced by the validation layers of Steps 3 and 6 and by the
conformance suite, since it is a property of behavior rather than of imports. In
Step 6 that means imputation is only ever something a recipe asks for:
`of.Impute` is a step the caller writes and the manifest records, and a
`MissingIndicator` placed after an imputation of the same columns is refused,
because it would come out constant and discard the missingness it was added to
preserve.

## The conformance suite

`tests/conformance/` is where rules 2 to 5 are checked as behavior. It is built
on named golden datasets whose every value is a function of the coordinates it
sits at — instance, event time, and for point-in-time data the origin that
issued it — so a leaked vintage can be identified rather than merely suspected.

```text
datasets.py   the golden semantic datasets, and the builders behind them
test_views.py both sources into all three fit views — the six materializations
test_point_in_time.py   leakage, sample counts, missingness, equivalence
test_backtest_leakage.py  the same leakage guarantee, through of.backtest
suite.py      the provider contract, generated from what a descriptor declares
```

The two leakage tests assert one property at two boundaries.
`test_point_in_time.py` poisons a vintage and materializes the earlier origin
directly, which holds the `ViewPlanner`. `test_backtest_leakage.py` runs a
backtest at that origin and searches every table of every view the provider was
handed, which holds the whole path a caller actually takes — a planner change
reaching one vintage past the origin fails there rather than surfacing as a
slightly different metric.

`suite.py` is the part integrations inherit. A descriptor states which view its
model trains on, which shapes and feature roles it accepts, which output kinds it
produces, whether it learns across origins and what it does about missing values;
the suite turns each of those statements into fits that must succeed and requests
that must be refused.
A model declaring `view=sequences` is therefore fitted from an event-time frame
and from real forecast vintages without either being written down, and in both
cases the provider is asserted to have received a `SequenceView` and nothing
else. A capability withheld is never one fewer check — it becomes a refusal that
has to happen before the provider is started.

`cases_for` and `refusals_for` take optional model parameters, which reach every
generated fit unchanged and are validated against the descriptor's own parameter
schema. They exist for models whose defaults are expensive rather than wrong —
`nixtla/nhits` runs the suite at two optimization steps, `darts/tide` at one
epoch, and `sktime/pooled-trees` and `sklearn/hist-gradient-boosting` at five
boosting iterations, which is the same statement in four libraries' own words —
and because they can only name parameters the model already advertises, they
cannot be used
to turn a capability on for the duration of the suite.

Rule 6 has one deliberate exception in the source: `recipes/nodes.py` contains
the provider spellings `input_size`, `input_chunk_length`, `hist_exog_list` and
friends in a *rejection* list. They appear there so that they cannot appear
anywhere else — passing one as a provider parameter raises an error naming the
OpenForecast field to use instead. Nothing constructs them, and no public object
serializes them.

Each integration runs the suite beside its own library, which is the arrangement
that keeps a provider's environment isolated. `tests/e2e/test_v1_experience.py`
is the opposite one and the only place the three meet: an OpenForecast install
that has never heard of any of them, reaching all three over the subprocess
protocol. It is where "provider-independent" stops being a property of each
integration in turn and becomes one of the surface — the same dataset, the same
plan and the same two calls, fitted by three libraries, with an ensemble
spanning two of them. It needs the environments installed and skips without
them, so CI installs them in a job of its own and sets `OPENFORECAST_E2E`, which
turns a skip into a failure.

## The execution views

The three fit views are named after the training unit they hold rather than
after a model family, because that is the only thing a provider needs to know
about the data it is handed:

| View           | Training unit                        | Typical models                |
| -------------- | ------------------------------------ | ----------------------------- |
| `SeriesView`   | one complete time series             | ARIMA, ETS, Theta             |
| `SequenceView` | many context → horizon sequences     | NHiTS, TFT, PatchTST          |
| `TabularView`  | individual supervised target rows    | HistGradientBoosting, LightGBM |

`ForecastView` is the inference counterpart of all three: one origin, one
horizon.

A `TabularView` is where the ownership boundary is sharpest, because a
supervised row has no time axis from which the semantics could be recovered
later. One row is one `instance × origin × lead`, and `X`, `y` and `keys` are
row-aligned: the features knowable at the origin, the outcome of the event time,
and `row_id, instance keys, origin_time, event_time, horizon_step` beside them
rather than inside `X`. Two vintages of one event time are two rows with the
same label, because their information vintages differ — a duplication that looks
like a bug to anyone who has only seen event-time tables, and is the point.

That view is why a forecasting framework's reduction API is the wrong place to
execute one. The chain `ForecastDataset → TabularView → framework → framework's
reduction → estimator` puts the forecast origin, the lead, the vintage and the
truth alignment in two places, with the framework's version winning silently.
OpenForecast already knows all four, so the chain is
`ForecastDataset → ViewPlanner → TabularView → estimator.fit(X, y)`, and
`integrations/sklearn` is the proof that nothing is missing from it: a library
that has never heard of a forecast origin executes point-in-time training
without reinterpreting one.

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

`PROTOCOL_VERSION` lives in `protocol/` rather than in the transport that
negotiates it, because the manifest needs it too and the two have to be one
number. An artifact written for another version is refused rather than
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

The provider interface is a structural `Protocol`, not a base class. A provider
may run in another process and another environment; what it shares with an
in-process one is the shape of three calls — `descriptors`, `fit`, `forecast` —
and not an inheritance chain it would have to import across a subprocess
boundary. Everything crossing those calls is either bulk data in an execution
view or a plain mapping (`params` as the user wrote them, `output` as it
serializes), which is exactly what becomes a JSON control message and an Arrow
bundle over the wire. `ProviderClient` therefore lives in `providers/` rather
than in `runtime/`: both sides of the boundary have to name it, and `runtime/` is
not on a provider's import surface, so a contract declared there could only be
duplicated.

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

## Provider isolation and the wire

An integration is never installed into the OpenForecast environment. Rule 1 says
the core depends on no forecasting framework, and two integrations can want
incompatible ones, so each gets its own `uv`-managed environment under
`~/.cache/openforecast/providers/<name>/<version>/` and is reached over a
subprocess protocol.

Three properties make that isolation cost nothing at the boundary.

**Discovery does not execute anything.** An environment is published only after
the provider inside it has answered a handshake, and what it said is written into
`environment.json`. Listing models and registering them therefore reads recorded
JSON and starts no process — the same reason `fit()` can plan without one. A
process starts when a model is actually executed, and the handshake is repeated
then: an environment whose contents no longer match its record is refused rather
than run as something it is not. For the same reason, a provider name OpenForecast
already ships cannot be installed over — a name is the namespace of the models it
advertises, so one name is one provider.

**Control and bulk data travel differently.** Requests and responses are JSON
Lines over stdin and stdout: small, ordered, one object per line. A view is an
Arrow IPC bundle in a directory the request points at, because a training set
does not belong inside a line of JSON. The bundle holds the view's *own* tables,
so reading it reconstructs the view through its ordinary constructor and every
invariant is enforced again on the far side of the process — a bundle truncated
in transit fails to load rather than training on a short window. Every message
declares its `PROTOCOL_VERSION`, and a peer speaking another one is refused at
the handshake, because the bundles either side writes are laid out by that
number.

**stdout is protocol only.** Forecasting libraries print, so the serving harness
replaces the provider's `sys.stdout` with the log stream for the duration of
every call and answers on the stream it captured at startup. A provider does not
have to be careful about it. On the engine's side, a line of stdout that is not a
response is a protocol violation rather than noise to skip, since the alternative
is a corrupted stream that eventually parses as something.

The failures that only exist once there is a boundary are named rather than
discovered: a process that dies is reported with its exit code and the tail of
its log, a request has a deadline after which the process is killed, and an error
envelope is re-raised as the error the same failure would have been in-process,
so a caller's handling does not depend on where the model ran. A failure does not
end the conversation, though — an exception becomes an error response and the
loop continues, because a provider that dies also loses the environment that took
a minute to start.

What the engine knows about all of this is nothing. `SubprocessProvider` answers
the same three calls the in-process provider does, and the test that says so is a
comparison rather than an assertion about plumbing: `builtin/seasonal-naive`
fitted and forecast over the wire produces the same Arrow table as the same model
fitted and forecast here.

The CLI exists for the one part of this that is not a forecasting operation:

```bash
openforecast providers install nixtla
openforecast providers list
openforecast providers inspect nixtla
openforecast providers remove nixtla
```

It is a projection over the same objects the Python API uses and computes nothing
of its own, and it keeps the protocol's stream contract — stdout is the answer,
`--json` for anything that parses it, stderr and a non-zero exit code for a
failure. It is built on `argparse`: a CLI framework would be a fourth runtime
dependency for a projection, and rule 1 makes that an architectural decision
rather than a convenience.

## The remote surface

Rule 7 says the dependency direction is semantics → engine → HTTP → OpenAPI →
generated SDKs. Three arrangements make that true rather than intended.

**Where a forecast runs is a client's transport, not a fact about the library.**

```python
client = of.OpenForecast(transport=of.LocalTransport())
client = of.OpenForecast(transport=of.HttpTransport("http://localhost:8321"))
```

`LocalTransport` owns an `Engine` and an artifact store; `HttpTransport` owns a
URL and knows nothing about either. The client above them turns what the caller
wrote into the request models in `server/wire.py`, hands them over, and turns
the answer back into a `Forecast` or a `ModelHandle` — with no branch anywhere
on where the model ran. The service in `server/app.py` is a router over the
*same* `Transport`, so "the same semantics remotely" is a property of the code
rather than a promise, and `tests/e2e/test_remote_transport.py` asserts it the
way Step 9's suite asserts the subprocess boundary: by comparison. Two clients,
one local and one over a real socket, are handed the same data and the same
calls, and the Arrow tables that come back have to be equal.

One thing legitimately changes shape across the boundary, and it is the thing
Step 7 already decided: **a fitted model is a resource with an identity, not a
value a caller holds.** A forecast therefore names it by reference —
`local/de-price@01K...` — and passing back the handle a fit returned means
sending that reference. The alias, the pinned revision and the handle name the
same artifact, so this is a spelling of the local API rather than a narrowing of
it.

**Control is JSON, bulk data is Arrow.** The same split the provider protocol
makes, for the same reason. A recipe, a plan, a horizon and an output spec are
small and worth having in a log, so they are Pydantic models and appear in the
OpenAPI document as themselves; a training set is not, so a dataset crosses as
the Arrow tables it already holds rather than as a hundred thousand nested JSON
objects. Today those tables are base64 in one opaque field, which is the interim
arrangement the step calls for — the honest fix is multipart or an uploaded
Arrow object the control message points at, and it can land without any control
model changing, because no row of data is described by one.

A payload is decoded through the *ordinary* constructors, so every invariant a
frame enforces is enforced again on the far side of the network. A truncated
table fails to load rather than being fitted as a shorter history, which is the
same property a view bundle has when it crosses a process.

**The document is generated, committed and diffed.**

```bash
uv run generate-openapi
git diff --exit-code spec/openapi/openapi.json
```

`document()` builds the application over a transport that raises if any route is
called, so the spec is a pure function of the route signatures and the model
schemas: no engine is consulted, no provider is started, and regenerating on a
machine with different providers installed produces the same bytes.

**The framework is optional, and only for serving.** FastAPI buys routing,
request validation against the models in `server/wire.py`, and an OpenAPI
document derived from those same models rather than written beside them — which
is rule 7 mechanically. But rule 1 makes a runtime dependency an architectural
decision, so it lives behind `openforecast[server]` and only `server/app.py` and
`server/openapi.py` import it. `HttpTransport` is `urllib`: a client that only
ever calls a remote service installs OpenForecast and nothing else, which is
what a transport abstraction is worth in the first place.

`openforecast serve` is the CLI half of it, and it binds to loopback by default.
A forecasting service has no authentication yet, so the default has to be the
one that does not publish an unauthenticated service to a network by accident.

## Probabilistic output

What kind of answer to produce is a request — `OutputSpec.point()`,
`OutputSpec.quantiles([...])`, `OutputSpec.samples(n)` — and what a model can
produce is a declaration, `OutputCapabilities`. The engine checks the one against
the other before a provider is started, which is the same rule instance counts,
target counts, feature roles and missing values already follow.

**One `Forecast`, three forms.** There is no `QuantileForecast` and no
`SampleForecast`. A predictive distribution is more *rows* of the canonical long
table — one per level, or one per draw, distinguished by `kind`, `quantile` and
`sample` — so application code downstream reads the same table whichever provider
answered and whichever form that provider is native in. A wide projection,
`to_wide()`, changes shape with the request; that is exactly why it is a
projection and not the representation.

**One conversion, one direction, asked for explicitly.**

```text
samples   ->  quantiles      the draws are the distribution; read it
quantiles ->  samples        refused: the paths would have to be invented
point     ->  anything       refused: there is no distribution to read
```

`OutputSpec.quantiles([...], from_samples=n)` is how the first is requested: the
provider is asked for `n` draws and OpenForecast reduces them, because the draw
count is part of what the quantiles are and because one estimator applied to
every provider is what makes two providers' quantiles comparable at all. It lives
in `protocol/quantiles.py` — the innermost layer — since the engine reducing a
sample forecast and a metric reading a quantile out of draws have to agree about
what "the 0.9 of these draws" is.

Nothing manufactures a distribution around a point forecast. A deterministic
model asked for quantiles is refused with a message naming what it does declare,
and a calibration layer that turns point forecasts into distributions is a thing
a caller can ask for explicitly, later, rather than something a request quietly
triggers.

**The conformance suite holds the claim.** A model declaring `quantiles` or
`samples` is asked for them by the inherited suite, and the answer has to cover
exactly the instances, event times and targets its point forecast covered, at the
levels or the draws requested — so "the same `Forecast` whoever produced it" is a
checked property of every provider rather than a promise in a docstring.

## Backtesting and point-in-time evaluation

`evaluation/` is where the abstraction stops being a way to call other people's
libraries and starts being worth something on its own. Its defining property is
negative: **there is no backtesting implementation in it.** No Nixtla
backtester, no Darts `historical_forecasts`, no sktime evaluation harness — and
not because they were reimplemented, but because there is nothing left for them
to do. Every question a backtest asks was already answered by a layer above:

```text
which origins exist        the validation strategy, over the source data
what was knowable at one    up_to / at_origin, on the data itself
what to materialize         the ViewPlanner, from each model's contract
what happened               the truth frame
```

So it lives in the outermost layer and imports `client.py`. That is the one
inward edge into the client, and it is the honest direction: backtesting is a
*user* of `of.fit` and `of.forecast`, which is why no provider — and nothing in
the engine — knows it is being backtested, and why a backtest against
`HttpTransport` runs on the service without a line of its own.

**A historical origin is an object, not an offset.** The leakage guarantee is
carried by the data rather than by the loop:

```python
TimeSeriesFrame.up_to(moment)    # simulated origins: the history, truncated
ForecastDataset.up_to(moment)    # observed origins: the vintages issued by then
```

A fold holds the *result* of one of those, so a later vintage is not merely
unused — it is absent from the object the model is handed, and there is nothing
for a bug in `evaluation/` to reach for. `up_to` on an event-time frame moves the
known features of the discarded rows into the future table, because a known
feature's later values are knowable in advance by definition; observed features
and targets do not move, and the future table refuses to carry either.

That pairing is also what makes the two validation strategies distinct rather
than two spellings of one thing: `RollingOrigin` folds a `TimeSeriesFrame` and
`ForecastOriginValidation` folds a `ForecastDataset`, and each refuses the other
source rather than inventing a vintage or an origin that never existed.

**Every row says what would make it incomparable.** `origin_fidelity`,
`provider` and `artifact` are read off the artifact the fold actually published,
never declared by the backtest, and `pairs` says how many outcomes a value was
computed over — so a fold scored on a third of its horizon is visible in the
result rather than only in the metric. `origin_fidelity` is the one that changes
conclusions, and carrying it per row is what turns "simulated availability versus
true point-in-time availability" into a comparison a caller can run rather than a
caveat they have to remember.

**The predictions are the primitive; the metrics are the summary.** A
`BacktestResult` holds both tables, and it keeps the larger one — one row per
model, fold, instance, event time and target — because the metric rows are
derivable from it and not the reverse. So `metrics_by("horizon_step")` regroups
what was already measured rather than re-running anything, and the group keys
are prediction columns, including the caller's own instance keys. A result that
kept only the means would make *does it degrade after horizon 48?* unanswerable
from the object it handed back, which is the most common question there is after
a backtest.

**A frozen artifact is evaluated, not refused.** A pinned revision names one
immutable fit, so it forecasts at every origin and `fit_seconds` is null; a
recipe or a bare reference is fitted per fold. Both are the same loop with the
fit made conditional, and which one a candidate is comes from what it *is*
rather than from a mode argument. The caveat — a frozen artifact was fitted on
data that may postdate the early origins, so its numbers are optimistic beside a
per-fold fit — is reported rather than enforced, for the same reason
`origin_fidelity` is: it is the caller's judgement, and refusing the run would
remove the one way to ask whether the model in production has drifted. What is
still refused is a pinned revision *inside* a recipe that is fitted per fold:
there is no way to fit a step that is already fitted.

**A metric is given a distribution, not a pair of numbers.** The rows about one
outcome are gathered back into one `Prediction` — a point, the levels the model
answered, or its draws — before anything is computed, which is what lets one
metric list score any provider: `PinballLoss(0.9)` reads the 0.9 of native
quantiles and of sample draws identically, and `MAE()` scores the median of
either. What a metric cannot read it does not score, and `pairs` reports how many
outcomes it did — a null value beside a zero count, never a zero score. Whether a
metric can score the requested output at all is answered from the *request*, so
`of.backtest` refuses a coverage of a point forecast before the first fit rather
than after the last one. `Coverage` is best at its nominal level rather than
highest and `IntervalWidth` is only readable beside it, which is why the first
ranks by distance and the second exists as its own metric instead of being folded
into a score.

One knob is deliberately a template rather than a literal. A backtest's `plan=`
has to reach candidates that do not share a contract, and a `WindowPlan` is a
field only a sequence model binds — `of.fit` refuses one handed to ARIMA, and
correctly, since somebody wrote it expecting an effect. So `plan_for` drops the
window for a candidate that binds none and nothing else is adapted, which is both
documented and the only way one plan can compare model families at all. A
candidate that needs something else states it with `of.Candidate(model, plan=...)`.

`of.eligible_models` is the screening half of `openforecast/auto`, and
eligibility means exactly one thing: **the fit would not be refused.** It
materializes the view the model's contract asks for and checks it against the
capabilities the model declared — the same two functions `fit()` runs — so
"AutoARIMA cannot learn across vintages" and "this model cannot consume missing
values" fall out as the sentences the fit would have failed with rather than
being written down a second time as heuristics. `openforecast/auto` itself is not
registered: a descriptor for it would have to name a view and a horizon before
the data has been seen, and the honest version is a policy over these pieces —
backtest, rank, fit the winner — rather than a model reference standing in front
of nothing.
