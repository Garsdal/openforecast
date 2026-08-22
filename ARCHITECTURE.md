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
   A provider receives a `SeriesView`, `WindowView`, `TabularView` or
   `ForecastView` — never a `TimeSeriesFrame`, `PointInTimeFrame` or
   `ForecastDataset`.
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
      runtime/  registry/  artifacts/
                        ↓
        client.py  commands/  server/
```

`protocol/` is the innermost layer and knows nothing about any provider.
`views/` sits below the semantic datasets it materializes from and above
everything that executes against them, which is what makes rule 2 mechanically
checkable rather than aspirational.

## Enforcement

These rules are tests, not documentation. `tests/unit/test_architecture.py`
AST-scans the package and fails on:

- any import of a forecasting framework, and any forecasting framework in the
  declared dependencies of `pyproject.toml` (rule 1);
- any runtime dependency beyond `pydantic`, `pyarrow` and `platformdirs`, and
  any import of `pandas` — a DataFrame is accepted at the edge and converted by
  `pyarrow`, never stored or depended on (rule 1);
- any import that points down the layer stack (rules 1, 2 and 7);
- provider terminology appearing in semantic protocol types (rule 6).

CI additionally greps `uv tree --no-dev` so that a framework cannot arrive as
somebody else's transitive dependency.

Two checks are named here but land with the code they constrain:

- the provider boundary test — integrations may import the four view types and
  must not import `ForecastDataset` or `PointInTimeFrame` — arrives with the
  views package in Step 4 (rules 2 and 3);
- the forbidden-terminology scan over serialized public objects arrives in
  Step 15 (rule 6).

Rule 4 is enforced by the point-in-time semantic model: `at_origin` matches an
origin exactly rather than approximately, a vintage is filtered before anything
downstream sees it, and an observed feature carrying a value for an event time
after its own origin is rejected. The property tests generate datasets whose
feature values name the origin that produced them, so a leaked vintage is
detectable rather than plausible.

Rule 5 is enforced by the validation layers of Steps 3 and 6 and by the
conformance suite in Step 10, since it is a property of behavior rather than of
imports.
