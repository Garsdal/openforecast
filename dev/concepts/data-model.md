# The data model

## Why two representations and not one

`TimeSeriesFrame` holds `instance × event_time × variable`: what happened.
`PointInTimeFrame` holds `instance × origin_time × event_time × variable`: what
was knowable, vintage by vintage. Forecast vintages are deliberately *not*
expressible on the first one — not as an optional `origin_time` column, not as a
flag.

A representation that can hold either quietly holds neither. Every consumer would
have to ask which mode it was in, and the ones that forgot would be the leakage
bugs. Keeping them apart means the object a provider is eventually handed cannot
be ambiguous about what it knew and when, and it means the code that *does* care —
the `ViewPlanner` — is one file rather than a habit.

`ForecastDataset` is the pair a vintage only means something as:

| Part | Type | Holds |
| --- | --- | --- |
| `information` | `PointInTimeFrame` | every vintage, exactly as it was issued |
| `truth` | `TimeSeriesFrame` | the realized outcome, once per event time |

and `ForecastContext` is exactly one origin of it: the shape production inference
always has.

## Roles, not column lists

A column is declared with two orthogonal axes — `kind` (temporal or static) and
`availability` (observed only up to the origin, or known into the future) — and the
categories people talk about are *derived* from those. That is why there is no
`PANEL_MULTIVARIATE` enum member anywhere: "panel" is more than one instance,
"multivariate" is more than one target, and both are questions the schema can
answer without a name being minted for the combination.

It is also what makes `up_to` correct without a special case. Truncating history
keeps the known features of the truncated rows, because a known feature's later
values are knowable in advance — that is what the role *means*, so the operation
follows from the declaration rather than from a rule beside it.

## Arrow all the way down

Three Arrow tables — history, future, static — against one schema. Arrow is the
canonical data-plane representation, so anything crossing a process or language
boundary is Arrow IPC rather than JSON: a view bundle handed to a provider
subprocess, a dataset in an HTTP body, a forecast coming back. A truncated table
fails to load instead of being fitted as a shorter history.

pandas is an edge format. `from_pandas` converts through `pyarrow` and stores
Arrow from then on, and OpenForecast itself never imports pandas — asserted by a
test rather than intended.

## Validate, never repair

Construction refuses each of these, because each silently changes what the data
means:

- duplicate instance/time rows
- timestamps off the declared frequency grid
- targets or observed features in the future table
- static features that vary within an instance
- truth labels that disagree between vintages of the same event time

Gaps and missing values are preserved as they are. A missing observation is
information: in point-in-time data it usually means the feature had not been
published at that origin, and imputing it would be inventing the past. A model
that cannot consume one says so in its capabilities, and the caller writes the
transform down as a recipe step where the artifact records it.

The general rule is that OpenForecast refuses ambiguity rather than resolving it.
`InconsistentTruthError` is the sharpest example: two vintages carrying different
outcomes for the same event time is a contradiction in the source data, and
picking one would produce a number nobody could reproduce.
