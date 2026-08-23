# Execution views

A provider never sees a semantic dataset. It is handed an **execution view**,
named after the training unit it holds rather than after a model family.

| View | Training unit | Typical models |
| --- | --- | --- |
| `SeriesView` | one complete time series | ARIMA, ETS, Theta |
| `SequenceView` | many context → horizon sequences | NHiTS, TFT, PatchTST |
| `TabularView` | individual supervised target rows | HistGradientBoosting, LightGBM, XGBoost |

`ForecastView` is the inference counterpart of all three: one origin, one horizon.

Naming them after training units rather than after libraries is what lets a
library be replaced without the vocabulary changing. "Sequence model" is a claim
about what a sample is; "Nixtla model" is a claim about who wrote the code.

## The planner is the only thing that knows where data came from

```python
from openforecast.views import ViewKind, ViewPlanner, ViewRequest

planner = ViewPlanner()
request = ViewRequest(kind=ViewKind.SEQUENCES, context=168, horizon=72)
```

<!-- docs-exec: skip — illustrative: `timeseries` and `forecast_dataset` stand for your own data -->

```python
from_event_time = planner.fit_view(timeseries, request)
from_vintages = planner.fit_view(forecast_dataset, request)
```

Both calls return a `SequenceView` with the same schema and the same sample
layout. What differs is provenance: windows cut out of one freshest series record
`OriginFidelity.SIMULATED`, and windows built from real vintages record
`OriginFidelity.OBSERVED`.

That is the whole reason the boundary exists. Point-in-time handling lives in one
file instead of being re-derived in every integration, and a provider physically
cannot branch on the source, because the source is not in the object it receives.
A test scans every provider's imports and fails on any mention of
`TimeSeriesFrame`, `PointInTimeFrame` or `ForecastDataset`.

## Supervised rows carry the most semantics

A `TabularView` holds no time axis to recover meaning from, so the meaning has to
be in the layout. One row is one `instance × origin × lead`, in three row-aligned
tables:

```text
X      the features knowable at the origin
y      what that event time turned out to be
keys   row_id, instance keys, origin_time, event_time, horizon_step
```

The keys are deliberately *not* in `X`. That is what stops an estimator from being
handed a timestamp or a zone as a feature by accident — and it is what makes a
fitted artifact able to forecast an instance it never saw.

## What a view guarantees

- **One sample is one origin.** A context window ending at the origin, a forecast
  window after it, validated rather than trusted.
- **Nothing is invented.** A window the data does not fully cover is dropped
  rather than padded; a missing value stays missing.
- **Identity is opaque.** Samples are keyed by a deterministic `sample_id`, with
  instance keys and origins in a separate `samples` table.
- **Inference names event times.** A `ForecastView`'s `future` table names exactly
  the event times being asked about, so a provider never derives them from a
  horizon count and a frequency.
- **The same invariants hold across a process boundary.** A view bundle is the
  same tables the in-process provider is handed, so reading one reconstructs a
  real view and re-checks every invariant on the far side. A bundle truncated in
  transit fails to load rather than training on a short window.

## Contracts are checked where they are declared

A model's `TrainingContract` says which view it consumes, whether several origins
may be learned from jointly, whether a horizon is bound at fit time, and whether
it can forecast an instance it never saw. Those interact, so they are checked
together: a `SeriesView` is one complete time series, so a series model cannot
claim to learn across origins or to generalize to unseen instances — it was fitted
per series and has nothing to generalize with.

Capability defaults are the conservative ones throughout. A descriptor that
declares nothing describes a single-series, univariate, point-forecast model that
cannot see a missing value. A capability is something a provider states, never
something it is assumed to have.
