# Point-in-time semantics

The rule the whole design is arranged around:

> A value known at origin time stays the value known at origin time. Leakage is a
> correctness bug, not a convenience.

## What a vintage is

A forecast issued at 08:00 for 12:00 is a different fact from one issued at 09:00
for 12:00. Both existed; both are what someone had to act on at the time. So both
are rows:

```text
zone origin_time event_time wind_fc load_fc
DE   08:00       12:00      10.1    54.2
DE   09:00       12:00      11.7    54.8
DE   10:00       12:00      12.4    55.1
```

Nothing deduplicates on event time, forward-fills across origins, or replaces an
early vintage with a later revision. Lead time is derived on request rather than
stored, because a third axis determined by the other two is a third axis free to
disagree with them.

## Simulated versus observed origins

There are two ways to get a training sample that represents "a forecast made at
time *t*":

| Fidelity | Where the sample came from |
| --- | --- |
| `SIMULATED` | a window cut out of one freshest series, pretending it was *t* |
| `OBSERVED` | the vintage that actually existed at *t* |

Both are legitimate and they are not the same. A model trained on simulated
origins was told the past was cleaner than it was — every feature at its final,
revised value — and a model trained on observed vintages saw the noise it will
actually face. So the artifact records which one it was, read off the materialized
view rather than declared by the caller, and a backtest reports it as a column:

```text
model  fold  origin  metric  value  ...  origin_fidelity  provider  artifact
```

which turns "simulated historical availability versus true point-in-time
availability" into a comparison you can run rather than a caveat you have to
remember.

A pretrained model reports a third value, `pretrained`, because there were no
training origins at all — which is a different thing from a frozen artifact that
may have seen data postdating the early origins.

## Where leakage is prevented

Not in a check, and not in every integration. In the *objects*:

- `dataset.at_origin(t)` yields a `ForecastContext` in which only that vintage
  exists. A feature revised at 12:00 is not filtered out of the 11:00 origin's
  context; it is absent from it.
- `dataset.up_to(t)` yields the vintages issued by then, and
  `frame.up_to(t)` the history known by then. A backtest fold holds the result of
  one of those, so there is nothing for a bug in the backtest loop to reach for.
- An observed feature holding a value for an event time *after* its own origin is
  rejected at construction: that combination cannot be anything but a mistake.
- A `SequenceView` sample is exactly one origin — a context window ending at the
  origin, a forecast window after it — and the view validates that rather than
  trusting it, so no integration can accidentally learn across two origins.

Samples are keyed by an opaque, deterministic `sample_id`, with instance keys and
origins in a separate table: a provider cannot condition on what it cannot see.

## What is never done to make data fit

A window the data does not fully cover is dropped rather than padded. A value the
source did not have stays missing rather than being imputed. Two vintages of the
same event time become two supervised rows, and their shared outcome is repeated
rather than reconciled — four distinct forecasting examples, because their
information vintages differ:

```text
X                     y
wind_fc  load_fc      price
NaN      54           80     <- 08:00 forecasting 12:00, wind not published yet
NaN      53           76     <- 08:00 forecasting 13:00
11       55           80     <- 09:00 forecasting 12:00, now it is
12       54           76     <- 09:00 forecasting 13:00
```

The conformance suite asserts these as behavior rather than documenting them as
intent: named golden datasets are materialized into all three fit views from both
semantic sources, and leakage, sample count, missingness and event-time
equivalence are checked on the result.
