# Providers

Nixtla wants one version of `torch`, Darts wants another, sktime wants
scikit-learn and statsmodels, the Chronos integration wants `torch` and
`transformers`, and OpenForecast wants none of it. So an integration is not
installed into the OpenForecast environment at all: it gets its own, built with
`uv`, and it is reached over a subprocess protocol.

```text
~/.cache/openforecast/providers/
    nixtla/
        0.1.0/
            environment.json     what the provider said when it was installed
            .venv/
```

An environment is published only once the provider inside it has answered a
handshake, and what it said is written down. That is what makes discovery cheap:
`of.models.list()` reads recorded JSON and starts no process. A process starts
when a model is actually fitted or forecast with — and the handshake is repeated
then, so an environment whose contents changed underneath its record is refused
rather than executed as something it no longer is.

## Two channels, and the split is the point

```text
control    JSON Lines over stdin/stdout — small, ordered, greppable
bulk       Arrow IPC bundles in a directory the message points at
```

```json
{"protocol_version": 1, "operation": "fit", "model": "nixtla/nhits",
 "view": {"kind": "sequences", "path": "/tmp/openforecast-nixtla-x/view"},
 "into": "/…/.tmp/01K5Z…/provider"}
```

A hundred thousand training rows do not belong in nested JSON, and a control
message that is one line of JSON can be logged, diffed and read by a person. So a
view bundle is the same tables the in-process provider is handed:

```text
sequences/                      tabular/
    schema.json                     schema.json
    provenance.json                 provenance.json
    temporal.arrow                  x.arrow
    samples.arrow                   y.arrow
    static.arrow                    keys.arrow
```

Reading one reconstructs a real view, so every invariant the view enforces is
enforced again on the far side of the process.

**stdout carries protocol and nothing else.** Forecasting libraries print, so the
serving harness redirects the provider's stdout to stderr for the duration of
every call and writes responses to the stream it captured at startup — a provider
does not have to be careful, it has to be correct. On the engine's side, a line of
stdout that is not a response is a protocol violation rather than noise to skip.

## Failures that only exist once there is a boundary

Named rather than discovered: a process that dies is reported with its exit code
and the tail of its log, a request that is never answered has a deadline and the
process is killed, a provider speaking another protocol version is refused, and an
error envelope is re-raised as the error the same failure would have been
in-process — so a caller's handling does not depend on where the model ran.

The engine, meanwhile, learns none of this. A `SubprocessProvider` answers the
same three calls the in-process one does, and `builtin/seasonal-naive` fitted over
the wire produces the same forecast as `builtin/seasonal-naive` fitted here —
which is the test that says the abstraction holds.

## Writing an integration

The harness plus a provider object:

<!-- docs-exec: skip — names a package that lives in its own environment -->

```python
from openforecast.providers import serve
from openforecast_nixtla.provider import NixtlaProvider

raise SystemExit(serve(NixtlaProvider()))
```

A provider's whole import surface is `openforecast.views`,
`openforecast.errors`, `openforecast.protocol`, `openforecast.models` and
`openforecast.providers` — the last two being how it declares what it provides and
how it is served. Nothing in that surface names a semantic source dataset, which
is what makes the rule mechanically checkable rather than aspirational; a test
scans every integration's source and fails on a violation.

What an integration writes is:

1. **descriptors** — for each model, which view it trains on, at how many origins,
   with which feature roles and output kinds, and what it does about missing
   values
2. **a parameter surface** — the provider's own knobs, declared, so that a
   parameter naming something OpenForecast owns can be refused with the field to
   use instead
3. **conversion** — view tables into whatever the library wants, and its answer
   back into a `Forecast`

Provider conformance is then *generated* rather than written: the suite turns each
statement in a descriptor into fits that must succeed and requests that must be
refused. Declaring `view=sequences` buys tests against an event-time frame and
against real forecast vintages, with the provider asserted to have received a
`SequenceView` in both.

`builtin/seasonal-naive` is the reference provider — a real local model with a real
contract, held to the same import boundary an external integration is, so the
engine can be proved end to end without a forecasting library installed.
