# Transports

*Generated from the code by `uv run generate-reference`. Do not edit by hand.*

Where a client executes, which is the only thing a transport decides.

## `HttpTransport`

*Class — `openforecast.server.transport`*

```python
HttpTransport(base_url: str, *, timeout: float = 600.0) -> None
```

Executes wherever ``base_url`` is serving, over ``/v1``.

Holds no engine, no catalog and no store: what a model means, whether it
needs fitting first and where its artifact lives are all decided by the
service. This side encodes the request, decodes the answer, and turns an
error envelope back into the exception it describes.

| Member | Kind | Summary |
| --- | --- | --- |
| `artifact(self, ref: str) -> ModelHandle` | method |  |
| `base_url` | property |  |
| `fit(self, body: FitBody) -> ModelHandle` | method |  |
| `forecast(self, body: ForecastBody) -> ForecastPayload` | method |  |
| `model(self, ref: str) -> ModelDescriptor` | method |  |
| `models(self) -> tuple[ModelDescriptor, ...]` | method |  |

## `LocalTransport`

*Class — `openforecast.server.transport`*

```python
LocalTransport(*, store: str | Path | ArtifactStore | None = None, catalog: ModelCatalog | None = None, providers: ProviderRegistry | None = None, engine: Engine | None = None) -> None
```

Executes in this process, against a store and a set of providers.

The default, and what ``of.fit`` has always done. It is written as a
transport so that the HTTP one has something to be the same as: the service
of :mod:`openforecast.server.app` is this class behind a router, which is
what makes "the same user-facing semantics" a fact about the code rather
than an intention.

| Member | Kind | Summary |
| --- | --- | --- |
| `artifact(self, ref: str) -> ModelHandle` | method |  |
| `engine` | property |  |
| `fit(self, body: FitBody) -> ModelHandle` | method |  |
| `forecast(self, body: ForecastBody) -> ForecastPayload` | method |  |
| `model(self, ref: str) -> ModelDescriptor` | method |  |
| `models(self) -> tuple[ModelDescriptor, ...]` | method |  |

## `Transport`

*Class — `openforecast.server.transport`*

```python
Transport(*args, **kwargs)
```

The five calls a forecasting service answers.

A structural protocol rather than a base class, for the same reason
:class:`~openforecast.runtime.provider.ProviderClient` is one: the two
implementations share the shape of these calls and nothing else, and one of
them is on the far side of a network.

| Member | Kind | Summary |
| --- | --- | --- |
| `artifact(self, ref: str) -> ModelHandle` | method | A fitted artifact, described without loading the model behind it. |
| `fit(self, body: FitBody) -> ModelHandle` | method | Fit, and return the artifact that was published. |
| `forecast(self, body: ForecastBody) -> ForecastPayload` | method | Forecast with a fitted reference, at one origin. |
| `model(self, ref: str) -> ModelDescriptor` | method | What one reference resolves to, without executing anything. |
| `models(self) -> tuple[ModelDescriptor, ...]` | method | Every model that can be fitted here. |
