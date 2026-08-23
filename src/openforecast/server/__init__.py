"""The HTTP projection of OpenForecast semantics.

```python
client = of.OpenForecast(transport=of.LocalTransport())
client = of.OpenForecast(transport=of.HttpTransport("http://localhost:8321"))
```

```bash
openforecast serve
```

Two halves, and only one of them needs a web framework.

The half exported here is the *semantics* of the remote surface: the request and
response models of :mod:`openforecast.server.wire` and the transports of
:mod:`openforecast.server.transport`. It is plain Pydantic and :mod:`urllib`, so
a client that only ever calls a remote service installs OpenForecast and nothing
else — which is what a transport abstraction is worth in the first place.

The other half is the service. :mod:`openforecast.server.app` builds a FastAPI
application over a transport and :mod:`openforecast.server.openapi` generates
``spec/openapi/openapi.json`` from it; both need the ``openforecast[server]``
extra and neither is imported here, so ``import openforecast`` stays free of a
web framework. They are imported by path:

```python
from openforecast.server.app import create_app
```

OpenAPI is generated *from* these models; it is never their source.
"""

from openforecast.server.transport import (
    DEFAULT_PORT,
    HttpTransport,
    LocalTransport,
    Transport,
    status_for,
)
from openforecast.server.wire import (
    DataKind,
    DataPayload,
    ErrorBody,
    ErrorInfo,
    FitBody,
    ForecastBody,
    ForecastContextPayload,
    ForecastDatasetPayload,
    ForecastPayload,
    ModelListing,
    PointInTimePayload,
    TimeSeriesPayload,
    decode_data,
    encode_data,
)

__all__ = [
    "DEFAULT_PORT",
    "DataKind",
    "DataPayload",
    "ErrorBody",
    "ErrorInfo",
    "FitBody",
    "ForecastBody",
    "ForecastContextPayload",
    "ForecastDatasetPayload",
    "ForecastPayload",
    "HttpTransport",
    "LocalTransport",
    "ModelListing",
    "PointInTimePayload",
    "TimeSeriesPayload",
    "Transport",
    "decode_data",
    "encode_data",
    "status_for",
]
