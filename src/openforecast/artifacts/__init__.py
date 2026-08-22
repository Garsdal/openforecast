"""Model artifact lifecycle: immutable revisions, manifests and aliases.

```text
ModelDefinition  ->  fit  ->  ModelArtifact  ->  forecast
```

A fitted model is a resource, not a variable. It has an identity that is
generated once, a manifest that says what it is, a directory that is written
exactly once, and a name that can be pointed at it:

```python
store = ArtifactStore()

with store.stage(artifact) as staging:
    provider.fit(view, into=staging.provider_path)

handle = staging.handle           # local/de-price@01K5Z6QK3M9TQK1W2E3R4T5Y6U
store.get("local/de-price")       # the same handle, through the alias
```

Three things make that hold:

*Immutability.* A revision is never rewritten, so a forecast made from
``local/de-price@01K...`` today is the forecast it would have made a month ago.
What moves is the alias: ``local/de-price`` means the latest selected revision,
which is why a scheduled job can name a model once and pick up retrainings.

*Atomicity.* A provider trains into ``.tmp/<artifact-id>`` and the directory is
renamed into place only after the fit succeeds. A half-written artifact that is
nevertheless resolvable would be worse than no artifact at all — it would
forecast.

*Provider ignorance.* The store creates the ``provider/`` subdirectory, hands it
over and never looks inside. Everything OpenForecast needs in order to decide
whether an artifact can answer a request — the view it was trained on, its
horizon, whether its origins were real vintages — is in the manifest, so no
provider has to be started to resolve, list, alias or delete anything.
"""

from openforecast.artifacts.artifact import ModelArtifact
from openforecast.artifacts.handle import ModelHandle
from openforecast.artifacts.identity import (
    ARTIFACT_ID_LENGTH,
    artifact_time,
    is_artifact_id,
    new_artifact_id,
)
from openforecast.artifacts.manifest import (
    COMPOSITE_PROVIDER,
    LOCAL_NAMESPACE,
    MissingValueTransform,
    ModelManifest,
    TrainedSchema,
    TrainingRecord,
    content_hash,
)
from openforecast.artifacts.store import ArtifactStaging, ArtifactStore, default_root

__all__ = [
    "ARTIFACT_ID_LENGTH",
    "ArtifactStaging",
    "ArtifactStore",
    "COMPOSITE_PROVIDER",
    "LOCAL_NAMESPACE",
    "MissingValueTransform",
    "ModelArtifact",
    "ModelHandle",
    "ModelManifest",
    "TrainedSchema",
    "TrainingRecord",
    "artifact_time",
    "content_hash",
    "default_root",
    "is_artifact_id",
    "new_artifact_id",
]
