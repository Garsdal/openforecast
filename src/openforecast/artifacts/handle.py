"""``ModelHandle``: a fitted model you can talk about without loading it.

```python
handle = store.get("local/de-price")

handle.ref                    # local/de-price@01K...
handle.manifest.training.view # ViewKind.SEQUENCES
handle.serves_horizon(72)     # True
```

A handle is a reference and a manifest, and deliberately nothing more. Loading a
native model is expensive — a Lightning checkpoint, a fitted StatsForecast
object — and almost every question asked of a fitted model is a question about
its manifest: which view it consumes, what horizon it was bound to, whether the
data at hand declares the features it was trained with. Listing ten artifacts
should not deserialize ten neural networks, and only the provider knows how to
deserialize one at all.

Reading the recipe and the training schema costs a file each, so those are
loaded on request rather than held.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from openforecast.artifacts.manifest import ModelManifest, TrainedSchema, TrainingRecord
from openforecast.models.ref import ModelRef

__all__ = ["ModelHandle"]


class ModelHandle(BaseModel):
    """One published artifact, as everything outside the provider sees it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest: ModelManifest
    #: Where the artifact lives. The provider subdirectory under it is opaque to
    #: everything in OpenForecast except the provider that wrote it.
    path: Path

    @property
    def ref(self) -> ModelRef:
        """``local/de-price@01K...`` — pinned, because a handle is one revision."""
        return self.manifest.ref

    @property
    def artifact_id(self) -> str:
        return self.manifest.artifact_id

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def provider(self) -> str:
        return self.manifest.provider

    @property
    def training(self) -> TrainingRecord:
        return self.manifest.training

    @property
    def data_schema(self) -> TrainedSchema:
        """What the data handed to a forecast has to declare."""
        return self.manifest.data_schema

    @property
    def provider_path(self) -> Path:
        """The provider's own directory, handed back to it unopened."""
        return self.path / "provider"

    def serves_horizon(self, horizon: int) -> bool:
        return self.manifest.training.serves_horizon(horizon)

    def __str__(self) -> str:
        return str(self.ref)

    def __repr__(self) -> str:
        training = self.manifest.training
        return (
            f"ModelHandle({self.ref}, source={self.manifest.source_model}, "
            f"view={training.view}, origin_fidelity={training.origin_fidelity}, "
            f"samples={training.samples})"
        )
