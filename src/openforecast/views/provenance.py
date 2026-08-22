"""Where a view's forecast origins came from.

Two views can hold numerically identical training samples and still mean
different things. Windows cut out of one freshest historical series describe
availability that was *reconstructed*; windows built from real vintages
describe availability that was *observed*. A model trained on the first is
being told the past was cleaner than it was, and that difference has to survive
into the artifact manifest, so it travels with the view rather than being
inferred later.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

__all__ = ["MATERIALIZER_VERSION", "OriginFidelity", "SourceKind", "ViewProvenance"]

#: Bumped whenever the same source and plan would materialize differently.
MATERIALIZER_VERSION = 1


class OriginFidelity(StrEnum):
    #: Origins were cut from an ordinary event-time series, so the feature
    #: values at each origin are today's values, not the ones then available.
    SIMULATED = "simulated"
    #: Origins are real forecast vintages: every value is the one that existed.
    OBSERVED = "observed"


class SourceKind(StrEnum):
    """Which semantic dataset a view was materialized from.

    Recorded for provenance only. A provider never sees it — that is the whole
    point of the view boundary — but an artifact has to be able to say what it
    was trained on.
    """

    TIME_SERIES = "time_series"
    FORECAST_DATASET = "forecast_dataset"
    FORECAST_CONTEXT = "forecast_context"


class ViewProvenance(BaseModel):
    """How a view was made, in terms a manifest can record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: SourceKind
    origin_fidelity: OriginFidelity
    materializer_version: int = MATERIALIZER_VERSION

    @property
    def is_observed(self) -> bool:
        return self.origin_fidelity is OriginFidelity.OBSERVED
