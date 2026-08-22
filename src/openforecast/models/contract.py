"""``TrainingContract``: which execution view a model trains on, and how.

This is the field the engine reads to decide what to materialize. Everything the
``ViewPlanner`` needs to know about the model is here, so that ``fit()`` can be
written once:

```python
view = planner.fit_view(data, contract=descriptor.training, plan=..., task=...)
```

The contract is about the *training unit*, not about the model family. Two
models that both learn from context -> horizon sequences declare the same view
however differently they are implemented, which is exactly what makes a
provider swappable.

The invariants of the views are enforced here rather than discovered at
materialization time. A ``SeriesView`` is one complete time series, so a series
model cannot learn across origins, cannot bind a horizon at fit time, and cannot
generalize to an instance it never saw — it has no shared parameters to
generalize with. A contract that claims otherwise is rejected when it is
declared, not when a user first hits it.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from openforecast.errors import SchemaError
from openforecast.protocol.vocabulary import ViewKind

__all__ = ["OriginScope", "TrainingContract"]


class OriginScope(StrEnum):
    #: Learns from one forecast origin. Point-in-time data must be narrowed to
    #: a single vintage before it reaches such a model.
    SINGLE = "single"
    #: Learns jointly across many origins, which is what makes real historical
    #: vintages usable as training samples.
    MULTIPLE = "multiple"


class TrainingContract(BaseModel):
    """How OpenForecast must materialize data before this model executes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    view: ViewKind
    origin_scope: OriginScope

    #: The caller has to state a context length; there is no defensible default.
    context_required: bool = False
    #: The native model fixes its horizon during training, so a forecast at a
    #: different horizon is a different model rather than a different call.
    horizon_bound_at_fit: bool = False
    #: A fitted artifact can forecast an instance that was not in its training data.
    supports_unseen_instances: bool = False

    @model_validator(mode="after")
    def _check_view_invariants(self) -> Self:
        if self.view is ViewKind.FORECAST:
            raise SchemaError(
                "a training contract names the view a model learns from; 'forecast' is "
                "the inference counterpart of the three training views, not one of them"
            )
        if self.view is ViewKind.SERIES:
            self._check_series_invariants()
        if self.view is not ViewKind.SEQUENCES and self.context_required:
            raise SchemaError(
                f"a {self.view} view sizes no context window, so a {self.view} model "
                f"cannot require one; lagged features are declared on the recipe"
            )
        return self

    def _check_series_invariants(self) -> None:
        if self.origin_scope is not OriginScope.SINGLE:
            raise SchemaError(
                "a series view is one complete time series and therefore holds one "
                "forecast origin; a model learning across origins consumes sequences "
                "or tabular rows"
            )
        if self.horizon_bound_at_fit:
            raise SchemaError(
                "a series view carries no horizon, so a series model cannot bind one "
                "at fit time; it is asked for a horizon at inference instead"
            )
        if self.supports_unseen_instances:
            raise SchemaError(
                "a series model is fitted per series, so an unseen instance has no "
                "fitted model to forecast it with"
            )

    @classmethod
    def series(cls) -> TrainingContract:
        """One complete time series per training unit — ARIMA, ETS, Theta."""
        return cls(view=ViewKind.SERIES, origin_scope=OriginScope.SINGLE)

    @classmethod
    def sequences(
        cls,
        *,
        origin_scope: OriginScope = OriginScope.MULTIPLE,
        horizon_bound_at_fit: bool = True,
        supports_unseen_instances: bool = False,
    ) -> TrainingContract:
        """Many context -> horizon sequences — NHiTS, TFT, PatchTST."""
        return cls(
            view=ViewKind.SEQUENCES,
            origin_scope=origin_scope,
            context_required=True,
            horizon_bound_at_fit=horizon_bound_at_fit,
            supports_unseen_instances=supports_unseen_instances,
        )

    @classmethod
    def tabular(
        cls,
        *,
        origin_scope: OriginScope = OriginScope.MULTIPLE,
        horizon_bound_at_fit: bool = False,
        supports_unseen_instances: bool = True,
    ) -> TrainingContract:
        """Individual supervised rows — LightGBM, XGBoost, CatBoost."""
        return cls(
            view=ViewKind.TABULAR,
            origin_scope=origin_scope,
            horizon_bound_at_fit=horizon_bound_at_fit,
            supports_unseen_instances=supports_unseen_instances,
        )

    @property
    def learns_across_origins(self) -> bool:
        """Whether several forecast origins may become training samples together."""
        return self.origin_scope is OriginScope.MULTIPLE
