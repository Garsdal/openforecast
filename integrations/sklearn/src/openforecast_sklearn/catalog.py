"""Reflect sklearn's regressor protocol into OpenForecast model descriptors.

```python
descriptors("sklearn")                            # every model, as the handshake reports it
adapter_for("sklearn/hist-gradient-boosting", "sklearn")  # the one that executes it
```

A catalog rather than a chain of ``if`` statements in the provider: a model is
added by naming its adapter here, and both the handshake and the dispatch read
the same table. Nothing else in the integration knows how many models there are.

``sklearn.utils.all_estimators`` is the catalog. Every ordinary regressor whose
public tags say it consumes a two-dimensional design matrix and one target is
wrapped by the same :class:`SklearnAdapter`. Constructor parameters are derived
from the estimator signature; ``random_state`` remains OpenForecast's fit seed.

The hand-tuned histogram-gradient-boosting entry is retained as an override: it
has stronger parameter descriptions and conformance coverage, but it executes
through exactly the same family adapter as discovered regressors.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import pyarrow as pa

from openforecast.errors import UnknownModelError
from openforecast.models import MissingValueSupport, ModelDescriptor, ModelRef
from openforecast.providers.native import class_slug, parameters_from_signature
from openforecast.views import FitView, ForecastView
from openforecast_sklearn.adapter import HIST_GRADIENT_BOOSTING, SklearnAdapter

__all__ = ["Adapter", "adapter_for", "descriptors", "model_names"]


class Adapter(Protocol):
    """One scikit-learn estimator, behind the three provider operations."""

    @property
    def name(self) -> str:
        """The model half of the reference: ``ridge`` in ``sklearn/ridge``."""
        ...

    def descriptor(self, provider: str) -> ModelDescriptor: ...

    def fit(
        self, view: FitView, params: Mapping[str, Any], into: Path, *, seed: int | None
    ) -> None: ...

    def forecast(self, view: ForecastView, output: Mapping[str, Any], state: Path) -> pa.Table: ...


#: Hand-tuned declarations override reflected defaults under the same name.
CERTIFIED_ADAPTERS: Mapping[str, Adapter] = {
    HIST_GRADIENT_BOOSTING.name: HIST_GRADIENT_BOOSTING,
}


@lru_cache(maxsize=1)
def _adapters() -> Mapping[str, Adapter]:
    adapters: dict[str, Adapter] = dict(CERTIFIED_ADAPTERS)
    try:
        from sklearn.utils import all_estimators, get_tags
    except ImportError:
        # Source-tree discovery remains usable without an integration
        # environment; the provider environment itself always has sklearn.
        return adapters

    for class_name, estimator_type in all_estimators(type_filter="regressor"):
        name = class_slug(class_name, suffixes=("Regressor", "Regression"))
        if name in adapters:
            continue
        discovered = parameters_from_signature(estimator_type, exclude=("random_state",))
        if not discovered.is_constructible:
            continue
        try:
            estimator = estimator_type()
            tags = get_tags(estimator)
        except (TypeError, ValueError):
            continue
        input_tags = tags.input_tags
        target_tags = tags.target_tags
        if not input_tags.two_d_array or not target_tags.single_output:
            continue
        missing = (
            MissingValueSupport.NATIVE
            if input_tags.allow_nan
            else MissingValueSupport.REQUIRES_TRANSFORM
        )
        adapters[name] = SklearnAdapter(
            name=name,
            display_name=class_name,
            estimator=lambda estimator_type=estimator_type: estimator_type,
            parameters=discovered.parameters,
            missing_values=missing,
            seeded="random_state" in _constructor_parameters(estimator_type),
        )
    return adapters


def _constructor_parameters(estimator_type: type[Any]) -> set[str]:
    import inspect

    try:
        return set(inspect.signature(estimator_type.__init__).parameters)
    except (TypeError, ValueError):
        return set()


def model_names() -> tuple[str, ...]:
    return tuple(_adapters())


def descriptors(provider: str) -> tuple[ModelDescriptor, ...]:
    """Every model this integration advertises, namespaced to ``provider``."""
    return tuple(adapter.descriptor(provider) for adapter in _adapters().values())


def adapter_for(model: ModelRef | str, provider: str) -> Adapter:
    """The adapter that executes ``model``, or a refusal naming what there is."""
    ref = ModelRef.parse(model)
    adapters = _adapters()
    adapter = adapters.get(ref.name) if ref.namespace == provider else None
    if adapter is None:
        raise UnknownModelError(
            f"{ref} is not a model of the {provider!r} provider; it provides "
            f"{[f'{provider}/{name}' for name in adapters]}"
        )
    return adapter
