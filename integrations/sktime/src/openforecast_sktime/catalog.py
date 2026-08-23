"""Reflect sktime's forecaster registry into OpenForecast model descriptors.

```python
descriptors("sktime")                          # every model, as the handshake reports it
adapter_for("sktime/pooled-trees", "sktime")    # the one that executes it
```

Every forecaster that can fit without receiving a horizon and can be constructed
from JSON parameters is exposed through the local-series protocol.  sktime's
own registry and class tags are authoritative for discovery and missing-value
support.  The pooled reduction remains a specialized global-sequence override.
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
from openforecast_sktime.adapters import local_models, panel_models

__all__ = ["Adapter", "adapter_for", "descriptors", "model_names"]


class Adapter(Protocol):
    """One sktime forecaster, behind the three provider operations."""

    @property
    def name(self) -> str:
        """The model half of the reference: ``theta`` in ``sktime/theta``."""
        ...

    def descriptor(self, provider: str) -> ModelDescriptor: ...

    def fit(
        self, view: FitView, params: Mapping[str, Any], into: Path, *, seed: int | None
    ) -> None: ...

    def forecast(self, view: ForecastView, output: Mapping[str, Any], state: Path) -> pa.Table: ...


#: Hand-tuned entries with full conformance coverage.
CERTIFIED_ADAPTERS: Mapping[str, Adapter] = {
    local_models.THETA.name: local_models.THETA,
    panel_models.POOLED_TREES.name: panel_models.POOLED_TREES,
}


@lru_cache(maxsize=1)
def _adapters() -> Mapping[str, Adapter]:
    adapters: dict[str, Adapter] = dict(CERTIFIED_ADAPTERS)
    try:
        from sktime.registry import all_estimators
        from sktime.utils.dependencies import _check_soft_dependencies
    except ImportError:
        return adapters

    for class_name, model_type in all_estimators(
        estimator_types="forecaster", return_names=True, suppress_import_stdout=True
    ):
        name = class_slug(class_name, suffixes=("Forecaster", "Model"))
        if name in adapters:
            continue
        if bool(model_type.get_class_tag("requires-fh-in-fit", True)):
            continue
        dependencies = model_type.get_class_tag("python_dependencies", None)
        required_packages = (
            [dependencies] if isinstance(dependencies, str) else list(dependencies or ())
        )
        if required_packages and not _check_soft_dependencies(
            *required_packages, severity="none"
        ):
            continue
        target_scitype = model_type.get_class_tag("scitype:y", "univariate")
        if target_scitype == "multivariate":
            continue
        discovered = parameters_from_signature(model_type, exclude=("random_state",))
        if not discovered.is_constructible:
            continue
        missing = (
            MissingValueSupport.NATIVE
            if bool(model_type.get_class_tag("capability:missing_values", False))
            else MissingValueSupport.REQUIRES_TRANSFORM
        )
        adapters[name] = local_models.SktimeLocalAdapter(
            name=name,
            display_name=class_name,
            model_type=lambda model_type=model_type: model_type,
            parameters=discovered.parameters,
            missing_values=missing,
            seeded="random_state" in _constructor_parameters(model_type),
        )
    return adapters


def _constructor_parameters(model_type: type[Any]) -> set[str]:
    import inspect

    try:
        return set(inspect.signature(model_type.__init__).parameters)
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
