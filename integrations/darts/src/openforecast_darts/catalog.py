"""Discover Darts forecasting classes and classify their native protocol.

```python
descriptors("darts")                  # every model, as the handshake reports it
adapter_for("darts/tide", "darts")    # the one that executes it
```

Darts already distinguishes local and global forecasting models. OpenForecast
uses that hierarchy directly: local classes share the series adapter and global
classes with the standard chunk-length contract share the sequence adapter.
Runtime ``supports_*`` properties supply covariate capabilities for global
models; constructor reflection supplies their JSON parameter schemas.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import pyarrow as pa

from openforecast.errors import UnknownModelError
from openforecast.models import FeatureCapabilities, ModelDescriptor, ModelRef
from openforecast.providers.native import class_slug, parameters_from_signature
from openforecast.views import FitView, ForecastView
from openforecast_darts.adapters import global_models, local_models

__all__ = ["Adapter", "adapter_for", "descriptors", "model_names"]


class Adapter(Protocol):
    """One Darts model, behind the three provider operations."""

    @property
    def name(self) -> str:
        """The model half of the reference: ``theta`` in ``darts/theta``."""
        ...

    def descriptor(self, provider: str) -> ModelDescriptor: ...

    def fit(
        self, view: FitView, params: Mapping[str, Any], into: Path, *, seed: int | None
    ) -> None: ...

    def forecast(self, view: ForecastView, output: Mapping[str, Any], state: Path) -> pa.Table: ...


#: Hand-tuned entries with full conformance coverage.
CERTIFIED_ADAPTERS: Mapping[str, Adapter] = {
    local_models.THETA.name: local_models.THETA,
    global_models.TIDE.name: global_models.TIDE,
    global_models.NHITS.name: global_models.NHITS,
}

_RESERVED = (
    "input_chunk_length",
    "output_chunk_length",
    "output_chunk_shift",
    "random_state",
    "pl_trainer_kwargs",
    "model_name",
    "work_dir",
    "save_checkpoints",
    "force_reset",
    "add_encoders",
    "likelihood",
)

_NAMES = {"TiDEModel": "tide", "NHiTSModel": "nhits", "Theta": "theta"}

# Darts exposes these in ``darts.models``, but constructing them resolves remote
# pretrained weights and their lifecycle is zero-shot rather than the trainable
# local/global protocols in this integration. The dedicated Chronos provider is
# the OpenForecast path for that lifecycle.
_PRETRAINED = {"Chronos2Model", "TimesFM2p5Model"}


@lru_cache(maxsize=1)
def _adapters() -> Mapping[str, Adapter]:
    adapters: dict[str, Adapter] = dict(CERTIFIED_ADAPTERS)
    try:
        import inspect

        from darts import models
        from darts.models.forecasting.forecasting_model import (
            GlobalForecastingModel,
            LocalForecastingModel,
        )
    except ImportError:
        return adapters

    seen: set[type[Any]] = set()
    for class_name in models.__all__:
        model_type = getattr(models, class_name, None)
        if (
            not inspect.isclass(model_type)
            or model_type in seen
            or "Classifier" in class_name
            or class_name in _PRETRAINED
            or not issubclass(model_type, LocalForecastingModel | GlobalForecastingModel)
        ):
            continue
        seen.add(model_type)
        name = _NAMES.get(
            class_name, class_slug(class_name, suffixes=("ForecastingModel", "Model"))
        )
        if name in adapters:
            continue
        constructor_parameters = set(inspect.signature(model_type.__init__).parameters)
        is_global = issubclass(model_type, GlobalForecastingModel)
        if is_global and "input_chunk_length" not in constructor_parameters:
            continue
        discovered = parameters_from_signature(model_type, exclude=_RESERVED)
        if not discovered.is_constructible:
            continue
        if not is_global:
            adapters[name] = local_models.DartsLocalAdapter(
                name=name,
                display_name=class_name,
                model_type=lambda model_type=model_type: model_type,
                parameters=discovered.parameters,
                seeded="random_state" in constructor_parameters,
            )
            continue
        probe_settings: dict[str, Any] = {"input_chunk_length": 24}
        if "output_chunk_length" in constructor_parameters:
            probe_settings["output_chunk_length"] = 3
        if "pl_trainer_kwargs" in constructor_parameters:
            probe_settings["pl_trainer_kwargs"] = dict(global_models.TRAINER_KWARGS)
        try:
            probe = model_type(**probe_settings)
        except (TypeError, ValueError):
            continue
        adapters[name] = global_models.DartsGlobalAdapter(
            name=name,
            display_name=class_name,
            model_type=lambda model_type=model_type: model_type,
            parameters=discovered.parameters,
            features=FeatureCapabilities(
                observed=bool(probe.supports_past_covariates),
                known=bool(probe.supports_future_covariates),
                static=bool(probe.supports_static_covariates),
            ),
            common_parameters=(),
            horizon_bound_at_fit="output_chunk_length" in constructor_parameters,
            constructor_parameters=tuple(constructor_parameters),
        )
    return adapters


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
