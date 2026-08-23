"""Discover Nixtla models and map them onto two native execution protocols.

```python
descriptors("nixtla")      # every model, as the handshake reports it
adapter_for("nixtla/nhits", "nixtla")   # the one that executes it
```

StatsForecast's public model classes all execute through one local-series
adapter; NeuralForecast's public model classes all execute through one global
sequence adapter. The upstream modules therefore supply the catalog, while the
two existing hand-tuned entries remain capability and documentation overrides.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import pyarrow as pa

from openforecast.errors import UnknownModelError
from openforecast.models import FeatureCapabilities, ModelDescriptor, ModelRef
from openforecast.providers.native import parameters_from_signature
from openforecast.views import FitView, ForecastView
from openforecast_nixtla.adapters import neuralforecast, statsforecast

__all__ = ["Adapter", "adapter_for", "descriptors", "model_names"]


class Adapter(Protocol):
    """One model of one Nixtla library, behind the three provider operations."""

    @property
    def name(self) -> str:
        """The model half of the reference: ``autoarima`` in ``nixtla/autoarima``."""
        ...

    def descriptor(self, provider: str) -> ModelDescriptor: ...

    def fit(
        self, view: FitView, params: Mapping[str, Any], into: Path, *, seed: int | None
    ) -> None: ...

    def forecast(self, view: ForecastView, output: Mapping[str, Any], state: Path) -> pa.Table: ...


#: Entries with conformance coverage and richer-than-conservative capabilities.
CERTIFIED_ADAPTERS: Mapping[str, Adapter] = {
    statsforecast.AUTOARIMA.name: statsforecast.AUTOARIMA,
    neuralforecast.NHITS.name: neuralforecast.NHITS,
}

_STATS_RESERVED = ("alias", "prediction_intervals")
_NEURAL_RESERVED = (
    "h",
    "input_size",
    "hist_exog_list",
    "futr_exog_list",
    "stat_exog_list",
    "alias",
    "loss",
    "valid_loss",
    "optimizer",
    "optimizer_kwargs",
    "lr_scheduler",
    "lr_scheduler_kwargs",
    "random_seed",
    "step_size",
    "start_padding_enabled",
    "training_data_availability_threshold",
    "dataloader_kwargs",
)


@lru_cache(maxsize=1)
def _adapters() -> Mapping[str, Adapter]:
    adapters: dict[str, Adapter] = dict(CERTIFIED_ADAPTERS)
    _discover_statsforecast(adapters)
    _discover_neuralforecast(adapters)
    return adapters


def _discover_statsforecast(adapters: dict[str, Adapter]) -> None:
    try:
        import inspect

        from statsforecast import models
    except ImportError:
        return
    excluded = {"ConstantModel", "NaNModel", "SklearnModel", "ZeroModel"}
    for class_name, model_type in inspect.getmembers(models, inspect.isclass):
        name = class_name.lower()
        if (
            class_name.startswith("_")
            or class_name in excluded
            or name in adapters
            or model_type.__module__ != models.__name__
            or not callable(getattr(model_type, "forecast", None))
        ):
            continue
        discovered = parameters_from_signature(model_type, exclude=_STATS_RESERVED)
        if not discovered.is_constructible:
            continue
        adapters[name] = statsforecast.StatsForecastAdapter(
            name=name,
            display_name=class_name,
            build=lambda model_type=model_type, **params: model_type(**params),
            parameters=discovered.parameters,
            exogenous=False,
            quantiles=False,
        )


def _discover_neuralforecast(adapters: dict[str, Adapter]) -> None:
    try:
        from neuralforecast import models
    except ImportError:
        return
    for class_name in models.__all__:
        model_type = getattr(models, class_name)
        name = class_name.lower()
        if name in adapters or bool(getattr(model_type, "MULTIVARIATE", False)):
            continue
        discovered = parameters_from_signature(model_type, exclude=_NEURAL_RESERVED)
        if not discovered.is_constructible:
            continue
        adapters[name] = neuralforecast.NeuralForecastAdapter(
            name=name,
            display_name=class_name,
            build=lambda model_type=model_type, **params: model_type(**params),
            parameters=discovered.parameters,
            features=FeatureCapabilities(
                observed=bool(getattr(model_type, "EXOGENOUS_HIST", False)),
                known=bool(getattr(model_type, "EXOGENOUS_FUTR", False)),
                static=bool(getattr(model_type, "EXOGENOUS_STAT", False)),
            ),
        )


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
