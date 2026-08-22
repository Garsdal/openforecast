"""A stub provider, for testing the engine rather than a model.

The built-in provider is a real one and is tested as such. What the engine needs
in addition is a provider it can be *lied* to by: one that consumes a view the
built-in model does not, that can be told to answer the wrong question, or that
simply records what it was handed so that a test can assert the engine gave it
what the descriptor asked for.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa

from openforecast.models import (
    FeatureCapabilities,
    InstanceCapabilities,
    MissingValueSupport,
    ModelCapabilities,
    ModelDescriptor,
    ModelLifecycle,
    ModelRef,
    OutputCapabilities,
    TargetCapabilities,
    TrainingContract,
)
from openforecast.protocol import ForecastColumn, forecast_columns
from openforecast.views import EVENT_TIME, FitView, ForecastView

STATE_FILENAME = "stub.txt"


def descriptor(
    name: str,
    *,
    provider: str = "stub",
    training: TrainingContract | None = None,
    capabilities: ModelCapabilities | None = None,
    lifecycle: ModelLifecycle | None = None,
) -> ModelDescriptor:
    """A descriptor that declares whatever a test needs it to declare."""
    return ModelDescriptor(
        ref=ModelRef.parse(f"{provider}/{name}"),
        provider=provider,
        display_name=f"Stub {name}",
        lifecycle=ModelLifecycle.trainable() if lifecycle is None else lifecycle,
        training=TrainingContract.series() if training is None else training,
        capabilities=ModelCapabilities(
            instances=InstanceCapabilities(single=True, panel=True),
            targets=TargetCapabilities(univariate=True, multivariate=True),
            features=FeatureCapabilities(observed=True, known=True, static=True),
            outputs=OutputCapabilities(point=True),
            missing_values=MissingValueSupport.NATIVE,
        )
        if capabilities is None
        else capabilities,
    )


@dataclass
class FitCall:
    """What the engine handed the provider for one leaf."""

    model: str
    params: Mapping[str, Any]
    view: FitView
    seed: int | None
    into: Path


@dataclass
class StubProvider:
    """Answers every request with ``value``, and remembers being asked."""

    name: str = "stub"
    version: str = "0.0.1"
    models: tuple[ModelDescriptor, ...] = ()
    value: float = 1.0
    #: Rewrites the answer, for testing what the engine does with a bad one.
    corrupt: Callable[[pa.Table], pa.Table] | None = None
    fits: list[FitCall] = field(default_factory=lambda: [])

    def descriptors(self) -> tuple[ModelDescriptor, ...]:
        return self.models

    def fit(
        self,
        *,
        model: ModelRef | str,
        params: Mapping[str, Any],
        view: FitView,
        seed: int | None,
        into: Path,
    ) -> None:
        self.fits.append(
            FitCall(model=str(model), params=dict(params), view=view, seed=seed, into=into)
        )
        (into / STATE_FILENAME).write_text(str(self.value), encoding="utf-8")

    def forecast(
        self,
        *,
        model: ModelRef | str,
        params: Mapping[str, Any],
        view: ForecastView,
        output: Mapping[str, Any],
        state: Path,
    ) -> pa.Table:
        del model, params, output
        value = float((state / STATE_FILENAME).read_text(encoding="utf-8"))
        answer = flat_answer(view, value)
        return answer if self.corrupt is None else self.corrupt(answer)


def flat_answer(view: ForecastView, value: float) -> pa.Table:
    keys = view.metadata.instance_keys
    rows = [
        (instance, moment, target)
        for instance in view.instances
        for moment in view.event_times
        for target in view.metadata.targets
    ]
    columns: dict[str, pa.Array[Any]] = {
        name: pa.array([instance[index] for instance, _, _ in rows])
        for index, name in enumerate(keys)
    }
    columns[ForecastColumn.EVENT_TIME.value] = pa.array(
        [moment for _, moment, _ in rows], type=view.future.column(EVENT_TIME).type
    )
    columns[ForecastColumn.TARGET.value] = pa.array([target for _, _, target in rows])
    columns[ForecastColumn.KIND.value] = pa.array(["point"] * len(rows))
    columns[ForecastColumn.QUANTILE.value] = pa.nulls(len(rows), type=pa.float64())
    columns[ForecastColumn.SAMPLE.value] = pa.nulls(len(rows), type=pa.int64())
    columns[ForecastColumn.VALUE.value] = pa.array([value] * len(rows), type=pa.float64())
    return pa.table({name: columns[name] for name in forecast_columns(keys)})
