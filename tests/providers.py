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
    """A descriptor that declares whatever a test needs it to declare.

    ``training`` defaults to a series contract for a model that can be fitted and
    to nothing at all for one that cannot — which is the invariant the descriptor
    enforces, so a pretrained lifecycle needs no second argument to express.
    """
    resolved = ModelLifecycle.trainable() if lifecycle is None else lifecycle
    contract = training
    if contract is None and resolved.supports_fit:
        contract = TrainingContract.series()
    return ModelDescriptor(
        ref=ModelRef.parse(f"{provider}/{name}"),
        provider=provider,
        display_name=f"Stub {name}",
        lifecycle=resolved,
        training=contract,
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
    #: The state directory each forecast was handed and what was in it *then*,
    #: so a test can check that a model nothing was fitted for was given nothing.
    #: Recorded rather than inspected afterwards: a zero-shot forecast's directory
    #: is temporary and is gone by the time the call returns.
    states: list[tuple[Path, tuple[str, ...]]] = field(default_factory=lambda: [])

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
        del model, params
        # A pretrained model is handed an empty directory, because nothing was
        # fitted for it. Falling back to ``value`` rather than raising is what
        # lets one stub serve both lifecycles.
        persisted = state / STATE_FILENAME
        value = float(persisted.read_text(encoding="utf-8")) if persisted.is_file() else self.value
        self.states.append((state, tuple(sorted(item.name for item in state.iterdir()))))
        answer = flat_answer(view, value, output)
        return answer if self.corrupt is None else self.corrupt(answer)


def flat_answer(
    view: ForecastView, value: float, output: Mapping[str, Any] | None = None
) -> pa.Table:
    """``value`` for every instance, event time and target, in the form asked for.

    A quantile is ``value`` shifted by how far the level is from the median and a
    sample path is ``value`` shifted by its draw index — arbitrary numbers, and
    deliberately so: what a test of the boundary can check is that the *shape* of
    the answer is the one that was requested, and arithmetic a provider invented
    would be the wrong thing to assert on.
    """
    kind = str((output or {}).get("kind", "point"))
    levels = [float(level) for level in (output or {}).get("levels", ())]
    draws = (output or {}).get("draws")

    keys = view.metadata.instance_keys
    described = [
        (instance, moment, target)
        for instance in view.instances
        for moment in view.event_times
        for target in view.metadata.targets
    ]
    if kind == "quantiles":
        parts = [("quantile", level, None, value + (level - 0.5) * 10.0) for level in levels]
    elif kind == "samples":
        parts = [("sample", None, draw, value + float(draw)) for draw in range(int(draws or 0))]
    else:
        parts = [("point", None, None, value)]
    rows = [
        (instance, moment, target, part) for instance, moment, target in described for part in parts
    ]

    columns: dict[str, pa.Array[Any]] = {
        name: pa.array([instance[index] for instance, _, _, _ in rows])
        for index, name in enumerate(keys)
    }
    columns[ForecastColumn.EVENT_TIME.value] = pa.array(
        [moment for _, moment, _, _ in rows], type=view.future.column(EVENT_TIME).type
    )
    columns[ForecastColumn.TARGET.value] = pa.array([target for _, _, target, _ in rows])
    columns[ForecastColumn.KIND.value] = pa.array([part[0] for *_, part in rows])
    columns[ForecastColumn.QUANTILE.value] = pa.array(
        [part[1] for *_, part in rows], type=pa.float64()
    )
    columns[ForecastColumn.SAMPLE.value] = pa.array([part[2] for *_, part in rows], type=pa.int64())
    columns[ForecastColumn.VALUE.value] = pa.array(
        [part[3] for *_, part in rows], type=pa.float64()
    )
    return pa.table({name: columns[name] for name in forecast_columns(keys)})
