"""One Chronos checkpoint, as OpenForecast advertises and executes it.

```text
descriptor   lifecycle=pretrained, training=None
forecast     ForecastView -> pipeline.predict_quantiles -> canonical columns
fit          there is none
```

The absence of a fit is the point of Step 23, and it is worth naming what the
absence buys. A trainable adapter has to persist native state into an artifact
directory, record which caller column was the target and in what order the
features were laid out, and rebuild exactly that at inference — because a fitted
model has positions rather than names. None of that exists here. The model is
the checkpoint, the checkpoint is pinned in the descriptor, and every forecast
starts from the same weights, so there is no state to write, no column order to
keep and nothing an artifact would have to record beyond the reference itself.

What replaces it is a *declaration*, and it is the only thing this adapter
promises:

```text
instances       single and panel      each series is one input in a batch
targets         univariate            a multivariate answer has to say which
                                      variate each number is about, and this
                                      integration does not yet
features        observed and known    past and future covariate slots
                static: no            Chronos-2 takes no static covariates
outputs         point and quantiles   Chronos-2 is a quantile model
missing values  native                a NaN is an unobserved step to it
```

Two of those deserve a sentence each.

**The point forecast is the median, not a mean.** Chronos-2 predicts quantiles
and nothing else, so a point forecast has to be read off the distribution. The
median is chosen because that is what a point metric reads a distribution at,
and because it is the level the model was trained to emit — a mean would have to
be integrated out of a piecewise-uniform approximation, which is a number the
model never said.

**A quantile level the model was not trained on is interpolated by the library,
and OpenForecast does not hide that.** Chronos-2 emits a fixed grid of levels
and interpolates between them for anything else; asking for 0.999 gets the
extreme it was trained on, with a warning on the log stream. That is the
library's own reading of its own distribution, which is exactly the thing
OpenForecast refuses to second-guess.

``chronos`` and ``torch`` are imported inside the calls that need them rather
than at module scope, for the reason the other integrations do it: a handshake
asks what this integration advertises, and answering should not pay for loading
a deep-learning stack. Here it matters more than elsewhere — importing ``torch``
is seconds, and ``openforecast providers list`` reads a recorded handshake.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa

from openforecast.errors import ProviderError
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
)
from openforecast.views import ForecastView
from openforecast_chronos import conversion

__all__ = ["CHECKPOINTS", "CHRONOS_2", "CheckpointSpec", "ChronosAdapter"]

#: The level a point forecast is read at. Chronos-2 answers quantiles, so a
#: point forecast is a reading of the distribution rather than a separate
#: prediction, and this is which reading.
MEDIAN = 0.5


@dataclass(frozen=True)
class CheckpointSpec:
    """One published checkpoint compatible with the Chronos-2 pipeline protocol."""

    name: str
    display_name: str
    checkpoint: str


class ChronosAdapter:
    """One pretrained Chronos checkpoint, behind the forecast operation."""

    def __init__(self, *, name: str, display_name: str, checkpoint: str) -> None:
        self._name = name
        self._display_name = display_name
        self._checkpoint = checkpoint
        self._pipeline: Any | None = None

    @classmethod
    def from_spec(cls, spec: CheckpointSpec) -> ChronosAdapter:
        return cls(name=spec.name, display_name=spec.display_name, checkpoint=spec.checkpoint)

    @property
    def name(self) -> str:
        return self._name

    @property
    def checkpoint(self) -> str:
        """The published weights this reference resolves to."""
        return self._checkpoint

    def descriptor(self, provider: str) -> ModelDescriptor:
        """What the catalog and the engine are told about this model.

        ``training=None`` is the declaration Step 23 exists for. Every other
        integration answers "which view do you learn from"; this one answers
        that there is nothing to learn, and the engine reads that rather than
        being told by a flag somewhere else what to do about it.
        """
        return ModelDescriptor(
            ref=ModelRef.parse(f"{provider}/{self._name}"),
            provider=provider,
            display_name=self._display_name,
            # Zero-shot, and frozen. Chronos-2 can be fine-tuned, and exposing
            # that would mean a training contract, an artifact, and a second
            # lifecycle for one model — none of which Step 23 is about.
            lifecycle=ModelLifecycle.pretrained(),
            training=None,
            capabilities=ModelCapabilities(
                instances=InstanceCapabilities(single=True, panel=True),
                targets=TargetCapabilities(univariate=True, multivariate=False),
                features=FeatureCapabilities(observed=True, known=True, static=False),
                outputs=OutputCapabilities(point=True, quantiles=True, samples=False),
                missing_values=MissingValueSupport.NATIVE,
            ),
        )

    # -- forecast -----------------------------------------------------------

    def forecast(self, view: ForecastView, output: Mapping[str, Any], state: Path) -> pa.Table:
        """The ``horizon`` steps after this origin, for every instance in the view.

        ``state`` is the empty directory the engine hands a model nothing was
        fitted for. It is deliberately not read: everything this model knows is
        in the checkpoint, and an adapter that reached into that directory would
        be one that had a fit it did not declare.
        """
        del state
        levels = _requested_levels(output, self._name)
        target = conversion.single_target(view.metadata.targets)
        inputs = [entry.as_mapping() for entry in conversion.inputs_for(view, target)]

        asked = [MEDIAN] if levels is None else list(levels)
        predicted = self._predict(inputs, horizon=len(view.event_times), levels=asked)
        return conversion.answer(
            view,
            predicted,
            target=target,
            levels=[None] if levels is None else list(levels),
        )

    def _predict(
        self, inputs: Sequence[Mapping[str, Any]], *, horizon: int, levels: Sequence[float]
    ) -> list[list[list[float]]]:
        """``quantiles`` for every instance, as nested lists of plain floats.

        The pipeline answers one tensor per input, shaped
        ``(variates, horizon, levels)``. This integration forecasts one variate,
        so the first axis is dropped here rather than in the conversion module:
        it is a property of what was asked for, and the conversion module should
        not have to know how a tensor is laid out.
        """
        pipeline = self._loaded()
        try:
            quantiles, _ = pipeline.predict_quantiles(
                inputs=list(inputs),
                prediction_length=horizon,
                quantile_levels=list(levels),
            )
        except Exception as error:
            # A library refusing this context is an execution failure the caller
            # can act on, not a bug in the boundary.
            raise ProviderError(
                f"{self._name} could not forecast this view: {type(error).__name__}: {error}"
            ) from error
        return [_one_variate(tensor, self._name) for tensor in quantiles]

    # -- the native pipeline -------------------------------------------------

    def _loaded(self) -> Any:
        """The pipeline, loaded once per process.

        A provider process serves many requests — a backtest over a hundred
        origins is a hundred forecasts down one pipe — and loading a checkpoint
        per request would dominate every one of them. Cached on the adapter
        rather than in a module-level dictionary because the adapter is already
        the one object per model.
        """
        if self._pipeline is None:
            self._pipeline = self._load()
        return self._pipeline

    def _load(self) -> Any:
        from chronos import BaseChronosPipeline

        try:
            return BaseChronosPipeline.from_pretrained(self._checkpoint)
        except Exception as error:
            raise ProviderError(
                f"the {self._checkpoint} checkpoint could not be loaded: "
                f"{type(error).__name__}: {error}. It is downloaded from the Hugging Face Hub "
                f"the first time it is used, so this is usually a network or cache problem"
            ) from error

    def __repr__(self) -> str:
        return f"ChronosAdapter({self._name}, checkpoint={self._checkpoint})"


def _requested_levels(output: Mapping[str, Any], model: str) -> tuple[float, ...] | None:
    """The levels asked for, or ``None`` for a point forecast.

    A request for anything else is refused here as well as by the engine. The
    engine checks it against the declaration, which is where a caller should
    meet it; this checks it against what the code below actually does, which is
    what stops the declaration and the implementation from drifting.
    """
    kind = str(output.get("kind", "point"))
    if kind == "point":
        return None
    if kind != "quantiles":
        raise ProviderError(
            f"{model} produces point and quantile forecasts, not {kind}; it predicts "
            f"quantiles directly and draws no sample paths"
        )
    levels = tuple(float(level) for level in output.get("levels", ()))
    if not levels:
        raise ProviderError(f"{model} was asked for quantiles and given no levels")
    return levels


def _one_variate(tensor: Any, model: str) -> list[list[float]]:
    """``(variates, horizon, levels)`` as ``horizon x levels``, for one variate."""
    rows: list[list[list[float]]] = tensor.tolist()
    if len(rows) != 1:
        raise ProviderError(f"{model} was asked about one target and answered {len(rows)} variates")
    return rows[0]


#: ``amazon/chronos-2``: the pretrained checkpoint published as
#: ``amazon/chronos-2`` on the Hugging Face Hub, which is deliberately the same
#: string. A reference a user already knows should not have to be translated.
CHRONOS_2_SPEC = CheckpointSpec(
    name="chronos-2",
    display_name="Chronos-2",
    checkpoint="amazon/chronos-2",
)

#: The lightweight checkpoint manifest. Adding another checkpoint that obeys
#: the same pipeline protocol is one data row, not another adapter.
CHECKPOINTS = (CHRONOS_2_SPEC,)

#: Backwards-compatible named adapter used by focused Chronos-2 tests.
CHRONOS_2 = ChronosAdapter.from_spec(CHRONOS_2_SPEC)
