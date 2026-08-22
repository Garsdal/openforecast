"""NeuralForecast models: a ``SequenceView`` in, a global neural model out.

```text
fit        SequenceView -> NeuralForecast(models=[NHITS(...)]).fit(long frame)
state      into/        -> neuralforecast/ + state.json
forecast   ForecastView -> predict(df=context, futr_df=future) -> the answer
```

A NeuralForecast model is *global*: one set of parameters is learned from every
training sample at once, and a sample is one ``context -> horizon`` window at one
forecast origin. That is what makes this the interesting half of the Nixtla
integration — it is the first model here that can learn from real point-in-time
vintages rather than from one of them.

The compilation is the point of the whole design, and it is three lines:

```text
WindowPlan(context=168)  ->  input_size=168
horizon=72               ->  h=72
sample_id                ->  unique_id
```

None of the three is something a caller states twice. The context length and the
horizon are OpenForecast's, because the ``ViewPlanner`` had to know both to cut
the samples in the first place; passing them again as native parameters is
refused by ``of.Model`` before this module is reached. And the identifier is the
view's, so this adapter never learns which instance or which origin a sample
came from — which is precisely why a ``TimeSeriesFrame`` and a ``ForecastDataset``
are indistinguishable from in here.

Two consequences of being global are worth naming.

**The horizon is bound at fit time.** NHiTS learns an output layer of exactly
``h`` steps, so an artifact trained for 72 cannot answer 48; the descriptor
declares ``horizon_bound_at_fit`` and the engine refuses the request with
``IncompatibleForecastTask`` before this module is reached.

**An unseen instance is forecastable.** Shared parameters are what makes that
true, so ``supports_unseen_instances`` is declared — and asserted in the tests,
because a capability nobody exercised is a claim rather than a capability.

``neuralforecast`` is imported inside the two calls that need it rather than at
module scope. A handshake — which is what installing a provider and listing
models does — only asks what this integration advertises, and importing PyTorch
to answer that would make discovery slow for no reason.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa

from openforecast.errors import DataError, ProviderError, RecipeError
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
from openforecast.views import FitView, ForecastView, SequenceView
from openforecast_nixtla import conversion
from openforecast_nixtla.parameters import Parameter, checked, named, schema_of
from openforecast_nixtla.state import STATE_FILENAME, read_state, write_state

__all__ = ["NHITS", "NeuralForecastAdapter"]

#: The directory NeuralForecast's own ``save`` writes its checkpoints into.
MODEL_DIRNAME = "neuralforecast"

#: What the trainer is not allowed to decide for itself. These are not modeling
#: choices: a provider runs inside somebody else's process and must not scatter
#: ``lightning_logs/`` and checkpoints through their working directory, nor print
#: progress bars over a stdout that is carrying the wire protocol.
TRAINER_KWARGS: Mapping[str, Any] = {
    "logger": False,
    "enable_checkpointing": False,
    "enable_progress_bar": False,
    "enable_model_summary": False,
}


class NeuralForecastAdapter:
    """One NeuralForecast model, as OpenForecast advertises and executes it."""

    def __init__(
        self,
        *,
        name: str,
        display_name: str,
        build: Callable[..., Any],
        parameters: Sequence[Parameter],
    ) -> None:
        self._name = name
        self._display_name = display_name
        self._build = build
        self._parameters = named(parameters)

    @property
    def name(self) -> str:
        return self._name

    def descriptor(self, provider: str) -> ModelDescriptor:
        """What the catalog and the engine are told about this model.

        Every capability is one the library actually has. It learns across
        origins because that is what a global model does with many windows; it
        needs a context length because ``input_size`` has no defensible default;
        it binds its horizon because the output layer has ``h`` units; it takes
        an unseen instance because the parameters are shared rather than fitted
        per series; and it consumes all three feature roles, because they are
        exactly the three covariate lists the library declares.

        Missing values are the one thing it cannot take as they come. Point-in-
        time data is full of them and a gradient step on a NaN produces NaN
        weights, so the declaration is ``REQUIRES_TRANSFORM``: the caller writes
        an imputation down, where the artifact records it, or the request is
        refused. Filling them in here is the silent imputation rule 5 forbids.
        """
        return ModelDescriptor(
            ref=ModelRef.parse(f"{provider}/{self._name}"),
            provider=provider,
            display_name=self._display_name,
            lifecycle=ModelLifecycle.trainable(),
            training=TrainingContract.sequences(supports_unseen_instances=True),
            capabilities=ModelCapabilities(
                instances=InstanceCapabilities(single=True, panel=True),
                targets=TargetCapabilities(univariate=True, multivariate=False),
                features=FeatureCapabilities(observed=True, known=True, static=True),
                outputs=OutputCapabilities(point=True),
                missing_values=MissingValueSupport.REQUIRES_TRANSFORM,
            ),
            parameters_schema=schema_of(self._parameters),
        )

    # -- fit ----------------------------------------------------------------

    def fit(
        self, view: FitView, params: Mapping[str, Any], into: Path, *, seed: int | None
    ) -> None:
        """Fit one global model over every sample, and persist what labels it."""
        from neuralforecast import NeuralForecast

        if not isinstance(view, SequenceView):
            raise ProviderError(
                f"{self._name} trains on context -> horizon sequences, so it cannot be "
                f"fitted from a {view.kind} view"
            )
        schema = view.schema
        prepared = conversion.sequence_frames(view)
        model = self._instantiate(
            params, prepared, context=schema.context, horizon=schema.horizon, seed=seed
        )
        forecaster = NeuralForecast(models=[model], freq=prepared.frequency)
        try:
            forecaster.fit(prepared.frame, static_df=prepared.static, val_size=0)
        except Exception as error:
            # A library refusing to train on these windows is an execution
            # failure the caller can act on, not a bug in the boundary.
            raise ProviderError(
                f"{self._name} could not be fitted on this data: {type(error).__name__}: {error}"
            ) from error
        forecaster.save(str(into / MODEL_DIRNAME), save_dataset=False, overwrite=True)
        write_state(
            into / STATE_FILENAME,
            {
                "model": self._name,
                "column": _column_of(model),
                "target": prepared.target,
                "hist_exog": list(prepared.hist_exog),
                "futr_exog": list(prepared.futr_exog),
                "stat_exog": list(prepared.stat_exog),
                "frequency": prepared.frequency,
                "context": schema.context,
                "horizon": schema.horizon,
                "samples": len(view.sample_ids),
            },
        )

    # -- forecast -----------------------------------------------------------

    def forecast(self, view: ForecastView, output: Mapping[str, Any], state: Path) -> pa.Table:
        """The ``horizon`` steps after this origin, for every instance in the view."""
        from neuralforecast import NeuralForecast

        kind = output.get("kind", "point")
        if kind != "point":
            raise ProviderError(f"{self._name} produces point forecasts, not {kind}")
        persisted = read_state(state / STATE_FILENAME, self._name)
        self._require_matching_window(view, persisted)

        frames = conversion.forecast_frames(
            view,
            target=str(persisted["target"]),
            hist_exog=[str(name) for name in persisted["hist_exog"]],
            futr_exog=[str(name) for name in persisted["futr_exog"]],
            stat_exog=[str(name) for name in persisted["stat_exog"]],
        )
        forecaster = NeuralForecast.load(str(state / MODEL_DIRNAME))
        try:
            predictions = forecaster.predict(
                df=frames.history, futr_df=frames.future, static_df=frames.static
            )
        except Exception as error:
            raise ProviderError(
                f"{self._name} could not forecast this view: {type(error).__name__}: {error}"
            ) from error
        return conversion.answer(
            view,
            frames.unique_ids,
            predictions,
            column=str(persisted["column"]),
            target=str(persisted["target"]),
        )

    def _require_matching_window(self, view: ForecastView, persisted: Mapping[str, Any]) -> None:
        """The window this model learned is the only one it can be asked about.

        The engine sizes the inference view from the artifact's own record, so
        reaching here with a different window means the artifact and the request
        disagree — and a mismatched context would be silently truncated by the
        library rather than refused.
        """
        metadata = view.metadata
        wanted = (int(persisted["context"]), int(persisted["horizon"]))
        given = (metadata.context, metadata.horizon)
        if given != wanted:
            raise DataError(
                f"{self._name} was fitted on {wanted[0]} context steps and a horizon of "
                f"{wanted[1]}, and this view holds {given[0]} and {given[1]}; a global "
                f"model learns one window and cannot be asked about another"
            )

    # -- parameters ---------------------------------------------------------

    def _instantiate(
        self,
        params: Mapping[str, Any],
        prepared: conversion.SequenceFrames,
        *,
        context: int,
        horizon: int,
        seed: int | None,
    ) -> Any:
        """The native model the caller's parameters and the view jointly describe.

        The caller supplies the modeling parameters; the view supplies the shape.
        They cannot collide, because the shape is not something ``of.Model`` lets
        a caller pass — ``input_size`` and ``h`` name concepts OpenForecast owns.
        """
        settings = checked(params, self._parameters, self._name)
        compiled: dict[str, Any] = {
            "h": horizon,
            "input_size": context,
            "hist_exog_list": list(prepared.hist_exog) or None,
            "futr_exog_list": list(prepared.futr_exog) or None,
            "stat_exog_list": list(prepared.stat_exog) or None,
            # The one-sequence invariant, in the library's own terms. Left at its
            # default, NeuralForecast right-pads every series by ``h`` and keeps
            # any window with a single real step in it — so a sample of exactly
            # ``input_size + h`` steps yields several shifted windows whose later
            # steps are padding rather than outcomes. Requiring a window to be
            # fully available makes the one window OpenForecast described the one
            # window the model trains on.
            "training_data_availability_threshold": 1.0,
            **TRAINER_KWARGS,
        }
        if seed is not None:
            compiled["random_seed"] = seed
        try:
            return self._build(**settings, **compiled)
        except (TypeError, ValueError) as error:
            raise RecipeError(f"{self._name} rejected {dict(params)}: {error}") from error

    def __repr__(self) -> str:
        return f"NeuralForecastAdapter({self._name})"


def _column_of(model: Any) -> str:
    """The column NeuralForecast will label this model's predictions with.

    Its own rule: the alias when one is set, and the class name otherwise. Read
    off the instance at fit time and persisted, so the forecast side reads the
    column the fit actually produced rather than guessing at it again.
    """
    alias: str | None = getattr(model, "alias", None)
    return alias if alias else type(model).__name__


def _nhits(**params: Any) -> Any:
    from neuralforecast.models import NHITS

    return NHITS(**params)


#: The parameters of ``NHITS`` a caller may set. Deliberately a subset: the
#: window, the covariate lists and the seed are OpenForecast's and are compiled
#: from the view, ``alias`` would rename the column the answer is read from, and
#: the distributed, dataloader and optimizer hooks are objects rather than the
#: JSON a recipe has to survive being written down as.
#:
#: ``start_padding_enabled`` and ``step_size`` are withheld for a different
#: reason than the rest: they govern how the library cuts windows out of a
#: series, and a sample already *is* one window. Padding one would manufacture a
#: second training sequence out of nothing, which is the one thing the sample
#: boundaries exist to prevent.
NHITS_PARAMETERS = (
    Parameter("max_steps", int, "Optimization steps to train for.", minimum=1),
    Parameter("learning_rate", float, "Step size of the optimizer.", minimum=0),
    Parameter("batch_size", int, "Training samples per batch.", minimum=1),
    Parameter("windows_batch_size", int, "Windows sampled per optimization step.", minimum=1),
    Parameter("num_lr_decays", int, "How many times the learning rate is decayed.", minimum=0),
    Parameter("dropout_prob_theta", float, "Dropout probability of the block MLPs.", minimum=0),
    Parameter(
        "activation",
        str,
        "Activation of the block MLPs.",
        choices=("ReLU", "Softplus", "Tanh", "SELU", "LeakyReLU", "PReLU", "Sigmoid"),
    ),
    Parameter(
        "pooling_mode", str, "How each block pools its input.", choices=("MaxPool1d", "AvgPool1d")
    ),
    Parameter(
        "interpolation_mode",
        str,
        "How each block interpolates its output.",
        choices=("linear", "nearest"),
    ),
    Parameter(
        "scaler_type",
        str,
        "Per-series scaling applied inside the model.",
        choices=("identity", "standard", "robust", "robust-iqr", "minmax", "minmax1", "revin"),
    ),
    Parameter(
        "exclude_insample_y", bool, "Condition on the covariates only, not on the target's past."
    ),
    Parameter("drop_last_loader", bool, "Drop the last, incomplete training batch."),
)

#: ``nixtla/nhits``: multi-rate hierarchical interpolation, fitted across samples.
NHITS = NeuralForecastAdapter(
    name="nhits",
    display_name="NHiTS",
    build=_nhits,
    parameters=NHITS_PARAMETERS,
)
