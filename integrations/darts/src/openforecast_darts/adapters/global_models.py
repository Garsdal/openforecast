"""Global Darts models: a ``SequenceView`` in, one set of shared parameters out.

```text
fit        SequenceView -> Model(input_chunk_length=..., output_chunk_length=...).fit([...])
state      into/        -> model.pt + model.pt.ckpt + state.json
forecast   ForecastView -> predict(n, series=[...], ...) -> the canonical columns
```

A Darts ``GlobalForecastingModel`` is *global*: one set of parameters is learned
from every training sample at once, and a sample is one ``context -> horizon``
window at one forecast origin. This is the half of the integration that Step 13
exists for — the same point-in-time claim as ``nixtla/nhits``, made by a library
that spells everything differently.

The compilation is the point of the whole design, and it is three lines:

```text
WindowPlan(context=168)  ->  input_chunk_length=168
horizon=72               ->  output_chunk_length=72, and predict(n=72)
one sample               ->  one TimeSeries in the list
```

None of the three is something a caller states twice. The context length and the
horizon are OpenForecast's, because the ``ViewPlanner`` had to know both to cut
the samples in the first place; passing them again as native parameters is
refused by ``of.Model`` before this module is reached. And the samples are the
view's, so this adapter never learns which instance or which origin one came
from — which is precisely why a ``TimeSeriesFrame`` and a ``ForecastDataset``
are indistinguishable from in here.

Three consequences of being global are worth naming.

**The horizon is bound at fit time.** ``output_chunk_length`` is part of the
architecture, so an artifact trained for 72 cannot answer 48; the descriptor
declares ``horizon_bound_at_fit`` and the engine refuses the request with
``IncompatibleForecastTask`` before this module is reached.

**An unseen instance is forecastable.** Shared parameters are what makes that
true, so ``supports_unseen_instances`` is declared — and asserted in the tests,
because a capability nobody exercised is a claim rather than a capability.

**Not every global model takes every covariate.** Darts is honest about this and
so is the catalog: ``TiDE`` conditions on past, future *and* static covariates,
while its ``NHiTS`` is a past-covariates model and says so
(``supports_future_covariates`` is ``False``). That difference is a
:class:`~openforecast.models.FeatureCapabilities` declaration rather than a
special case in the code, which is what stops a known feature from being quietly
handed over as a past covariate — a value known ahead of its event time, used as
if it were not.

The catalog discovers Darts' global classes and runtime capabilities once and
injects the selected class into this adapter. Fit and forecast contain no
model-name dispatch.
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
from openforecast_darts import conversion
from openforecast_darts.parameters import Parameter, checked, named, schema_of
from openforecast_darts.state import STATE_FILENAME, read_state, write_state

__all__ = ["NHITS", "TIDE", "DartsGlobalAdapter"]

#: What Darts' own ``save`` is pointed at. It writes the model beside a
#: ``.ckpt`` of the Lightning weights, which is why this is a name in a
#: directory the engine owns rather than a directory of its own.
MODEL_FILENAME = "model.pt"

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

#: The parameters every Darts torch model takes, whatever its architecture.
TORCH_PARAMETERS = (
    Parameter("n_epochs", int, "Passes over the training samples.", minimum=1),
    Parameter("batch_size", int, "Training samples per batch.", minimum=1),
    Parameter("dropout", float, "Dropout probability.", minimum=0, maximum=1),
)


class DartsGlobalAdapter:
    """One global Darts model, as OpenForecast advertises and executes it."""

    def __init__(
        self,
        *,
        name: str,
        display_name: str,
        model_type: Callable[[], Any],
        parameters: Sequence[Parameter],
        features: FeatureCapabilities,
        common_parameters: Sequence[Parameter] = TORCH_PARAMETERS,
        horizon_bound_at_fit: bool = True,
        constructor_parameters: Sequence[str] = (
            "input_chunk_length",
            "output_chunk_length",
            "pl_trainer_kwargs",
            "random_state",
        ),
    ) -> None:
        self._name = name
        self._display_name = display_name
        self._model_type = model_type
        self._parameters = named((*common_parameters, *parameters))
        self._features = features
        self._horizon_bound_at_fit = horizon_bound_at_fit
        self._constructor_parameters = set(constructor_parameters)

    @property
    def name(self) -> str:
        return self._name

    def descriptor(self, provider: str) -> ModelDescriptor:
        """What the catalog and the engine are told about this model.

        Every capability is one the library actually has. It learns across
        origins because that is what a global model does with many windows; it
        needs a context length because ``input_chunk_length`` has no defensible
        default; it binds its horizon because ``output_chunk_length`` is part of
        the architecture; it takes an unseen instance because the parameters are
        shared rather than fitted per series; and it consumes the covariate
        kinds this particular Darts model declares support for, no more.

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
            training=TrainingContract.sequences(
                horizon_bound_at_fit=self._horizon_bound_at_fit,
                supports_unseen_instances=True,
            ),
            capabilities=ModelCapabilities(
                instances=InstanceCapabilities(single=True, panel=True),
                targets=TargetCapabilities(univariate=True, multivariate=False),
                features=self._features,
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
        if not isinstance(view, SequenceView):
            raise ProviderError(
                f"{self._name} trains on context -> horizon sequences, so it cannot be "
                f"fitted from a {view.kind} view"
            )
        schema = view.schema
        prepared = conversion.sequence_series(view, features=self._features)
        model = self._instantiate(params, context=schema.context, horizon=schema.horizon, seed=seed)
        try:
            model.fit(
                series=prepared.targets,
                past_covariates=prepared.past_covariates,
                future_covariates=prepared.future_covariates,
            )
        except Exception as error:
            # A library refusing to train on these windows is an execution
            # failure the caller can act on, not a bug in the boundary.
            raise ProviderError(
                f"{self._name} could not be fitted on this data: {type(error).__name__}: {error}"
            ) from error
        model.save(str(into / MODEL_FILENAME))
        write_state(
            into / STATE_FILENAME,
            {
                "model": self._name,
                "target": prepared.target,
                "observed": list(prepared.observed),
                "known": list(prepared.known),
                "static": list(prepared.static),
                "frequency": prepared.frequency,
                "context": schema.context,
                "horizon": schema.horizon,
                "samples": len(prepared.sample_ids),
            },
        )

    # -- forecast -----------------------------------------------------------

    def forecast(self, view: ForecastView, output: Mapping[str, Any], state: Path) -> pa.Table:
        """The ``horizon`` steps after this origin, for every instance in the view."""
        kind = output.get("kind", "point")
        if kind != "point":
            raise ProviderError(f"{self._name} produces point forecasts, not {kind}")
        persisted = read_state(state / STATE_FILENAME, self._name)
        self._require_matching_window(view, persisted)

        target = str(persisted["target"])
        prepared = conversion.forecast_series(
            view,
            target=target,
            observed=[str(name) for name in persisted["observed"]],
            known=[str(name) for name in persisted["known"]],
            static=[str(name) for name in persisted["static"]],
        )
        model = self._load(state / MODEL_FILENAME)
        try:
            predictions = model.predict(
                n=view.metadata.horizon,
                series=prepared.history,
                past_covariates=prepared.past_covariates,
                future_covariates=prepared.future_covariates,
            )
        except Exception as error:
            raise ProviderError(
                f"{self._name} could not forecast this view: {type(error).__name__}: {error}"
            ) from error
        return conversion.answer(
            view,
            _as_list(predictions),
            instances=prepared.instances,
            target=target,
        )

    def _require_matching_window(self, view: ForecastView, persisted: Mapping[str, Any]) -> None:
        """The window this model learned is the only one it can be asked about.

        The engine sizes the inference view from the artifact's own record, so
        reaching here with a different window means the artifact and the request
        disagree — and a mismatched context is something Darts would truncate
        rather than refuse.
        """
        metadata = view.metadata
        wanted = (int(persisted["context"]), int(persisted["horizon"]))
        given = (metadata.context, metadata.horizon)
        mismatch = given[0] != wanted[0] or (self._horizon_bound_at_fit and given[1] != wanted[1])
        if mismatch:
            raise DataError(
                f"{self._name} was fitted on {wanted[0]} context steps and a horizon of "
                f"{wanted[1]}, and this view holds {given[0]} and {given[1]}; a global "
                f"model learns one window and cannot be asked about another"
            )

    # -- the native model ---------------------------------------------------

    def _instantiate(
        self, params: Mapping[str, Any], *, context: int, horizon: int, seed: int | None
    ) -> Any:
        """The native model the caller's parameters and the view jointly describe.

        The caller supplies the modeling parameters; the view supplies the shape.
        They cannot collide, because the shape is not something ``of.Model`` lets
        a caller pass — ``input_chunk_length`` and ``output_chunk_length`` name
        concepts OpenForecast owns.
        """
        settings = checked(params, self._parameters, self._name)
        compiled: dict[str, Any] = {}
        if "input_chunk_length" in self._constructor_parameters:
            compiled["input_chunk_length"] = context
        if "output_chunk_length" in self._constructor_parameters:
            compiled["output_chunk_length"] = horizon
        if "pl_trainer_kwargs" in self._constructor_parameters:
            compiled["pl_trainer_kwargs"] = dict(TRAINER_KWARGS)
        if seed is not None and "random_state" in self._constructor_parameters:
            compiled["random_state"] = seed
        try:
            return self._model_type()(**settings, **compiled)
        except (TypeError, ValueError) as error:
            raise RecipeError(f"{self._name} rejected {dict(params)}: {error}") from error

    def _load(self, path: Path) -> Any:
        """The fitted model a previous fit saved, or a refusal saying what is there."""
        try:
            return self._model_type().load(str(path))
        except Exception as error:
            raise ProviderError(
                f"the fitted state of {self._name} at {path} could not be loaded: "
                f"{type(error).__name__}: {error}"
            ) from error

    def __repr__(self) -> str:
        return f"DartsGlobalAdapter({self._name})"


def _as_list(predictions: Any) -> list[Any]:
    """Darts answers a list of series for a list of series, and one for one."""
    return list(predictions) if isinstance(predictions, list) else [predictions]


def _tide() -> Any:
    from darts.models import TiDEModel

    return TiDEModel


def _nhits() -> Any:
    from darts.models import NHiTSModel

    return NHiTSModel


#: The parameters of ``TiDEModel`` a caller may set. Deliberately a subset: the
#: window and the seed are OpenForecast's and are compiled from the view and the
#: fit plan, ``add_encoders`` would manufacture covariates OpenForecast did not
#: declare, and the optimizer, dataloader and trainer hooks are objects rather
#: than the JSON a recipe has to survive being written down as.
TIDE_PARAMETERS = (
    Parameter("hidden_size", int, "Width of the encoder and decoder layers.", minimum=1),
    Parameter("num_encoder_layers", int, "Dense encoder layers.", minimum=1),
    Parameter("num_decoder_layers", int, "Dense decoder layers.", minimum=1),
    Parameter("decoder_output_dim", int, "Width of the decoder's output.", minimum=1),
    Parameter("temporal_decoder_hidden", int, "Width of the temporal decoder.", minimum=1),
    Parameter("temporal_width_past", int, "Width past covariates are projected to.", minimum=0),
    Parameter("temporal_width_future", int, "Width future covariates are projected to.", minimum=0),
    Parameter("use_layer_norm", bool, "Apply layer normalization in the residual blocks."),
)

#: ``darts/tide``: dense encoder-decoder over the whole window, fitted across
#: every sample. The global model that consumes all three feature roles, which
#: is what makes it the one the point-in-time conformance cases run against.
TIDE = DartsGlobalAdapter(
    name="tide",
    display_name="TiDE",
    model_type=_tide,
    parameters=TIDE_PARAMETERS,
    features=FeatureCapabilities(observed=True, known=True, static=True),
)

#: The parameters of ``NHiTSModel`` a caller may set. ``pooling_kernel_sizes``
#: and ``n_freq_downsample`` are withheld because they are nested tuples whose
#: shape has to agree with the stack count — a recipe is a document, and a
#: parameter that can only be written down as a matrix does not belong in one.
NHITS_PARAMETERS = (
    Parameter("num_stacks", int, "Stacks of blocks.", minimum=1),
    Parameter("num_blocks", int, "Blocks per stack.", minimum=1),
    Parameter("num_layers", int, "Fully connected layers per block.", minimum=1),
    Parameter("layer_widths", int, "Width of every fully connected layer.", minimum=1),
    Parameter(
        "activation",
        str,
        "Activation of the block MLPs.",
        choices=(
            "ReLU",
            "RReLU",
            "PReLU",
            "ELU",
            "Softplus",
            "Tanh",
            "SELU",
            "LeakyReLU",
            "Sigmoid",
        ),
    ),
    Parameter("MaxPool1d", bool, "Pool with max rather than with an average."),
)

#: ``darts/nhits``: multi-rate hierarchical interpolation. Darts implements it as
#: a past-covariates model, so unlike ``nixtla/nhits`` it declares no support for
#: values known ahead of their event time — the same architecture, a different
#: capability, and the descriptor is where the difference is stated rather than
#: discovered.
NHITS = DartsGlobalAdapter(
    name="nhits",
    display_name="NHiTS",
    model_type=_nhits,
    parameters=NHITS_PARAMETERS,
    features=FeatureCapabilities(observed=True, known=False, static=False),
)
