"""Panel sktime forecasters: a ``SequenceView`` in, one pooled model out.

```text
fit        SequenceView -> make_reduction(..., pooling="global").fit(panel, fh)
state      into/        -> model.zip + state.json
forecast   ForecastView -> update(new panel), predict(fh) -> canonical columns
```

sktime is the ecosystem that says out loud what the other two leave implicit: a
forecaster handed a panel is *vectorized* over its instances unless it pools
across them. ``pooling="global"`` is that statement, and it is what makes this a
global model — one set of parameters learned from every sample at once, where a
sample is one ``context -> horizon`` window at one forecast origin.

The compilation is the whole point of the design, and it is three lines:

```text
WindowPlan(context=168)  ->  window_length=168
one sample               ->  one panel unit
horizon=72               ->  ForecastingHorizon(1..72)
```

None of the three is something a caller states twice. The context length is
OpenForecast's, because the ``ViewPlanner`` had to know it to cut the samples at
all; passing it again as a native parameter is refused by ``of.Model`` before
this module is reached. And the samples are the view's, so this adapter never
learns which instance or which origin one came from — which is precisely why a
``TimeSeriesFrame`` and a ``ForecastDataset`` are indistinguishable from in here.

Three consequences of *this* global model are worth naming, and two of them
differ from the neural models of Steps 12 and 13.

**The horizon is not bound at fit.** A recursive reduction learns one step and
rolls, so the artifact answers whatever horizon it is asked for — where
``nixtla/nhits`` and ``darts/tide`` bake theirs into the architecture. The
descriptor is where that difference is stated, and the engine reads it rather
than assuming a global model is horizon-bound.

**An unseen instance is forecastable.** Pooled parameters are what makes that
true, so ``supports_unseen_instances`` is declared — and asserted in the tests,
because a capability nobody exercised is a claim rather than a capability.

**A forecast at a new origin is an update, not a refit.** sktime forecasters
hold the series they were fitted on, and inference happens by handing the fitted
model the panel of the origin being asked about with ``update_params=False`` —
the parameters are untouched, only the windows the forecast rolls from are new.
The answer therefore also covers the training samples, and the rows of the
instances actually asked about are selected by label rather than assumed.

This specialized pooled protocol remains beside the reflected local forecaster
driver. The catalog selects it as an override; fit and forecast contain no
model-name dispatch.
"""

from __future__ import annotations

import copy
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
from openforecast_sktime import conversion
from openforecast_sktime.parameters import Parameter, checked, named, schema_of
from openforecast_sktime.state import STATE_FILENAME, read_state, write_state

__all__ = ["POOLED_TREES", "SktimePanelAdapter"]

#: What sktime's own ``save`` is pointed at. It writes one zip, so this is a name
#: in the directory the engine owns rather than a directory of its own.
MODEL_FILENAME = "model.zip"


class SktimePanelAdapter:
    """One pooled sktime forecaster, as OpenForecast advertises and executes it."""

    def __init__(
        self,
        *,
        name: str,
        display_name: str,
        regressor: Callable[[], Any],
        parameters: Sequence[Parameter],
        features: FeatureCapabilities,
    ) -> None:
        self._name = name
        self._display_name = display_name
        self._regressor = regressor
        self._parameters = named(parameters)
        self._features = features

    @property
    def name(self) -> str:
        return self._name

    def descriptor(self, provider: str) -> ModelDescriptor:
        """What the catalog and the engine are told about this model.

        Every capability is one the library actually has. It learns across
        origins because that is what pooling does with many windows; it needs a
        context length because ``window_length`` has no defensible default; it
        does *not* bind its horizon, because a recursive reduction rolls its own
        prediction forward; it takes an unseen instance because the parameters
        are pooled rather than fitted per series; and it consumes the feature
        roles that survive into an exogenous frame reaching past the origin.

        Missing values are the one thing it cannot take as they come. Point-in-
        time data is full of them and a regressor fitted on a lagged window of
        NaNs learns nothing, so the declaration is ``REQUIRES_TRANSFORM``: the
        caller writes an imputation down, where the artifact records it, or the
        request is refused. Filling them in here is the silent imputation rule 5
        forbids.
        """
        return ModelDescriptor(
            ref=ModelRef.parse(f"{provider}/{self._name}"),
            provider=provider,
            display_name=self._display_name,
            lifecycle=ModelLifecycle.trainable(),
            training=TrainingContract.sequences(
                horizon_bound_at_fit=False, supports_unseen_instances=True
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
        """Fit one pooled model over every sample, and persist what labels it."""
        if not isinstance(view, SequenceView):
            raise ProviderError(
                f"{self._name} trains on context -> horizon sequences, so it cannot be "
                f"fitted from a {view.kind} view"
            )
        schema = view.schema
        prepared = conversion.sequence_panel(view, features=self._features)
        model = self._instantiate(params, context=schema.context, seed=seed)
        try:
            model.fit(y=prepared.y, X=prepared.X, fh=_relative_horizon(schema.horizon))
        except Exception as error:
            # A library refusing to train on these windows is an execution
            # failure the caller can act on, not a bug in the boundary.
            raise ProviderError(
                f"{self._name} could not be fitted on this data: {type(error).__name__}: {error}"
            ) from error
        model.save(_saved_as(into / MODEL_FILENAME))
        write_state(
            into / STATE_FILENAME,
            {
                "model": self._name,
                "target": prepared.target,
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
        self._require_matching_context(view, persisted)

        target = str(persisted["target"])
        prepared = conversion.forecast_panel(
            view,
            target=target,
            known=[str(name) for name in persisted["known"]],
            static=[str(name) for name in persisted["static"]],
        )
        # Deep-copied because the fitted artifact is read once and asked about
        # many origins: updating the loaded object in place would make the
        # second question depend on the first.
        model = copy.deepcopy(self._load(state / MODEL_FILENAME))
        try:
            model.update(y=prepared.y, X=prepared.X, update_params=False)
            predicted = model.predict(fh=_relative_horizon(view.metadata.horizon), X=prepared.X)
        except Exception as error:
            raise ProviderError(
                f"{self._name} could not forecast this view: {type(error).__name__}: {error}"
            ) from error
        return conversion.answer(view, predicted, instances=prepared.instances, target=target)

    def _require_matching_context(self, view: ForecastView, persisted: Mapping[str, Any]) -> None:
        """The window this model rolls from is the one it learned to roll from.

        The engine sizes the inference view from the artifact's own record, so
        reaching here with a different context means the artifact and the request
        disagree — and a short window is something the library would pad or
        truncate rather than refuse. The horizon is deliberately not checked: a
        recursive reduction answers any horizon, which is what the descriptor
        declares.
        """
        wanted = int(persisted["context"])
        given = view.metadata.context
        if given != wanted:
            raise DataError(
                f"{self._name} rolls its forecast from {wanted} context steps and this view "
                f"holds {given}; a pooled model learns one window length and cannot be asked "
                f"about another"
            )

    # -- the native model ---------------------------------------------------

    def _instantiate(self, params: Mapping[str, Any], *, context: int, seed: int | None) -> Any:
        """The pooled forecaster the caller's parameters and the view jointly describe.

        The caller supplies the regressor's parameters; the view supplies the
        shape. They cannot collide, because the shape is not something
        ``of.Model`` lets a caller pass — ``window_length`` names a concept
        OpenForecast owns.
        """
        from sktime.forecasting.compose import make_reduction

        settings = checked(params, self._parameters, self._name)
        try:
            regressor = self._regressor()(**settings, random_state=seed)
        except (TypeError, ValueError) as error:
            raise RecipeError(f"{self._name} rejected {dict(params)}: {error}") from error
        return make_reduction(
            regressor,
            strategy="recursive",
            window_length=context,
            pooling="global",
        )

    def _load(self, path: Path) -> Any:
        """The fitted model a previous fit saved, or a refusal saying what is there."""
        from sktime.base import load

        try:
            return load(path)
        except Exception as error:
            raise ProviderError(
                f"the fitted state of {self._name} at {path} could not be loaded: "
                f"{type(error).__name__}: {error}"
            ) from error

    def __repr__(self) -> str:
        return f"SktimePanelAdapter({self._name})"


def _saved_as(path: Path) -> Path:
    """What to hand ``save``, which appends the suffix this file already names."""
    return path.with_suffix("")


def _relative_horizon(horizon: int) -> Any:
    """``1 .. horizon``, as the steps-after-the-cutoff sktime calls a horizon."""
    from sktime.forecasting.base import ForecastingHorizon

    return ForecastingHorizon(list(range(1, horizon + 1)), is_relative=True)


def _hist_gradient_boosting() -> Any:
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor


#: The parameters of ``HistGradientBoostingRegressor`` a caller may set.
#: Deliberately a subset: the window and the seed are OpenForecast's and are
#: compiled from the view and the fit plan, and the callbacks and warm-start
#: hooks are objects rather than the JSON a recipe has to survive being written
#: down as.
POOLED_TREES_PARAMETERS = (
    Parameter("max_iter", int, "Boosting iterations.", minimum=1),
    Parameter("learning_rate", float, "Shrinkage applied to each tree.", minimum=0),
    Parameter("max_depth", int, "Maximum depth of a tree.", minimum=1),
    Parameter("max_leaf_nodes", int, "Maximum leaves of a tree.", minimum=2),
    Parameter("min_samples_leaf", int, "Minimum training rows in a leaf.", minimum=1),
    Parameter("l2_regularization", float, "L2 penalty on the leaf values.", minimum=0),
)

#: ``sktime/pooled-trees``: gradient-boosted trees, reduced recursively and
#: pooled across every sample in the panel. The global model of this integration,
#: and the one whose generated conformance cases include real forecast vintages.
POOLED_TREES = SktimePanelAdapter(
    name="pooled-trees",
    display_name="Pooled gradient-boosted trees",
    regressor=_hist_gradient_boosting,
    parameters=POOLED_TREES_PARAMETERS,
    features=FeatureCapabilities(observed=False, known=True, static=True),
)
