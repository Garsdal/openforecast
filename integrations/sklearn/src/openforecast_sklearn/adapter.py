"""One scikit-learn estimator, as OpenForecast advertises and executes it.

```text
fit        TabularView   -> estimator.fit(X, y)
state      into/         -> estimator.pkl + metadata.json
forecast   ForecastView  -> estimator.predict(X) -> canonical columns
```

Those really are the two lines that execute a model. Everything else in this file
is the declaration — what the model can be given, which parameters it takes — and
the bookkeeping that labels an answer, because ``predict`` returns numbers and no
statement about what they are about.

The reason it is two lines is the point of the step. A ``TabularView`` already
holds one row per ``instance × origin × lead``, carrying the feature values that
existed *at that origin* and the outcome that event time turned out to have. The
reduction — forecast origin, target time, lead, information vintage, truth
alignment — happened in the ``ViewPlanner``, once, on OpenForecast's side of the
boundary. There is nothing left for a provider to reinterpret, which is why this
adapter never learns which origin a row came from and cannot tell an event-time
frame from real forecast vintages.

Three properties of *this* execution path are worth naming.

**The horizon is not bound at fit.** One row is one lead, and the lead is not a
feature, so a fitted estimator answers a row about lead 3 exactly as it answers a
row about lead 96. The descriptor declares that, and the engine reads it rather
than assuming that a model fitted with ``horizon=72`` can only be asked for 72 —
which is the same declaration ``sktime/pooled-trees`` makes for a completely
different reason.

**An unseen instance is forecastable.** There is one set of parameters, learned
from every row, and an instance the fit never saw is a row like any other. That
holds only because the instance keys are *not* in ``X``: a ``TabularView`` keeps
them in its ``keys`` table, so an estimator cannot have learned a zone as a
feature by accident. A caller who wants that asks for it as a static feature.

**Missing values arrive as they are.** Point-in-time data is full of them — a
feed that had not published yet at an origin — and ``HistGradientBoostingRegressor``
routes ``NaN`` down a learned default branch rather than needing it filled in. So
the declaration is ``NATIVE`` and no imputation happens anywhere on this path.
That is the one capability that makes this estimator, rather than a ridge
regression, the honest first one to expose.

The catalog discovers sklearn once and injects the selected estimator class
into this protocol adapter. Fit and forecast contain no model-name dispatch.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa

from openforecast.errors import ProviderError, RecipeError
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
from openforecast.views import FitView, ForecastView, TabularView
from openforecast_sklearn import conversion
from openforecast_sklearn.parameters import Parameter, checked, named, schema_of
from openforecast_sklearn.state import (
    ESTIMATOR_FILENAME,
    METADATA_FILENAME,
    read_estimator,
    read_metadata,
    write_estimator,
    write_metadata,
)

__all__ = ["HIST_GRADIENT_BOOSTING", "SklearnAdapter"]


class SklearnAdapter:
    """A scikit-learn regressor fitted on supervised forecasting rows."""

    def __init__(
        self,
        *,
        name: str,
        display_name: str,
        estimator: Callable[[], Any],
        parameters: Sequence[Parameter],
        missing_values: MissingValueSupport,
        seeded: bool = True,
    ) -> None:
        self._name = name
        self._display_name = display_name
        self._estimator = estimator
        self._parameters = named(parameters)
        self._missing_values = missing_values
        self._seeded = seeded

    @property
    def name(self) -> str:
        return self._name

    def descriptor(self, provider: str) -> ModelDescriptor:
        """What the catalog and the engine are told about this model.

        Every capability is one this execution path actually has. It learns
        across origins, because that is what a tabular view of several vintages
        *is*; it sizes no context window, because a row is not a window; it does
        not bind its horizon, because a lead is not a feature; and it takes an
        instance it never saw, because the parameters are shared across every row.

        The three feature roles are all declared, and the observed one deserves a
        sentence. A tabular row describes an event time *after* its origin, so an
        observed feature — a measurement — has no value there and a
        ``TabularView`` does not offer it as a column at all. Declaring
        ``observed=False`` would advertise a refusal OpenForecast does not make:
        data carrying observed features is accepted, and what those features hold
        reaches a row only if the caller carries it as a known feature. Declaring
        it here says the data is welcome, not that the column becomes a feature.
        """
        return ModelDescriptor(
            ref=ModelRef.parse(f"{provider}/{self._name}"),
            provider=provider,
            display_name=self._display_name,
            lifecycle=ModelLifecycle.trainable(),
            training=TrainingContract.tabular(),
            capabilities=ModelCapabilities(
                instances=InstanceCapabilities(single=True, panel=True),
                targets=TargetCapabilities(univariate=True, multivariate=False),
                features=FeatureCapabilities(observed=True, known=True, static=True),
                outputs=OutputCapabilities(point=True),
                missing_values=self._missing_values,
            ),
            parameters_schema=schema_of(self._parameters),
        )

    # -- fit ----------------------------------------------------------------

    def fit(
        self, view: FitView, params: Mapping[str, Any], into: Path, *, seed: int | None
    ) -> None:
        """Fit the estimator on every supervised row, and persist what labels ``X``."""
        if not isinstance(view, TabularView):
            raise ProviderError(
                f"{self._name} trains on supervised rows, so it cannot be fitted from a "
                f"{view.kind} view"
            )
        prepared = conversion.design_matrix(view)
        estimator = self._instantiate(params, seed=seed)
        try:
            estimator.fit(prepared.X, prepared.y)
        except Exception as error:
            # A library refusing to train on these rows is an execution failure
            # the caller can act on, not a bug in the boundary.
            raise ProviderError(
                f"{self._name} could not be fitted on this data: {type(error).__name__}: {error}"
            ) from error
        write_estimator(into / ESTIMATOR_FILENAME, estimator)
        write_metadata(
            into / METADATA_FILENAME,
            {
                "model": self._name,
                "target": prepared.target,
                # The order is the contract: an estimator has positions, not names.
                "features": list(prepared.features),
                "known": list(prepared.known),
                "static": list(prepared.static),
                "rows": int(prepared.X.shape[0]),
                "horizon": view.schema.horizon,
            },
        )

    # -- forecast -----------------------------------------------------------

    def forecast(self, view: ForecastView, output: Mapping[str, Any], state: Path) -> pa.Table:
        """The ``horizon`` steps after this origin, for every instance in the view."""
        kind = output.get("kind", "point")
        if kind != "point":
            raise ProviderError(f"{self._name} produces point forecasts, not {kind}")
        persisted = read_metadata(state / METADATA_FILENAME, self._name)

        rows = conversion.inference_matrix(
            view,
            features=[str(name) for name in persisted["features"]],
            known=[str(name) for name in persisted["known"]],
            static=[str(name) for name in persisted["static"]],
        )
        estimator = read_estimator(state / ESTIMATOR_FILENAME, self._name)
        try:
            predicted = estimator.predict(rows.X)
        except Exception as error:
            raise ProviderError(
                f"{self._name} could not forecast this view: {type(error).__name__}: {error}"
            ) from error
        return conversion.answer(view, predicted, rows=rows, target=str(persisted["target"]))

    # -- the native estimator -----------------------------------------------

    def _instantiate(self, params: Mapping[str, Any], *, seed: int | None) -> Any:
        """The estimator the caller's parameters describe.

        The seed is OpenForecast's — it is a property of the fit rather than of
        the model, and ``of.FitPlan(seed=...)`` is where a caller states it — so
        ``random_state`` is not among the parameters advertised and cannot be
        passed twice.
        """
        settings = checked(params, self._parameters, self._name)
        if self._seeded:
            settings["random_state"] = seed
        try:
            return self._estimator()(**settings)
        except (TypeError, ValueError) as error:
            raise RecipeError(f"{self._name} rejected {dict(params)}: {error}") from error

    def __repr__(self) -> str:
        return f"SklearnAdapter({self._name})"


def _hist_gradient_boosting() -> Any:
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor


#: The parameters of ``HistGradientBoostingRegressor`` a caller may set.
#: Deliberately a subset: the seed is OpenForecast's and comes from the fit plan,
#: and the callbacks, warm-start hooks and monotonic constraints are objects
#: rather than the JSON a recipe has to survive being written down as.
HIST_GRADIENT_BOOSTING_PARAMETERS = (
    Parameter("max_iter", int, "Boosting iterations.", minimum=1),
    Parameter("learning_rate", float, "Shrinkage applied to each tree.", minimum=0),
    Parameter("max_depth", int, "Maximum depth of a tree.", minimum=1),
    Parameter("max_leaf_nodes", int, "Maximum leaves of a tree.", minimum=2),
    Parameter("min_samples_leaf", int, "Minimum training rows in a leaf.", minimum=1),
    Parameter("l2_regularization", float, "L2 penalty on the leaf values.", minimum=0),
    Parameter("max_bins", int, "Histogram bins per feature.", minimum=2, maximum=255),
)

#: ``sklearn/hist-gradient-boosting``: histogram-based gradient boosting, fitted
#: on the supervised rows of a ``TabularView``. The first estimator this
#: integration exposes, chosen for one capability: it reads ``NaN`` as a branch
#: rather than as an error, which is what a point-in-time design matrix is full
#: of.
HIST_GRADIENT_BOOSTING = SklearnAdapter(
    name="hist-gradient-boosting",
    display_name="Histogram-based gradient boosting",
    estimator=_hist_gradient_boosting,
    parameters=HIST_GRADIENT_BOOSTING_PARAMETERS,
    missing_values=MissingValueSupport.NATIVE,
)
