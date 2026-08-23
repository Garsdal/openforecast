"""Local Darts models: a ``SeriesView`` in, one fitted model per series out.

```text
fit        SeriesView   -> Model().fit(TimeSeries) once per series
state      into/        -> series-<n>.pkl per series + state.json
forecast   ForecastView -> predict(n) -> the canonical forecast columns
```

A Darts ``LocalForecastingModel`` is *local*: every series is fitted on its own,
and what is learned about one says nothing about another. That is why the
training contract is a series view with a single origin, why the artifact cannot
forecast an instance it never saw, and why the horizon is asked for at inference
rather than bound at fit — all of which the descriptor states, so the engine
refuses those requests before this module is reached.

It is also the same shape as ``nixtla/autoarima``, which is the claim: a local
model reaches point-in-time data at one selected origin and no more, whichever
library it came from.

One consequence of locality is worth being strict about. A fitted Theta
continues the series it was fitted on, so ``predict(n)`` means "the n steps
after the last observation seen at fit time". A forecast asked for at a *later*
origin is therefore not the model applied to newer data — it is the same
extrapolation, mislabeled. So the last event time of every series is persisted
and an origin that does not match it is refused with an explanation, rather than
answered with numbers that quietly forecast the wrong steps.

``Theta`` is the first model exposed. Adding ``FourTheta`` or ``ExponentialSmoothing``
is another :class:`DartsLocalAdapter` beside it, which is the point of the
parameters being declared as data.

The catalog discovers Darts' local classes once and injects the selected class
into this adapter. Fit and forecast contain no model-name dispatch.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
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
from openforecast.views import FitView, ForecastView, SeriesView
from openforecast_darts import conversion
from openforecast_darts.parameters import Parameter, checked, named, schema_of
from openforecast_darts.state import STATE_FILENAME, read_state, write_state

__all__ = ["THETA", "DartsLocalAdapter"]


def _model_filename(index: int) -> str:
    """Where the model fitted on the ``index``-th series is pickled.

    Numbered rather than named after the series: a ``series_id`` is opaque and an
    instance key is the caller's, and neither is something to build a path out
    of. The state file holds the mapping.
    """
    return f"series-{index}.pkl"


class DartsLocalAdapter:
    """One local Darts model, as OpenForecast advertises and executes it."""

    def __init__(
        self,
        *,
        name: str,
        display_name: str,
        model_type: Callable[[], Any],
        parameters: Sequence[Parameter],
        compile_params: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
        missing_values: MissingValueSupport = MissingValueSupport.UNSUPPORTED,
        seeded: bool = False,
    ) -> None:
        self._name = name
        self._display_name = display_name
        self._model_type = model_type
        self._parameters = named(parameters)
        self._compile = compile_params
        self._missing_values = missing_values
        self._seeded = seeded

    @property
    def name(self) -> str:
        return self._name

    def descriptor(self, provider: str) -> ModelDescriptor:
        """What the catalog and the engine are told about this model.

        Every capability is one the library actually has. It is univariate
        because Theta forecasts one column; it takes a panel because a panel is
        many independent series to a model fitted per series; it takes no
        feature at all, because a Theta forecast is a function of the target's
        own history and nothing else; and it cannot see a missing value, so data
        with gaps is refused before it gets here unless the caller asked for an
        imputation.
        """
        return ModelDescriptor(
            ref=ModelRef.parse(f"{provider}/{self._name}"),
            provider=provider,
            display_name=self._display_name,
            lifecycle=ModelLifecycle.trainable(),
            training=TrainingContract.series(),
            capabilities=ModelCapabilities(
                instances=InstanceCapabilities(single=True, panel=True),
                targets=TargetCapabilities(univariate=True, multivariate=False),
                features=FeatureCapabilities(observed=False, known=False, static=False),
                outputs=OutputCapabilities(point=True),
                missing_values=self._missing_values,
            ),
            parameters_schema=schema_of(self._parameters),
        )

    # -- fit ----------------------------------------------------------------

    def fit(
        self, view: FitView, params: Mapping[str, Any], into: Path, *, seed: int | None
    ) -> None:
        """Fit one native model per series, and persist each with what labels it."""
        if not isinstance(view, SeriesView):
            raise ProviderError(
                f"{self._name} trains on one complete time series, so it cannot be fitted "
                f"from a {view.kind} view"
            )
        prepared = conversion.training_series(view)
        entries: list[dict[str, Any]] = []
        for index, (series_id, series) in enumerate(prepared.series.items()):
            model = self._instantiate(params, seed=seed)
            try:
                model.fit(series)
            except Exception as error:
                # A library refusing to fit this series is an execution failure
                # the caller can act on, not a bug in the boundary.
                raise ProviderError(
                    f"{self._name} could not be fitted on the series of instance "
                    f"{prepared.instances[series_id]}: {type(error).__name__}: {error}"
                ) from error
            model.save(str(into / _model_filename(index)))
            entries.append(
                {
                    "file": _model_filename(index),
                    "key": list(prepared.instances[series_id]),
                    "last_event_time": prepared.last_event_times[series_id].isoformat(),
                }
            )
        write_state(
            into / STATE_FILENAME,
            {
                "model": self._name,
                "target": prepared.target,
                "frequency": prepared.frequency,
                "series": entries,
            },
        )

    # -- forecast -----------------------------------------------------------

    def forecast(self, view: ForecastView, output: Mapping[str, Any], state: Path) -> pa.Table:
        """The next ``horizon`` steps of every series the view asks about."""
        kind = output.get("kind", "point")
        if kind != "point":
            raise ProviderError(f"{self._name} produces point forecasts, not {kind}")
        persisted = read_state(state / STATE_FILENAME, self._name)
        fitted = {
            tuple(entry["key"]): entry
            for entry in persisted["series"]  # pyright: ignore[reportUnknownVariableType]
        }

        instances = view.instances
        predictions: list[Any] = []
        for instance in instances:
            entry = self._entry_for(instance, fitted, view)
            model = self._load(state / str(entry["file"]))
            try:
                predictions.append(model.predict(view.metadata.horizon))
            except Exception as error:
                raise ProviderError(
                    f"{self._name} could not forecast instance {instance}: "
                    f"{type(error).__name__}: {error}"
                ) from error
        return conversion.answer(
            view, predictions, instances=instances, target=str(persisted["target"])
        )

    def _entry_for(
        self,
        instance: conversion.InstanceKey,
        fitted: Mapping[tuple[Any, ...], Mapping[str, Any]],
        view: ForecastView,
    ) -> Mapping[str, Any]:
        """The fitted series this instance is asking about, at this origin.

        A local model continues the series it saw; it does not re-read a new one.
        ``predict`` extrapolates from the last observation of the fit, so
        answering at a different origin would produce the right numbers for the
        wrong event times. Refusing is the only honest answer available without
        fitting again, which is a fit and belongs in ``fit``.
        """
        entry = fitted.get(instance)
        if entry is None:
            raise DataError(
                f"{self._name} is fitted per series, so it has no model for instance "
                f"{instance}; it was fitted on {sorted(str(key) for key in fitted)}"
            )
        end = datetime.fromisoformat(str(entry["last_event_time"]))
        if end != view.origin_time:
            raise DataError(
                f"{self._name} forecasts the steps after the last observation it was "
                f"fitted on, which for instance {instance} is {end.isoformat()}; this "
                f"forecast is made at the origin {view.origin_time.isoformat()}. Fit at "
                f"that origin instead — a local model is refitted rather than reused"
            )
        return entry

    # -- the native model ---------------------------------------------------

    def _instantiate(self, params: Mapping[str, Any], *, seed: int | None) -> Any:
        """The native model the caller's parameters describe."""
        settings = checked(params, self._parameters, self._name)
        compiled = dict(self._compile(settings)) if self._compile is not None else settings
        if self._seeded:
            compiled["random_state"] = seed
        try:
            return self._model_type()(**compiled)
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
        return f"DartsLocalAdapter({self._name})"


def _theta() -> Any:
    from darts.models import Theta

    return Theta


def _theta_settings(settings: Mapping[str, Any]) -> Mapping[str, Any]:
    """``season_mode`` as the enum Darts takes, from the string a recipe holds.

    The only translation either local model needs: a recipe is a document, and a
    Python enum member is not something a document can carry — so the choices the
    descriptor advertises are strings and this is where they land.
    """
    from darts.utils.utils import SeasonalityMode

    modes = {mode.name.lower(): mode for mode in SeasonalityMode}
    compiled = dict(settings)
    mode = compiled.get("season_mode")
    if mode is not None:
        compiled["season_mode"] = modes[str(mode)]
    return compiled


#: The parameters of ``Theta`` a caller may set. All of them: Theta is a small
#: model and none of its arguments name something OpenForecast owns.
THETA_PARAMETERS = (
    Parameter("theta", int, "Weight of the second theta line. 2 is the classic method."),
    Parameter("seasonality_period", int, "Steps in one season. Inferred when unset.", minimum=1),
    Parameter(
        "season_mode",
        str,
        "How the seasonal component combines with the trend.",
        choices=("multiplicative", "additive", "none"),
    ),
)

#: ``darts/theta``: the Theta method, fitted per series.
THETA = DartsLocalAdapter(
    name="theta",
    display_name="Theta",
    model_type=_theta,
    parameters=THETA_PARAMETERS,
    compile_params=_theta_settings,
)
