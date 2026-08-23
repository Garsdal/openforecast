"""Local sktime forecasters: a ``SeriesView`` in, one fitted model per series out.

```text
fit        SeriesView   -> Forecaster().fit(series, fh) once per series
state      into/        -> series-<n>.zip per series + state.json
forecast   ForecastView -> predict(fh) -> the canonical forecast columns
```

A local forecaster is fitted on one series and what it learned about that series
says nothing about another. That is why the training contract is a series view
with a single origin, why the artifact cannot forecast an instance it never saw,
and why the horizon is asked for at inference rather than bound at fit — all of
which the descriptor states, so the engine refuses those requests before this
module is reached.

It is the same shape as ``nixtla/autoarima`` and ``darts/theta``, which is the
claim Step 14 makes about a third ecosystem: a local model reaches
point-in-time data at one selected origin and no more, whichever library it came
from.

One consequence of locality is worth being strict about. A fitted forecaster
continues the series it was fitted on, and ``predict`` with a relative horizon
means "the steps after the cutoff of that series". A forecast asked for at a
*later* origin is therefore not the model applied to newer data — it is the same
extrapolation, mislabeled. So the last event time of every series is persisted
and an origin that does not match it is refused with an explanation, rather than
answered with numbers that quietly forecast the wrong steps.

``sktime`` is imported inside the calls that need it rather than at module
scope: a handshake asks what this integration advertises, and answering that
should not pay for a library import.
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
from openforecast_sktime import conversion
from openforecast_sktime.parameters import Parameter, checked, named, schema_of
from openforecast_sktime.state import STATE_FILENAME, read_state, write_state

__all__ = ["THETA", "SktimeLocalAdapter"]


def _model_filename(index: int) -> str:
    """Where the forecaster fitted on the ``index``-th series is saved.

    Numbered rather than named after the series: a ``series_id`` is opaque and an
    instance key is the caller's, and neither is something to build a path out
    of. The state file holds the mapping. ``sktime``'s own ``save`` appends the
    ``.zip``, which is why the suffix is written down here too.
    """
    return f"series-{index}.zip"


class SktimeLocalAdapter:
    """One local sktime forecaster, as OpenForecast advertises and executes it."""

    def __init__(
        self,
        *,
        name: str,
        display_name: str,
        model_type: Callable[[], Any],
        parameters: Sequence[Parameter],
        defaults: Mapping[str, Any] | None = None,
    ) -> None:
        self._name = name
        self._display_name = display_name
        self._model_type = model_type
        self._parameters = named(parameters)
        self._defaults = dict(defaults or {})

    @property
    def name(self) -> str:
        return self._name

    def descriptor(self, provider: str) -> ModelDescriptor:
        """What the catalog and the engine are told about this model.

        Every capability is one the library actually has. It is univariate
        because the forecaster models one column; it takes a panel because a
        panel is many independent series to a model fitted per series; it takes
        no feature at all, because the Theta method is a function of the target's
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
                missing_values=MissingValueSupport.UNSUPPORTED,
            ),
            parameters_schema=schema_of(self._parameters),
        )

    # -- fit ----------------------------------------------------------------

    def fit(
        self, view: FitView, params: Mapping[str, Any], into: Path, *, seed: int | None
    ) -> None:
        """Fit one native forecaster per series, and persist each with what labels it."""
        del seed  # the forecasters exposed here are deterministic
        if not isinstance(view, SeriesView):
            raise ProviderError(
                f"{self._name} trains on one complete time series, so it cannot be fitted "
                f"from a {view.kind} view"
            )
        prepared = conversion.training_series(view)
        entries: list[dict[str, Any]] = []
        for index, (series_id, series) in enumerate(prepared.series.items()):
            model = self._instantiate(params)
            try:
                model.fit(series)
            except Exception as error:
                # A library refusing to fit this series is an execution failure
                # the caller can act on, not a bug in the boundary.
                raise ProviderError(
                    f"{self._name} could not be fitted on the series of instance "
                    f"{prepared.instances[series_id]}: {type(error).__name__}: {error}"
                ) from error
            model.save(_saved_as(into / _model_filename(index)))
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
        target = str(persisted["target"])

        horizon = view.metadata.horizon
        answers: dict[str, Any] = {}
        instances: dict[str, conversion.InstanceKey] = {}
        labels = conversion.instance_labels(view.instances)
        for label, instance in zip(labels, view.instances, strict=True):
            entry = self._entry_for(instance, fitted, view)
            model = self._load(state / str(entry["file"]))
            try:
                answers[label] = model.predict(fh=_relative_horizon(horizon))
            except Exception as error:
                raise ProviderError(
                    f"{self._name} could not forecast instance {instance}: "
                    f"{type(error).__name__}: {error}"
                ) from error
            instances[label] = instance
        return conversion.answer(
            view, _as_panel(answers, target), instances=instances, target=target
        )

    def _entry_for(
        self,
        instance: conversion.InstanceKey,
        fitted: Mapping[tuple[Any, ...], Mapping[str, Any]],
        view: ForecastView,
    ) -> Mapping[str, Any]:
        """The fitted series this instance is asking about, at this origin.

        A local model continues the series it saw; it does not re-read a new one.
        ``predict`` extrapolates from the cutoff of the fit, so answering at a
        different origin would produce the right numbers for the wrong event
        times. Refusing is the only honest answer available without fitting
        again, which is a fit and belongs in ``fit``.
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

    def _instantiate(self, params: Mapping[str, Any]) -> Any:
        """The native forecaster the caller's parameters describe.

        Over the adapter's own defaults, which are a *narrower* starting point
        than the library's rather than a different model: what they change is
        documented on the parameter they set, and stating the parameter puts the
        library's behavior back. Nothing here is silently unreachable.
        """
        settings = {**self._defaults, **checked(params, self._parameters, self._name)}
        try:
            return self._model_type()(**settings)
        except (TypeError, ValueError) as error:
            raise RecipeError(f"{self._name} rejected {dict(params)}: {error}") from error

    def _load(self, path: Path) -> Any:
        """The forecaster a previous fit saved, or a refusal saying what is there."""
        from sktime.base import load

        try:
            return load(path)
        except Exception as error:
            raise ProviderError(
                f"the fitted state of {self._name} at {path} could not be loaded: "
                f"{type(error).__name__}: {error}"
            ) from error

    def __repr__(self) -> str:
        return f"SktimeLocalAdapter({self._name})"


def _saved_as(path: Path) -> Path:
    """What to hand ``save``, which appends the suffix this file already names."""
    return path.with_suffix("")


def _relative_horizon(horizon: int) -> Any:
    """``1 .. horizon``, as the steps-after-the-cutoff sktime calls a horizon."""
    from sktime.forecasting.base import ForecastingHorizon

    return ForecastingHorizon(list(range(1, horizon + 1)), is_relative=True)


def _as_panel(answers: Mapping[str, Any], target: str) -> Any:
    """The per-series answers, as the one panel :func:`conversion.answer` reads.

    A local fit is many forecasters and a panel is how their answers are handed
    back together, so the two execution models converge on one shape here rather
    than in two copies of the labeling code.
    """
    import pandas as pd

    frames = [
        pd.DataFrame(
            {target: [float(value) for value in series.to_numpy()]},
            index=pd.MultiIndex.from_product(
                [[label], pd.DatetimeIndex(series.index)], names=conversion.PANEL_LEVELS
            ),
        )
        for label, series in answers.items()
    ]
    return pd.concat(frames)


def _theta() -> Any:
    from sktime.forecasting.theta import ThetaForecaster

    return ThetaForecaster


#: The parameters of ``ThetaForecaster`` a caller may set. ``sp`` is the season
#: length in steps of the data's frequency, which is a modeling choice rather
#: than something OpenForecast owns — unlike the horizon, which is the forecast
#: task's, and is therefore not here.
THETA_PARAMETERS = (
    Parameter("sp", int, "Steps in one seasonal period. 1 fits no seasonality.", minimum=1),
    Parameter(
        "deseasonalize",
        bool,
        "Remove seasonality before fitting the trend. Off by default here: "
        "sktime's deseasonalization is multiplicative and undefined on a series "
        "that touches zero.",
    ),
)

#: ``sktime/theta``: the Theta method, fitted per series.
THETA = SktimeLocalAdapter(
    name="theta",
    display_name="Theta",
    model_type=_theta,
    parameters=THETA_PARAMETERS,
    # A multiplicative deseasonalization raises on a series holding a zero, and
    # a load that goes to zero is a perfectly ordinary series. So a caller asking
    # for one asks for it by name; the alternative is a model that refuses data
    # for a reason nobody stated.
    defaults={"deseasonalize": False},
)
