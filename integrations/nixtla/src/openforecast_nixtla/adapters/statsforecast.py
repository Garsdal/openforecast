"""StatsForecast models: a ``SeriesView`` in, point forecasts or quantiles out.

```text
fit        SeriesView   -> StatsForecast(models=[AutoARIMA(...)]).fit(long frame)
state      into/        -> statsforecast.pkl + state.json
forecast   ForecastView -> predict(h)            -> the canonical point rows
                        -> predict(h, level=[…]) -> the canonical quantile rows
```

A StatsForecast prediction interval *is* a pair of quantiles of the predictive
distribution, under another name and in percent. So a quantile request is served
by asking the library for the intervals that carry the levels asked for — the 0.1
and the 0.9 are the bounds of its 80% interval, and the 0.5 of a symmetric
distribution is the point forecast itself — and nothing is invented on the way.
Samples are refused: what the fitted model can produce is interval bounds, and
paths drawn through them would be paths it never produced.

A StatsForecast model is *local*: every series is fitted on its own, and what is
learned about one says nothing about another. That is why the training contract
is a series view with a single origin, why the artifact cannot forecast an
instance it never saw, and why the horizon is asked for at inference rather than
bound at fit — all of which the descriptor states, so the engine refuses those
requests before this module is reached.

One consequence of locality is worth being strict about. A fitted ARIMA
continues the series it was fitted on, so ``predict(h)`` means "the h steps
after the last observation seen at fit time". A forecast asked for at a *later*
origin is therefore not the model applied to newer data — it is the same
extrapolation, mislabeled. So the last event time of every series is persisted
and an origin that does not match it is refused with an explanation, rather than
answered with numbers that quietly forecast the wrong steps.

``AutoARIMA`` is the first model exposed. Adding ``AutoETS`` or ``AutoTheta`` is
another :class:`StatsForecastAdapter` beside it, which is the point of the
parameters being declared as data.

``statsforecast`` is imported inside the two calls that need it rather than at
module scope. A handshake — which is what installing a provider and listing
models does — only asks what this integration advertises, and paying for a JIT
compiler to answer that would make discovery slow for no reason.
"""

from __future__ import annotations

import re
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
from openforecast_nixtla import conversion
from openforecast_nixtla.parameters import Parameter, checked, named, schema_of
from openforecast_nixtla.state import STATE_FILENAME, read_state, write_state

__all__ = ["AUTOARIMA", "StatsForecastAdapter"]

#: The pickled ``StatsForecast`` object, written by its own ``save``.
MODEL_FILENAME = "statsforecast.pkl"

#: The quantile a symmetric predictive distribution has at its point forecast.
MEDIAN = 0.5

#: Decimal places an interval level is rounded to before it is matched against
#: the columns the library returned. ``(1 - 2 * 0.1) * 100`` is 80.00000000000001
#: in binary floating point, and that is not a different interval.
_CONFIDENCE_PRECISION = 6

#: ``AutoARIMA-lo-80``, ``AutoARIMA-hi-95.5`` — one bound of one interval.
_INTERVAL_COLUMN = re.compile(r".*-(?P<side>lo|hi)-(?P<confidence>\d+(?:\.\d+)?)$")


class StatsForecastAdapter:
    """One StatsForecast model, as OpenForecast advertises and executes it."""

    def __init__(
        self,
        *,
        name: str,
        display_name: str,
        build: Callable[..., Any],
        parameters: Sequence[Parameter],
        exogenous: bool,
        quantiles: bool,
    ) -> None:
        self._name = name
        self._display_name = display_name
        self._build = build
        self._parameters = named(parameters)
        self._exogenous = exogenous
        self._quantiles = quantiles

    @property
    def name(self) -> str:
        return self._name

    def descriptor(self, provider: str) -> ModelDescriptor:
        """What the catalog and the engine are told about this model.

        Every capability is one the library actually has. It is univariate
        because a StatsForecast model forecasts one column; it takes a panel
        because a panel is many independent series to it; it takes known
        features as exogenous regressors and no others, because a value that
        stops at the forecast origin is not something an ARIMA can condition a
        future step on; and it cannot see a missing value, so data with gaps is
        refused before it gets here unless the caller asked for an imputation.

        Quantiles are declared where the fitted model actually has a predictive
        distribution to read them from — ``AutoARIMA`` does, through the
        prediction intervals it can produce. Samples are declared by nothing
        here: drawing paths through fitted interval bounds would be inventing
        paths the model never produced.
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
                features=FeatureCapabilities(observed=False, known=self._exogenous, static=False),
                outputs=OutputCapabilities(point=True, quantiles=self._quantiles),
                missing_values=MissingValueSupport.UNSUPPORTED,
            ),
            parameters_schema=schema_of(self._parameters),
        )

    # -- fit ----------------------------------------------------------------

    def fit(
        self, view: FitView, params: Mapping[str, Any], into: Path, *, seed: int | None
    ) -> None:
        """Fit one native model per series, and persist it with what labels it."""
        from statsforecast import StatsForecast

        del seed  # the models exposed here are deterministic order searches
        if not isinstance(view, SeriesView):
            raise ProviderError(
                f"{self._name} trains on one complete time series, so it cannot be fitted "
                f"from a {view.kind} view"
            )
        prepared = conversion.training_frame(view)
        model = self._instantiate(params)
        forecaster = StatsForecast(models=[model], freq=prepared.frequency, n_jobs=1)
        try:
            forecaster.fit(prepared.frame)
        except Exception as error:
            # A library refusing to fit these series is an execution failure the
            # caller can act on, not a bug in the boundary.
            raise ProviderError(
                f"{self._name} could not be fitted on this data: {type(error).__name__}: {error}"
            ) from error
        forecaster.save(str(into / MODEL_FILENAME))
        write_state(
            into / STATE_FILENAME,
            {
                "model": self._name,
                "column": str(model.alias),
                "target": prepared.target,
                "exogenous": list(prepared.exogenous),
                "frequency": prepared.frequency,
                "series": [
                    {
                        "unique_id": series_id,
                        "key": list(instance),
                        "last_event_time": prepared.last_event_times[series_id].isoformat(),
                    }
                    for series_id, instance in prepared.instances.items()
                ],
            },
        )

    # -- forecast -----------------------------------------------------------

    def forecast(self, view: ForecastView, output: Mapping[str, Any], state: Path) -> pa.Table:
        """The next ``horizon`` steps of every series the view asks about."""
        from statsforecast import StatsForecast

        kind = str(output.get("kind", "point"))
        levels = self._requested_levels(kind, output)
        persisted = read_state(state / STATE_FILENAME, self._name)
        unique_ids = {tuple(entry["key"]): str(entry["unique_id"]) for entry in persisted["series"]}
        exogenous = tuple(str(name) for name in persisted["exogenous"])
        self._require_matching_origin(view, unique_ids, persisted)

        forecaster = StatsForecast.load(str(state / MODEL_FILENAME))
        future = conversion.future_frame(view, unique_ids, exogenous)
        column = str(persisted["column"])
        try:
            predictions = forecaster.predict(
                h=view.metadata.horizon,
                X_df=future,
                **({"level": _confidences(levels)} if levels else {}),
            )
        except Exception as error:
            raise ProviderError(
                f"{self._name} could not forecast this view: {type(error).__name__}: {error}"
            ) from error
        if not levels:
            return conversion.answer(
                view, unique_ids, predictions, column=column, target=str(persisted["target"])
            )
        return conversion.quantile_answer(
            view,
            unique_ids,
            predictions,
            columns=_quantile_columns(levels, column, list(predictions.columns)),
            target=str(persisted["target"]),
        )

    def _requested_levels(self, kind: str, output: Mapping[str, Any]) -> tuple[float, ...]:
        """The quantile levels this request asks for, or ``()`` for a point forecast.

        A sample forecast is refused rather than approximated by drawing from the
        fitted intervals: what this library returns is a set of interval bounds,
        and paths through them would be paths this model never produced.
        """
        if kind == "point":
            return ()
        if kind == "quantiles" and self._quantiles:
            return tuple(float(level) for level in output.get("levels", ()))
        produces = "point forecasts and quantiles" if self._quantiles else "point forecasts"
        raise ProviderError(f"{self._name} produces {produces}, not {kind}")

    def _require_matching_origin(
        self,
        view: ForecastView,
        unique_ids: Mapping[tuple[Any, ...], str],
        persisted: Mapping[str, Any],
    ) -> None:
        """A local model continues the series it saw; it does not re-read it.

        ``predict(h)`` extrapolates from the last observation of the fit, so an
        origin that is not that observation would produce the right numbers for
        the wrong event times. Refusing is the only honest answer available
        without fitting again, which is a fit and belongs in ``fit``.
        """
        ends = {
            str(entry["unique_id"]): datetime.fromisoformat(str(entry["last_event_time"]))
            for entry in persisted["series"]
        }
        for instance in view.instances:
            series_id = unique_ids.get(instance)
            if series_id is None:
                raise DataError(
                    f"{self._name} is fitted per series, so it has no model for instance "
                    f"{instance}; it was fitted on {sorted(str(key) for key in unique_ids)}"
                )
            end = ends[series_id]
            if end != view.origin_time:
                raise DataError(
                    f"{self._name} forecasts the steps after the last observation it was "
                    f"fitted on, which for instance {instance} is {end.isoformat()}; this "
                    f"forecast is made at the origin {view.origin_time.isoformat()}. Fit at "
                    f"that origin instead — a local model is refitted rather than reused"
                )

    # -- parameters ---------------------------------------------------------

    def _instantiate(self, params: Mapping[str, Any]) -> Any:
        """The native model the caller's parameters describe."""
        settings = checked(params, self._parameters, self._name)
        try:
            return self._build(**settings)
        except (TypeError, ValueError) as error:
            raise RecipeError(f"{self._name} rejected {dict(params)}: {error}") from error

    def __repr__(self) -> str:
        return f"StatsForecastAdapter({self._name})"


def _confidences(levels: Sequence[float]) -> list[float]:
    """The interval levels that carry these quantiles, as percentages.

    A StatsForecast interval is symmetric around the point forecast, so one
    confidence level carries two quantiles: the 80% interval of a distribution is
    its 0.1 and its 0.9. So the requested levels are folded onto the confidences
    that hold them, and a level of exactly 0.5 needs none — the median of a
    symmetric predictive distribution is the point forecast itself.

    ```text
    0.1  ->  80        0.9  ->  80        0.5  ->  the point forecast
    ```
    """
    wanted = {_confidence(level) for level in levels if level != MEDIAN}
    return sorted(wanted)


def _confidence(level: float) -> float:
    """The interval level, in percent, whose bound is this quantile."""
    return round(abs(2.0 * level - 1.0) * 100.0, _CONFIDENCE_PRECISION)


def _quantile_columns(
    levels: Sequence[float], point_column: str, answered: Sequence[str]
) -> list[tuple[float, str]]:
    """Which returned column holds each requested quantile level.

    Read off the columns the library actually returned rather than rebuilt from
    the values passed to it: StatsForecast names an interval column after the
    number it was given, and ``80`` and ``80.0`` are the same request spelled two
    ways. Matching numerically is what makes that not matter.
    """
    bounds: dict[tuple[str, float], str] = {}
    for column in answered:
        parsed = _INTERVAL_COLUMN.fullmatch(column)
        if parsed is not None:
            bounds[parsed.group("side"), float(parsed.group("confidence"))] = column

    resolved: list[tuple[float, str]] = []
    for level in levels:
        if level == MEDIAN:
            resolved.append((level, point_column))
            continue
        side = "lo" if level < MEDIAN else "hi"
        column = bounds.get((side, _confidence(level)))
        if column is None:
            raise ProviderError(
                f"the fitted model was asked for the {_confidence(level)}% interval that "
                f"carries the quantile {level} and answered {list(answered)}"
            )
        resolved.append((level, column))
    return resolved


def _auto_arima(**params: Any) -> Any:
    from statsforecast.models import AutoARIMA

    return AutoARIMA(**params)


#: The parameters of ``AutoARIMA`` a caller may set. Deliberately a subset:
#: ``alias`` would rename the column the answer is read from, ``trace`` prints,
#: and the interval parameters are not a caller's to set here — which levels an
#: interval covers is what ``of.OutputSpec.quantiles([...])`` says, and a second
#: place to say it would be a second answer to one question.
AUTOARIMA_PARAMETERS = (
    Parameter("season_length", int, "Steps of the data's frequency in one season.", minimum=1),
    Parameter("d", int, "Order of first differencing. Selected when unset.", minimum=0),
    Parameter("D", int, "Order of seasonal differencing. Selected when unset.", minimum=0),
    Parameter("max_p", int, "Largest non-seasonal AR order to consider.", minimum=0),
    Parameter("max_q", int, "Largest non-seasonal MA order to consider.", minimum=0),
    Parameter("max_P", int, "Largest seasonal AR order to consider.", minimum=0),
    Parameter("max_Q", int, "Largest seasonal MA order to consider.", minimum=0),
    Parameter("max_order", int, "Largest total order of the selected model.", minimum=0),
    Parameter("max_d", int, "Largest order of first differencing to consider.", minimum=0),
    Parameter("max_D", int, "Largest order of seasonal differencing to consider.", minimum=0),
    Parameter("start_p", int, "Non-seasonal AR order the search starts at.", minimum=0),
    Parameter("start_q", int, "Non-seasonal MA order the search starts at.", minimum=0),
    Parameter("start_P", int, "Seasonal AR order the search starts at.", minimum=0),
    Parameter("start_Q", int, "Seasonal MA order the search starts at.", minimum=0),
    Parameter("stationary", bool, "Restrict the search to stationary models."),
    Parameter("seasonal", bool, "Allow seasonal terms."),
    Parameter("ic", str, "Information criterion used to select.", choices=("aicc", "aic", "bic")),
    Parameter("stepwise", bool, "Search stepwise rather than over the whole grid."),
    Parameter("nmodels", int, "How many models the stepwise search may try.", minimum=1),
    Parameter("approximation", bool, "Approximate the likelihood while searching."),
    Parameter("allowdrift", bool, "Allow a drift term."),
    Parameter("allowmean", bool, "Allow a non-zero mean."),
    Parameter("biasadj", bool, "Bias-adjust the back-transformed forecast."),
)

#: ``nixtla/autoarima``: order selection over ARIMA models, per series.
AUTOARIMA = StatsForecastAdapter(
    name="autoarima",
    display_name="AutoARIMA",
    build=_auto_arima,
    parameters=AUTOARIMA_PARAMETERS,
    exogenous=True,
    quantiles=True,
)
