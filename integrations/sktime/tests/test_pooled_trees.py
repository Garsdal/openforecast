"""``sktime/pooled-trees`` through the public API: a global model on real vintages.

Step 14 is the third-ecosystem stage, so what is asserted here is the claim
rather than the numbers — and the claims are deliberately the same ones the
Nixtla and Darts integrations assert about ``nixtla/nhits`` and ``darts/tide``:

```text
learning       every historical origin becomes one training sample
compilation    WindowPlan(context) -> window_length, horizon -> ForecastingHorizon
panel          sample_id x event_time -> one sktime MultiIndex panel
isolation      one sample is one origin, and the library cannot see past it
roles          known/static -> the one exogenous frame sktime has
generalization an instance the artifact never saw is still forecastable
missingness    a real point-in-time gap is never quietly filled in
```

Plus the one claim that is *new* here, and is the reason a third global model
was worth adding rather than a third spelling of the second: this model does not
bind its horizon at fit, because a recursive reduction rolls one step at a time.
A capability that differs between two global models is exactly the kind of thing
a descriptor has to be able to say.

How well a hundred trees fit six-step windows is not one of those claims, and
the fits below are deliberately small: what is being checked is what the model
was handed and how its answer came back labeled.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import golden
import pytest
from golden import FAST, POOLED_TREES, at

import openforecast as of
from openforecast.errors import DataError, ModelRequiresFit, RecipeError
from openforecast.models import ModelDescriptor, ModelRef
from openforecast.views import (
    EVENT_TIME,
    SAMPLE_ID,
    FitView,
    ForecastView,
    SequenceView,
    ViewKind,
    ViewPlanner,
    ViewRequest,
)

CONTEXT = 3
HORIZON = 3
ORIGINS = 6
FIRST_ORIGIN = 2


def plan(*, context: int = CONTEXT, seed: int | None = 11) -> of.FitPlan:
    """Learn from every vintage, over a window of ``context`` steps."""
    return of.FitPlan(origins=of.AllOrigins(), window=of.WindowPlan(context=context), seed=seed)


# -- learning from real vintages ---------------------------------------------


def test_every_historical_origin_becomes_one_training_sample(tmp_path: Path) -> None:
    """The claim Step 14 repeats from Steps 12 and 13, in the artifact's own record."""
    dataset = golden.point_in_time_dataset(instances=2, origins=ORIGINS, horizon=HORIZON)
    client = golden.client(tmp_path)

    handle = client.fit(
        POOLED_TREES, dataset, horizon=HORIZON, params=FAST, plan=plan(), name="de-price"
    )

    assert handle.training.view == "sequences"
    assert handle.training.source == "forecast_dataset"
    # Real vintages, not windows cut out of one freshest series.
    assert handle.training.origin_fidelity == "observed"
    assert handle.training.context == CONTEXT
    assert handle.training.horizon == HORIZON
    assert handle.training.samples == 2 * ORIGINS
    assert handle.manifest.provider == "sktime"
    assert handle.manifest.provider_version == golden.PROVIDER.version


def test_an_event_time_frame_reaches_the_same_provider_call(tmp_path: Path) -> None:
    """The two semantic sources are one view type; only the fidelity differs."""
    frame = golden.event_time_frame(instances=2, periods=12, future_periods=HORIZON, known=True)
    recording = Recording()
    client = golden.client(tmp_path, recording)

    handle = client.fit(
        POOLED_TREES, frame, horizon=HORIZON, params=FAST, plan=plan(), name="zones"
    )

    assert [type(view) for view in recording.fit_views] == [SequenceView]
    assert handle.training.source == "time_series"
    assert handle.training.origin_fidelity == "simulated"


def test_the_forecast_is_made_at_one_named_origin(tmp_path: Path) -> None:
    dataset = golden.point_in_time_dataset(instances=2, origins=ORIGINS, horizon=HORIZON)
    origin = at(FIRST_ORIGIN + ORIGINS - 1)
    recording = Recording()
    client = golden.client(tmp_path, recording)

    handle = client.fit(
        POOLED_TREES, dataset, horizon=HORIZON, params=FAST, plan=plan(), name="de-price"
    )
    forecast = client.forecast(handle, dataset.at_origin(origin), horizon=HORIZON)

    # The provider receives a ForecastView, never the ForecastDataset behind it.
    assert [type(view) for view in recording.forecast_views] == [ForecastView]
    assert forecast.origin_time == origin
    assert forecast.event_times == tuple(at(FIRST_ORIGIN + ORIGINS + step) for step in range(3))
    assert forecast.instance_keys == ("zone",)
    assert forecast.table.num_rows == 2 * HORIZON
    assert all(value == value for value in golden.values(forecast)), "the answer holds NaNs"


# -- semantic compilation ----------------------------------------------------


def test_the_window_and_the_horizon_are_compiled_into_the_native_model(
    tmp_path: Path, reductions: list[dict[str, Any]], fits: list[dict[str, Any]]
) -> None:
    """``WindowPlan(context=...)`` is ``window_length``; the task's horizon is ``fh``.

    The only line of the design that differs from Nixtla's and Darts' is which
    words the library uses. That the caller says none of them twice is the same.
    Pooling is compiled too: a forecaster handed a panel is vectorized over its
    instances unless it is told to pool, and a *global* model is one that pools.
    """
    dataset = golden.point_in_time_dataset(instances=1, origins=ORIGINS, horizon=HORIZON)
    client = golden.client(tmp_path)

    client.fit(
        POOLED_TREES, dataset, horizon=HORIZON, params=FAST, plan=plan(context=4), name="compiled"
    )

    (built,) = reductions
    assert built["window_length"] == 4
    assert built["pooling"] == "global"
    assert built["strategy"] == "recursive"
    # And the fit plan's seed, rather than the library picking one.
    assert built["estimator"].random_state == 11

    (call,) = fits
    assert list(call["fh"]) == [1, 2, 3]


def test_the_native_spellings_of_the_window_cannot_be_passed_twice() -> None:
    """The user must not state the context length or the horizon as parameters."""
    with pytest.raises(RecipeError, match="of.WindowPlan"):
        of.Model(POOLED_TREES, params={"window_length": 168})

    with pytest.raises(RecipeError, match="ForecastTask"):
        of.Model(POOLED_TREES, params={"fh": 72})

    with pytest.raises(RecipeError, match="of.FitPlan"):
        of.Model(POOLED_TREES, params={"random_state": 3})


def test_a_model_learning_from_sequences_cannot_guess_a_context_length(tmp_path: Path) -> None:
    dataset = golden.point_in_time_dataset(instances=1, origins=ORIGINS, horizon=HORIZON)
    client = golden.client(tmp_path)

    with pytest.raises(RecipeError, match="of.WindowPlan"):
        client.fit(POOLED_TREES, dataset, horizon=HORIZON, params=FAST, name="no-window")


def test_the_feature_roles_become_the_one_exogenous_frame_sktime_has(
    tmp_path: Path, fits: list[dict[str, Any]]
) -> None:
    """``known`` and ``static`` are columns of ``X``; there is nowhere else to put them.

    Read off the call the library actually received, because the mapping is the
    claim: sktime has one exogenous frame rather than three kinds of covariate,
    so a value knowable ahead of its event time is a column of it and a value
    with no time axis is a column of it that is constant within its unit.
    """
    dataset = golden.point_in_time_dataset(
        instances=2, origins=ORIGINS, horizon=HORIZON, static=True
    )
    client = golden.client(tmp_path)

    client.fit(POOLED_TREES, dataset, horizon=HORIZON, params=FAST, plan=plan(), name="roles")

    (call,) = fits
    exogenous = call["X"]
    assert list(exogenous.columns) == [golden.KNOWN, golden.STATIC]
    # A static feature has no time axis, so it is constant within every unit.
    per_unit = exogenous.groupby(level=0)[golden.STATIC].nunique()
    assert set(per_unit) == {1}


# -- the panel, and the one-sequence invariant --------------------------------


def test_the_panel_is_one_unit_per_sample_and_one_row_per_event_time(tmp_path: Path) -> None:
    """sktime's explicit panel format, built from the view and nothing else.

    ``sample_id`` becomes the outer index level and ``event_time`` the inner one,
    which is the mapping Step 14 exists to check. The count is the invariant the
    whole ``SequenceView`` design holds: a library given one long series per
    instance would slide a window along it and learn from sequences nobody
    described, so it is given one unit per ``instance × origin`` of exactly
    ``context + horizon`` steps instead.
    """
    dataset = golden.point_in_time_dataset(instances=2, origins=ORIGINS, horizon=HORIZON)
    recording = Recording()
    client = golden.client(tmp_path, recording)

    client.fit(POOLED_TREES, dataset, horizon=HORIZON, params=FAST, plan=plan(), name="panel")

    from openforecast_sktime import conversion

    (view,) = recording.fit_views
    assert isinstance(view, SequenceView)
    prepared = conversion.sequence_panel(
        view, features=_descriptor(POOLED_TREES).capabilities.features
    )

    assert prepared.y.index.names == [SAMPLE_ID, EVENT_TIME]
    assert len(prepared.sample_ids) == 2 * ORIGINS
    units = prepared.y.groupby(level=0).size()
    assert len(units) == 2 * ORIGINS
    assert set(units) == {CONTEXT + HORIZON}
    assert len(view.origins) == ORIGINS


def test_one_unit_of_the_panel_never_spans_two_forecast_origins(tmp_path: Path) -> None:
    """Two origins of one instance are two units, even where they overlap in time.

    Point-in-time samples share event times by construction — the same hour is
    described by several vintages — so the panel holds the same event time in
    several units and never twice in one. That is what stops a pooled reducer
    from cutting a window across the boundary between two origins.
    """
    dataset = golden.point_in_time_dataset(instances=1, origins=ORIGINS, horizon=HORIZON)
    recording = Recording()
    client = golden.client(tmp_path, recording)

    client.fit(POOLED_TREES, dataset, horizon=HORIZON, params=FAST, plan=plan(), name="isolated")

    from openforecast_sktime import conversion

    (view,) = recording.fit_views
    assert isinstance(view, SequenceView)
    panel = conversion.sequence_panel(
        view, features=_descriptor(POOLED_TREES).capabilities.features
    ).y

    assert not panel.index.has_duplicates
    moments = panel.index.get_level_values(1)
    assert len(set(moments)) < len(moments), "no event time was described by two vintages"


# -- what a global model can be asked --------------------------------------


def test_an_instance_the_artifact_never_saw_is_still_forecastable(tmp_path: Path) -> None:
    """``supports_unseen_instances``, exercised rather than merely declared.

    Pooled parameters are what makes this possible, and the panel label an
    instance gets at inference is what makes it expressible: sktime is handed the
    window of a series it has never been shown, with the parameters untouched.
    """
    fitted_on = golden.point_in_time_dataset(instances=2, origins=ORIGINS, horizon=HORIZON)
    asked_about = golden.point_in_time_dataset(instances=3, origins=ORIGINS, horizon=HORIZON)
    origin = at(FIRST_ORIGIN + ORIGINS - 1)
    client = golden.client(tmp_path)

    handle = client.fit(
        POOLED_TREES, fitted_on, horizon=HORIZON, params=FAST, plan=plan(), name="global"
    )
    forecast = client.forecast(handle, asked_about.at_origin(origin), horizon=HORIZON)

    zones: list[str] = forecast.table.column("zone").to_pylist()
    assert set(zones) == {"DE", "FR", "NL"}, "the third zone was never in the training data"
    assert forecast.table.num_rows == 3 * HORIZON


def test_a_horizon_the_artifact_was_not_fitted_for_is_answered(tmp_path: Path) -> None:
    """The capability that distinguishes this global model from the neural ones.

    ``nixtla/nhits`` and ``darts/tide`` bake the horizon into an architecture and
    refuse another one with ``IncompatibleForecastTask``. A recursive reduction
    learns one step and rolls, so it answers whatever it is asked — and the
    descriptor says so, which is how the engine knows not to refuse.
    """
    dataset = golden.point_in_time_dataset(instances=1, origins=ORIGINS, horizon=HORIZON)
    origin = at(FIRST_ORIGIN + ORIGINS - 1)
    client = golden.client(tmp_path)

    handle = client.fit(
        POOLED_TREES, dataset, horizon=HORIZON, params=FAST, plan=plan(), name="rolling"
    )
    assert handle.serves_horizon(HORIZON - 1)

    forecast = client.forecast(handle, dataset.at_origin(origin), horizon=HORIZON - 1)

    assert forecast.event_times == tuple(
        at(FIRST_ORIGIN + ORIGINS + step) for step in range(HORIZON - 1)
    )
    assert forecast.table.num_rows == HORIZON - 1


def test_a_context_the_artifact_was_not_fitted_for_is_refused(tmp_path: Path) -> None:
    """The window it rolls from is the one it learned to roll from.

    The engine sizes the inference view from the artifact's own record, so this
    is the adapter's guard against an artifact and a request that disagree — a
    short window is something the library would pad rather than refuse.
    """
    from openforecast_sktime.adapters.panel_models import POOLED_TREES as ADAPTER

    dataset = golden.point_in_time_dataset(instances=1, origins=ORIGINS, horizon=HORIZON)
    origin = at(FIRST_ORIGIN + ORIGINS - 1)
    client = golden.client(tmp_path)
    handle = client.fit(
        POOLED_TREES, dataset, horizon=HORIZON, params=FAST, plan=plan(), name="windowed"
    )
    view = ViewPlanner().forecast_view(
        dataset.at_origin(origin),
        ViewRequest(kind=ViewKind.FORECAST, horizon=HORIZON, context=CONTEXT + 1),
    )

    with pytest.raises(DataError, match="rolls its forecast from"):
        ADAPTER.forecast(view, {"kind": "point"}, handle.provider_path)


def test_a_model_reference_that_was_never_fitted_is_not_fitted_here(tmp_path: Path) -> None:
    dataset = golden.point_in_time_dataset(instances=1, origins=ORIGINS, horizon=HORIZON)
    client = golden.client(tmp_path)

    with pytest.raises(ModelRequiresFit, match="sktime/pooled-trees"):
        client.forecast(POOLED_TREES, dataset.at_origin(at(FIRST_ORIGIN)), horizon=HORIZON)


# -- missing values ----------------------------------------------------------


def test_a_point_in_time_gap_is_refused_rather_than_quietly_filled(tmp_path: Path) -> None:
    """An observed feature stops at its own origin, which is information.

    A regressor fitted on a window of NaNs learns nothing, so the model declares
    ``REQUIRES_TRANSFORM`` and the request is refused with the imputation the
    caller would have to write down. What it never is, is filled in here.
    """
    dataset = golden.point_in_time_dataset(
        instances=1, origins=ORIGINS, horizon=HORIZON, observed=True
    )
    client = golden.client(tmp_path)

    with pytest.raises(DataError, match="cannot be given the features") as refusal:
        client.fit(POOLED_TREES, dataset, horizon=HORIZON, params=FAST, plan=plan(), name="gaps")

    assert golden.OBSERVED in str(refusal.value)
    assert not list((tmp_path / "models").glob("*")), "a refused fit left an artifact"


# -- parameters --------------------------------------------------------------


def test_a_parameter_the_model_does_not_have_is_refused_by_name(tmp_path: Path) -> None:
    dataset = golden.point_in_time_dataset(instances=1, origins=ORIGINS, horizon=HORIZON)
    client = golden.client(tmp_path)

    def fit(params: dict[str, object]) -> None:
        client.fit(
            POOLED_TREES, dataset, horizon=HORIZON, params=params, plan=plan(), name="broken"
        )

    with pytest.raises(RecipeError, match=r"no parameter \['nonsense'\]"):
        fit({"nonsense": 1})

    with pytest.raises(RecipeError, match="max_iter of at least 1"):
        fit({"max_iter": 0})

    with pytest.raises(RecipeError, match="learning_rate of at least 0"):
        fit({"learning_rate": -1.0})

    with pytest.raises(RecipeError, match="max_iter as integer"):
        fit({"max_iter": True})

    assert not list((tmp_path / "models").glob("*")), "a refused fit left an artifact"


def test_the_parameters_are_compiled_into_the_native_regressor(
    tmp_path: Path, reductions: list[dict[str, Any]]
) -> None:
    dataset = golden.point_in_time_dataset(instances=1, origins=ORIGINS, horizon=HORIZON)
    client = golden.client(tmp_path)

    client.fit(
        POOLED_TREES,
        dataset,
        horizon=HORIZON,
        params={"max_iter": 7, "max_depth": 2, "learning_rate": 0.5},
        plan=plan(),
        name="tuned",
    )

    (built,) = reductions
    regressor = built["estimator"]
    assert regressor.max_iter == 7
    assert regressor.max_depth == 2
    assert regressor.learning_rate == 0.5


# -- the harness -------------------------------------------------------------


def _descriptor(model: str) -> ModelDescriptor:
    (found,) = [
        candidate for candidate in golden.PROVIDER.descriptors() if str(candidate.ref) == model
    ]
    return found


class Recording:
    """The provider, wrapped so a test can see what the engine handed it."""

    def __init__(self) -> None:
        self.inner = golden.PROVIDER
        self.fit_views: list[FitView] = []
        self.forecast_views: list[ForecastView] = []

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def version(self) -> str:
        return self.inner.version

    def descriptors(self) -> tuple[ModelDescriptor, ...]:
        return self.inner.descriptors()

    def fit(
        self,
        *,
        model: ModelRef | str,
        params: Mapping[str, Any],
        view: FitView,
        seed: int | None,
        into: Path,
    ) -> None:
        self.fit_views.append(view)
        self.inner.fit(model=model, params=params, view=view, seed=seed, into=into)

    def forecast(
        self,
        *,
        model: ModelRef | str,
        params: Mapping[str, Any],
        view: ForecastView,
        output: Mapping[str, Any],
        state: Path,
    ) -> Any:
        self.forecast_views.append(view)
        return self.inner.forecast(
            model=model, params=params, view=view, output=output, state=state
        )


@pytest.fixture
def reductions(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[dict[str, Any]]]:
    """Every keyword argument ``make_reduction`` was actually called with."""
    import sktime.forecasting.compose

    built: list[dict[str, Any]] = []
    original = sktime.forecasting.compose.make_reduction

    def record(estimator: Any, **kwargs: Any) -> Any:
        built.append({"estimator": estimator, **kwargs})
        return original(estimator, **kwargs)

    monkeypatch.setattr(sktime.forecasting.compose, "make_reduction", record)
    yield built


@pytest.fixture
def fits(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[dict[str, Any]]]:
    """Every keyword argument the native ``fit`` was handed, per fit.

    Around the library's own call rather than around the provider: the panel and
    the exogenous mapping are only visible in the shape of the call into sktime.
    """
    from sktime.forecasting.base import BaseForecaster

    calls: list[dict[str, Any]] = []
    original = BaseForecaster.fit

    def record(self: Any, y: Any, X: Any = None, fh: Any = None) -> Any:
        calls.append({"y": y, "X": X, "fh": fh})
        return original(self, y, X=X, fh=fh)

    monkeypatch.setattr(BaseForecaster, "fit", record)
    yield calls
