"""``darts/tide`` through the public API: a global model on real vintages.

Step 13 is the library-neutrality stage, so what is asserted here is the claim
rather than the numbers — and the claims are deliberately the same seven the
Nixtla integration asserts about ``nixtla/nhits``:

```text
learning       every historical origin becomes one training sample
compilation    WindowPlan(context) -> input_chunk_length, horizon -> output_chunk_length
isolation      one sample is one origin, and the library cannot see past it
roles          observed/known/static -> past/future/static covariates
generalization an instance the artifact never saw is still forecastable
horizon        a horizon the artifact was not fitted for is refused
missingness    a real point-in-time gap is never quietly filled in
```

If this file reads like ``integrations/nixtla/tests/test_nhits.py`` with two
words changed, that is the result: the same plans, the same data and the same
assertions, against a library that spells every one of its own concepts
differently.

How well a neural network fits six-step windows is not one of those claims, and
the fits below are deliberately one epoch long: what is being checked is what
the model was handed and how its answer came back labeled.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import golden
import pytest
from golden import FAST, TIDE, at

import openforecast as of
from openforecast.errors import (
    DataError,
    IncompatibleForecastTask,
    ModelRequiresFit,
    RecipeError,
)
from openforecast.models import ModelDescriptor, ModelRef
from openforecast.views import FitView, ForecastView, SequenceView

CONTEXT = 3
HORIZON = 3
ORIGINS = 6
FIRST_ORIGIN = 2


def plan(*, context: int = CONTEXT, seed: int | None = 11) -> of.FitPlan:
    """Learn from every vintage, over a window of ``context`` steps."""
    return of.FitPlan(origins=of.AllOrigins(), window=of.WindowPlan(context=context), seed=seed)


# -- learning from real vintages ---------------------------------------------


def test_every_historical_origin_becomes_one_training_sample(tmp_path: Path) -> None:
    """The claim Step 13 repeats from Step 12, in the artifact's own record."""
    dataset = golden.point_in_time_dataset(instances=2, origins=ORIGINS, horizon=HORIZON)
    client = golden.client(tmp_path)

    handle = client.fit(TIDE, dataset, horizon=HORIZON, params=FAST, plan=plan(), name="de-price")

    assert handle.training.view == "sequences"
    assert handle.training.source == "forecast_dataset"
    # Real vintages, not windows cut out of one freshest series.
    assert handle.training.origin_fidelity == "observed"
    assert handle.training.context == CONTEXT
    assert handle.training.horizon == HORIZON
    assert handle.training.samples == 2 * ORIGINS
    assert handle.manifest.provider == "darts"
    assert handle.manifest.provider_version == golden.PROVIDER.version


def test_an_event_time_frame_reaches_the_same_provider_call(tmp_path: Path) -> None:
    """The two semantic sources are one view type; only the fidelity differs."""
    frame = golden.event_time_frame(instances=2, periods=12, future_periods=HORIZON, known=True)
    recording = Recording()
    client = golden.client(tmp_path, recording)

    handle = client.fit(TIDE, frame, horizon=HORIZON, params=FAST, plan=plan(), name="zones")

    assert [type(view) for view in recording.fit_views] == [SequenceView]
    assert handle.training.source == "time_series"
    assert handle.training.origin_fidelity == "simulated"


def test_the_forecast_is_made_at_one_named_origin(tmp_path: Path) -> None:
    dataset = golden.point_in_time_dataset(instances=2, origins=ORIGINS, horizon=HORIZON)
    origin = at(FIRST_ORIGIN + ORIGINS - 1)
    recording = Recording()
    client = golden.client(tmp_path, recording)

    handle = client.fit(TIDE, dataset, horizon=HORIZON, params=FAST, plan=plan(), name="de-price")
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
    tmp_path: Path, native: list[dict[str, Any]]
) -> None:
    """``WindowPlan(context=...)`` is ``input_chunk_length``; the task's horizon
    is ``output_chunk_length``.

    The one line of the design that differs from Nixtla's is which words the
    library uses. That the caller says neither of them twice is the same.
    """
    dataset = golden.point_in_time_dataset(instances=1, origins=ORIGINS, horizon=HORIZON)
    client = golden.client(tmp_path)

    client.fit(TIDE, dataset, horizon=HORIZON, params=FAST, plan=plan(context=4), name="compiled")

    (built,) = native
    assert built["input_chunk_length"] == 4
    assert built["output_chunk_length"] == HORIZON
    # And the fit plan's seed, rather than the library picking one.
    assert built["random_state"] == 11
    # A provider runs inside somebody else's process: no logs, no checkpoints,
    # no progress bar over a stdout that may be carrying the wire protocol.
    assert built["pl_trainer_kwargs"]["logger"] is False
    assert built["pl_trainer_kwargs"]["enable_checkpointing"] is False


def test_the_native_spellings_of_the_window_cannot_be_passed_twice() -> None:
    """The user must not state the context length or the horizon as parameters."""
    with pytest.raises(RecipeError, match="of.WindowPlan"):
        of.Model(TIDE, params={"input_chunk_length": 168})

    with pytest.raises(RecipeError, match="ForecastTask"):
        of.Model(TIDE, params={"output_chunk_length": 72})


def test_a_model_learning_from_sequences_cannot_guess_a_context_length(tmp_path: Path) -> None:
    dataset = golden.point_in_time_dataset(instances=1, origins=ORIGINS, horizon=HORIZON)
    client = golden.client(tmp_path)

    with pytest.raises(RecipeError, match="of.WindowPlan"):
        client.fit(TIDE, dataset, horizon=HORIZON, params=FAST, name="no-window")


def test_the_three_feature_roles_become_the_three_kinds_of_covariate(
    tmp_path: Path, handed: list[dict[str, Any]]
) -> None:
    """``observed``, ``known`` and ``static`` are Darts' three covariate kinds.

    Read off the call the library actually received, because the mapping is the
    claim: a value that stops at the origin is a past covariate, a value known
    ahead of its event time is a future covariate, and a value with no time axis
    rides on the series as a static one.
    """
    dataset = golden.point_in_time_dataset(
        instances=2, origins=ORIGINS, horizon=HORIZON, static=True
    )
    client = golden.client(tmp_path)

    client.fit(TIDE, dataset, horizon=HORIZON, params=FAST, plan=plan(), name="roles")

    (call,) = handed
    assert [list(series.components) for series in call["future_covariates"][:1]] == [[golden.KNOWN]]
    # No observed feature in this dataset, and an empty list is not the same
    # thing to the library as no list at all.
    assert call["past_covariates"] is None
    # A static feature has no time axis, so Darts carries it on the series.
    statics = call["series"][0].static_covariates
    assert statics is not None
    assert list(statics.columns) == [golden.STATIC]


# -- the one-sequence invariant ----------------------------------------------


def test_no_training_sample_ever_spans_two_forecast_origins(
    tmp_path: Path, windows: list[int]
) -> None:
    """The invariant the whole ``SequenceView`` design exists to hold.

    A library given one long series per instance would slide a window along it
    and learn from sequences nobody described. Darts is given one window per
    ``instance x origin`` instead, of exactly ``input_chunk_length +
    output_chunk_length`` steps, so the number of windows it can cut is the
    number of samples and no more — which is what this asserts, from inside the
    library's own training dataset.
    """
    dataset = golden.point_in_time_dataset(instances=2, origins=ORIGINS, horizon=HORIZON)
    client = golden.client(tmp_path)

    client.fit(TIDE, dataset, horizon=HORIZON, params=FAST, plan=plan(), name="isolated")

    assert windows == [2 * ORIGINS], "the library built more windows than there are samples"


def test_each_sample_is_handed_over_as_its_own_series(tmp_path: Path) -> None:
    """The series themselves, before any of them reaches a tensor.

    One ``TimeSeries`` per sample, ``context + horizon`` steps in each, and no
    series shared between two origins of the same instance.
    """
    dataset = golden.point_in_time_dataset(instances=2, origins=ORIGINS, horizon=HORIZON)
    recording = Recording()
    client = golden.client(tmp_path, recording)

    client.fit(TIDE, dataset, horizon=HORIZON, params=FAST, plan=plan(), name="frames")

    from openforecast_darts import conversion

    (view,) = recording.fit_views
    assert isinstance(view, SequenceView)
    prepared = conversion.sequence_series(view, features=_descriptor(TIDE).capabilities.features)

    assert len(prepared.targets) == 2 * ORIGINS
    assert {len(series) for series in prepared.targets} == {CONTEXT + HORIZON}
    assert len(view.origins) == ORIGINS


# -- what a global model can be asked --------------------------------------


def test_an_instance_the_artifact_never_saw_is_still_forecastable(tmp_path: Path) -> None:
    """``supports_unseen_instances``, exercised rather than merely declared."""
    fitted_on = golden.point_in_time_dataset(instances=2, origins=ORIGINS, horizon=HORIZON)
    asked_about = golden.point_in_time_dataset(instances=3, origins=ORIGINS, horizon=HORIZON)
    origin = at(FIRST_ORIGIN + ORIGINS - 1)
    client = golden.client(tmp_path)

    handle = client.fit(TIDE, fitted_on, horizon=HORIZON, params=FAST, plan=plan(), name="global")
    forecast = client.forecast(handle, asked_about.at_origin(origin), horizon=HORIZON)

    zones: list[str] = forecast.table.column("zone").to_pylist()
    assert set(zones) == {"DE", "FR", "NL"}, "the third zone was never in the training data"
    assert forecast.table.num_rows == 3 * HORIZON


def test_a_horizon_the_artifact_was_not_fitted_for_is_refused(tmp_path: Path) -> None:
    """``output_chunk_length`` is part of the architecture, so 2 is another model."""
    dataset = golden.point_in_time_dataset(instances=1, origins=ORIGINS, horizon=HORIZON)
    origin = at(FIRST_ORIGIN + ORIGINS - 1)
    client = golden.client(tmp_path)

    handle = client.fit(TIDE, dataset, horizon=HORIZON, params=FAST, plan=plan(), name="bound")

    with pytest.raises(IncompatibleForecastTask, match="horizon bound to"):
        client.forecast(handle, dataset.at_origin(origin), horizon=HORIZON - 1)


def test_a_model_reference_that_was_never_fitted_is_not_fitted_here(tmp_path: Path) -> None:
    dataset = golden.point_in_time_dataset(instances=1, origins=ORIGINS, horizon=HORIZON)
    client = golden.client(tmp_path)

    with pytest.raises(ModelRequiresFit, match="darts/tide"):
        client.forecast(TIDE, dataset.at_origin(at(FIRST_ORIGIN)), horizon=HORIZON)


# -- missing values ----------------------------------------------------------


def test_a_point_in_time_gap_is_refused_rather_than_quietly_filled(tmp_path: Path) -> None:
    """An observed feature stops at its own origin, which is information.

    A gradient step on a NaN produces NaN weights, so the model declares
    ``REQUIRES_TRANSFORM`` and the request is refused with the imputation the
    caller would have to write down. What it never is, is filled in here.
    """
    dataset = golden.point_in_time_dataset(
        instances=1, origins=ORIGINS, horizon=HORIZON, observed=True
    )
    client = golden.client(tmp_path)

    with pytest.raises(DataError, match="of.Impute") as refusal:
        client.fit(TIDE, dataset, horizon=HORIZON, params=FAST, plan=plan(), name="gaps")

    assert golden.OBSERVED in str(refusal.value)
    assert not list((tmp_path / "models").glob("*")), "a refused fit left an artifact"


# -- parameters --------------------------------------------------------------


def test_a_parameter_the_model_does_not_have_is_refused_by_name(tmp_path: Path) -> None:
    dataset = golden.point_in_time_dataset(instances=1, origins=ORIGINS, horizon=HORIZON)
    client = golden.client(tmp_path)

    with pytest.raises(RecipeError, match=r"no parameter \['nonsense'\]"):
        client.fit(TIDE, dataset, horizon=HORIZON, params={"nonsense": 1}, plan=plan(), name="x")

    with pytest.raises(RecipeError, match="n_epochs of at least 1"):
        client.fit(TIDE, dataset, horizon=HORIZON, params={"n_epochs": 0}, plan=plan(), name="x")

    with pytest.raises(RecipeError, match="dropout of at most 1"):
        client.fit(TIDE, dataset, horizon=HORIZON, params={"dropout": 2.0}, plan=plan(), name="x")

    assert not list((tmp_path / "models").glob("*")), "a refused fit left an artifact"


def test_the_parameters_are_compiled_into_the_native_model(
    tmp_path: Path, native: list[dict[str, Any]]
) -> None:
    dataset = golden.point_in_time_dataset(instances=1, origins=ORIGINS, horizon=HORIZON)
    client = golden.client(tmp_path)

    client.fit(
        TIDE,
        dataset,
        horizon=HORIZON,
        params={"n_epochs": 1, "hidden_size": 8, "use_layer_norm": True},
        plan=plan(),
        name="architecture",
    )

    (built,) = native
    assert built["hidden_size"] == 8
    assert built["use_layer_norm"] is True


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
def native(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[dict[str, Any]]]:
    """Every keyword argument ``TiDEModel`` was actually constructed with."""
    import darts.models

    built: list[dict[str, Any]] = []
    original = darts.models.TiDEModel

    def record(**kwargs: Any) -> Any:
        built.append(dict(kwargs))
        return original(**kwargs)

    monkeypatch.setattr(darts.models, "TiDEModel", record)
    yield built


@pytest.fixture
def handed(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[dict[str, Any]]]:
    """Every keyword argument the native ``fit`` was handed, per fit.

    Around the library's own call rather than around the provider: the covariate
    mapping is only visible in the shape of the call into Darts.
    """
    from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel

    calls: list[dict[str, Any]] = []
    original = TorchForecastingModel.fit

    def record(self: Any, **kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        return original(self, **kwargs)

    monkeypatch.setattr(TorchForecastingModel, "fit", record)
    yield calls


@pytest.fixture
def windows(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[int]]:
    """How many training windows Darts built, per fit.

    Read from inside Darts' own training dataset, because the invariant is about
    what the library ends up learning from rather than about what this
    integration believes it handed over.
    """
    from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel

    seen: list[int] = []
    original = TorchForecastingModel._build_train_dataset

    def record(self: Any, *args: Any, **kwargs: Any) -> Any:
        dataset = original(self, *args, **kwargs)
        seen.append(len(dataset))
        return dataset

    monkeypatch.setattr(TorchForecastingModel, "_build_train_dataset", record)
    yield seen
