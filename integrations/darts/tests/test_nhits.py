"""``darts/nhits``: the same architecture as ``nixtla/nhits``, not the same model.

This is the file that answers the question Step 13 asks. Two libraries implement
NHiTS; one of them wires future covariates into it and the other does not. So the
two ``nhits`` references OpenForecast exposes agree about everything except one
capability — and *because* a capability is a declaration, that difference costs
no code anywhere:

```text
same    the sequences contract, the bound horizon, the unseen instance,
        the window compiled from the plan, one sample per origin
differs Darts' NHiTS takes no value known ahead of its event time,
        so a known feature is refused rather than quietly used as a past one
```

The refusal is the interesting assertion. Handing a known feature to a
past-covariates model as a past covariate would train on it happily and silently
ignore everything it says about the future, which is the kind of quiet
wrong-answer the capability declarations exist to make impossible.

One consequence follows and is asserted below: a model whose only feature role
is ``observed`` cannot read point-in-time vintages today. An observed feature
stops at its own origin by definition, so a sequence sample always carries gaps
in it, and this model declares ``REQUIRES_TRANSFORM`` — so the fit is refused,
naming the imputation the caller would have to write down. ``darts/tide`` is the
model the point-in-time cases run against, and ``test_tide.py`` is where they
are.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import golden
import pytest
from golden import FAST, NHITS, at

import openforecast as of
from openforecast.errors import DataError, IncompatibleForecastTask

CONTEXT = 3
HORIZON = 3


def plan(*, context: int = CONTEXT, seed: int | None = 11) -> of.FitPlan:
    return of.FitPlan(origins=of.AllOrigins(), window=of.WindowPlan(context=context), seed=seed)


def test_a_global_model_of_the_target_alone_is_fitted_and_forecast(tmp_path: Path) -> None:
    """The UX the plan asks for: one reference away from ``nixtla/nhits``."""
    frame = golden.event_time_frame(instances=2, periods=12)
    client = golden.client(tmp_path)

    handle = client.fit(NHITS, frame, horizon=HORIZON, params=FAST, plan=plan(), name="zones")
    forecast = client.forecast(handle, frame, horizon=HORIZON)

    assert handle.training.view == "sequences"
    assert handle.training.context == CONTEXT
    assert handle.training.horizon == HORIZON
    assert forecast.origin_time == at(11)
    assert forecast.event_times == (at(12), at(13), at(14))
    assert forecast.table.num_rows == 2 * HORIZON
    assert all(value == value for value in golden.values(forecast)), "the answer holds NaNs"


def test_the_window_is_compiled_from_the_plan_and_the_task(
    tmp_path: Path, native: list[dict[str, Any]]
) -> None:
    """The compilation is the adapter's, so both global models share it."""
    frame = golden.event_time_frame(instances=1, periods=12)
    client = golden.client(tmp_path)

    client.fit(NHITS, frame, horizon=HORIZON, params=FAST, plan=plan(context=4), name="compiled")

    (built,) = native
    assert built["input_chunk_length"] == 4
    assert built["output_chunk_length"] == HORIZON
    assert built["random_state"] == 11


def test_a_value_known_ahead_of_its_event_time_is_refused_rather_than_misused(
    tmp_path: Path,
) -> None:
    """The capability difference, where it is visible: at the engine boundary.

    ``nixtla/nhits`` takes this frame. This one does not, because Darts' NHiTS
    has no future covariate to put the feature in — and the answer to that is a
    refusal naming the feature, not a past covariate holding half of it.
    """
    frame = golden.event_time_frame(periods=12, future_periods=HORIZON, known=True)
    client = golden.client(tmp_path)

    with pytest.raises(DataError, match="cannot be given the features") as refusal:
        client.fit(NHITS, frame, horizon=HORIZON, params=FAST, plan=plan(), name="futures")

    assert golden.KNOWN in str(refusal.value)
    assert "known=False" in str(refusal.value)
    assert not list((tmp_path / "models").glob("*")), "a refused fit left an artifact"


def test_an_observed_feature_in_a_sequence_sample_needs_a_stated_imputation(
    tmp_path: Path,
) -> None:
    """An observed feature has no value past the origin, in either semantic source.

    The planner masks it beyond each sample's origin — that is what makes a
    simulated origin honest — so a sequence sample carrying one always has gaps
    in its forecast half, and this model cannot take a gap through a gradient
    step. Refused with the transform the caller would have to write down.
    """
    frame = golden.event_time_frame(periods=12, observed=True)
    client = golden.client(tmp_path)

    with pytest.raises(DataError, match="of.Impute") as refusal:
        client.fit(NHITS, frame, horizon=HORIZON, params=FAST, plan=plan(), name="gaps")

    assert golden.OBSERVED in str(refusal.value)


def test_a_horizon_the_artifact_was_not_fitted_for_is_refused(tmp_path: Path) -> None:
    frame = golden.event_time_frame(periods=12)
    client = golden.client(tmp_path)

    handle = client.fit(NHITS, frame, horizon=HORIZON, params=FAST, plan=plan(), name="bound")

    with pytest.raises(IncompatibleForecastTask, match="horizon bound to"):
        client.forecast(handle, frame, horizon=HORIZON - 1)


def test_an_instance_the_artifact_never_saw_is_still_forecastable(tmp_path: Path) -> None:
    """Shared parameters, exercised on the second global model too."""
    client = golden.client(tmp_path)

    handle = client.fit(
        NHITS,
        golden.event_time_frame(instances=2, periods=12),
        horizon=HORIZON,
        params=FAST,
        plan=plan(),
        name="global",
    )
    forecast = client.forecast(
        handle, golden.event_time_frame(instances=3, periods=12), horizon=HORIZON
    )

    zones: list[str] = forecast.table.column("zone").to_pylist()
    assert set(zones) == {"DE", "FR", "NL"}, "the third zone was never in the training data"


@pytest.fixture
def native(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[dict[str, Any]]]:
    """Every keyword argument ``NHiTSModel`` was actually constructed with."""
    import darts.models

    built: list[dict[str, Any]] = []
    original = darts.models.NHiTSModel

    def record(**kwargs: Any) -> Any:
        built.append(dict(kwargs))
        return original(**kwargs)

    monkeypatch.setattr(darts.models, "NHiTSModel", record)
    yield built
