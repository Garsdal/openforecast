"""``darts/theta`` through the public API, on both semantic sources.

The local half of the integration, and the claim is the one ``nixtla/autoarima``
makes:

```text
event time     fit a frame, forecast the steps after it
point in time  fit one selected vintage, and be refused every vintage at once
lifecycle      a model reference that was never fitted is not silently fitted
```

The targets are straight lines, which the Theta method continues as a line of
its own — damped rather than exact, because that is what the method does — so the
assertions below are about the shape of the answer rather than about a recorded
blob: the values grow by one step of the line, in the right order, labeled with
the right instance.
"""

from __future__ import annotations

from pathlib import Path

import golden
import pytest
from golden import THETA, at

import openforecast as of
from openforecast.errors import DataError, ModelRequiresFit, OriginScopeError, RecipeError
from openforecast.protocol import ForecastColumn

HORIZON = 3


def test_a_straight_line_is_continued_from_the_end_of_the_history(tmp_path: Path) -> None:
    frame = golden.event_time_frame(periods=24)
    client = golden.client(tmp_path)

    handle = client.fit(THETA, frame, name="de-load")
    forecast = client.forecast(handle, frame, horizon=HORIZON)

    assert forecast.origin_time == at(23)
    assert forecast.event_times == (at(24), at(25), at(26))
    # The last observation is 230 and the line rises by 10 a step, so 240, 250,
    # 260 is what an undamped continuation would be. Theta damps the trend, so
    # the assertion is that it continued the line at all — within a tenth.
    assert golden.values(forecast) == pytest.approx([240.0, 250.0, 260.0], rel=0.1)
    assert handle.manifest.provider == "darts"
    assert handle.manifest.provider_version == golden.PROVIDER.version


def test_a_panel_is_one_model_per_series_labeled_with_its_instance(tmp_path: Path) -> None:
    """A local model fitted on three series answers as three series."""
    frame = golden.event_time_frame(instances=3, periods=24)
    client = golden.client(tmp_path)

    forecast = client.forecast(client.fit(THETA, frame, name="zones"), frame, horizon=HORIZON)
    table = forecast.table

    assert forecast.instance_keys == ("zone",)
    assert table.num_rows == 3 * HORIZON
    zones: list[str] = table.column("zone").to_pylist()
    values: list[float] = table.column(ForecastColumn.VALUE.value).to_pylist()
    by_zone = {
        zone: [value for value, name in zip(values, zones, strict=True) if name == zone]
        for zone in ("DE", "FR", "NL")
    }
    for index, zone in enumerate(("DE", "FR", "NL")):
        expected = [golden.target_value(index, step) for step in (24, 25, 26)]
        assert by_zone[zone] == pytest.approx(expected, rel=0.1)
        assert by_zone[zone] == sorted(by_zone[zone]), "a rising line came back falling"


def test_learning_across_every_vintage_at_once_is_refused(tmp_path: Path) -> None:
    """Theta does not learn jointly across historical forecast origins.

    The same refusal ``nixtla/autoarima`` gets for the same request, from the
    same place: a series view holds one forecast origin, so the *selection* is
    what is impossible, and the engine says so before a provider is started.
    """
    dataset = golden.point_in_time_dataset()
    client = golden.client(tmp_path)

    with pytest.raises(OriginScopeError, match="one forecast origin"):
        client.fit(THETA, dataset, plan=of.FitPlan(origins=of.AllOrigins()), name="all")


def test_a_vintage_this_model_cannot_read_is_refused_for_its_features(tmp_path: Path) -> None:
    """One selected vintage is an ordinary series — of features Theta cannot take.

    A point-in-time dataset holds at least one feature by construction, because
    an origin and an event time on their own carry no information. Theta takes
    none, so the vintage a single origin selects is refused for its features
    rather than for its origins: the narrowing to one origin succeeded, and the
    model is simply the wrong one for data whose whole content is a covariate.

    Which is why ``darts/tide`` is the model the point-in-time cases run
    against, and why a local model that conditions on nothing is a local model
    that cannot read vintages — a fact about this model, not about the boundary.
    """
    dataset = golden.point_in_time_dataset(origins=6, first_origin=2)
    client = golden.client(tmp_path)

    with pytest.raises(DataError, match="cannot be given the features") as refusal:
        client.fit(THETA, dataset, plan=of.FitPlan(origins=of.AtOrigin(at(7))), name="vintage")

    assert golden.KNOWN in str(refusal.value)


def test_a_feature_this_model_cannot_consume_is_refused(tmp_path: Path) -> None:
    """A Theta forecast is a function of the target's own past, and says so.

    Refused by the engine against the descriptor, before the provider is
    started — which is the difference between a capability and a hope.
    """
    client = golden.client(tmp_path)

    with pytest.raises(DataError, match="cannot be given the features"):
        client.fit(THETA, golden.event_time_frame(periods=24, known=True), name="exog")


def test_a_model_reference_that_was_never_fitted_is_not_fitted_here(tmp_path: Path) -> None:
    """``of.forecast(model="darts/theta", ...)`` is a lifecycle error."""
    client = golden.client(tmp_path)

    with pytest.raises(ModelRequiresFit, match="darts/theta"):
        client.forecast(THETA, golden.event_time_frame(periods=24), horizon=HORIZON)


def test_forecasting_from_an_origin_the_model_was_not_fitted_at_is_refused(
    tmp_path: Path,
) -> None:
    """A local model continues the series it saw; it does not re-read a new one.

    ``predict`` extrapolates from the last observation of the fit, so answering
    at a different origin would produce the right numbers for the wrong event
    times. The alternative to refusing is fitting again, which is a fit.
    """
    client = golden.client(tmp_path)
    handle = client.fit(THETA, golden.event_time_frame(periods=24), name="de-load")

    with pytest.raises(DataError, match="fitted on"):
        client.forecast(handle, golden.event_time_frame(periods=20), horizon=HORIZON)


def test_an_instance_the_artifact_never_saw_has_no_model_to_forecast_it(tmp_path: Path) -> None:
    """The other side of ``supports_unseen_instances`` being ``False``."""
    client = golden.client(tmp_path)
    handle = client.fit(THETA, golden.event_time_frame(instances=2, periods=24), name="two-zones")

    with pytest.raises(DataError, match="no model for instance"):
        client.forecast(handle, golden.event_time_frame(instances=3, periods=24), horizon=HORIZON)


def test_a_parameter_the_model_does_not_have_is_refused_by_name(tmp_path: Path) -> None:
    client = golden.client(tmp_path)
    frame = golden.event_time_frame(periods=24)

    with pytest.raises(RecipeError, match=r"no parameter \['nonsense'\]"):
        client.fit(THETA, frame, params={"nonsense": 1}, name="broken")

    with pytest.raises(RecipeError, match="seasonality_period of at least 1"):
        client.fit(THETA, frame, params={"seasonality_period": 0}, name="broken")

    with pytest.raises(RecipeError, match="theta as integer"):
        client.fit(THETA, frame, params={"theta": True}, name="broken")

    with pytest.raises(RecipeError, match=r"season_mode in \["):
        client.fit(THETA, frame, params={"season_mode": "sometimes"}, name="broken")

    assert not list((tmp_path / "models").glob("*")), "a refused fit left an artifact"


def test_the_parameters_are_compiled_into_the_native_model(tmp_path: Path) -> None:
    """A different theta is a different line, which is how we know it arrived.

    ``theta=2`` is the classic method and damps the trend by half; ``theta=1``
    is a plain exponential smoothing of the line and damps it much harder. The
    difference between the two answers is the parameter taking effect.
    """
    frame = golden.event_time_frame(periods=48)
    client = golden.client(tmp_path)

    classic = client.forecast(
        client.fit(THETA, frame, params={"theta": 2}, name="classic"), frame, horizon=HORIZON
    )
    flat = client.forecast(
        client.fit(THETA, frame, params={"theta": 1}, name="flat"), frame, horizon=HORIZON
    )

    assert golden.values(classic) == pytest.approx([480.0, 490.0, 500.0], rel=0.1)
    assert golden.values(flat) != pytest.approx(golden.values(classic), abs=1.0)


def test_the_season_mode_a_recipe_holds_is_a_string(tmp_path: Path) -> None:
    """A recipe is a document, so an enum member is not something it can carry."""
    frame = golden.event_time_frame(periods=48)
    client = golden.client(tmp_path)

    handle = client.fit(
        THETA, frame, params={"season_mode": "additive", "seasonality_period": 24}, name="seasonal"
    )
    forecast = client.forecast(handle, frame, horizon=HORIZON)

    assert forecast.event_times == (at(48), at(49), at(50))
    assert all(value == value for value in golden.values(forecast)), "the answer holds NaNs"
