"""What the estimator is handed, and what comes back labeled with.

The claims of Step 18, in the order they matter:

```text
one row per instance, origin and lead      and no deduplication by event time
the values of *that* origin in that row    vintages do not leak sideways
missing values arrive missing              nothing is imputed on this path
the label is the outcome of the event time repeated where two origins share one
the answer maps back                       instance, origin and event time
```

Everything below goes through the public client. The design matrix is inspected
by materializing the same view the engine materializes and asking this
integration's own conversion for it — which is the boundary the step is about, so
it is checked where it is rather than inferred from a forecast.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import golden
import numpy as np
import pytest
from golden import FAST, HIST_GRADIENT_BOOSTING, at
from openforecast_sklearn import conversion
from openforecast_sklearn.state import ESTIMATOR_FILENAME, METADATA_FILENAME

import openforecast as of
from openforecast.errors import DataError, ProviderError, RecipeError
from openforecast.views import EVENT_TIME, HORIZON_STEP, ORIGIN_TIME, ROW_ID, ViewKind
from openforecast.views.planner import ViewPlanner, ViewRequest

HORIZON = 3
ORIGINS = 6
INSTANCES = 2

#: Enough boosting to actually learn the golden line, which is what makes an
#: assertion about *which* instance a row came back for meaningful. Only the
#: tests that read forecast values pay for it.
LEARNS = {"max_iter": 200, "min_samples_leaf": 1}

#: The golden values are exact functions of their coordinates, so a fitted
#: estimator lands on them to within a rounding error rather than approximately.
TOLERANCE = 1.0


def tabular_view(data: object, *, horizon: int = HORIZON) -> Any:
    """The view the engine would materialize for this model, from this data."""
    return ViewPlanner().fit_view(data, ViewRequest(kind=ViewKind.TABULAR, horizon=horizon))


def column(table: Any, name: str) -> list[Any]:
    values: list[Any] = table.column(name).to_pylist()
    return values


def step(moment: Any) -> int:
    """Which step of the golden grid ``moment`` sits at."""
    return int((moment - golden.START) // golden.HOUR)


# -- the supervised rows -----------------------------------------------------


def test_two_origins_forecasting_the_same_event_time_are_two_rows() -> None:
    """The whole point of a point-in-time fit, and it looks like a mistake.

    Origin 2 and origin 3 both describe 05:00. They knew different things when
    they did, so they are two distinct forecasting examples that happen to share
    an outcome — and no deduplication by event time happens anywhere.
    """
    data = golden.point_in_time_dataset(instances=1, origins=ORIGINS, horizon=HORIZON)

    view = tabular_view(data)

    assert view.num_rows == ORIGINS * HORIZON
    origins = column(view.keys, ORIGIN_TIME)
    events = column(view.keys, EVENT_TIME)
    shared = [origin for origin, event in zip(origins, events, strict=True) if event == at(5)]
    assert sorted(shared) == [at(2), at(3), at(4)]
    # Distinct rows, and the identifier says so rather than the ordering.
    assert len(set(column(view.keys, ROW_ID))) == view.num_rows


def test_a_row_carries_the_values_that_existed_at_its_own_origin() -> None:
    """A vintage that leaked into another origin's row is identifiable arithmetic."""
    data = golden.point_in_time_dataset(instances=1, origins=ORIGINS, horizon=HORIZON)

    view = tabular_view(data)
    prepared = conversion.design_matrix(view)

    assert prepared.features == (golden.KNOWN,)
    origins = column(view.keys, ORIGIN_TIME)
    events = column(view.keys, EVENT_TIME)
    expected = [
        golden.known_value(0, step(event), step(origin))
        for origin, event in zip(origins, events, strict=True)
    ]
    assert prepared.X[:, 0].tolist() == expected


def test_the_label_is_the_outcome_of_the_event_time_the_row_describes() -> None:
    """``y`` is joined on the event time, so two vintages share one outcome."""
    data = golden.point_in_time_dataset(instances=INSTANCES, origins=ORIGINS, horizon=HORIZON)

    view = tabular_view(data)
    prepared = conversion.design_matrix(view)

    assert prepared.target == golden.TARGET
    instances = column(view.keys, golden.ZONE)
    events = column(view.keys, EVENT_TIME)
    expected = [
        golden.target_value(golden.ZONES.index(zone), step(event))
        for zone, event in zip(instances, events, strict=True)
    ]
    assert prepared.y.tolist() == expected


def test_the_lead_is_not_a_feature_and_neither_is_the_timestamp() -> None:
    """What the ``keys`` table is for: an estimator has positions, not meanings.

    A row that carried its own origin would let the estimator learn the calendar
    instead of the relationship, and one that carried its instance key would make
    an unseen instance unforecastable — which this model declares it is not.
    """
    data = golden.point_in_time_dataset(instances=INSTANCES, origins=ORIGINS, static=True)

    view = tabular_view(data)

    assert conversion.design_matrix(view).features == (golden.KNOWN, golden.STATIC)
    for reserved in (ROW_ID, ORIGIN_TIME, EVENT_TIME, HORIZON_STEP, golden.ZONE):
        assert reserved not in view.X.column_names


def test_the_missing_values_of_a_feed_that_had_not_published_reach_the_estimator() -> None:
    """Nothing is imputed on this path, and the estimator is the reason it can be.

    The known feed starts publishing at the third origin, so every row of the
    first two carries ``NaN`` — which is what point-in-time data looks like, and
    what ``HistGradientBoostingRegressor`` reads as a branch rather than an error.
    """
    data = golden.point_in_time_dataset(
        instances=1, origins=ORIGINS, horizon=HORIZON, published_from=2
    )

    view = tabular_view(data)
    prepared = conversion.design_matrix(view)

    missing = [math.isnan(value) for value in prepared.X[:, 0].tolist()]
    assert missing.count(True) == 2 * HORIZON
    origins = column(view.keys, ORIGIN_TIME)
    assert {origin for origin, absent in zip(origins, missing, strict=True) if absent} == {
        at(2),
        at(3),
    }
    # And they are still missing by the time the whole fit has run.
    assert not np.isnan(prepared.y).any(), "a label went missing"


def test_a_feature_that_is_not_a_number_is_refused_by_name() -> None:
    """An estimator takes a matrix; a string column is not one.

    Refused rather than silently label-encoded: an encoding this integration
    invented would be a modeling decision the artifact does not record.
    """
    data = golden.point_in_time_dataset(instances=1, origins=ORIGINS, horizon=HORIZON)
    view = tabular_view(data)
    labeled = view.X.set_column(
        0, golden.KNOWN, [[f"zone-{index}" for index in range(view.num_rows)]]
    )

    with pytest.raises(ProviderError, match=golden.KNOWN):
        conversion.design_matrix(
            type(view)(
                X=labeled, y=view.y, keys=view.keys, schema=view.schema, provenance=view.provenance
            )
        )


def test_a_row_with_no_outcome_to_learn_from_is_refused_rather_than_dropped(
    tmp_path: Path,
) -> None:
    """The asymmetry between a missing feature and a missing label.

    A gap in a feature is information — this estimator branches on it, which is
    why it is the one exposed first. A gap in the *label* is a row with nothing
    to learn from, and dropping it silently would make the row count the manifest
    records a number about a training set that was never used.
    """
    data = _target_gap_at(5)

    with pytest.raises(ProviderError, match="no load to learn from"):
        golden.client(tmp_path).fit(HIST_GRADIENT_BOOSTING, data, horizon=HORIZON, params=FAST)


# -- fit, forecast and the labels on the answer ------------------------------


def test_the_fit_is_one_row_per_instance_origin_and_lead(tmp_path: Path) -> None:
    data = golden.point_in_time_dataset(
        instances=INSTANCES, origins=ORIGINS, horizon=HORIZON, static=True
    )

    handle = golden.client(tmp_path).fit(
        HIST_GRADIENT_BOOSTING, data, horizon=HORIZON, params=FAST, name="de-price"
    )

    assert handle.training.view == "tabular"
    assert handle.training.source == "forecast_dataset"
    assert handle.training.origin_fidelity == "observed"
    assert handle.training.samples == INSTANCES * ORIGINS * HORIZON
    # A tabular view sizes no context window; a row is not a window.
    assert handle.training.context is None
    assert not handle.training.horizon_bound


def test_a_forecast_comes_back_labeled_with_the_instance_and_event_time(
    tmp_path: Path,
) -> None:
    """The mapping back, asserted on values rather than only on labels.

    The golden target is ``instance * 1000 + event * 10``, so a row that came
    back for the wrong zone or the wrong hour is off by a thousand or by ten
    rather than being subtly wrong. That is what makes this an assertion about
    the mapping and not about how well five trees fit a line.
    """
    data = golden.point_in_time_dataset(
        instances=INSTANCES, origins=8, horizon=HORIZON, static=True
    )
    origin = at(9)
    client = golden.client(tmp_path)

    handle = client.fit(HIST_GRADIENT_BOOSTING, data, horizon=HORIZON, params=LEARNS)
    forecast = client.forecast(handle, data.at_origin(origin), horizon=HORIZON)

    assert forecast.origin_time == origin
    assert forecast.event_times == tuple(at(9 + step) for step in range(1, HORIZON + 1))
    assert forecast.instance_keys == (golden.ZONE,)
    zones = column(forecast.table, golden.ZONE)
    events = column(forecast.table, "event_time")
    for zone, event, value in zip(zones, events, golden.values(forecast), strict=True):
        expected = golden.target_value(golden.ZONES.index(zone), step(event))
        assert value == pytest.approx(expected, abs=TOLERANCE)


def test_a_horizon_the_estimator_never_bound_is_still_servable(tmp_path: Path) -> None:
    """One row is one lead, and the lead is not a feature.

    So there is nothing in the fitted estimator that knows how far ahead it was
    trained — the descriptor says so, and the engine reads that rather than
    assuming a model fitted at ``horizon=3`` answers only three steps.
    """
    data = golden.point_in_time_dataset(
        instances=1, origins=ORIGINS, horizon=HORIZON + 2, static=True
    )
    client = golden.client(tmp_path)

    handle = client.fit(HIST_GRADIENT_BOOSTING, data, horizon=HORIZON, params=FAST)
    forecast = client.forecast(handle, data.at_origin(at(7)), horizon=HORIZON + 2)

    assert handle.serves_horizon(HORIZON + 2)
    assert forecast.num_rows == HORIZON + 2


def test_an_instance_the_fit_never_saw_is_forecastable(tmp_path: Path) -> None:
    """Shared parameters, and no instance key in ``X`` to have overfitted to."""
    client = golden.client(tmp_path)
    handle = client.fit(
        HIST_GRADIENT_BOOSTING,
        golden.point_in_time_dataset(instances=2, origins=ORIGINS, static=True),
        horizon=HORIZON,
        params=FAST,
    )

    unseen = golden.point_in_time_dataset(instances=3, origins=ORIGINS, static=True)
    forecast = client.forecast(handle, unseen.at_origin(at(7)), horizon=HORIZON)

    assert set(column(forecast.table, golden.ZONE)) == set(golden.ZONES)
    assert forecast.num_rows == 3 * HORIZON


def test_an_event_time_frame_and_real_vintages_are_the_same_call(tmp_path: Path) -> None:
    """A provider cannot tell them apart; only the recorded fidelity differs."""
    client = golden.client(tmp_path)

    simulated = client.fit(
        HIST_GRADIENT_BOOSTING,
        golden.event_time_frame(instances=1, periods=12, future_periods=HORIZON),
        horizon=HORIZON,
        params=FAST,
        name="simulated",
    )
    observed = client.fit(
        HIST_GRADIENT_BOOSTING,
        golden.point_in_time_dataset(instances=1, origins=ORIGINS, horizon=HORIZON),
        horizon=HORIZON,
        params=FAST,
        name="observed",
    )

    assert simulated.training.origin_fidelity == "simulated"
    assert observed.training.origin_fidelity == "observed"
    assert simulated.training.view == observed.training.view == "tabular"


def test_the_artifact_holds_the_estimator_and_what_labels_its_columns(
    tmp_path: Path,
) -> None:
    """The column order is the contract between a fit and every forecast from it."""
    client = golden.client(tmp_path)
    handle = client.fit(
        HIST_GRADIENT_BOOSTING,
        golden.point_in_time_dataset(instances=INSTANCES, origins=ORIGINS, static=True),
        horizon=HORIZON,
        params=FAST,
    )

    provider_directory = handle.provider_path
    assert (provider_directory / ESTIMATOR_FILENAME).is_file()
    metadata = (provider_directory / METADATA_FILENAME).read_text(encoding="utf-8")
    assert f'"features": [\n    "{golden.KNOWN}",\n    "{golden.STATIC}"\n  ]' in metadata
    assert handle.manifest.provider == "sklearn"
    assert handle.manifest.provider_version == golden.PROVIDER.version


# -- refusals ----------------------------------------------------------------


def test_a_dataset_with_nothing_knowable_at_the_origin_has_no_row_to_build() -> None:
    """A supervised row is features and a label; the other views have a history.

    A property of the ``TabularView`` rather than of this provider, which is why
    it is refused by the planner before a provider is started.
    """
    with pytest.raises(DataError, match="no known or static feature"):
        tabular_view(golden.event_time_frame(instances=1, periods=12, known=False))


def test_an_observed_feature_is_accepted_data_and_not_a_feature(tmp_path: Path) -> None:
    """A tabular row is after its origin, so a measurement has no value there.

    The data is accepted — refusing it would refuse most real datasets — and the
    column simply is not offered as a feature. The descriptor declares
    ``observed`` for exactly that reason.
    """
    data = golden.point_in_time_dataset(instances=1, origins=ORIGINS, observed=True)

    view = tabular_view(data)
    handle = golden.client(tmp_path).fit(HIST_GRADIENT_BOOSTING, data, horizon=HORIZON, params=FAST)

    assert golden.OBSERVED not in view.X.column_names
    assert conversion.design_matrix(view).features == (golden.KNOWN,)
    assert handle.training.samples == ORIGINS * HORIZON


def test_a_second_target_is_refused_before_the_provider_is_started(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="cannot be fitted on 2 targets"):
        golden.client(tmp_path).fit(
            HIST_GRADIENT_BOOSTING,
            _two_targets(),
            horizon=HORIZON,
            params=FAST,
        )


def test_a_parameter_the_estimator_rejects_is_a_recipe_error(tmp_path: Path) -> None:
    client = golden.client(tmp_path)
    data = golden.point_in_time_dataset(instances=1, origins=ORIGINS)

    with pytest.raises(RecipeError, match="max_leaf_nodes"):
        client.fit(HIST_GRADIENT_BOOSTING, data, horizon=HORIZON, params={"max_leaf_nodes": 1})
    with pytest.raises(RecipeError, match="takes no parameter"):
        client.fit(HIST_GRADIENT_BOOSTING, data, horizon=HORIZON, params={"n_estimators": 10})

    assert not list((tmp_path / "models").glob("*")), "a failed fit left an artifact"


def test_the_context_length_is_not_something_a_row_can_be_given(tmp_path: Path) -> None:
    """``WindowPlan`` sizes a sequence; a row is one event time and has no window."""
    with pytest.raises(of.RecipeError, match="binds no context length"):
        golden.client(tmp_path).fit(
            HIST_GRADIENT_BOOSTING,
            golden.point_in_time_dataset(instances=1, origins=ORIGINS),
            horizon=HORIZON,
            plan=of.FitPlan(window=of.WindowPlan(context=3)),
            params=FAST,
        )


def _two_targets() -> of.ForecastDataset:
    """The golden vintages, with a second target column beside the first."""
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for step in range(ORIGINS):
        origin = 2 + step
        for event in range(origin + HORIZON + 1):
            rows.append(
                {
                    golden.ORIGIN_TIME: at(origin),
                    golden.EVENT_TIME: at(event),
                    golden.TARGET: golden.target_value(0, event),
                    "wind": golden.target_value(0, event) + 1,
                    golden.KNOWN: golden.known_value(0, event, origin),
                }
            )
    return of.ForecastDataset.from_pandas(
        pd.DataFrame(rows),
        origin_time=golden.ORIGIN_TIME,
        event_time=golden.EVENT_TIME,
        event_frequency=golden.FREQUENCY,
        origin_frequency=golden.FREQUENCY,
        targets=[golden.TARGET, "wind"],
        known_features=[golden.KNOWN],
    )


def _target_gap_at(gap: int) -> of.ForecastDataset:
    """The golden vintages, with one event time whose outcome was never measured."""
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for index in range(ORIGINS):
        origin = 2 + index
        for event in range(origin + HORIZON + 1):
            rows.append(
                {
                    golden.ORIGIN_TIME: at(origin),
                    golden.EVENT_TIME: at(event),
                    golden.TARGET: (golden.NAN if event == gap else golden.target_value(0, event)),
                    golden.KNOWN: golden.known_value(0, event, origin),
                }
            )
    return of.ForecastDataset.from_pandas(
        pd.DataFrame(rows),
        origin_time=golden.ORIGIN_TIME,
        event_time=golden.EVENT_TIME,
        event_frequency=golden.FREQUENCY,
        origin_frequency=golden.FREQUENCY,
        targets=[golden.TARGET],
        known_features=[golden.KNOWN],
    )
