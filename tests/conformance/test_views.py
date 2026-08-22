"""Both semantic sources, all three fit views — the six materializations.

```text
TimeSeriesFrame  -> SeriesView      SequenceView    TabularView
ForecastDataset  -> SeriesView (1)  SequenceView    TabularView
```

Every one of them is checked for the same things: the view holds the training
units the source implies, its values are the arithmetic the golden datasets are
built from, and its provenance says where the origins came from. The unit tests
of ``views/`` prove the invariants hold; this suite proves the six combinations
a provider can actually be handed are all reachable and all mean what they say.

(1) A series is one time axis and therefore one origin, so a point-in-time
dataset reaches this view at a selected vintage and nowhere else.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

import openforecast as of
from openforecast.views import (
    EVENT_TIME,
    ORIGIN_TIME,
    ROW_ID,
    SAMPLE_ID,
    SERIES_ID,
    AtOrigin,
    OriginFidelity,
    SequenceView,
    SeriesView,
    SourceKind,
    TabularView,
    ViewKind,
    ViewPlanner,
    ViewRequest,
)
from tests.conformance import datasets
from tests.conformance.datasets import SemanticDataset, column

CONTEXT = 3
HORIZON = 2

planner = ViewPlanner()


def series_view(data: SemanticDataset, **request: Any) -> SeriesView:
    view = planner.fit_view(data, ViewRequest(kind=ViewKind.SERIES, **request))
    assert isinstance(view, SeriesView)
    return view


def sequence_view(data: SemanticDataset, **request: Any) -> SequenceView:
    view = planner.fit_view(
        data, ViewRequest(kind=ViewKind.SEQUENCES, context=CONTEXT, horizon=HORIZON, **request)
    )
    assert isinstance(view, SequenceView)
    return view


def tabular_view(data: SemanticDataset, **request: Any) -> TabularView:
    view = planner.fit_view(data, ViewRequest(kind=ViewKind.TABULAR, horizon=HORIZON, **request))
    assert isinstance(view, TabularView)
    return view


def value_at(view: SequenceView, sample: str, moment: datetime, name: str) -> Any:
    """One cell of a sequence view's temporal table."""
    rows = zip(
        column(view.temporal, SAMPLE_ID),
        column(view.temporal, EVENT_TIME),
        column(view.temporal, name),
        strict=True,
    )
    return next(value for held, event, value in rows if held == sample and event == moment)


# -- TimeSeriesFrame --------------------------------------------------------


def test_a_frame_becomes_one_series_per_instance() -> None:
    data = datasets.panel_univariate()

    view = series_view(data)

    assert len(view.series_ids) == 3
    assert view.temporal.num_rows == 3 * 24
    assert view.schema.targets == ("load",)
    assert view.schema.feature_names == ("temp", "temp_fc", "capacity")
    # An event-time frame is its own newest vintage, so no origin was selected.
    assert view.schema.origin_time is None
    assert view.provenance.source is SourceKind.TIME_SERIES
    assert view.provenance.origin_fidelity is OriginFidelity.SIMULATED
    assert column(view.temporal, "load")[:3] == [
        datasets.target_value(0, step) for step in range(3)
    ]
    assert set(column(view.series, SERIES_ID)) == set(view.series_ids)


def test_a_frame_becomes_one_sequence_per_instance_and_simulated_origin() -> None:
    data = datasets.panel_univariate()

    view = sequence_view(data)

    # 24 event times, of which those with 3 context and 2 forecast steps around
    # them can carry a sample: steps 2 through 21.
    assert len(view.sample_ids) == 3 * 20
    assert view.temporal.num_rows == 3 * 20 * (CONTEXT + HORIZON)
    assert view.schema.context == CONTEXT
    assert view.schema.horizon == HORIZON
    assert view.provenance.origin_fidelity is OriginFidelity.SIMULATED
    assert sorted(set(column(view.samples, ORIGIN_TIME))) == [
        datasets.at(step) for step in range(2, 22)
    ]


def test_a_frame_becomes_one_supervised_row_per_origin_and_horizon_step() -> None:
    data = datasets.panel_univariate()

    view = tabular_view(data)

    # Every origin with an outcome within 2 steps: 23 for step 1, 22 for step 2.
    assert view.num_rows == 3 * (23 + 22)
    assert view.schema.x_columns == ("temp_fc", "capacity")
    assert view.schema.y_columns == ("load",)
    assert len(set(column(view.keys, ROW_ID))) == view.num_rows
    assert view.provenance.origin_fidelity is OriginFidelity.SIMULATED
    # A tabular row describes an event time after its origin, so an observed
    # feature has no value there and is not offered as one.
    assert "temp" not in view.X.column_names


def test_a_multivariate_frame_carries_every_target_into_every_view() -> None:
    data = datasets.panel_multivariate()

    assert series_view(data).schema.targets == ("load", "wind")
    assert sequence_view(data).schema.targets == ("load", "wind")
    assert tabular_view(data).schema.y_columns == ("load", "wind")


def test_a_single_series_frame_materializes_without_instance_keys() -> None:
    data = datasets.single_univariate()

    view = series_view(data)

    assert view.schema.instance_keys == ()
    assert len(view.series_ids) == 1
    assert view.static is None


# -- ForecastDataset --------------------------------------------------------


def test_a_dataset_becomes_one_series_at_one_selected_origin() -> None:
    data = datasets.point_in_time(instances=3, cumulative=True, static=True)
    origin = data.origins[-1]

    view = series_view(data, origins=AtOrigin(origin))

    assert len(view.series_ids) == 3
    assert view.schema.origin_time == origin
    assert view.provenance.source is SourceKind.FORECAST_DATASET
    assert view.provenance.origin_fidelity is OriginFidelity.OBSERVED
    # Every value in it was issued by the vintage that was asked for.
    assert {datasets.origin_of(value) for value in column(view.temporal, "wind_fc")} == {
        data.origins.index(origin) + 2
    }


def test_a_dataset_becomes_one_sequence_per_instance_and_observed_origin() -> None:
    data = datasets.pit_panel_univariate()

    view = sequence_view(data)

    assert len(view.sample_ids) == 3 * len(data.origins)
    assert view.origins == data.origins
    assert view.provenance.source is SourceKind.FORECAST_DATASET
    assert view.provenance.origin_fidelity is OriginFidelity.OBSERVED


def test_a_dataset_becomes_supervised_rows_carrying_their_own_vintage() -> None:
    data = datasets.pit_panel_univariate()

    view = tabular_view(data)

    assert view.num_rows == 3 * len(data.origins) * HORIZON
    assert view.origins == data.origins
    assert view.provenance.origin_fidelity is OriginFidelity.OBSERVED
    origins = column(view.keys, ORIGIN_TIME)
    features = column(view.X, "wind_fc")
    assert [datasets.origin_of(value) for value in features] == [
        data.origins.index(origin) + 2 for origin in origins
    ]


def test_a_multivariate_dataset_carries_every_target_into_every_view() -> None:
    data = datasets.pit_panel_multivariate()

    assert sequence_view(data).schema.targets == ("price", "volume")
    assert tabular_view(data).schema.y_columns == ("price", "volume")
    assert series_view(data, origins=AtOrigin(data.origins[-1])).schema.targets == (
        "price",
        "volume",
    )


def test_a_series_view_of_many_vintages_is_refused_rather_than_flattened() -> None:
    """There is no honest way to put several vintages on one time axis."""
    with pytest.raises(of.OriginScopeError):
        series_view(datasets.pit_panel_univariate())


def test_the_static_features_of_a_panel_reach_both_sources_alike() -> None:
    frame = datasets.panel_univariate()
    dataset = datasets.pit_panel_univariate()

    from_frame = sequence_view(frame)
    from_dataset = sequence_view(dataset)

    for view in (from_frame, from_dataset):
        static = view.static
        assert static is not None
        assert static.num_rows == len(view.sample_ids)
        assert set(column(static, "capacity")) == {
            datasets.capacity_value(index) for index in range(3)
        }
