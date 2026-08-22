"""The golden datasets are what everything else is measured against.

Every other conformance test asserts something about a materialized view, so a
fixture that quietly stopped being a panel, or stopped varying by vintage, would
turn those tests into assertions about nothing. This module is the check on the
fixtures themselves: each golden dataset is named, has the shape its name
claims, and — for the point-in-time ones — actually disagrees between vintages
where it says it does.
"""

from __future__ import annotations

from datetime import datetime

import pytest

import openforecast as of
from tests.conformance import datasets
from tests.conformance.datasets import EVENT_TIME, GOLDEN_DATASETS, SemanticDataset

EXPECTED = (
    "single_univariate",
    "single_multivariate",
    "panel_univariate",
    "panel_multivariate",
    "pit_panel_univariate",
    "pit_panel_multivariate",
    "pit_missingness",
    "pit_varying_vintages",
    "pit_known_future",
    "pit_observed_features",
)


def build(name: str) -> SemanticDataset:
    return datasets.golden(name)


@pytest.fixture(params=EXPECTED)
def dataset(request: pytest.FixtureRequest) -> SemanticDataset:
    return build(str(request.param))


def test_every_golden_dataset_the_suite_names_exists() -> None:
    assert tuple(GOLDEN_DATASETS) == EXPECTED


def test_a_golden_dataset_is_built_fresh_each_time() -> None:
    """Two callers must not be able to see each other's fixture."""
    assert build("panel_univariate") is not build("panel_univariate")


def test_an_unknown_name_is_not_silently_empty() -> None:
    with pytest.raises(KeyError, match="no golden dataset"):
        datasets.golden("pit_nonexistent")


def test_every_golden_dataset_is_a_semantic_source(dataset: SemanticDataset) -> None:
    assert isinstance(dataset, of.TimeSeriesFrame | of.ForecastDataset)


@pytest.mark.parametrize(
    ("name", "instances", "targets"),
    [
        ("single_univariate", 1, ("load",)),
        ("single_multivariate", 1, ("load", "wind")),
        ("panel_univariate", 3, ("load",)),
        ("panel_multivariate", 3, ("load", "wind")),
    ],
)
def test_the_event_time_datasets_have_the_shape_their_names_claim(
    name: str, instances: int, targets: tuple[str, ...]
) -> None:
    frame = build(name)
    assert isinstance(frame, of.TimeSeriesFrame)

    assert len(frame.instances) == instances
    assert frame.schema.targets == targets
    # A single series is keyless, not a panel of one.
    assert frame.schema.instance_keys == (("zone",) if instances > 1 else ())


@pytest.mark.parametrize(
    ("name", "instances", "targets"),
    [
        ("pit_panel_univariate", 3, ("price",)),
        ("pit_panel_multivariate", 3, ("price", "volume")),
        ("pit_missingness", 1, ("price",)),
        ("pit_varying_vintages", 2, ("price",)),
        ("pit_known_future", 2, ("price",)),
        ("pit_observed_features", 2, ("price",)),
    ],
)
def test_the_point_in_time_datasets_have_the_shape_their_names_claim(
    name: str, instances: int, targets: tuple[str, ...]
) -> None:
    data = build(name)
    assert isinstance(data, of.ForecastDataset)

    assert len(data.instances) == instances
    assert data.targets == targets
    assert len(data.origins) > 1


def test_a_known_value_names_the_vintage_that_issued_it() -> None:
    """The decoding the leakage assertions rely on, checked on its own."""
    value = datasets.known_value(2, 7, origin=9)

    assert datasets.origin_of(value) == 9
    assert datasets.origin_of(datasets.known_value(2, 7)) == -1


def test_the_vintages_of_pit_varying_vintages_actually_disagree() -> None:
    data = build("pit_varying_vintages")
    assert isinstance(data, of.ForecastDataset)
    event = datasets.at(4)

    published = published_for(data, event)

    assert len(set(published.values())) == len(published) > 1


def test_a_stable_dataset_carries_no_vintage_at_all() -> None:
    """What makes the event-time equivalence comparison meaningful."""
    data = datasets.point_in_time(stable=True)
    event = datasets.at(4)

    published = published_for(data, event)

    assert {datasets.origin_of(value) for value in published.values()} == {-1}
    assert len(set(published.values())) == 1


def test_pit_missingness_withholds_exactly_the_cell_it_says_it_does() -> None:
    data = build("pit_missingness")
    assert isinstance(data, of.ForecastDataset)
    event = datasets.at(datasets.MISSING_EVENT)

    published = published_for(data, event)

    assert [datasets.is_missing(value) for _, value in sorted(published.items())] == [
        True,
        True,
        False,
    ]
    assert published[datasets.at(datasets.AVAILABLE_ORIGIN)] == datasets.AVAILABLE_VALUE


def test_an_observed_feature_stops_at_its_own_origin() -> None:
    """The semantic model enforces it; the fixture has to actually exercise it."""
    data = build("pit_observed_features")
    assert isinstance(data, of.ForecastDataset)
    origin = data.origins[-1]
    vintage = data.information.at_origin(origin).table

    events = datasets.column(vintage, "target_time")
    measured = datasets.column(vintage, "temp")

    assert [datasets.is_missing(value) for value in measured] == [
        moment > origin for moment in events
    ]


def test_the_leakage_sentinel_is_the_dataset_the_plan_describes() -> None:
    data = datasets.leakage_sentinel()
    event = datasets.at(datasets.SENTINEL_EVENT)

    published = published_for(data, event)

    assert [value for _, value in sorted(published.items())] == [10.0, 20.0, datasets.POISON]


def published_for(data: of.ForecastDataset, event: datetime) -> dict[datetime, float]:
    """What each vintage said the known feature would be at ``event``."""
    published: dict[datetime, float] = {}
    for origin in data.origins:
        table = data.information.at_origin(origin).table
        rows = zip(
            datasets.column(table, EVENT_TIME),
            datasets.column(table, "wind_fc"),
            strict=True,
        )
        # Not every vintage describes every event time, and one that does not is
        # silent rather than missing: it simply has nothing to say about it.
        for moment, value in rows:
            if moment == event:
                published[origin] = value
    return published
