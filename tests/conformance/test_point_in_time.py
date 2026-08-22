"""The four point-in-time properties every future provider inherits.

```text
leakage           a later vintage is unreachable from an earlier origin
sample count      one instance and origin is one sequence, and nothing else is
missingness       an availability that improves is preserved exactly
equivalence       identical vintages materialize identically to event-time data
```

None of them is a property of any provider, and all of them are properties every
provider depends on. They are asserted against the materialized views because
that is the only surface a provider sees: whatever an integration does with a
``SequenceView``, it cannot recover a vintage the planner never put in it, and
it cannot repair one the planner did.
"""

from __future__ import annotations

from typing import Any

import pytest

import openforecast as of
from openforecast.views import (
    EVENT_TIME,
    ORIGIN_TIME,
    SAMPLE_ID,
    AllOrigins,
    AtOrigin,
    OriginFidelity,
    OriginsBetween,
    SequenceView,
    TabularView,
    ViewKind,
    ViewPlanner,
    ViewRequest,
)
from tests.conformance import datasets
from tests.conformance.datasets import SemanticDataset, column

planner = ViewPlanner()


def materialize(data: SemanticDataset, **request: Any) -> Any:
    return planner.fit_view(data, ViewRequest(**request))


def values(view: Any, name: str) -> list[Any]:
    """Every value of a named column, whichever table the view keeps it in."""
    if isinstance(view, TabularView):
        return column(view.X, name)
    return column(view.temporal, name)


# -- leakage ----------------------------------------------------------------

#: The vintage the sentinel is materialized at, and the one that is poisoned.
MATERIALIZED_ORIGIN = datasets.at(9)
POISONED_ORIGIN = datasets.at(10)


@pytest.mark.parametrize(
    ("request_kwargs", "reaches_the_sentinel"),
    [
        ({"kind": ViewKind.SERIES}, False),
        ({"kind": ViewKind.SEQUENCES, "context": 2, "horizon": 3}, True),
        ({"kind": ViewKind.TABULAR, "horizon": 3}, True),
    ],
    ids=["series", "sequences", "tabular"],
)
def test_a_later_vintage_is_unreachable_from_an_earlier_origin(
    request_kwargs: dict[str, Any], reaches_the_sentinel: bool
) -> None:
    """The sentinel: origin 09 sees the 20 it published and never the 999999.

    ```text
    origin 08 -> target 12 -> wind = 10
    origin 09 -> target 12 -> wind = 20
    origin 10 -> target 12 -> wind = 999999
    ```

    A series holds only event times up to its origin, so the sentinel event is
    not in it at all; what has to hold there is that nothing else of the later
    vintage arrived either. The two forward-looking views do cover the sentinel
    and must find the 20.
    """
    data = datasets.leakage_sentinel()

    view = materialize(data, origins=AtOrigin(MATERIALIZED_ORIGIN), **request_kwargs)

    wind = values(view, "wind_fc")
    assert datasets.POISON not in wind
    assert (20.0 in wind) is reaches_the_sentinel
    # Nor did anything else of the poisoned vintage get in on the side: every
    # value that carries a vintage carries the one that was asked for.
    assert {datasets.origin_of(value) for value in wind if value not in (10.0, 20.0)} == {9}


def test_the_sentinel_is_reachable_at_the_origin_that_published_it() -> None:
    """The negative above is only worth having if the positive holds."""
    data = datasets.leakage_sentinel()

    view = materialize(data, kind=ViewKind.TABULAR, horizon=2, origins=AtOrigin(POISONED_ORIGIN))

    assert datasets.POISON in values(view, "wind_fc")


# -- sequence sample count --------------------------------------------------


def test_a_hundred_origins_of_three_instances_are_three_hundred_samples() -> None:
    """One ``instance x origin`` is one sample; ``AllOrigins`` takes every one."""
    data = datasets.point_in_time(instances=3, origins=100, context=2, horizon=2)

    view = materialize(data, kind=ViewKind.SEQUENCES, context=2, horizon=2, origins=AllOrigins())

    assert isinstance(view, SequenceView)
    assert len(view.sample_ids) == 300
    assert len(view.origins) == 100
    assert view.temporal.num_rows == 300 * 4
    # And each of them is one instance at one origin, not a merge of several.
    pairs = zip(column(view.samples, ORIGIN_TIME), column(view.samples, "zone"), strict=True)
    assert len(set(pairs)) == 300


def test_a_stride_thins_the_samples_and_not_the_sequences() -> None:
    data = datasets.point_in_time(instances=3, origins=100, context=2, horizon=2)

    view = materialize(
        data, kind=ViewKind.SEQUENCES, context=2, horizon=2, origins=AllOrigins(stride=4)
    )

    assert isinstance(view, SequenceView)
    assert len(view.sample_ids) == 75
    assert view.temporal.num_rows == 75 * 4


# -- missingness ------------------------------------------------------------


def test_an_availability_that_improves_is_preserved_exactly() -> None:
    """``NaN, NaN, 42`` is what the feed did, so it is what the view holds."""
    data = datasets.pit_missingness()

    published = [_sentinel_row(data, datasets.at(origin)) for origin in (8, 9, 10)]

    assert [datasets.is_missing(value) for value in published] == [True, True, False]
    assert published[-1] == datasets.AVAILABLE_VALUE


def test_a_gap_is_materialized_rather_than_dropped_or_filled() -> None:
    """A row that says nothing is not the same as no row, and not a number either.

    Dropping it would hide from the model that the origin had no forecast; the
    neighbouring vintages both hold a value, so filling it in would be exactly
    the silent repair the architecture forbids.
    """
    data = datasets.pit_missingness()

    view = materialize(data, kind=ViewKind.TABULAR, horizon=4, origins=AtOrigin(datasets.at(8)))

    wind = values(view, "wind_fc")
    assert len(wind) == 4
    assert sum(datasets.is_missing(value) for value in wind) == 1


def _sentinel_row(data: SemanticDataset, origin: Any) -> float | None:
    """What one vintage published for the event time the feed is late on."""
    view = materialize(data, kind=ViewKind.TABULAR, horizon=4, origins=AtOrigin(origin))
    moment = datasets.at(datasets.MISSING_EVENT)
    rows = zip(column(view.keys, EVENT_TIME), column(view.X, "wind_fc"), strict=True)
    return next(value for event, value in rows if event == moment)


# -- event-time equivalence -------------------------------------------------

EQUIVALENCE_ORIGINS = 6
EQUIVALENCE_CONTEXT = 3
EQUIVALENCE_HORIZON = 3


def test_identical_vintages_materialize_exactly_like_event_time_data() -> None:
    """The two sources differ in fidelity, and — given equal values — in nothing else.

    Both are materialized at the same origins with the same window, so any
    difference left is a difference the planner introduced. Only
    ``OriginFidelity`` may differ, because only it is about where the origins
    came from rather than about what the data says.
    """
    dataset = datasets.point_in_time(
        instances=2,
        origins=EQUIVALENCE_ORIGINS,
        context=EQUIVALENCE_CONTEXT,
        horizon=EQUIVALENCE_HORIZON,
        stable=True,
    )
    frame = datasets.event_time(
        instances=2,
        targets=("price",),
        known=("wind_fc",),
        periods=EQUIVALENCE_CONTEXT + EQUIVALENCE_ORIGINS + EQUIVALENCE_HORIZON,
    )
    origins = AllOrigins()
    window: dict[str, Any] = {
        "kind": ViewKind.SEQUENCES,
        "context": EQUIVALENCE_CONTEXT,
        "horizon": EQUIVALENCE_HORIZON,
    }

    from_dataset = materialize(dataset, origins=origins, **window)
    from_frame = materialize(
        frame,
        origins=_between(dataset),
        **window,
    )

    assert isinstance(from_dataset, SequenceView)
    assert isinstance(from_frame, SequenceView)
    assert from_dataset.sample_ids == from_frame.sample_ids
    assert from_dataset.temporal.equals(from_frame.temporal)
    assert from_dataset.samples.equals(from_frame.samples)
    assert from_dataset.schema == from_frame.schema
    assert from_dataset.provenance.origin_fidelity is OriginFidelity.OBSERVED
    assert from_frame.provenance.origin_fidelity is OriginFidelity.SIMULATED


def test_the_sequences_of_the_two_sources_are_not_equal_when_the_vintages_disagree() -> None:
    """The equivalence above is a property of the data, not of the planner."""
    dataset = datasets.point_in_time(
        instances=2,
        origins=EQUIVALENCE_ORIGINS,
        context=EQUIVALENCE_CONTEXT,
        horizon=EQUIVALENCE_HORIZON,
    )
    frame = datasets.event_time(
        instances=2,
        targets=("price",),
        known=("wind_fc",),
        periods=EQUIVALENCE_CONTEXT + EQUIVALENCE_ORIGINS + EQUIVALENCE_HORIZON,
    )
    window: dict[str, Any] = {
        "kind": ViewKind.SEQUENCES,
        "context": EQUIVALENCE_CONTEXT,
        "horizon": EQUIVALENCE_HORIZON,
    }

    from_dataset = materialize(dataset, origins=AllOrigins(), **window)
    from_frame = materialize(frame, origins=_between(dataset), **window)

    assert isinstance(from_dataset, SequenceView)
    assert isinstance(from_frame, SequenceView)
    # Same samples, same targets, different information at each origin.
    assert from_dataset.sample_ids == from_frame.sample_ids
    assert column(from_dataset.temporal, "price") == column(from_frame.temporal, "price")
    assert column(from_dataset.temporal, "wind_fc") != column(from_frame.temporal, "wind_fc")


def test_a_sample_id_names_the_instance_and_origin_and_not_the_source() -> None:
    """Which is why the two views above can be compared row by row at all."""
    dataset = datasets.point_in_time(instances=2, stable=True)
    frame = datasets.event_time(instances=2, targets=("price",), known=("wind_fc",), periods=12)
    window: dict[str, Any] = {"kind": ViewKind.SEQUENCES, "context": 3, "horizon": 3}

    from_dataset = materialize(dataset, origins=AllOrigins(), **window)
    from_frame = materialize(frame, origins=_between(dataset), **window)

    assert set(column(from_dataset.temporal, SAMPLE_ID)) == set(
        column(from_frame.temporal, SAMPLE_ID)
    )


def _between(dataset: of.ForecastDataset) -> OriginsBetween:
    """The vintages of ``dataset``, as a selection an event-time frame understands.

    An event-time frame can simulate an origin at every event time it holds, so
    comparing the two sources means asking the frame for exactly the origins the
    vintages exist at.
    """
    return OriginsBetween(dataset.origins[0], dataset.origins[-1])
