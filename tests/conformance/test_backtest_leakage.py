"""The leakage guarantee, asserted where a leak would be visible: the provider.

```text
origin 08 -> target 10 -> wind = the vintage of 08
origin 09 -> target 10 -> wind = 999999
```

Backtesting at origin 08 must hand the provider what 08 published for event time
10, and can never hand it the 999999 that 09 published for the same event —
*hand it* being the whole point of this file. Step 17 asserts the guarantee at
the manifest, which records the origin an artifact was materialized from, and at
the forecast value, which is a number a model computed. Neither observes what
the provider was actually given, so a planner change that reached one vintage
past the origin would surface as a slightly different metric and nothing else.

Here the provider is wrapped in the conformance suite's
:class:`~tests.conformance.suite.Recording` client, and every table of every
view it received is searched for the poison. It is the crudest possible leakage
test and the one worth having permanently: it fails at the boundary the leak
would cross rather than behind a plausible number.

:mod:`tests.conformance.test_point_in_time` asserts the same property directly
on the ``ViewPlanner``. This one asserts it through ``of.backtest``, which is
how a caller reaches the planner at all — via ``of.fit`` and ``of.forecast``, at
a fold whose training data is a *different dataset* rather than the same one
with a cut-off remembered alongside it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

import openforecast as of
from openforecast.models import TrainingContract
from openforecast.views import ViewKind
from tests import providers
from tests.conformance import datasets, suite

#: The event step every vintage sits at. Each publishes the event times up to
#: itself and the ``HORIZON`` after it, so the origins evaluated below have a
#: full history behind them and covered forecast windows ahead of them.
VINTAGES = tuple(range(2, 14))
HORIZON = 3

#: The origin the backtest is run at, the vintage published immediately after it
#: — the poisoned one — and the event time the two disagree about.
EVALUATED = 8
POISONED = 9
SENTINEL = 10

#: What the evaluated vintage published for that event time, and therefore the
#: only value of it any view may hold.
PUBLISHED = datasets.known_value(0, SENTINEL, EVALUATED)

CONTRACTS = {
    ViewKind.SERIES: TrainingContract.series(),
    ViewKind.SEQUENCES: TrainingContract.sequences(),
    ViewKind.TABULAR: TrainingContract.tabular(),
}

#: What each contract can be fitted with. A series model holds one complete
#: series and so learns from one origin by declaration; the two forward-looking
#: views learn from every vintage whose window the training data covers.
PLANS = {
    ViewKind.SERIES: of.FitPlan(origins=of.LatestOrigin()),
    ViewKind.SEQUENCES: of.FitPlan(origins=of.AllOrigins(), window=of.WindowPlan(context=3)),
    ViewKind.TABULAR: of.FitPlan(origins=of.AllOrigins()),
}

#: Every table any of the four views keeps its columns in. A view is searched by
#: attribute rather than by type so that one sweep covers all of them — and a
#: view that grows a table has to be named here to stay covered.
VIEW_TABLES = ("X", "y", "keys", "temporal", "samples", "static", "history", "future")


def poisoned_vintages() -> of.ForecastDataset:
    """Real vintages, one of which publishes a value no feed ever would.

    Every known value names the origin that issued it, so a leak is legible
    beyond the sentinel: :func:`~tests.conformance.datasets.origin_of` reports
    which vintage any value in a view came from.
    """
    rows = [
        {
            datasets.ORIGIN_TIME: datasets.at(origin),
            datasets.EVENT_TIME: datasets.at(event),
            "price": datasets.target_value(0, event),
            "wind_fc": (
                datasets.POISON
                if (origin, event) == (POISONED, SENTINEL)
                else datasets.known_value(0, event, origin)
            ),
        }
        for origin in VINTAGES
        for event in range(origin + HORIZON + 1)
    ]
    return datasets.forecast_dataset(rows)


def feature_values(view: object, name: str) -> list[Any]:
    """Every value of one feature column, wherever in a view it is kept."""
    found: list[Any] = []
    for attribute in VIEW_TABLES:
        table = getattr(view, attribute, None)
        if isinstance(table, pa.Table) and name in table.column_names:
            found.extend(datasets.column(table, name))
    return found


def handed_to(kind: ViewKind, store: Path, *, origin: int = EVALUATED) -> list[Any]:
    """Every ``wind_fc`` value a backtest at one origin gave the provider."""
    descriptor = providers.descriptor(f"{kind}-consumer", training=CONTRACTS[kind])
    recording = suite.Recording(providers.StubProvider(models=(descriptor,)))
    client = suite.client_for(descriptor, recording, store)

    of.backtest(
        models=[of.Candidate(str(descriptor.ref), plan=PLANS[kind])],
        data=poisoned_vintages(),
        validation=of.ForecastOriginValidation(
            horizon=HORIZON, origins=of.AtOrigin(datasets.at(origin))
        ),
        metrics=[of.MAE()],
        client=client,
    )

    views = [*recording.fit_views, *recording.forecast_views]
    assert views, "the backtest never reached the provider, so it proved nothing"
    return [value for view in views for value in feature_values(view, "wind_fc")]


@pytest.mark.parametrize("kind", list(CONTRACTS), ids=str)
def test_a_poisoned_vintage_never_reaches_the_provider_of_an_earlier_origin(
    kind: ViewKind, tmp_path: Path
) -> None:
    """Whatever view the model declared, and whether it was to fit or to forecast."""
    handed = handed_to(kind, tmp_path)

    assert datasets.POISON not in handed
    # Nor did anything else of a later vintage arrive on the side: every value
    # names the origin that issued it, and none of them is after this one.
    assert max(datasets.origin_of(value) for value in handed) <= EVALUATED


def test_the_value_the_evaluated_vintage_published_does_reach_it(tmp_path: Path) -> None:
    """The negative above is only worth having if the positive holds.

    A backtest that materialized nothing forward-looking at all would satisfy
    every assertion above, so the vintage of the evaluated origin has to be
    shown reaching the provider for exactly the event time the poison replaced.
    """
    handed = handed_to(ViewKind.TABULAR, tmp_path)

    assert PUBLISHED in handed
    assert datasets.origin_of(PUBLISHED) == EVALUATED


def test_the_poison_does_reach_the_origin_that_published_it(tmp_path: Path) -> None:
    """And the sweep can see it, which is what makes its absence above evidence.

    Backtesting at the poisoned vintage itself is not a leak — it is that
    vintage's own information — so the same search that finds nothing at 08
    finds the 999999 at 09.
    """
    handed = handed_to(ViewKind.TABULAR, tmp_path, origin=POISONED)

    assert datasets.POISON in handed
