"""The suite itself, against declarations the built-in provider does not make.

``builtin/seasonal-naive`` consumes a ``SeriesView`` and declares every
capability there is, so running the harness against it exercises one of the
three view branches and none of the refusals. The models of Steps 11 to 14 will
exercise the rest — a ``SequenceView`` consumer that learns across vintages, a
``TabularView`` consumer, a model that takes no features — and the harness has
to be known to work *before* it is the thing judging an integration.

So the declarations below are made by a stub whose only job is to consume
whatever it is handed. What is under test here is the suite: that a contract
naming sequences is fitted from both semantic sources and hands the provider a
``SequenceView`` in both, and that a capability withheld turns into a refusal
rather than into one fewer check.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openforecast.models import (
    FeatureCapabilities,
    InstanceCapabilities,
    MissingValueSupport,
    ModelCapabilities,
    ModelDescriptor,
    TargetCapabilities,
    TrainingContract,
)
from openforecast.views import SequenceView, SeriesView, SourceKind, TabularView, ViewKind
from tests import providers
from tests.conformance import suite

EVERYTHING = ModelCapabilities(
    instances=InstanceCapabilities(single=True, panel=True),
    targets=TargetCapabilities(univariate=True, multivariate=True),
    features=FeatureCapabilities(observed=True, known=True, static=True),
    missing_values=MissingValueSupport.NATIVE,
)


def stub(
    name: str,
    *,
    training: TrainingContract | None = None,
    capabilities: ModelCapabilities | None = None,
) -> ModelDescriptor:
    return providers.descriptor(
        name,
        training=training,
        capabilities=EVERYTHING if capabilities is None else capabilities,
    )


def run(descriptor: ModelDescriptor, store: Path) -> None:
    """Every case and every refusal the declaration implies, against the stub."""
    provider = providers.StubProvider(models=(descriptor,))
    for case in suite.cases_for(descriptor):
        suite.run_case(case, descriptor=descriptor, provider=provider, store=store / case.name)
    for refusal in suite.refusals_for(descriptor):
        suite.run_refusal(
            refusal, descriptor=descriptor, provider=provider, store=store / refusal.name
        )


# -- one contract per view --------------------------------------------------

CONTRACTS = {
    ViewKind.SERIES: TrainingContract.series(),
    ViewKind.SEQUENCES: TrainingContract.sequences(),
    ViewKind.TABULAR: TrainingContract.tabular(),
}
VIEWS = {
    ViewKind.SERIES: SeriesView,
    ViewKind.SEQUENCES: SequenceView,
    ViewKind.TABULAR: TabularView,
}


@pytest.mark.parametrize("kind", list(CONTRACTS), ids=str)
def test_a_model_is_conformance_tested_on_whichever_view_it_declares(
    kind: ViewKind, tmp_path: Path
) -> None:
    run(stub(f"{kind}-consumer", training=CONTRACTS[kind]), tmp_path)


@pytest.mark.parametrize("kind", list(CONTRACTS), ids=str)
def test_both_semantic_sources_reach_every_declared_view(kind: ViewKind) -> None:
    """The Step 10 promise: declaring a view buys tests against both sources."""
    descriptor = stub(f"{kind}-consumer", training=CONTRACTS[kind])

    cases = suite.cases_for(descriptor)

    assert {case.source for case in cases} == {
        SourceKind.TIME_SERIES,
        SourceKind.FORECAST_DATASET,
    }
    assert suite.VIEW_TYPES[kind] is VIEWS[kind]


@pytest.mark.parametrize("kind", list(CONTRACTS), ids=str)
def test_the_provider_receives_only_the_view_it_declared(kind: ViewKind, tmp_path: Path) -> None:
    """Whatever the source was, and it cannot tell which it was."""
    descriptor = stub(f"{kind}-consumer", training=CONTRACTS[kind])
    provider = providers.StubProvider(models=(descriptor,))
    recorded: list[type[object]] = []

    for case in suite.cases_for(descriptor):
        recording = suite.Recording(provider)
        client = suite.client_for(descriptor, recording, tmp_path / case.name)
        client.fit(str(descriptor.ref), case.data(), horizon=case.horizon, plan=case.plan)
        recorded.extend(type(view) for view in recording.fit_views)

    assert set(recorded) == {VIEWS[kind]}


# -- a capability withheld is a refusal, not a gap --------------------------


def test_a_single_series_model_refuses_a_panel(tmp_path: Path) -> None:
    descriptor = stub(
        "single-only",
        capabilities=EVERYTHING.model_copy(
            update={"instances": InstanceCapabilities(single=True, panel=False)}
        ),
    )

    assert "a panel" in [refusal.name for refusal in suite.refusals_for(descriptor)]
    assert {case.instances for case in suite.cases_for(descriptor)} == {1}
    run(descriptor, tmp_path)


def test_a_univariate_model_refuses_a_second_target(tmp_path: Path) -> None:
    descriptor = stub(
        "univariate-only",
        capabilities=EVERYTHING.model_copy(
            update={"targets": TargetCapabilities(univariate=True, multivariate=False)}
        ),
    )

    assert "two targets" in [refusal.name for refusal in suite.refusals_for(descriptor)]
    assert {case.targets for case in suite.cases_for(descriptor)} == {("load",)}
    run(descriptor, tmp_path)


def test_a_model_that_takes_no_features_refuses_them_and_has_no_vintages_to_read(
    tmp_path: Path,
) -> None:
    """A point-in-time frame holds at least one feature, so there is nothing to offer."""
    descriptor = stub(
        "no-features",
        capabilities=EVERYTHING.model_copy(update={"features": FeatureCapabilities()}),
    )

    assert "a known feature" in [refusal.name for refusal in suite.refusals_for(descriptor)]
    assert {case.source for case in suite.cases_for(descriptor)} == {SourceKind.TIME_SERIES}
    run(descriptor, tmp_path)


def test_a_model_that_cannot_see_a_gap_refuses_data_that_has_one(tmp_path: Path) -> None:
    descriptor = stub(
        "no-missing",
        capabilities=EVERYTHING.model_copy(
            update={"missing_values": MissingValueSupport.UNSUPPORTED}
        ),
    )

    assert "a gap in the data" in [refusal.name for refusal in suite.refusals_for(descriptor)]
    run(descriptor, tmp_path)


def test_a_series_model_refuses_to_learn_across_vintages(tmp_path: Path) -> None:
    """Declared by ``origin_scope``, refused by the planner, generated from neither."""
    descriptor = stub("series-consumer", training=TrainingContract.series())

    assert "every vintage at once" in [refusal.name for refusal in suite.refusals_for(descriptor)]
    run(descriptor, tmp_path)


def test_a_model_that_learns_across_origins_is_given_every_vintage(tmp_path: Path) -> None:
    descriptor = stub("sequences-consumer", training=TrainingContract.sequences())

    refusals = [refusal.name for refusal in suite.refusals_for(descriptor)]

    assert "every vintage at once" not in refusals
    run(descriptor, tmp_path)
