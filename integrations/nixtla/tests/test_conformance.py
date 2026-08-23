"""The OpenForecast conformance suite, applied to this provider.

Nothing here is written per model. Certified descriptors exercise every
capability of the shared StatsForecast and NeuralForecast protocol adapters —
over an event-time frame *and* over real forecast vintages — and everything
they withhold becomes a request that must be refused before the provider is
started. Reflected models inherit those adapters; discovery tests separately
verify their native family, capabilities and constructor schemas.

The suite ships with the OpenForecast repository rather than with the
distribution, so a run without the checkout beside it skips this module instead
of failing; see ``conftest.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openforecast_nixtla import NixtlaProvider

from openforecast.models import ModelDescriptor

suite = pytest.importorskip(
    "tests.conformance.suite",
    reason="the conformance suite lives in the OpenForecast checkout",
)

PROVIDER = NixtlaProvider()

#: What the suite must not be made to pay for. A neural model's default is a
#: thousand optimization steps, and none of them say anything about whether it
#: consumes a panel or refuses a second target — which is all the generated
#: cases assert. Only parameters the descriptor already advertises are accepted
#: here, so this cannot quietly change what is being conformance-tested.
PARAMETERS: dict[str, dict[str, object]] = {
    "autoarima": {},
    "nhits": {"max_steps": 2},
}

DESCRIPTORS = tuple(
    descriptor for descriptor in PROVIDER.descriptors() if descriptor.ref.name in PARAMETERS
)

CASES = [
    pytest.param(descriptor, case, id=f"{descriptor.ref.name}-{case.name}")
    for descriptor in DESCRIPTORS
    for case in suite.cases_for(descriptor, PARAMETERS.get(descriptor.ref.name))
]

REFUSALS = [
    pytest.param(descriptor, refusal, id=f"{descriptor.ref.name}-{refusal.name}")
    for descriptor in DESCRIPTORS
    for refusal in suite.refusals_for(descriptor, PARAMETERS.get(descriptor.ref.name))
]


def test_this_provider_advertises_something_to_conform_to() -> None:
    assert DESCRIPTORS
    assert CASES


@pytest.mark.parametrize(("descriptor", "case"), CASES)
def test_every_declared_capability_is_one_the_provider_has(
    descriptor: ModelDescriptor, case: object, tmp_path: Path
) -> None:
    suite.run_case(case, descriptor=descriptor, provider=PROVIDER, store=tmp_path)


@pytest.mark.parametrize(("descriptor", "refusal"), REFUSALS)
def test_everything_it_declares_it_cannot_do_is_refused(
    descriptor: ModelDescriptor, refusal: object, tmp_path: Path
) -> None:
    suite.run_refusal(refusal, descriptor=descriptor, provider=PROVIDER, store=tmp_path)


def test_each_model_is_fitted_from_both_semantic_sources() -> None:
    """The claim the view boundary exists to make, per certified model.

    A ``SeriesView`` from an event-time frame and a ``SeriesView`` from real
    vintages at a selected origin are the same thing to this integration, and
    the suite asserts it was handed exactly that in both cases.
    """
    for descriptor in DESCRIPTORS:
        sources = {case.source for case in suite.cases_for(descriptor)}
        assert len(sources) == 2, f"{descriptor.ref} is only conformance-tested on {sources}"
