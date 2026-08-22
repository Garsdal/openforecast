"""The OpenForecast conformance suite, applied to this provider.

Nothing here is written per model. The parameters are generated from the
descriptors :class:`NixtlaProvider` advertises, so every capability it declares
becomes a fit that must succeed — over an event-time frame *and* over real
forecast vintages — and everything it withholds becomes a request that must be
refused before the provider is started. Advertising a second model exercises it
here the moment it appears in the catalog.

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
DESCRIPTORS = PROVIDER.descriptors()

CASES = [
    pytest.param(descriptor, case, id=f"{descriptor.ref.name}-{case.name}")
    for descriptor in DESCRIPTORS
    for case in suite.cases_for(descriptor)
]

REFUSALS = [
    pytest.param(descriptor, refusal, id=f"{descriptor.ref.name}-{refusal.name}")
    for descriptor in DESCRIPTORS
    for refusal in suite.refusals_for(descriptor)
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
    """The claim the view boundary exists to make, per advertised model.

    A ``SeriesView`` from an event-time frame and a ``SeriesView`` from real
    vintages at a selected origin are the same thing to this integration, and
    the suite asserts it was handed exactly that in both cases.
    """
    for descriptor in DESCRIPTORS:
        sources = {case.source for case in suite.cases_for(descriptor)}
        assert len(sources) == 2, f"{descriptor.ref} is only conformance-tested on {sources}"
