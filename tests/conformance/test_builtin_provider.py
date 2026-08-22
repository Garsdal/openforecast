"""Step 10's "done when": the reference provider passes every capability it declares.

The parameters are generated from the descriptors ``BUILTIN_PROVIDER``
advertises, so this module has nothing model-specific in it and never will. A
model added to the built-in provider is fitted and forecast here the moment it
is advertised, over every shape and both semantic sources its own declaration
implies — and if it declares something it cannot do, this is where that is
found.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openforecast.models import ModelDescriptor
from openforecast.providers.builtin.provider import BUILTIN_PROVIDER
from tests.conformance import suite

DESCRIPTORS = BUILTIN_PROVIDER.descriptors()

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


def test_the_builtin_provider_advertises_something_to_conform_to() -> None:
    assert DESCRIPTORS
    assert CASES


@pytest.mark.parametrize(("descriptor", "case"), CASES)
def test_every_declared_capability_is_one_the_provider_has(
    descriptor: ModelDescriptor, case: suite.Case, tmp_path: Path
) -> None:
    suite.run_case(case, descriptor=descriptor, provider=BUILTIN_PROVIDER, store=tmp_path)


@pytest.mark.parametrize(("descriptor", "refusal"), REFUSALS)
def test_everything_it_declares_it_cannot_do_is_refused(
    descriptor: ModelDescriptor, refusal: suite.Refusal, tmp_path: Path
) -> None:
    suite.run_refusal(refusal, descriptor=descriptor, provider=BUILTIN_PROVIDER, store=tmp_path)


def test_the_same_model_is_fitted_from_both_semantic_sources() -> None:
    """The claim the view boundary exists to make, per advertised model."""
    for descriptor in DESCRIPTORS:
        sources = {case.source for case in suite.cases_for(descriptor)}
        assert len(sources) == 2, f"{descriptor.ref} is only conformance-tested on {sources}"
