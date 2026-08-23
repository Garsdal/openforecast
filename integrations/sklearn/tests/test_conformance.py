"""The OpenForecast conformance suite, applied to this provider.

Nothing here is written per model. The certified descriptor exercises every
capability of the shared sklearn protocol adapter — over an event-time frame
*and* over real forecast vintages — and everything it withholds becomes a
request that must be refused before the provider is started. Reflected
estimators inherit that adapter; discovery tests separately verify their native
tags and constructor schemas.

This is the file that makes Step 18 a step rather than an addition. The suite was
written against three sequence and series providers, and it is generated from
declarations rather than from libraries — so a ``TabularView`` consumer that had
to be special-cased anywhere in it would mean the third view was never really a
peer of the other two. It is not special-cased anywhere: the tabular cases below
are the ones the suite already knew how to generate.

The suite ships with the OpenForecast repository rather than with the
distribution, so a run without the checkout beside it skips this module instead
of failing; see ``conftest.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openforecast_sklearn import SklearnProvider

from openforecast.models import ModelDescriptor
from openforecast.protocol.vocabulary import ViewKind

suite = pytest.importorskip(
    "tests.conformance.suite",
    reason="the conformance suite lives in the OpenForecast checkout",
)

PROVIDER = SklearnProvider()

#: What the suite must not be made to pay for. A boosted model's default is a
#: hundred iterations, and none of them say anything about whether it consumes a
#: panel or refuses a second target — which is all the generated cases assert.
#: Only parameters the descriptor already advertises are accepted here, so this
#: cannot quietly change what is being conformance-tested.
PARAMETERS: dict[str, dict[str, object]] = {
    "hist-gradient-boosting": {"max_iter": 5},
}

# Full conformance is a certification tier, not the discovery mechanism. Every
# reflected regressor shares the already-conformant tabular driver; these are
# the hand-tuned entries whose stronger capability declarations are exercised
# exhaustively on every run.
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


def test_the_tabular_model_is_tested_on_real_vintages() -> None:
    """The claim the view boundary exists to make, on the view it was made for.

    A ``TabularView`` of real vintages is the shape the whole design was drawn
    for — one row per instance, origin and lead, holding what was knowable at
    that origin — so a generated suite that only ever built rows out of one
    freshest series would be testing the easy half. Two sources, asserted here
    rather than assumed, so a regression that stopped generating the second shows
    up as a failure instead of a suite that quietly got smaller.
    """
    (estimator,) = DESCRIPTORS

    sources = {case.source for case in suite.cases_for(estimator, PARAMETERS[estimator.ref.name])}

    assert estimator.training.view is ViewKind.TABULAR
    assert len(sources) == 2, f"{estimator.ref} is only conformance-tested on {sources}"
