"""The OpenForecast conformance suite, applied to this provider.

Nothing here is written per model. Certified descriptors exercise every
capability of the shared Darts local and global protocol adapters — over an
event-time frame *and* over real forecast vintages — and everything they
withhold becomes a request that must be refused before the provider is started.
Reflected models inherit those adapters; discovery tests separately verify
their native family, capabilities and constructor schemas.

This is the file Step 13 is really about: it is the *same* suite the Nixtla
integration runs, generated from declarations rather than written per library, so
``darts/tide`` is held to exactly the point-in-time contract ``nixtla/nhits`` is
held to without a line of it being restated here.

The suite ships with the OpenForecast repository rather than with the
distribution, so a run without the checkout beside it skips this module instead
of failing; see ``conftest.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openforecast_darts import DartsProvider

from openforecast.models import ModelDescriptor
from openforecast.protocol.vocabulary import ViewKind

suite = pytest.importorskip(
    "tests.conformance.suite",
    reason="the conformance suite lives in the OpenForecast checkout",
)

PROVIDER = DartsProvider()

#: What the suite must not be made to pay for. A neural model's default is a
#: hundred epochs, and none of them say anything about whether it consumes a
#: panel or refuses a second target — which is all the generated cases assert.
#: Only parameters the descriptor already advertises are accepted here, so this
#: cannot quietly change what is being conformance-tested.
PARAMETERS: dict[str, dict[str, object]] = {
    "theta": {},
    "tide": {"n_epochs": 1},
    "nhits": {"n_epochs": 1},
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


def test_the_sequence_model_that_takes_known_features_is_tested_on_real_vintages() -> None:
    """The claim the view boundary exists to make, on the model that can carry it.

    A point-in-time dataset holds at least one feature, and the only role that
    survives to a forecast origin without an imputation is the known one — so
    ``darts/tide`` is the model whose generated cases include real vintages, and
    a regression that stopped generating them would show up here rather than as
    a suite that quietly got smaller.
    """
    (tide,) = [descriptor for descriptor in DESCRIPTORS if descriptor.ref.name == "tide"]

    sources = {case.source for case in suite.cases_for(tide, PARAMETERS["tide"])}

    assert tide.training.view is ViewKind.SEQUENCES
    assert len(sources) == 2, f"{tide.ref} is only conformance-tested on {sources}"
