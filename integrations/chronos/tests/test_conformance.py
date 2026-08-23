"""The OpenForecast conformance suite, applied to this provider.

Nothing here is written per model. The parameters are generated from the
descriptors :class:`ChronosProvider` advertises, so every capability it declares
becomes a forecast that must succeed — over an event-time frame *and* over real
forecast vintages — and everything it withholds becomes a request that must be
refused before the provider is started.

This is the file that makes Step 23 a step rather than an addition. The suite
was written against four trainable providers, and it is generated from
declarations rather than from lifecycles — so a pretrained model that had to be
special-cased anywhere in it would mean zero-shot use was bolted on rather than
expressed. It is not special-cased: a descriptor with ``training=None`` gets the
same cases with the fit half removed, the same refusals moved to the forecast,
and one refusal the others cannot have — being fitted at all.

The pipeline is a stand-in, because a checkpoint is a download and none of the
generated assertions are about the numbers. ``test_chronos2.py`` runs the real
one.

The suite ships with the OpenForecast repository rather than with the
distribution, so a run without the checkout beside it skips this module instead
of failing; see ``conftest.py``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fake import FakePipeline
from openforecast_chronos import ChronosProvider
from openforecast_chronos.adapter import CHRONOS_2

from openforecast.models import ModelDescriptor
from openforecast.models.capabilities import MissingValueSupport

suite = pytest.importorskip(
    "tests.conformance.suite",
    reason="the conformance suite lives in the OpenForecast checkout",
)

PROVIDER = ChronosProvider()
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


@pytest.fixture(autouse=True)
def _pipeline() -> Iterator[FakePipeline]:
    """The loaded-pipeline cache, filled with a stand-in for the duration.

    Reaching into the adapter's own cache rather than adding a seam to the
    production class: the cache exists because a provider process serves many
    requests, and a test that primes it is using it for what it is.
    """
    fake = FakePipeline()
    CHRONOS_2._pipeline = fake
    try:
        yield fake
    finally:
        CHRONOS_2._pipeline = None


@pytest.mark.parametrize(("descriptor", "case"), CASES)
def test_a_declared_capability_is_one_the_provider_serves(
    descriptor: ModelDescriptor, case: object, tmp_path: Path
) -> None:
    suite.run_case(case, descriptor=descriptor, provider=PROVIDER, store=tmp_path)


@pytest.mark.parametrize(("descriptor", "refusal"), REFUSALS)
def test_an_undeclared_capability_is_refused(
    descriptor: ModelDescriptor, refusal: object, tmp_path: Path
) -> None:
    suite.run_refusal(refusal, descriptor=descriptor, provider=PROVIDER, store=tmp_path)


def test_the_suite_generates_the_zero_shot_shape_of_the_cases() -> None:
    """The generated cases exist, and none of them is a fit.

    A suite that silently generated nothing for a pretrained model would pass by
    not running, which is the failure mode a declaration-driven suite is most
    prone to.
    """
    assert CASES
    assert all(descriptor.training is None for descriptor in DESCRIPTORS)
    assert all(not descriptor.is_fittable for descriptor in DESCRIPTORS)
    names = {refusal.values[1].name for refusal in REFUSALS}
    assert "being fitted" in names
    # Both semantic sources reach a model that never learns from either.
    case_names = {case.values[1].name for case in CASES}
    assert any(name.startswith("event-time") for name in case_names)
    assert any(name.startswith("point-in-time") for name in case_names)


def test_missing_values_are_declared_native_so_a_gap_is_not_refused() -> None:
    """The one capability that changes which cases the suite generates.

    A model that could not consume a ``NaN`` would get a refusal for a gap and
    no observed-feature cases at all, since an observed feature has no value past
    its own origin. Chronos reads an unobserved step as an unobserved step, so
    the suite hands it both.
    """
    (descriptor,) = DESCRIPTORS
    assert descriptor.capabilities.missing_values is MissingValueSupport.NATIVE
    assert "a gap in the data" not in {refusal.values[1].name for refusal in REFUSALS}
