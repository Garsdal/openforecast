"""What the provider says about itself, before anything is fitted.

A handshake is the only thing that happens at installation time, so what it
reports has to be right and it has to be cheap: the descriptors below are what
the engine plans every fit against, and answering them may not import a
forecasting library.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from openforecast_darts import PROVIDER_NAME, PROVIDER_VERSION, DartsProvider, catalog
from openforecast_darts.adapters.global_models import NHITS, TIDE
from openforecast_darts.adapters.local_models import THETA
from openforecast_darts.conversion import pandas_frequency

from openforecast.errors import UnknownModelError
from openforecast.models import ModelDescriptor
from openforecast.models.capabilities import MissingValueSupport
from openforecast.models.contract import OriginScope
from openforecast.protocol.vocabulary import ViewKind
from openforecast.views import Frequency

PROVIDER = DartsProvider()


def descriptor_for(name: str) -> ModelDescriptor:
    (found,) = [candidate for candidate in PROVIDER.descriptors() if candidate.ref.name == name]
    return found


def test_the_provider_is_the_namespace_of_the_models_it_advertises() -> None:
    assert PROVIDER.name == PROVIDER_NAME == "darts"
    assert PROVIDER.version == PROVIDER_VERSION
    assert {str(descriptor.ref) for descriptor in PROVIDER.descriptors()} == {
        "darts/theta",
        "darts/tide",
        "darts/nhits",
    }
    assert all(descriptor.provider == "darts" for descriptor in PROVIDER.descriptors())


def test_theta_declares_what_a_local_statistical_model_can_do() -> None:
    """The same declaration ``nixtla/autoarima`` makes, from another library.

    Which is the point: "local" is a statement about how a model learns, and two
    libraries that agree about that produce the same contract without either of
    them knowing about the other.
    """
    descriptor = descriptor_for("theta")
    contract = descriptor.training
    capabilities = descriptor.capabilities

    assert contract.view is ViewKind.SERIES
    assert contract.origin_scope is OriginScope.SINGLE
    assert not contract.horizon_bound_at_fit
    assert not contract.supports_unseen_instances

    assert (capabilities.instances.single, capabilities.instances.panel) == (True, True)
    assert (capabilities.targets.univariate, capabilities.targets.multivariate) == (True, False)
    # A Theta forecast is a function of the target's own history and nothing else.
    assert not capabilities.features.observed
    assert not capabilities.features.known
    assert not capabilities.features.static
    assert capabilities.missing_values is MissingValueSupport.UNSUPPORTED
    assert descriptor.lifecycle.requires_fit


def test_tide_declares_what_a_global_neural_model_can_do() -> None:
    """The declaration Step 13 exists to make good on, twice over.

    Line for line the declaration of ``nixtla/nhits``, which is the whole claim:
    switching libraries changes the reference and nothing else about how the
    engine plans a point-in-time fit.
    """
    descriptor = descriptor_for("tide")
    contract = descriptor.training
    capabilities = descriptor.capabilities

    assert contract.view is ViewKind.SEQUENCES
    assert contract.origin_scope is OriginScope.MULTIPLE
    assert contract.learns_across_origins
    assert contract.context_required
    assert contract.horizon_bound_at_fit
    assert contract.supports_unseen_instances

    assert (capabilities.instances.single, capabilities.instances.panel) == (True, True)
    assert (capabilities.targets.univariate, capabilities.targets.multivariate) == (True, False)
    assert capabilities.features.observed
    assert capabilities.features.known
    assert capabilities.features.static
    assert capabilities.missing_values is MissingValueSupport.REQUIRES_TRANSFORM
    assert descriptor.lifecycle.requires_fit


def test_two_models_of_one_architecture_may_declare_different_capabilities() -> None:
    """Darts' NHiTS is a past-covariates model; Nixtla's takes all three roles.

    Same architecture, same name, different capability — and the descriptor is
    where the difference is stated rather than discovered. Everything else about
    the two global models here is identical, which is what makes the feature
    declaration the only thing a caller has to read.
    """
    nhits = descriptor_for("nhits")
    tide = descriptor_for("tide")

    assert nhits.training == tide.training
    assert nhits.capabilities.features.observed
    assert not nhits.capabilities.features.known, "Darts' NHiTS takes no future covariate"
    assert not nhits.capabilities.features.static


def test_the_window_of_a_sequence_model_is_never_a_parameter() -> None:
    """The user must not state the context length or the horizon twice.

    Refused at the recipe boundary for every provider, and absent from what this
    one advertises — so there is nowhere for a second copy to be written down.
    Darts spells them differently from Nixtla, and both spellings are withheld.
    """
    for name in ("tide", "nhits"):
        schema = descriptor_for(name).parameters_schema
        for owned in ("input_chunk_length", "output_chunk_length", "random_state", "add_encoders"):
            assert owned not in schema["properties"], f"{owned} is OpenForecast's, not a parameter"


def test_the_declared_parameters_are_the_ones_that_are_accepted() -> None:
    """The schema a caller reads and the check a caller hits are one table."""
    schema = descriptor_for("theta").parameters_schema

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["properties"]["seasonality_period"] == {
        "type": "integer",
        "description": "Steps in one season. Inferred when unset.",
        "minimum": 1,
    }
    assert schema["properties"]["season_mode"]["enum"] == ["multiplicative", "additive", "none"]

    torch_schema = descriptor_for("tide").parameters_schema
    assert torch_schema["properties"]["dropout"] == {
        "type": "number",
        "description": "Dropout probability.",
        "minimum": 0,
        "maximum": 1,
    }


def test_a_model_this_provider_does_not_have_is_named_as_such() -> None:
    with pytest.raises(UnknownModelError, match=r"darts/tide"):
        catalog.adapter_for("darts/tft", "darts")

    with pytest.raises(UnknownModelError, match=r"not a model of the 'darts' provider"):
        catalog.adapter_for("nixtla/nhits", "darts")


def test_the_handshake_imports_no_forecasting_library() -> None:
    """Discovery is a question about descriptors, and it should stay cheap.

    In a fresh interpreter, because by the time the rest of this suite has run
    the libraries are loaded and the question would answer itself. ``darts``
    pulls in PyTorch and Lightning, and paying seconds of import to list three
    model names would make every ``openforecast providers list`` feel broken.
    """
    probe = (
        "import sys\n"
        "from openforecast_darts import DartsProvider\n"
        "DartsProvider().descriptors()\n"
        "print([name for name in ('darts', 'torch', 'pytorch_lightning') "
        "if name in sys.modules])\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert completed.stdout.strip() == "[]", "answering a handshake imported a library"


@pytest.mark.parametrize(
    ("frequency", "alias"),
    [
        ("15m", "15min"),
        ("1h", "1h"),
        ("30s", "30s"),
        ("1d", "1D"),
        # Pandas anchors a weekly offset to a weekday and a weekly series need
        # not start on the one it would pick, so a week is seven days.
        ("2w", "14D"),
        ("1mo", "1MS"),
    ],
)
def test_a_frequency_is_translated_into_the_alias_a_library_accepts(
    frequency: str, alias: str
) -> None:
    assert pandas_frequency(Frequency.parse(frequency)) == alias


def test_the_adapter_says_which_model_it_is() -> None:
    assert THETA.name == "theta"
    assert "theta" in repr(THETA)
    assert TIDE.name == "tide"
    assert "tide" in repr(TIDE)
    assert NHITS.name == "nhits"
    assert repr(PROVIDER) == f"DartsProvider(version={PROVIDER_VERSION}, models=3)"
