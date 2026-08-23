"""What the provider says about itself, before anything is fitted.

A handshake is the only thing that happens at installation time, so what it
reports has to be right and it has to be cheap: the descriptors below are what
the engine plans every fit against, and answering reflects the installed
StatsForecast and NeuralForecast catalogs without fitting models.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from openforecast_nixtla import PROVIDER_NAME, PROVIDER_VERSION, NixtlaProvider, catalog
from openforecast_nixtla.adapters.neuralforecast import NHITS
from openforecast_nixtla.adapters.statsforecast import AUTOARIMA
from openforecast_nixtla.conversion import pandas_frequency

from openforecast.errors import UnknownModelError
from openforecast.models import ModelDescriptor
from openforecast.models.capabilities import MissingValueSupport
from openforecast.models.contract import OriginScope
from openforecast.protocol.vocabulary import ViewKind
from openforecast.views import Frequency

PROVIDER = NixtlaProvider()


def descriptor_for(name: str) -> ModelDescriptor:
    (found,) = [candidate for candidate in PROVIDER.descriptors() if candidate.ref.name == name]
    return found


def test_the_provider_is_the_namespace_of_the_models_it_advertises() -> None:
    assert PROVIDER.name == PROVIDER_NAME == "nixtla"
    assert PROVIDER.version == PROVIDER_VERSION
    refs = {str(descriptor.ref) for descriptor in PROVIDER.descriptors()}
    assert {"nixtla/autoarima", "nixtla/autoets", "nixtla/nhits", "nixtla/patchtst"} <= refs
    assert all(descriptor.provider == "nixtla" for descriptor in PROVIDER.descriptors())


def test_autoarima_declares_what_a_local_statistical_model_can_do() -> None:
    descriptor = descriptor_for("autoarima")
    contract = descriptor.training
    capabilities = descriptor.capabilities

    assert contract.view is ViewKind.SERIES
    assert contract.origin_scope is OriginScope.SINGLE
    assert not contract.horizon_bound_at_fit
    assert not contract.supports_unseen_instances

    assert (capabilities.instances.single, capabilities.instances.panel) == (True, True)
    assert (capabilities.targets.univariate, capabilities.targets.multivariate) == (True, False)
    assert capabilities.features.known
    assert not capabilities.features.observed
    assert not capabilities.features.static
    assert capabilities.missing_values is MissingValueSupport.UNSUPPORTED
    assert descriptor.lifecycle.requires_fit


def test_nhits_declares_what_a_global_neural_model_can_do() -> None:
    """The declaration Step 12 exists to make good on.

    Every line of it is exercised: the sequences contract by the conformance
    suite, the covariate roles and the unseen instance by ``test_nhits.py``, and
    the bound horizon by the engine refusing a request for another one.
    """
    descriptor = descriptor_for("nhits")
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


def test_the_window_of_a_sequence_model_is_never_a_parameter() -> None:
    """The user must not state the context length or the horizon twice.

    Refused at the recipe boundary for every provider, and absent from what this
    one advertises — so there is nowhere for a second copy to be written down.
    """
    schema = descriptor_for("nhits").parameters_schema

    for owned in ("h", "input_size", "random_seed", "futr_exog_list", "hist_exog_list"):
        assert owned not in schema["properties"], f"{owned} is OpenForecast's, not a parameter"


def test_the_declared_parameters_are_the_ones_that_are_accepted() -> None:
    """The schema a caller reads and the check a caller hits are one table."""
    descriptor = descriptor_for("autoarima")
    schema = descriptor.parameters_schema

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert "season_length" in schema["properties"]
    assert schema["properties"]["season_length"] == {
        "type": "integer",
        "description": "Steps of the data's frequency in one season.",
        "minimum": 1,
    }
    assert schema["properties"]["ic"]["enum"] == ["aicc", "aic", "bic"]
    # ``alias`` would rename the column the answer is read from, so it is not
    # something a caller may set.
    assert "alias" not in schema["properties"]


def test_a_model_this_provider_does_not_have_is_named_as_such() -> None:
    with pytest.raises(UnknownModelError, match=r"nixtla/nhits"):
        catalog.adapter_for("nixtla/not-a-model", "nixtla")

    with pytest.raises(UnknownModelError, match=r"not a model of the 'nixtla' provider"):
        catalog.adapter_for("darts/tcn", "nixtla")


def test_the_handshake_discovers_from_the_installed_forecasting_libraries() -> None:
    """The installed Nixtla versions, rather than copied class lists, are authoritative."""
    probe = (
        "import sys\n"
        "from openforecast_nixtla import NixtlaProvider\n"
        "NixtlaProvider().descriptors()\n"
        "print([name for name in ('statsforecast', 'neuralforecast', 'torch') "
        "if name in sys.modules])\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert "statsforecast" in completed.stdout
    assert "neuralforecast" in completed.stdout


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
    assert AUTOARIMA.name == "autoarima"
    assert "autoarima" in repr(AUTOARIMA)
    assert NHITS.name == "nhits"
    assert "nhits" in repr(NHITS)
    assert repr(PROVIDER) == (
        f"NixtlaProvider(version={PROVIDER_VERSION}, models={len(catalog.model_names())})"
    )


def test_discovered_models_inherit_their_native_protocol_and_parameters() -> None:
    autoets = descriptor_for("autoets")
    patchtst = descriptor_for("patchtst")

    assert autoets.training.view is ViewKind.SERIES
    assert "season_length" in autoets.parameters_schema["properties"]
    assert patchtst.training.view is ViewKind.SEQUENCES
    assert "max_steps" in patchtst.parameters_schema["properties"]
