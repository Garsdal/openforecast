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
from openforecast_nixtla import PROVIDER_NAME, PROVIDER_VERSION, NixtlaProvider, catalog
from openforecast_nixtla.adapters.statsforecast import AUTOARIMA
from openforecast_nixtla.conversion import pandas_frequency

from openforecast.errors import UnknownModelError
from openforecast.models.capabilities import MissingValueSupport
from openforecast.models.contract import OriginScope
from openforecast.protocol.vocabulary import ViewKind
from openforecast.views import Frequency

PROVIDER = NixtlaProvider()


def test_the_provider_is_the_namespace_of_the_models_it_advertises() -> None:
    assert PROVIDER.name == PROVIDER_NAME == "nixtla"
    assert PROVIDER.version == PROVIDER_VERSION
    assert {str(descriptor.ref) for descriptor in PROVIDER.descriptors()} == {"nixtla/autoarima"}
    assert all(descriptor.provider == "nixtla" for descriptor in PROVIDER.descriptors())


def test_autoarima_declares_what_a_local_statistical_model_can_do() -> None:
    (descriptor,) = [
        candidate for candidate in PROVIDER.descriptors() if candidate.ref.name == "autoarima"
    ]
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


def test_the_declared_parameters_are_the_ones_that_are_accepted() -> None:
    """The schema a caller reads and the check a caller hits are one table."""
    (descriptor,) = PROVIDER.descriptors()
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
    with pytest.raises(UnknownModelError, match=r"nixtla/autoarima"):
        catalog.adapter_for("nixtla/autoets", "nixtla")

    with pytest.raises(UnknownModelError, match=r"not a model of the 'nixtla' provider"):
        catalog.adapter_for("darts/tcn", "nixtla")


def test_the_handshake_imports_no_forecasting_library() -> None:
    """Discovery is a question about descriptors, and it should stay cheap.

    In a fresh interpreter, because by the time the rest of this suite has run
    the library is loaded and the question would answer itself.
    """
    probe = (
        "import sys\n"
        "from openforecast_nixtla import NixtlaProvider\n"
        "NixtlaProvider().descriptors()\n"
        "print('statsforecast' in sys.modules)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert completed.stdout.strip() == "False", "answering a handshake imported the library"


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
    assert repr(PROVIDER) == f"NixtlaProvider(version={PROVIDER_VERSION}, models=1)"
