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
from openforecast_sktime import PROVIDER_NAME, PROVIDER_VERSION, SktimeProvider, catalog
from openforecast_sktime.adapters.local_models import THETA
from openforecast_sktime.adapters.panel_models import POOLED_TREES
from openforecast_sktime.conversion import instance_labels, pandas_frequency

from openforecast.errors import UnknownModelError
from openforecast.models import ModelDescriptor
from openforecast.models.capabilities import MissingValueSupport
from openforecast.models.contract import OriginScope
from openforecast.protocol.vocabulary import ViewKind
from openforecast.views import Frequency

PROVIDER = SktimeProvider()


def descriptor_for(name: str) -> ModelDescriptor:
    (found,) = [candidate for candidate in PROVIDER.descriptors() if candidate.ref.name == name]
    return found


def test_the_provider_is_the_namespace_of_the_models_it_advertises() -> None:
    assert PROVIDER.name == PROVIDER_NAME == "sktime"
    assert PROVIDER.version == PROVIDER_VERSION
    refs = {str(descriptor.ref) for descriptor in PROVIDER.descriptors()}
    assert {"sktime/theta", "sktime/pooled-trees"} <= refs
    assert len(refs) > 2
    assert all(descriptor.provider == "sktime" for descriptor in PROVIDER.descriptors())


def test_theta_declares_what_a_local_statistical_model_can_do() -> None:
    """The same declaration ``nixtla/autoarima`` and ``darts/theta`` make.

    Which is the point: "local" is a statement about how a model learns, and
    three libraries that agree about that produce the same contract without any
    of them knowing about the others.
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


def test_the_pooled_model_declares_what_a_global_panel_model_can_do() -> None:
    """A global model, with the one difference this library actually has.

    Everything ``darts/tide`` declares about learning across origins, needing a
    context length and generalizing to an unseen instance is declared here too —
    and the horizon is *not* bound, because a recursive reduction rolls one step
    at a time rather than baking the horizon into an architecture. That is a
    capability difference between two global models, stated in the descriptor
    rather than discovered at inference.
    """
    descriptor = descriptor_for("pooled-trees")
    contract = descriptor.training
    capabilities = descriptor.capabilities

    assert contract.view is ViewKind.SEQUENCES
    assert contract.origin_scope is OriginScope.MULTIPLE
    assert contract.learns_across_origins
    assert contract.context_required
    assert not contract.horizon_bound_at_fit
    assert contract.supports_unseen_instances

    assert (capabilities.instances.single, capabilities.instances.panel) == (True, True)
    assert (capabilities.targets.univariate, capabilities.targets.multivariate) == (True, False)
    # sktime has one exogenous frame, and a value in it has to exist at the event
    # time being forecast — which an observed feature does not.
    assert not capabilities.features.observed
    assert capabilities.features.known
    assert capabilities.features.static
    assert capabilities.missing_values is MissingValueSupport.REQUIRES_TRANSFORM
    assert descriptor.lifecycle.requires_fit


def test_the_window_of_a_panel_model_is_never_a_parameter() -> None:
    """The user must not state the context length or the horizon twice.

    Refused at the recipe boundary for every provider, and absent from what this
    one advertises — so there is nowhere for a second copy to be written down.
    sktime spells them differently again, and those spellings are withheld too.
    """
    schema = descriptor_for("pooled-trees").parameters_schema

    for owned in ("window_length", "fh", "pooling", "strategy", "random_state"):
        assert owned not in schema["properties"], f"{owned} is OpenForecast's, not a parameter"


def test_the_declared_parameters_are_the_ones_that_are_accepted() -> None:
    """The schema a caller reads and the check a caller hits are one table."""
    schema = descriptor_for("theta").parameters_schema

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["properties"]["sp"] == {
        "type": "integer",
        "description": "Steps in one seasonal period. 1 fits no seasonality.",
        "minimum": 1,
    }

    pooled = descriptor_for("pooled-trees").parameters_schema
    assert pooled["properties"]["learning_rate"] == {
        "type": "number",
        "description": "Shrinkage applied to each tree.",
        "minimum": 0,
    }


def test_a_model_this_provider_does_not_have_is_named_as_such() -> None:
    with pytest.raises(UnknownModelError, match=r"sktime/theta"):
        catalog.adapter_for("sktime/not-a-model", "sktime")

    with pytest.raises(UnknownModelError, match=r"not a model of the 'sktime' provider"):
        catalog.adapter_for("darts/theta", "sktime")


def test_the_handshake_discovers_from_the_installed_forecasting_library() -> None:
    """The installed sktime registry, rather than a copied class list, is authoritative."""
    probe = (
        "import sys\n"
        "from openforecast_sktime import SktimeProvider\n"
        "SktimeProvider().descriptors()\n"
        "print([name for name in ('sktime', 'sklearn', 'statsmodels') if name in sys.modules])\n"
    )
    completed = subprocess.run(  # noqa: S603 - the command is this interpreter
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert "sktime" in completed.stdout


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


def test_a_panel_label_stands_for_a_position_rather_than_for_a_key() -> None:
    """An instance key is the caller's data; a pandas index level is not.

    Two keys that stringify the same would collide in a panel index, and the
    collision would silently mix two series. The position cannot collide, and it
    is what maps a returned row back to the instance it is about.
    """
    assert instance_labels([("DE",), ("FR",)]) == ("instance-0", "instance-1")
    assert instance_labels([]) == ()


def test_the_adapter_says_which_model_it_is() -> None:
    assert THETA.name == "theta"
    assert "theta" in repr(THETA)
    assert POOLED_TREES.name == "pooled-trees"
    assert "pooled-trees" in repr(POOLED_TREES)
    assert repr(PROVIDER) == (
        f"SktimeProvider(version={PROVIDER_VERSION}, models={len(catalog.model_names())})"
    )
