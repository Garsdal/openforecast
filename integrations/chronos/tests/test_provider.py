"""What the provider says about itself, before anything is forecast.

A handshake is the only thing that happens at installation time, so what it
reports has to be right and it has to be cheap. Cheap matters more here than in
the other integrations: this one depends on ``torch``, and a handshake that
imported it would make ``openforecast providers install amazon`` and every
``providers list`` pay seconds for a question answered by a dataclass.

The declaration matters more too. A trainable model's descriptor is checked
against reality at every fit; a pretrained model is only ever checked at the
forecast, so the descriptor is the whole of what a caller can plan against.

The last section is the boundary claim, checked by parsing this distribution's
own imports rather than by trusting them: a provider that imported a
``ForecastDataset`` would have been handed something the view abstraction was
supposed to absorb.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest
from openforecast_chronos import PROVIDER_NAME, PROVIDER_VERSION, ChronosProvider, catalog
from openforecast_chronos.adapter import CHRONOS_2

from openforecast.errors import ModelDoesNotSupportFit, UnknownModelError
from openforecast.models import ModelDescriptor
from openforecast.models.capabilities import MissingValueSupport

PROVIDER = ChronosProvider()

#: ``integrations/chronos/tests`` -> the code that ships in the distribution.
SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"

#: Semantic source datasets. A provider that names one has reached past the view.
SOURCE_TYPES = (
    "ForecastContext",
    "ForecastDataset",
    "PointInTimeFrame",
    "PointInTimeSchema",
    "TimeSeriesFrame",
    "TimeSeriesSchema",
)


def descriptor_for(name: str) -> ModelDescriptor:
    (found,) = [candidate for candidate in PROVIDER.descriptors() if candidate.ref.name == name]
    return found


# -- identity ----------------------------------------------------------------


def test_the_provider_is_namespaced_after_the_models_it_advertises() -> None:
    """The provider name is the namespace, which is what installation checks."""
    assert PROVIDER.name == PROVIDER_NAME == "amazon"
    assert PROVIDER.version == PROVIDER_VERSION
    assert all(descriptor.provider == "amazon" for descriptor in PROVIDER.descriptors())
    assert [str(descriptor.ref) for descriptor in PROVIDER.descriptors()] == ["amazon/chronos-2"]


def test_the_reference_is_the_published_checkpoint() -> None:
    """A user should not have to translate a name they already know."""
    assert CHRONOS_2.checkpoint == "amazon/chronos-2"


def test_an_unknown_model_is_refused_naming_what_there_is() -> None:
    with pytest.raises(UnknownModelError, match="amazon/chronos-2"):
        catalog.adapter_for("amazon/nothing", "amazon")
    with pytest.raises(UnknownModelError):
        catalog.adapter_for("nixtla/nhits", "amazon")


# -- the declaration ---------------------------------------------------------


def test_the_model_declares_the_pretrained_lifecycle_and_no_training_contract() -> None:
    """Step 23's declaration, and the whole of what makes zero-shot use work."""
    descriptor = descriptor_for("chronos-2")

    assert descriptor.lifecycle.is_zero_shot
    assert not descriptor.lifecycle.requires_fit
    assert not descriptor.lifecycle.supports_fit
    assert not descriptor.is_fittable
    assert descriptor.training is None


def test_the_declared_capabilities_are_the_ones_this_path_has() -> None:
    capabilities = descriptor_for("chronos-2").capabilities

    assert capabilities.instances.single and capabilities.instances.panel
    assert capabilities.targets.univariate and not capabilities.targets.multivariate
    assert capabilities.features.observed and capabilities.features.known
    # Chronos-2 takes past and future covariates and no static ones.
    assert not capabilities.features.static
    assert capabilities.outputs.point and capabilities.outputs.quantiles
    # It predicts quantiles directly; it draws no sample paths.
    assert not capabilities.outputs.samples
    assert capabilities.missing_values is MissingValueSupport.NATIVE


def test_the_model_advertises_no_parameters() -> None:
    """Nothing is compiled at fit time, because there is no fit.

    A forecast carries no parameters through the wire, so a parameter advertised
    here would be one a caller could set and nothing would read.
    """
    assert descriptor_for("chronos-2").parameters_schema == {}


# -- the operation it does not have ------------------------------------------


def test_fitting_is_refused_at_the_provider_as_well_as_at_the_registry() -> None:
    """Unreachable through ``of.fit``, and still a refusal rather than a no-op."""
    with pytest.raises(ModelDoesNotSupportFit, match="cannot be fitted"):
        PROVIDER.fit(
            model="amazon/chronos-2",
            params={},
            view=None,  # pyright: ignore[reportArgumentType]
            seed=None,
            into=Path("/nonexistent"),
        )


# -- the boundary ------------------------------------------------------------


def _imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.add(node.module or "")
            found.update(alias.name for alias in node.names)
    return found


def test_no_module_of_this_distribution_names_a_semantic_source_type() -> None:
    """The view boundary, checked rather than trusted.

    A pretrained model is the one most tempting to hand raw data to — it has no
    fit view, so "just give it the frame" looks harmless. It is not: the origin,
    the vintage and the information set would then be this integration's problem
    rather than OpenForecast's, and a point-in-time backtest would silently leak.
    """
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        named = _imported_names(path)
        leaked = sorted(named & set(SOURCE_TYPES))
        assert not leaked, f"{path.name} imports {leaked}"


def test_a_handshake_imports_neither_chronos_nor_torch() -> None:
    """Discovery is a dataclass, and it may not pay for a deep-learning stack."""
    script = (
        "import sys",
        "from openforecast_chronos import ChronosProvider",
        "descriptors = ChronosProvider().descriptors()",
        "assert descriptors",
        "loaded = sorted(name for name in sys.modules if name in {'chronos', 'torch'})",
        "assert loaded == [], loaded",
    )
    completed = subprocess.run(  # noqa: S603 - this interpreter, a literal script
        [sys.executable, "-c", "\n".join(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
