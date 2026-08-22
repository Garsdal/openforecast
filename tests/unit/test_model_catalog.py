"""``ModelDescriptor`` and the catalog a reference resolves against.

The descriptor is the contract between a provider and the engine: if it is
complete, ``fit()`` can materialize data without asking the provider anything
first. These tests hold it to that.
"""

from __future__ import annotations

import pytest

from openforecast import DuplicateModelError, SchemaError, UnknownModelError
from openforecast.models import (
    DEFAULT_CATALOG,
    FeatureCapabilities,
    MissingValueSupport,
    ModelCapabilities,
    ModelCatalog,
    ModelDescriptor,
    ModelLifecycle,
    ModelRef,
    TrainingContract,
    ViewKind,
)

REVISION = "01K5Z6QK3M9TQK1W2E3R4T5Y6U"

NHITS = ModelDescriptor(
    ref=ModelRef.parse("nixtla/nhits"),
    provider="nixtla",
    display_name="NHiTS",
    lifecycle=ModelLifecycle.trainable(),
    training=TrainingContract.sequences(supports_unseen_instances=True),
    capabilities=ModelCapabilities(
        features=FeatureCapabilities(observed=True, known=True, static=True),
        missing_values=MissingValueSupport.REQUIRES_TRANSFORM,
    ),
    parameters_schema={"type": "object", "properties": {"max_steps": {"type": "integer"}}},
)

AUTOARIMA = ModelDescriptor(
    ref=ModelRef.parse("nixtla/autoarima"),
    provider="nixtla",
    display_name="AutoARIMA",
    lifecycle=ModelLifecycle.trainable(),
    training=TrainingContract.series(),
)

DARTS_NHITS = ModelDescriptor(
    ref=ModelRef.parse("darts/nhits"),
    provider="darts",
    display_name="NHiTS (Darts)",
    lifecycle=ModelLifecycle.trainable(),
    training=TrainingContract.sequences(),
)


# -- the descriptor ---------------------------------------------------------


def test_a_descriptor_explains_how_to_materialize_data_for_the_model() -> None:
    """Everything the planner needs, without a round trip to the provider."""
    assert NHITS.ref == ModelRef.parse("nixtla/nhits")
    assert NHITS.training.view is ViewKind.SEQUENCES
    assert NHITS.training.context_required
    assert NHITS.training.learns_across_origins
    assert NHITS.capabilities.requires_missing_value_transform
    assert NHITS.lifecycle.requires_fit
    assert str(NHITS) == "nixtla/nhits"


def test_a_descriptor_round_trips_through_json() -> None:
    """It travels over provider RPC and HTTP, so serialization is part of the type."""
    restored = ModelDescriptor.model_validate_json(NHITS.model_dump_json())

    assert restored == NHITS
    assert restored.model_dump()["ref"] == {
        "namespace": "nixtla",
        "name": "nhits",
        "revision": None,
    }


def test_a_descriptor_accepts_the_reference_as_a_string() -> None:
    """A provider catalog is written the way a user would type the reference."""
    descriptor = ModelDescriptor(
        ref="builtin/seasonal-naive",  # pyright: ignore[reportArgumentType]
        provider="builtin",
        display_name="Seasonal naive",
        lifecycle=ModelLifecycle.trainable(),
        training=TrainingContract.series(),
    )

    assert descriptor.ref == ModelRef.parse("builtin/seasonal-naive")


def test_a_provider_may_only_advertise_its_own_namespace() -> None:
    with pytest.raises(SchemaError, match="may only advertise its own models"):
        ModelDescriptor(
            ref=ModelRef.parse("darts/nhits"),
            provider="nixtla",
            display_name="NHiTS",
            lifecycle=ModelLifecycle.trainable(),
            training=TrainingContract.sequences(),
        )


def test_a_descriptor_cannot_pin_a_revision() -> None:
    """A descriptor describes a model; a revision names one fitted artifact of it."""
    with pytest.raises(SchemaError, match="pins a revision"):
        ModelDescriptor(
            ref=ModelRef.parse(f"local/de-price@{REVISION}"),
            provider="local",
            display_name="DE price",
            lifecycle=ModelLifecycle.trainable(),
            training=TrainingContract.sequences(),
        )


def test_a_descriptor_needs_a_display_name() -> None:
    with pytest.raises(SchemaError, match="display name"):
        ModelDescriptor(
            ref=ModelRef.parse("nixtla/nhits"),
            provider="nixtla",
            display_name="  ",
            lifecycle=ModelLifecycle.trainable(),
            training=TrainingContract.sequences(),
        )


def test_a_parameter_schema_must_be_a_json_schema_object() -> None:
    with pytest.raises(SchemaError, match="type 'object'"):
        ModelDescriptor(
            ref=ModelRef.parse("nixtla/nhits"),
            provider="nixtla",
            display_name="NHiTS",
            lifecycle=ModelLifecycle.trainable(),
            training=TrainingContract.sequences(),
            parameters_schema={"max_steps": {"type": "integer"}},
        )


def test_declaring_no_parameters_is_allowed() -> None:
    assert AUTOARIMA.parameters_schema == {}


# -- the catalog ------------------------------------------------------------


def test_a_catalog_resolves_a_string_to_a_descriptor() -> None:
    catalog = ModelCatalog([NHITS, AUTOARIMA])

    assert catalog.get("nixtla/nhits") is NHITS
    assert catalog.get(ModelRef.parse("nixtla/autoarima")) is AUTOARIMA
    assert "nixtla/nhits" in catalog
    assert "darts/nhits" not in catalog
    assert len(catalog) == 2


def test_listing_is_ordered_by_reference_and_filterable_by_provider() -> None:
    catalog = ModelCatalog([NHITS, DARTS_NHITS, AUTOARIMA])

    assert [str(descriptor) for descriptor in catalog.list()] == [
        "darts/nhits",
        "nixtla/autoarima",
        "nixtla/nhits",
    ]
    assert catalog.list(provider="nixtla") == (AUTOARIMA, NHITS)
    assert catalog.list(provider="sktime") == ()
    assert catalog.providers() == ("darts", "nixtla")
    assert tuple(catalog) == catalog.list()


def test_an_unknown_reference_names_what_is_known() -> None:
    catalog = ModelCatalog([NHITS])

    with pytest.raises(UnknownModelError, match="nixtla/nhits"):
        catalog.get("darts/nhits")


def test_an_empty_catalog_says_so_rather_than_listing_nothing() -> None:
    with pytest.raises(UnknownModelError, match="no models are registered yet"):
        ModelCatalog().get("nixtla/nhits")


def test_asking_for_a_revision_points_at_the_artifact_registry() -> None:
    """A descriptor is never pinned, so a pinned reference cannot name one."""
    with pytest.raises(UnknownModelError, match="fitted revision"):
        ModelCatalog([NHITS]).get(f"local/de-price@{REVISION}")


def test_registering_a_reference_twice_is_refused() -> None:
    """Otherwise which model you get would depend on provider load order."""
    catalog = ModelCatalog([NHITS])

    with pytest.raises(DuplicateModelError, match="already registered"):
        catalog.register(NHITS)


def test_catalogs_do_not_share_state() -> None:
    first = ModelCatalog()
    first.register(NHITS)

    assert len(ModelCatalog()) == 0
    assert len(first) == 1


# -- the default catalog behind ``of.models`` --------------------------------


def test_the_module_functions_read_the_default_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openforecast import models

    monkeypatch.setattr(models, "DEFAULT_CATALOG", ModelCatalog())

    assert models.list() == ()
    assert models.register(NHITS) is NHITS
    assert models.get("nixtla/nhits") is NHITS
    assert models.list(provider="nixtla") == (NHITS,)


def test_the_default_catalog_is_empty_until_a_provider_fills_it() -> None:
    """Step 8 registers the built-in reference provider; nothing does yet."""
    assert DEFAULT_CATALOG.list() == ()
