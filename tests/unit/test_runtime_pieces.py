"""The smaller pieces the engine is assembled from.

A forecast's own representation, the provider registry, the capability checks
against a materialized view, and the manifest of a composite artifact — each
tested where it lives rather than through a fit that happens to exercise it.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

import openforecast as of
from openforecast.artifacts import ModelArtifact, TrainedSchema
from openforecast.errors import (
    ArtifactError,
    DataError,
    DuplicateModelError,
    ProviderError,
    SchemaError,
)
from openforecast.models import (
    DEFAULT_CATALOG,
    InstanceCapabilities,
    MissingValueSupport,
    ModelCapabilities,
    ModelCatalog,
    TargetCapabilities,
)
from openforecast.protocol import ForecastColumn, forecast_columns
from openforecast.providers.builtin import BUILTIN_PROVIDER
from openforecast.runtime import (
    Forecast,
    ProviderClient,
    ProviderRegistry,
    default_providers,
    install_default_providers,
    validate_view,
)
from openforecast.runtime.providers import register_descriptors
from openforecast.runtime.validation import view_tables
from tests import artifacts, factories, providers

ORIGIN = datetime(2026, 1, 1, 12)


def answer(**overrides: Any) -> pa.Table:
    columns: dict[str, list[Any]] = {
        "zone": ["DE", "DE"],
        ForecastColumn.EVENT_TIME.value: [ORIGIN, ORIGIN],
        ForecastColumn.TARGET.value: ["load", "load"],
        ForecastColumn.KIND.value: ["point", "point"],
        ForecastColumn.QUANTILE.value: [None, None],
        ForecastColumn.SAMPLE.value: [None, None],
        ForecastColumn.VALUE.value: [1.0, 2.0],
    }
    columns.update(overrides)
    return pa.table(columns)


def forecast(table: pa.Table | None = None) -> Forecast:
    return Forecast(
        answer() if table is None else table,
        origin_time=ORIGIN,
        horizon=2,
        targets=("load",),
        instance_keys=("zone",),
        model="local/de-load@01K5Z6QK3M9TQK1W2E3R4T5Y6U",
    )


# -- the forecast -----------------------------------------------------------


def test_a_forecast_is_put_in_canonical_column_order() -> None:
    reordered = answer().select([ForecastColumn.VALUE.value, "zone", *forecast_columns()[:-1]][:7])

    assert forecast(reordered).table.column_names == list(forecast_columns(("zone",)))


def test_a_table_that_is_not_a_forecast_is_refused() -> None:
    """By the time one is built the request was checked; the answer was not."""
    with pytest.raises(ProviderError, match="missing the columns"):
        forecast(answer().drop_columns([ForecastColumn.KIND.value]))


def test_a_forecast_reports_what_it_is() -> None:
    result = forecast()

    assert result.num_rows == 2
    assert result.horizon == 2
    assert result.targets == ("load",)
    assert result.instance_keys == ("zone",)
    assert result.event_times == (ORIGIN,)
    assert "local/de-load@" in repr(result)


def test_point_drops_the_columns_that_describe_no_point() -> None:
    kept = forecast().point()

    assert kept.column_names == ["zone", "event_time", "target", "value"]
    assert kept.num_rows == 2


def test_a_probabilistic_row_is_not_a_point_forecast() -> None:
    mixed = answer(
        kind=["point", "quantile"],
        quantile=[None, 0.5],
    )

    assert forecast(mixed).point().num_rows == 1


def test_two_forecasts_are_equal_when_they_say_the_same_thing() -> None:
    assert forecast() == forecast()
    assert forecast() != forecast(answer(value=[1.0, 3.0]))
    assert forecast().__eq__(object()) is NotImplemented


def test_a_forecast_converts_to_pandas_without_pandas_being_a_dependency() -> None:
    frame = forecast().to_pandas()

    assert list(frame.columns) == list(forecast_columns(("zone",)))


# -- the provider registry --------------------------------------------------


def test_the_shipped_providers_are_installed_once(tmp_path: Path) -> None:
    """Installing twice is not a conflict; two providers claiming a name is."""
    catalog = ModelCatalog()
    install_default_providers(catalog)
    install_default_providers(catalog)

    assert [str(ref) for ref in catalog.refs()] == ["builtin/seasonal-naive"]
    assert "builtin" in default_providers()
    assert isinstance(BUILTIN_PROVIDER, ProviderClient)


def test_two_providers_may_not_claim_one_name() -> None:
    registry = ProviderRegistry([providers.StubProvider()])

    with pytest.raises(ProviderError, match="already registered"):
        registry.register(providers.StubProvider())


def test_a_second_model_under_one_reference_is_refused() -> None:
    """Which model a reference means may not depend on provider load order."""
    catalog = ModelCatalog()
    install_default_providers(catalog)
    impostor = providers.StubProvider(
        name="other", models=(providers.descriptor("seasonal-naive", provider="builtin"),)
    )

    with pytest.raises(DuplicateModelError, match="already registered"):
        register_descriptors([impostor], catalog)


def test_the_registry_names_what_it_holds() -> None:
    registry = default_providers()

    assert len(registry) == 1
    assert [provider.name for provider in registry] == ["builtin"]
    assert "builtin" in repr(registry)
    with pytest.raises(ProviderError, match="no provider named 'nixtla'"):
        registry.get("nixtla")


def test_the_built_in_provider_only_executes_its_own_models(tmp_path: Path) -> None:
    with pytest.raises(of.UnknownModelError, match="not a model of the 'builtin' provider"):
        BUILTIN_PROVIDER.fit(
            model="nixtla/nhits",
            params={},
            view=artifacts.series_view(artifacts.frame()),
            seed=None,
            into=tmp_path,
        )
    assert "BuiltinProvider" in repr(BUILTIN_PROVIDER)
    assert DEFAULT_CATALOG.get("builtin/seasonal-naive").provider == "builtin"


# -- capability checks against a view ---------------------------------------


def descriptor(**capabilities: Any) -> Any:
    return providers.descriptor("m", capabilities=ModelCapabilities(**capabilities))


def test_a_multivariate_view_needs_a_multivariate_model() -> None:
    frame = of.TimeSeriesFrame.from_pandas(
        history=factories.history(
            instances=("DE",), instance_key="zone", periods=6, targets=("load", "wind")
        ),
        time="timestamp",
        frequency="1h",
        instance_keys=["zone"],
        targets=["load", "wind"],
    )
    view = artifacts.series_view(frame)

    with pytest.raises(DataError, match="cannot be fitted on 2 targets"):
        validate_view(
            view,
            descriptor(
                instances=InstanceCapabilities(single=True, panel=True),
                targets=TargetCapabilities(univariate=True, multivariate=False),
                missing_values=MissingValueSupport.NATIVE,
            ),
        )


def test_a_declared_imputation_is_what_lets_a_strict_model_see_a_gap() -> None:
    """The transform is the difference, and it is recorded in the artifact."""
    history = factories.history(instances=("DE",), instance_key="zone", periods=8)
    history.loc[history.index[0], "load"] = factories.NAN
    gapped = of.TimeSeriesFrame.from_pandas(
        history=history,
        time="timestamp",
        frequency="1h",
        instance_keys=["zone"],
        targets=["load"],
    )
    view = artifacts.series_view(gapped)
    strict = descriptor(
        instances=InstanceCapabilities(single=True, panel=True),
        missing_values=MissingValueSupport.REQUIRES_TRANSFORM,
    )

    with pytest.raises(DataError, match="of.Impute"):
        validate_view(view, strict)
    impute = of.Impute(columns=of.ColumnSet.TARGETS, method=of.ImputeMethod.MEDIAN)
    validate_view(view, strict, (impute,))


def test_a_model_that_cannot_see_a_gap_at_all_says_so_differently() -> None:
    history = factories.history(instances=("DE",), instance_key="zone", periods=8)
    history.loc[history.index[0], "load"] = factories.NAN
    gapped = of.TimeSeriesFrame.from_pandas(
        history=history,
        time="timestamp",
        frequency="1h",
        instance_keys=["zone"],
        targets=["load"],
    )

    with pytest.raises(DataError, match="remove them from the data"):
        validate_view(
            artifacts.series_view(gapped),
            descriptor(
                instances=InstanceCapabilities(single=True, panel=True),
                missing_values=MissingValueSupport.UNSUPPORTED,
            ),
        )


def test_a_tabular_views_values_are_its_features_and_labels() -> None:
    """The keys are identifiers OpenForecast wrote, so they are not scanned."""
    view = artifacts.tabular_view()

    assert view_tables(view) == (view.X, view.y)


def test_a_sequence_view_is_checked_the_same_way() -> None:
    view = artifacts.sequence_view()

    validate_view(
        view,
        descriptor(
            instances=InstanceCapabilities(single=True, panel=True),
            features=of.models.FeatureCapabilities(known=True),
            missing_values=MissingValueSupport.NATIVE,
        ),
    )
    assert view_tables(view) == (view.temporal,)


# -- what a composite artifact records --------------------------------------


def test_a_composite_artifact_names_no_source_model() -> None:
    recipe = of.Ensemble(models=(of.Model("builtin/seasonal-naive"), of.Model("nixtla/nhits")))
    views = [artifacts.series_view(artifacts.frame()), artifacts.sequence_view()]

    artifact = ModelArtifact.of_composite(
        name="blend",
        recipe=recipe,
        views=views,
        data_schema=TrainedSchema.merge([TrainedSchema.of_view(view.schema) for view in views]),
        openforecast_version=of.__version__,
    )

    assert artifact.manifest.source_model is None
    assert artifact.manifest.is_composite
    assert len(artifact.manifest.training_records) == 2
    assert artifact.manifest.provider == "openforecast"


def test_a_composite_artifact_holds_at_least_one_model() -> None:
    with pytest.raises(ArtifactError, match="at least one"):
        ModelArtifact.of_composite(
            name="empty",
            recipe=of.Model("builtin/seasonal-naive"),
            views=[],
            data_schema=TrainedSchema(frequency=of.Frequency.parse("1h"), targets=("load",)),
            openforecast_version=of.__version__,
        )


def test_an_artifact_records_either_one_fit_or_several() -> None:
    """Both would be two accounts of one fit, free to disagree."""
    manifest = artifacts.artifact().manifest
    payload = manifest.model_dump()

    with pytest.raises(ArtifactError, match="either how its one model"):
        type(manifest).model_validate(payload | {"members": [payload["training"]]})


def test_a_composite_names_no_source_model_and_a_leaf_always_does() -> None:
    manifest = artifacts.artifact().manifest
    payload = manifest.model_dump()

    with pytest.raises(ArtifactError, match="names a source model if and only if"):
        type(manifest).model_validate(
            payload | {"training": None, "members": [payload["training"]]}
        )


def test_members_fitted_on_different_data_cannot_be_one_artifact() -> None:
    """They were materialized from the same source, so this is a bug, not a case."""
    hourly = TrainedSchema(frequency=of.Frequency.parse("1h"), targets=("load",))
    daily = TrainedSchema(frequency=of.Frequency.parse("1d"), targets=("load",))

    with pytest.raises(SchemaError, match="different data"):
        TrainedSchema.merge([hourly, daily])
    with pytest.raises(SchemaError, match="at least one view"):
        TrainedSchema.merge([])


def test_the_data_schema_of_a_composite_is_what_any_member_needs() -> None:
    """A tabular member consumes no observed feature; a sequence one may."""
    observed = of.FeatureSpec(name="temp", availability=of.FeatureAvailability.OBSERVED)
    known = of.FeatureSpec(name="wind_fc", availability=of.FeatureAvailability.KNOWN)
    hourly = of.Frequency.parse("1h")

    merged = TrainedSchema.merge(
        [
            TrainedSchema(frequency=hourly, targets=("load",), features=(observed, known)),
            TrainedSchema(frequency=hourly, targets=("load",), features=(known,)),
        ]
    )

    assert merged.features == (observed, known)


def test_a_composite_serves_a_horizon_only_if_every_member_does() -> None:
    manifest = artifacts.artifact().manifest
    payload = manifest.model_dump()
    record = manifest.training
    assert record is not None and record.horizon == artifacts.HORIZON

    composite = type(manifest).model_validate(
        payload
        | {
            "training": None,
            "source_model": None,
            "members": [payload["training"], payload["training"] | {"horizon": record.horizon + 1}],
        }
    )

    assert composite.is_composite
    assert not composite.serves_horizon(record.horizon)
    assert not composite.serves_horizon(record.horizon + 1)
