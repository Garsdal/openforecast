"""``Engine``: fit and forecast, with nothing left to decide.

```python
handle = engine.fit(model="builtin/seasonal-naive", data=train, params={"season_length": 24})
forecast = engine.forecast(model=handle, data=context, horizon=48)
```

The fit is five steps and none of them branch on who provides the model:

```text
normalize the recipe        a string, a Model, a Pipeline, an Ensemble
resolve every model         the registry answers what each reference means
materialize each view       the ViewPlanner, from the model's own contract
check it against the model  capabilities meeting data, before anything starts
hand it to the provider     into a staging directory, published on success
```

There is no place in that sequence for ``if provider == "nixtla"``, because
there is nothing for it to decide: a descriptor says which view to materialize
and what the model can be given, and the provider registry says who executes it.
That is the whole point of Steps 4 to 7 arriving before this one.

Point-in-time is invisible here too. A ``ForecastDataset`` and a
``TimeSeriesFrame`` both go to :meth:`ViewPlanner.fit_view`, and the only thing
that comes back different is the ``OriginFidelity`` recorded in the manifest. A
series model asked to learn from every vintage of point-in-time data raises
``OriginScopeError`` — from the planner, where the one branch on source type
lives, and not from here.

A recipe that is not a single model — a pipeline, an ensemble — is executed by
OpenForecast itself. Each leaf is materialized, transformed and fitted on its
own, into its own directory inside the one artifact, and their forecasts are
combined on the way back out.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa

from openforecast.artifacts.artifact import ModelArtifact
from openforecast.artifacts.handle import ModelHandle
from openforecast.artifacts.manifest import TrainedSchema, TrainingRecord
from openforecast.artifacts.store import ArtifactStore
from openforecast.data._arrow import InstanceKey, column_values, key_rows
from openforecast.data.forecast_context import ForecastContext
from openforecast.data.frame import TimeSeriesFrame
from openforecast.data.point_in_time import parse_moment
from openforecast.errors import (
    DataError,
    IncompatibleForecastTask,
    ProviderError,
    RecipeError,
    UnsupportedPlanError,
)
from openforecast.models.catalog import DEFAULT_CATALOG, ModelCatalog
from openforecast.models.descriptor import ModelDescriptor
from openforecast.models.ref import ModelRef
from openforecast.protocol.vocabulary import ForecastColumn, ViewKind
from openforecast.recipes.nodes import Ensemble, Mean, Model, Pipeline, Recipe, Reduction
from openforecast.recipes.transforms import Transform
from openforecast.registry.models import ModelRegistry
from openforecast.runtime.forecast import Forecast
from openforecast.runtime.provider import ProviderClient, ProviderRegistry
from openforecast.runtime.transforms import (
    STATE_FILENAME,
    TransformState,
    apply_to_forecast_view,
    fit_transforms,
    invert_forecast,
    read_state,
    write_state,
)
from openforecast.runtime.validation import validate_view
from openforecast.tasks.forecast import ForecastTask, OutputSpec
from openforecast.tasks.plan import FitPlan
from openforecast.views.forecast import ForecastView
from openforecast.views.planner import FitView, ViewPlanner, ViewRequest

__all__ = ["Engine", "Leaf", "leaves", "normalize_forecast_context", "normalize_recipe"]

#: One leaf of a composite recipe, inside the artifact's provider directory.
LEAF_DIRNAME = "leaf"
#: The provider's own directory inside a leaf. Opaque, like every provider
#: directory — a composite artifact just has more than one of them.
STATE_DIRNAME = "state"

#: What ``of.fit(model=...)`` and ``of.forecast(model=...)`` accept.
ModelInput = Recipe | ModelHandle | ModelRef | str


@dataclass(frozen=True)
class Leaf:
    """One model of a recipe, with the transforms that run before it."""

    model: Model
    transforms: tuple[Transform, ...] = ()


class Engine:
    """Fits models and forecasts with them, over a set of providers."""

    def __init__(
        self,
        *,
        store: ArtifactStore | None = None,
        catalog: ModelCatalog | None = None,
        providers: ProviderRegistry | None = None,
        planner: ViewPlanner | None = None,
    ) -> None:
        self._catalog = catalog if catalog is not None else DEFAULT_CATALOG
        self._store = store if store is not None else ArtifactStore()
        self._providers = providers if providers is not None else ProviderRegistry()
        self._planner = planner if planner is not None else ViewPlanner()
        self._registry = ModelRegistry(catalog=self._catalog, store=self._store)

    # -- accessors ---------------------------------------------------------

    @property
    def registry(self) -> ModelRegistry:
        return self._registry

    @property
    def store(self) -> ArtifactStore:
        return self._store

    @property
    def catalog(self) -> ModelCatalog:
        return self._catalog

    @property
    def providers(self) -> ProviderRegistry:
        return self._providers

    # -- fit ---------------------------------------------------------------

    def fit(
        self,
        model: ModelInput,
        data: object,
        *,
        horizon: int | None = None,
        plan: FitPlan | None = None,
        name: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> ModelHandle:
        """Fit ``model`` on ``data`` and publish the artifact it produced."""
        recipe = normalize_recipe(model, params)
        plan = FitPlan() if plan is None else plan
        task = None if horizon is None else ForecastTask(horizon)
        fitted = [self._materialize(leaf, data, plan, task) for leaf in leaves(recipe)]
        artifact = self._describe(recipe, fitted, name=name, plan=plan)

        with self._store.stage(artifact) as staging:
            composite = artifact.manifest.is_composite
            for index, prepared in enumerate(fitted):
                paths = _leaf_paths(staging.provider_path, index, composite=composite)
                paths.state.mkdir(parents=True, exist_ok=True)
                if prepared.transforms.steps:
                    write_state(paths.transforms, prepared.transforms)
                prepared.provider.fit(
                    model=prepared.leaf.model.ref,
                    params=prepared.leaf.model.params,
                    view=prepared.view,
                    seed=plan.seed,
                    into=paths.state,
                )
        return staging.handle

    def _materialize(
        self, leaf: Leaf, data: object, plan: FitPlan, task: ForecastTask | None
    ) -> _Prepared:
        """Everything that has to be true before a provider is started."""
        descriptor = self._registry.for_fit(leaf.model.ref)
        request = ViewRequest.for_contract(descriptor.training, plan=plan, task=task)
        view = self._planner.fit_view(data, request)
        view, transforms = fit_transforms(view, leaf.transforms)
        validate_view(view, descriptor, leaf.transforms)
        return _Prepared(
            leaf=leaf,
            provider=self._providers.get(descriptor.provider),
            view=view,
            transforms=transforms,
        )

    def _describe(
        self,
        recipe: Recipe,
        fitted: Sequence[_Prepared],
        *,
        name: str | None,
        plan: FitPlan,
    ) -> ModelArtifact:
        """The artifact this fit will produce, before any of it is written."""
        chosen = name if name is not None else fitted[0].leaf.model.ref.name
        if isinstance(recipe, Model):
            one = fitted[0]
            return ModelArtifact.of_fit(
                name=chosen,
                source_model=recipe.ref,
                recipe=recipe,
                view=one.view,
                provider=one.provider.name,
                provider_version=one.provider.version,
                openforecast_version=_version(),
                plan=plan,
            )
        return ModelArtifact.of_composite(
            name=chosen,
            recipe=recipe,
            views=[prepared.view for prepared in fitted],
            data_schema=TrainedSchema.merge(
                [TrainedSchema.of_view(prepared.view.schema) for prepared in fitted]
            ),
            openforecast_version=_version(),
            plan=plan,
        )

    # -- forecast ----------------------------------------------------------

    def forecast(
        self,
        model: ModelInput,
        data: object,
        *,
        horizon: int,
        output: OutputSpec | None = None,
        origin_time: str | datetime | None = None,
    ) -> Forecast:
        """Forecast ``horizon`` steps with a fitted artifact."""
        task = ForecastTask(horizon)
        output = OutputSpec.point() if output is None else output
        handle = self._resolve_artifact(model)
        if not handle.serves_horizon(horizon):
            bound = [record.horizon for record in handle.training_records]
            raise IncompatibleForecastTask(
                f"{handle.ref} was fitted with its horizon bound to {bound} and cannot "
                f"forecast {horizon} steps; fit it for the horizon you need"
            )
        artifact = self._store.read(handle.ref)
        context = normalize_forecast_context(data, origin_time=origin_time)
        _check_data_schema(handle.data_schema, context, handle.ref)

        answers = [
            self._answer(leaf, record, context, task, output, paths)
            for leaf, record, paths in self._leaf_state(handle, artifact.recipe)
        ]
        combined = _combine(artifact.recipe, iter(answers))
        return Forecast(
            combined,
            origin_time=context.origin_time,
            horizon=horizon,
            targets=context.schema.targets,
            instance_keys=context.schema.instance_keys,
            model=str(handle.ref),
        )

    def _answer(
        self,
        leaf: Leaf,
        record: TrainingRecord,
        context: ForecastContext,
        task: ForecastTask,
        output: OutputSpec,
        paths: _LeafPaths,
    ) -> pa.Table:
        """One leaf's forecast, on the scale the caller's data was on."""
        descriptor = self._catalog.get(leaf.model.ref)
        if not output.is_supported_by(descriptor.capabilities.outputs):
            raise DataError(
                f"{descriptor.ref} cannot produce a {output.kind} forecast; it declares "
                f"point={descriptor.capabilities.outputs.point}, quantiles="
                f"{descriptor.capabilities.outputs.quantiles}, samples="
                f"{descriptor.capabilities.outputs.samples}"
            )
        request = ViewRequest(kind=ViewKind.FORECAST, horizon=task.horizon, context=record.context)
        view = self._planner.forecast_view(context, request)
        transforms = read_state(paths.transforms)
        answer = self._providers.get(descriptor.provider).forecast(
            model=leaf.model.ref,
            params=leaf.model.params,
            view=apply_to_forecast_view(view, transforms),
            output=output.model_dump(mode="json"),
            state=paths.state,
        )
        _check_answer(answer, view, output, descriptor)
        return invert_forecast(answer, context.schema.instance_keys, transforms)

    def _leaf_state(
        self, handle: ModelHandle, recipe: Recipe
    ) -> list[tuple[Leaf, TrainingRecord, _LeafPaths]]:
        """Every fitted leaf, with the record and the directory belonging to it."""
        found = leaves(recipe)
        records = handle.training_records
        if len(found) != len(records):
            raise ProviderError(
                f"{handle.ref} records {len(records)} fitted models but its recipe names "
                f"{len(found)}; the artifact was modified after it was written"
            )
        return [
            (
                leaf,
                record,
                _leaf_paths(handle.provider_path, index, composite=handle.is_composite),
            )
            for index, (leaf, record) in enumerate(zip(found, records, strict=True))
        ]

    def _resolve_artifact(self, model: ModelInput) -> ModelHandle:
        if isinstance(model, ModelHandle):
            return model
        if isinstance(model, Pipeline | Ensemble | Reduction):
            raise RecipeError(
                "a forecast is made with a fitted model, not with a recipe; fit the recipe "
                "and forecast with the local/... reference that comes back"
            )
        ref = model.ref if isinstance(model, Model) else ModelRef.parse(model)
        resolved = self._registry.resolve(ref)
        if isinstance(resolved, ModelDescriptor):
            raise UnsupportedPlanError(
                f"{ref} is used zero-shot, and executing a model that was never fitted "
                f"arrives with the first pretrained model that needs it"
            )
        return resolved

    def __repr__(self) -> str:
        return (
            f"Engine(models={len(self._catalog)}, providers={len(self._providers)}, "
            f"store={self._store.root})"
        )


@dataclass(frozen=True)
class _Prepared:
    """A leaf that has been materialized and checked, and not yet fitted."""

    leaf: Leaf
    provider: ProviderClient
    view: FitView
    transforms: TransformState


@dataclass(frozen=True)
class _LeafPaths:
    """Where one leaf's provider state and fitted transforms live."""

    state: Path
    transforms: Path


def _leaf_paths(provider_path: Path, index: int, *, composite: bool) -> _LeafPaths:
    """A leaf model owns the artifact's provider directory; a composite shares it.

    One model means the promise Step 7 made holds exactly: the ``provider/``
    directory is the provider's and nothing else is in it. A composite is
    executed by OpenForecast, so that directory becomes OpenForecast's, and each
    leaf gets one of its own inside.
    """
    if not composite:
        return _LeafPaths(state=provider_path, transforms=provider_path / STATE_FILENAME)
    root = provider_path / f"{LEAF_DIRNAME}-{index}"
    return _LeafPaths(state=root / STATE_DIRNAME, transforms=root / STATE_FILENAME)


# -- recipes ----------------------------------------------------------------


def normalize_recipe(model: ModelInput, params: dict[str, Any] | None = None) -> Recipe:
    """The recipe ``model`` means.

    ``of.fit("builtin/seasonal-naive", params={...})`` is the short spelling of
    ``of.fit(of.Model("builtin/seasonal-naive", params={...}))``; passing both a
    recipe and parameters names them twice, so it is refused.
    """
    if isinstance(model, ModelHandle):
        raise RecipeError(
            f"{model.ref} is a fitted artifact, not a model to fit; fit the recipe it "
            f"records, on data of your choosing"
        )
    if isinstance(model, ModelRef | str):
        return Model(model, params=dict(params or {}))
    if params:
        raise RecipeError(
            "params was given alongside a recipe that already carries its own; put them "
            "on the of.Model they belong to"
        )
    return model


def leaves(recipe: Recipe, transforms: Sequence[Transform] = ()) -> tuple[Leaf, ...]:
    """Every model in ``recipe``, in the order it is fitted and forecast.

    The transforms of an enclosing pipeline travel with each leaf: they are
    fitted against the view that leaf consumes, since two members of an ensemble
    may not consume the same one.
    """
    if isinstance(recipe, Model):
        return (Leaf(model=recipe, transforms=tuple(transforms)),)
    if isinstance(recipe, Pipeline):
        return leaves(recipe.estimator, (*transforms, *recipe.transforms))
    if isinstance(recipe, Ensemble):
        return tuple(leaf for member in recipe.models for leaf in leaves(member, transforms))
    raise UnsupportedPlanError(
        "of.Reduction is part of the recipe protocol but is not executable yet; a "
        "reduction materializes a TabularView, and the estimators that consume one "
        "arrive with their integration"
    )


def _combine(recipe: Recipe, answers: Iterator[pa.Table]) -> pa.Table:
    """Fold the leaves' answers back up the recipe tree."""
    if isinstance(recipe, Model):
        return next(answers)
    if isinstance(recipe, Pipeline):
        return _combine(recipe.estimator, answers)
    if isinstance(recipe, Ensemble):
        parts = [_combine(member, answers) for member in recipe.models]
        weights = (
            [1.0 / len(parts)] * len(parts)
            if isinstance(recipe.combine, Mean)
            else list(recipe.combine.normalized)
        )
        return _weighted_mean(parts, weights)
    raise UnsupportedPlanError("of.Reduction is not executable yet")  # pragma: no cover


def _weighted_mean(parts: Sequence[pa.Table], weights: Sequence[float]) -> pa.Table:
    """Average the members row by row, matched on what each row describes.

    A member that answered a different set of rows is a bug rather than a partial
    ensemble, and a missing value in one member makes the combination missing:
    the mean of "we do not know" and a number is not that number.
    """
    keys = [_answer_keys(part) for part in parts]
    if any(key != keys[0] for key in keys[1:]):
        raise ProviderError(
            "the members of this ensemble answered different event times or targets, so "
            "their forecasts cannot be combined"
        )
    values = [column_values(part, ForecastColumn.VALUE.value) for part in parts]
    combined: list[float | None] = []
    for position in range(parts[0].num_rows):
        row = [value[position] for value in values]
        combined.append(
            None
            if any(item is None for item in row)
            else sum(float(item) * weight for item, weight in zip(row, weights, strict=True))
        )
    index = parts[0].column_names.index(ForecastColumn.VALUE.value)
    return parts[0].set_column(
        index, ForecastColumn.VALUE.value, pa.array(combined, type=pa.float64())
    )


def _answer_keys(table: pa.Table) -> list[tuple[Any, ...]]:
    columns = [name for name in table.column_names if name != ForecastColumn.VALUE.value]
    return key_rows(table, columns)


# -- forecast data ----------------------------------------------------------


def normalize_forecast_context(
    data: object, *, origin_time: str | datetime | None = None
) -> ForecastContext:
    """The single inference origin ``data`` describes.

    A ``ForecastContext`` already is one. A ``TimeSeriesFrame`` becomes one at
    the end of its history, which is the only origin it can describe without
    discarding data. A ``ForecastDataset`` holds many vintages and is not
    narrowed here: choosing one silently would forecast from information the
    caller never named, so it is asked for by ``dataset.at_origin(t)``.
    """
    if isinstance(data, ForecastContext):
        wanted = None if origin_time is None else parse_moment(origin_time, "origin_time")
        if wanted is not None and wanted != data.origin_time:
            raise DataError(
                f"this context is at origin {data.origin_time.isoformat()} and "
                f"{wanted.isoformat()} was asked for; a context is one origin, and moving "
                f"it here would forecast from information it does not describe"
            )
        return data
    if isinstance(data, TimeSeriesFrame):
        moments: list[datetime] = column_values(data.history, data.schema.time)
        if not moments:
            raise DataError("a forecast needs history to forecast from; this frame has none")
        return ForecastContext(
            origin_time=max(moments) if origin_time is None else origin_time, frame=data
        )
    raise DataError(
        f"cannot forecast from {type(data).__name__}; a forecast is made at one origin, "
        f"so pass a ForecastContext, a TimeSeriesFrame whose history ends at the origin, "
        f"or one origin of a ForecastDataset with dataset.at_origin(t)"
    )


def _check_data_schema(expected: TrainedSchema, context: ForecastContext, ref: ModelRef) -> None:
    """The data a forecast is made from has to be the data the model was fitted on."""
    schema = context.schema
    if expected.frequency != schema.frequency:
        raise DataError(
            f"{ref} was fitted on {expected.frequency} data and this context is {schema.frequency}"
        )
    if expected.instance_keys != schema.instance_keys:
        raise DataError(
            f"{ref} was fitted on instance keys {list(expected.instance_keys)} and this "
            f"context declares {list(schema.instance_keys)}"
        )
    if expected.targets != schema.targets:
        raise DataError(
            f"{ref} was fitted to forecast {list(expected.targets)} and this context "
            f"declares {list(schema.targets)}; the extra or missing targets would be "
            f"asked of a model that was never fitted for them"
        )
    absent = [
        feature.name
        for feature in expected.features
        if feature.name not in {declared.name for declared in schema.features}
    ]
    if absent:
        raise DataError(
            f"{ref} was fitted with the features {absent}, which this context does not "
            f"declare; a model cannot be asked to condition on what it is not given"
        )


def _check_answer(
    answer: pa.Table, view: ForecastView, output: OutputSpec, descriptor: ModelDescriptor
) -> None:
    """A provider answered the question it was asked, or it did not answer.

    Checked here rather than trusted, because a provider that returns a shorter
    horizon or a target it invented produces a forecast that looks exactly like a
    correct one.
    """
    metadata = view.metadata
    expected = {
        (instance, moment, target)
        for instance in view.instances
        for moment in view.event_times
        for target in metadata.targets
    }
    seen = _described_rows(answer, metadata.instance_keys)
    if seen != expected:
        raise ProviderError(
            f"{descriptor.ref} was asked for {len(expected)} forecasts and answered "
            f"{len(seen)}, for different instances, event times or targets"
        )
    kinds = set(column_values(answer, ForecastColumn.KIND.value))
    if kinds != {output.kind.value}:
        raise ProviderError(
            f"{descriptor.ref} was asked for a {output.kind} forecast and answered {sorted(kinds)}"
        )


def _described_rows(
    answer: pa.Table, instance_keys: Sequence[str]
) -> set[tuple[InstanceKey, datetime, str]]:
    instances = key_rows(answer, instance_keys)
    moments: list[datetime] = column_values(answer, ForecastColumn.EVENT_TIME.value)
    targets: list[str] = column_values(answer, ForecastColumn.TARGET.value)
    return set(zip(instances, moments, targets, strict=True))


def _version() -> str:
    """The OpenForecast version, read where it is defined.

    Imported here rather than at module scope: ``openforecast/__init__.py``
    imports the client, which imports this, so naming it at import time would
    close a cycle.
    """
    from openforecast import __version__

    return __version__
