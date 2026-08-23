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
combined on the way back out. Every leaf is checked before any of them runs, at
fit and at forecast alike: an ensemble whose second member cannot consume the
data is refused whole, rather than half-trained and then abandoned.

The combination is a weighted mean and nothing else. A quantile forecast is
averaged level by level, which is quantile averaging rather than the quantile of
a mixture; sample paths are not combined at all, because draw *i* of one member
has nothing to do with draw *i* of another.

Since Step 23 a forecast has two shapes, and :meth:`Engine.forecast` is one
branch wide because of it:

```text
local/de-price@01K...   an artifact -> its recipe, its leaves, its fitted state
amazon/chronos-2        a descriptor -> one provider call, no artifact at all
```

Which one a reference means is the registry's answer rather than a flag on the
call, and everything after the branch is shared: the same forecast view, the
same output check, the same validation of what came back. Nothing downstream of
here can tell whether the numbers came from a model fitted this morning or from
one pretrained a year ago.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
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
from openforecast.models.contract import TrainingContract
from openforecast.models.descriptor import ModelDescriptor
from openforecast.models.ref import ModelRef
from openforecast.protocol.vocabulary import ForecastColumn, ViewKind
from openforecast.recipes.nodes import Ensemble, Model, Pipeline, Recipe, Reduction
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
from openforecast.runtime.validation import validate_forecast_view, validate_view
from openforecast.tasks.forecast import ForecastTask, OutputKind, OutputSpec
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
        found = leaves(recipe)
        # Every member is resolved before any of them is materialized, and every
        # one materialized before any provider starts: an ensemble one member of
        # which cannot be trained is refused whole rather than half-trained.
        descriptors = [self._registry.for_fit(leaf.model.ref) for leaf in found]
        _check_shared_plan(plan, descriptors)
        fitted = [
            self._materialize(leaf, descriptor, data, plan, task, shared=len(found) > 1)
            for leaf, descriptor in zip(found, descriptors, strict=True)
        ]
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
        self,
        leaf: Leaf,
        descriptor: ModelDescriptor,
        data: object,
        plan: FitPlan,
        task: ForecastTask | None,
        *,
        shared: bool,
    ) -> _Prepared:
        """Everything that has to be true before a provider is started."""
        request = ViewRequest.for_contract(
            descriptor.required_training, plan=plan, task=task, shared_plan=shared
        )
        view = self._planner.fit_view(data, request)
        view, transforms = fit_transforms(view, leaf.transforms)
        validate_view(view, descriptor, leaf.transforms)
        return _Prepared(
            leaf=leaf,
            provider=self._providers.get(descriptor.provider),
            view=view,
            transforms=transforms,
            contract=descriptor.required_training,
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
                contract=one.contract,
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
            contracts=[prepared.contract for prepared in fitted],
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
        """Forecast ``horizon`` steps, with a fitted artifact or a pretrained model."""
        task = ForecastTask(horizon)
        output = OutputSpec.point() if output is None else output
        resolved = self._resolve(model)
        context = normalize_forecast_context(data, origin_time=origin_time)
        combined = (
            self._zero_shot(resolved, context, task, output)
            if isinstance(resolved, ModelDescriptor)
            else self._fitted(resolved, context, task, output)
        )
        answer = Forecast(
            combined,
            origin_time=context.origin_time,
            horizon=horizon,
            targets=context.schema.targets,
            instance_keys=context.schema.instance_keys,
            model=str(resolved.ref),
        )
        # `quantiles(..., from_samples=n)` was executed as a sample forecast, so
        # the reduction happens here rather than in the provider: one estimator,
        # applied to whoever drew the paths, is the whole of what makes two
        # providers' quantiles comparable.
        return answer.to_quantiles(output.levels) if output.derived_from_samples else answer

    def _fitted(
        self,
        handle: ModelHandle,
        context: ForecastContext,
        task: ForecastTask,
        output: OutputSpec,
    ) -> pa.Table:
        """The forecast of an artifact: every leaf answered from its own state."""
        horizon = task.horizon
        if not handle.serves_horizon(horizon):
            bound = [record.horizon for record in handle.training_records]
            raise IncompatibleForecastTask(
                f"{handle.ref} was fitted with its horizon bound to {bound} and cannot "
                f"forecast {horizon} steps; fit it for the horizon you need"
            )
        artifact = self._store.read(handle.ref)
        _check_data_schema(handle.data_schema, context, handle.ref)

        members = self._leaf_state(handle, artifact.recipe)
        _check_outputs(output, [self._catalog.get(leaf.model.ref) for leaf, _, _ in members])
        answers = [
            self._answer(leaf, record, context, task, output, paths)
            for leaf, record, paths in members
        ]
        return _combine(artifact.recipe, iter(answers))

    def _zero_shot(
        self,
        descriptor: ModelDescriptor,
        context: ForecastContext,
        task: ForecastTask,
        output: OutputSpec,
    ) -> pa.Table:
        """The forecast of a pretrained model, which was never fitted here.

        Everything a fitted forecast reads out of the artifact is absent, and
        each absence is a fact rather than a gap. There is no horizon bound,
        because nothing bound one. There is no fitted schema to check the
        context against, because the model was not fitted on any data — so what
        the declaration is checked against is the forecast view itself, which is
        the only data this model will ever see. There are no transforms, because
        a recipe is fitted and this is a reference.

        The provider is still handed a state directory, and it is empty on
        purpose: the contract is "the directory belonging to this model", and
        for a model nothing was fitted for there is nothing in it. Whatever
        pretrained weights the integration needs are the integration's own, in
        its own environment, and are not part of an OpenForecast artifact.
        """
        _check_output(output, descriptor)
        executed = output.as_executed()
        view = self._planner.forecast_view(
            context, ViewRequest(kind=ViewKind.FORECAST, horizon=task.horizon)
        )
        validate_forecast_view(view, descriptor)
        with TemporaryDirectory(prefix="openforecast-zero-shot-") as empty:
            answer = self._providers.get(descriptor.provider).forecast(
                model=descriptor.ref,
                params={},
                view=view,
                output=executed.model_dump(mode="json"),
                state=Path(empty),
            )
        _check_answer(answer, view, executed, descriptor)
        return answer

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
        executed = output.as_executed()
        request = ViewRequest(kind=ViewKind.FORECAST, horizon=task.horizon, context=record.context)
        view = self._planner.forecast_view(context, request)
        transforms = read_state(paths.transforms)
        answer = self._providers.get(descriptor.provider).forecast(
            model=leaf.model.ref,
            params=leaf.model.params,
            view=apply_to_forecast_view(view, transforms),
            output=executed.model_dump(mode="json"),
            state=paths.state,
        )
        _check_answer(answer, view, executed, descriptor)
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

    def _resolve(self, model: ModelInput) -> ModelHandle | ModelDescriptor:
        """What this reference forecasts with: an artifact, or a pretrained model.

        The registry answers both, and which one came back is the whole of the
        difference between the two lifecycles. A recipe is neither — it names
        models to fit rather than a model to forecast with.
        """
        if isinstance(model, ModelHandle):
            return model
        if isinstance(model, Pipeline | Ensemble | Reduction):
            raise RecipeError(
                "a forecast is made with a fitted model, not with a recipe; fit the recipe "
                "and forecast with the local/... reference that comes back"
            )
        ref = model.ref if isinstance(model, Model) else ModelRef.parse(model)
        return self._registry.resolve(ref)

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
    #: What the model said it learns from. The manifest needs the one part of it
    #: the materialized view cannot answer: whether the horizon is bound.
    contract: TrainingContract


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
        "of.Reduction is part of the recipe protocol but is not executable yet. What it "
        "adds is generating lagged features from an ordinary event-time series; a "
        "ForecastDataset already carries the features a supervised row needs, so a tabular "
        "model is fitted on one directly: of.fit('sklearn/hist-gradient-boosting', "
        "data=dataset, horizon=...)"
    )


def _combine(recipe: Recipe, answers: Iterator[pa.Table]) -> pa.Table:
    """Fold the leaves' answers back up the recipe tree."""
    if isinstance(recipe, Model):
        return next(answers)
    if isinstance(recipe, Pipeline):
        return _combine(recipe.estimator, answers)
    if isinstance(recipe, Ensemble):
        parts = [_combine(member, answers) for member in recipe.models]
        return _weighted_mean(parts, recipe.normalized_weights)
    raise UnsupportedPlanError("of.Reduction is not executable yet")  # pragma: no cover


def _weighted_mean(parts: Sequence[pa.Table], weights: Sequence[float]) -> pa.Table:
    """Average the members row by row, matched on what each row describes.

    A member that answered a different set of rows is a bug rather than a partial
    ensemble, and a missing value in one member makes the combination missing:
    the mean of "we do not know" and a number is not that number.

    A row is matched on everything that is not the value, so a quantile forecast
    is combined level by level: the ensemble's P10 is the weighted mean of its
    members' P10. That is *quantile averaging*, and it is not the quantile of the
    mixture of the members' distributions — the two agree only when the members
    agree. It is the combination that needs nothing from the members beyond the
    levels that were asked of all of them.
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


def _check_shared_plan(plan: FitPlan, descriptors: Sequence[ModelDescriptor]) -> None:
    """A plan written for several models has to reach at least one of them.

    One model refuses a ``WindowPlan`` it cannot use, because nobody writes one
    expecting it to do nothing. Several models share one plan, so a member that
    binds no context window is simply not who the window was addressed to — but
    a window no member binds is still the mistake the single-model rule catches.
    """
    if len(descriptors) < 2 or plan.window is None:
        return
    views = {descriptor.required_training.view for descriptor in descriptors}
    if ViewKind.SEQUENCES not in views:
        raise RecipeError(
            "no member of this recipe learns from context -> horizon sequences, so the "
            "WindowPlan would have no effect on any of them. Drop it, or ensemble in a "
            "model that sizes a context window"
        )


def _check_outputs(output: OutputSpec, descriptors: Sequence[ModelDescriptor]) -> None:
    """Every member can answer the request, before the first of them is asked.

    The same rule the fit follows: an ensemble is checked whole. A member that
    cannot produce what was asked for would otherwise be discovered after its
    siblings had already run, and the answer thrown away.
    """
    for descriptor in descriptors:
        _check_output(output, descriptor)
    if len(descriptors) > 1 and output.as_executed().kind is OutputKind.SAMPLES:
        raise UnsupportedPlanError(
            "an ensemble does not combine sample paths yet: averaging draw 3 of one member "
            "with draw 3 of another combines two unrelated draws, and pooling them would "
            "weight the members by how many each took rather than by their weights. Ask it "
            "for quantiles, which are combined level by level"
        )


def _check_output(output: OutputSpec, descriptor: ModelDescriptor) -> None:
    """A model produces what it declares, and is not asked for anything else.

    Checked from the declaration, before a provider is started: an unanswerable
    request is a mismatch between what was asked and what the model says it can
    do, and finding that out from a provider stack trace after a fit would be
    finding it out in the wrong place.
    """
    outputs = descriptor.capabilities.outputs
    if output.is_supported_by(outputs):
        return
    asked_for_native_quantiles = (
        output.kind is OutputKind.QUANTILES and not output.derived_from_samples
    )
    remedy = (
        " It draws samples, so of.OutputSpec.quantiles("
        f"{list(output.levels)}, from_samples=n) asks for quantiles of n draws."
        if asked_for_native_quantiles and outputs.samples
        else ""
    )
    raise DataError(
        f"{descriptor.ref} cannot produce a {output.kind} forecast; it declares "
        f"point={outputs.point}, quantiles={outputs.quantiles}, samples={outputs.samples}."
        f"{remedy}"
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
    if kinds != {output.kind.row_kind}:
        raise ProviderError(
            f"{descriptor.ref} was asked for a {output.kind} forecast and answered {sorted(kinds)}"
        )
    _check_distribution(answer, output, descriptor)


def _check_distribution(answer: pa.Table, output: OutputSpec, descriptor: ModelDescriptor) -> None:
    """A probabilistic answer describes the distribution it was asked for.

    The levels that came back are the levels that were requested, and a sample
    forecast holds the draws it was asked to take. Neither is a formality: a
    provider that answered ``0.9`` where ``0.95`` was asked for produces a table
    a caller then scores as if it held a 0.95, and no later check would notice.
    """
    if output.kind is OutputKind.QUANTILES:
        levels = column_values(answer, ForecastColumn.QUANTILE.value)
        found = tuple(sorted({level for level in levels if level is not None}))
        if found != output.levels:
            raise ProviderError(
                f"{descriptor.ref} was asked for the quantiles {list(output.levels)} and "
                f"answered {list(found)}"
            )
    if output.kind is OutputKind.SAMPLES:
        draws = column_values(answer, ForecastColumn.SAMPLE.value)
        found_draws = {draw for draw in draws if draw is not None}
        if found_draws != set(range(output.draws or 0)):
            raise ProviderError(
                f"{descriptor.ref} was asked for {output.draws} sample paths and answered "
                f"{len(found_draws)}, indexed {sorted(found_draws)[:5]}...; the draws of a "
                f"sample forecast are numbered from 0"
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
