"""Provider conformance: the contract a model has to satisfy to be advertised.

```python
for case in suite.cases_for(descriptor):
    suite.run_case(case, descriptor=descriptor, provider=provider, store=tmp_path)
```

Nothing here is written per provider. A descriptor states which view its model
trains on, how many series and targets it takes, which feature roles it
consumes, whether it learns across origins and what it does about missing
values — and every one of those statements implies something that can be
checked. The suite reads the declaration and generates:

```text
cases       fits that must succeed, over both semantic sources
refusals    requests the declaration says the model cannot serve
```

So a model declaring ``view=sequences`` is automatically fitted from an
event-time frame *and* from real forecast vintages, and in both cases the
provider is handed a ``SequenceView`` and nothing else. That is the boundary
this whole design exists for, and it is asserted rather than assumed: the
provider under test is wrapped in a :class:`Recording` client that remembers
what it was actually given.

A model that declares less gets fewer positive cases and more refusals — never
fewer checks. Declaring no panel support does not exempt a provider from panel
data; it changes what panel data must do, which is fail as a declaration
mismatch before the provider is started.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

import openforecast as of
from openforecast.errors import DataError, OpenForecastError, OriginScopeError
from openforecast.models import ModelDescriptor, ModelRef
from openforecast.models.capabilities import MissingValueSupport
from openforecast.models.catalog import ModelCatalog
from openforecast.models.contract import TrainingContract
from openforecast.protocol import ForecastColumn
from openforecast.runtime.provider import ProviderClient, ProviderRegistry
from openforecast.views import (
    FitView,
    ForecastView,
    OriginFidelity,
    SequenceView,
    SeriesView,
    SourceKind,
    TabularView,
    ViewKind,
)
from tests.conformance import datasets
from tests.conformance.datasets import SemanticDataset

#: The view type a contract's declared kind must materialize into.
VIEW_TYPES: Mapping[ViewKind, type[FitView]] = {
    ViewKind.SERIES: SeriesView,
    ViewKind.SEQUENCES: SequenceView,
    ViewKind.TABULAR: TabularView,
}

#: One shape of data, used by every case so that the counts below are readable.
PERIODS = 24
ORIGINS = 6
CONTEXT = 3
HORIZON = 3
PANEL_INSTANCES = 3


# -- what a provider is handed ----------------------------------------------


@dataclass
class Recording:
    """A provider, wrapped so the suite can see what the engine gave it.

    The point of the view boundary is that a provider cannot tell an event-time
    frame from real vintages. That is only true if it is handed a view and never
    a source dataset, which is a claim about the engine — so it is checked here
    rather than trusted.
    """

    inner: ProviderClient
    fit_views: list[FitView] = field(default_factory=lambda: [])
    forecast_views: list[ForecastView] = field(default_factory=lambda: [])

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def version(self) -> str:
        return self.inner.version

    def descriptors(self) -> tuple[ModelDescriptor, ...]:
        return self.inner.descriptors()

    def fit(
        self,
        *,
        model: ModelRef | str,
        params: Mapping[str, Any],
        view: FitView,
        seed: int | None,
        into: Path,
    ) -> None:
        self.fit_views.append(view)
        self.inner.fit(model=model, params=params, view=view, seed=seed, into=into)

    def forecast(
        self,
        *,
        model: ModelRef | str,
        params: Mapping[str, Any],
        view: ForecastView,
        output: Mapping[str, Any],
        state: Path,
    ) -> pa.Table:
        self.forecast_views.append(view)
        return self.inner.forecast(
            model=model, params=params, view=view, output=output, state=state
        )


# -- the cases ---------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One fit that a model's own declaration says must succeed."""

    name: str
    build: Callable[[], SemanticDataset]
    plan: of.FitPlan
    source: SourceKind
    fidelity: OriginFidelity
    instances: int
    targets: tuple[str, ...]
    horizon: int = HORIZON

    def data(self) -> SemanticDataset:
        return self.build()


@dataclass(frozen=True)
class Refusal:
    """One request a model's own declaration says it cannot serve."""

    name: str
    build: Callable[[], SemanticDataset]
    plan: of.FitPlan
    error: type[OpenForecastError]
    match: str
    horizon: int = HORIZON

    def data(self) -> SemanticDataset:
        return self.build()


def cases_for(descriptor: ModelDescriptor) -> tuple[Case, ...]:
    """Every fit ``descriptor`` claims it can serve, over both semantic sources."""
    cases: list[Case] = []
    for instances in _instance_counts(descriptor):
        for targets in _target_sets(descriptor):
            shape = f"{'panel' if instances > 1 else 'single'}-{len(targets)}-target"
            cases.append(
                Case(
                    name=f"event-time-{shape}",
                    build=partial(_frame, descriptor, instances=instances, targets=targets),
                    plan=_plan(descriptor.training),
                    source=SourceKind.TIME_SERIES,
                    fidelity=OriginFidelity.SIMULATED,
                    instances=instances,
                    targets=targets,
                )
            )
            if not _consumes_point_in_time(descriptor):
                continue
            cases.append(
                Case(
                    name=f"point-in-time-{shape}",
                    build=partial(_dataset, descriptor, instances=instances, targets=targets),
                    plan=_plan(descriptor.training, origins=_origins(descriptor.training)),
                    source=SourceKind.FORECAST_DATASET,
                    fidelity=OriginFidelity.OBSERVED,
                    instances=instances,
                    targets=targets,
                )
            )
    return tuple(cases)


def refusals_for(descriptor: ModelDescriptor) -> tuple[Refusal, ...]:
    """Every request ``descriptor`` says it cannot serve, and how it must say so."""
    capabilities = descriptor.capabilities
    contract = descriptor.training
    refusals: list[Refusal] = []

    if not capabilities.instances.panel:
        refusals.append(
            Refusal(
                name="a panel",
                build=partial(
                    _frame, descriptor, instances=PANEL_INSTANCES, targets=_target_names(1)
                ),
                plan=_plan(contract),
                error=DataError,
                match="cannot be fitted on a panel",
            )
        )
    if not capabilities.targets.multivariate:
        refusals.append(
            Refusal(
                name="two targets",
                build=partial(_frame, descriptor, instances=1, targets=_target_names(2)),
                plan=_plan(contract),
                error=DataError,
                match="cannot be fitted on 2 targets",
            )
        )
    unsupported = _unsupported_role(descriptor)
    if unsupported is not None:
        refusals.append(
            Refusal(
                name=f"a {unsupported} feature",
                build=partial(_frame_with_role, unsupported),
                plan=_plan(contract),
                error=DataError,
                match="cannot be given the features",
            )
        )
    if capabilities.missing_values is MissingValueSupport.UNSUPPORTED:
        refusals.append(
            Refusal(
                name="a gap in the data",
                build=partial(_frame_with_gap, descriptor),
                plan=_plan(contract),
                error=DataError,
                match="has missing values",
            )
        )
    if not contract.learns_across_origins and _consumes_point_in_time(descriptor):
        refusals.append(
            Refusal(
                name="every vintage at once",
                build=partial(_dataset, descriptor, instances=1, targets=_target_names(1)),
                plan=_plan(contract, origins=of.AllOrigins()),
                error=OriginScopeError,
                match="one forecast origin",
            )
        )
    return tuple(refusals)


# -- running them ------------------------------------------------------------


def run_case(
    case: Case, *, descriptor: ModelDescriptor, provider: ProviderClient, store: Path
) -> None:
    """Fit and forecast one case, asserting everything the declaration implies."""
    contract = descriptor.training
    recording = Recording(provider)
    client = client_for(descriptor, recording, store)
    data = case.data()

    handle = client.fit(str(descriptor.ref), data, horizon=case.horizon, plan=case.plan)

    # The provider was handed the view its contract names, and only that.
    assert [type(view) for view in recording.fit_views] == [VIEW_TYPES[contract.view]]
    assert recording.fit_views[0].provenance.source is case.source
    assert recording.fit_views[0].provenance.origin_fidelity is case.fidelity

    # And the artifact says the same thing, so it survives the process.
    assert handle.training.view == contract.view
    assert handle.training.source == case.source
    assert handle.training.origin_fidelity == case.fidelity
    assert handle.training.context == (CONTEXT if contract.view is ViewKind.SEQUENCES else None)
    assert handle.training.samples >= 1

    if not descriptor.capabilities.outputs.point:
        # Everything above is about the fit; a model that produces no point
        # forecast is asked for one it can produce by its own suite.
        return

    forecast = client.forecast(handle, _inference_data(data, case), horizon=case.horizon)
    answer = forecast.table

    assert [type(view) for view in recording.forecast_views] == [ForecastView]
    assert answer.num_rows == case.instances * case.horizon * len(case.targets)
    assert set(datasets.column(answer, ForecastColumn.TARGET.value)) == set(case.targets)
    assert set(datasets.column(answer, ForecastColumn.KIND.value)) == {"point"}
    assert forecast.origin_time == _origin_of(data, case)
    assert len(forecast.event_times) == case.horizon
    # A forecast comes back labeled with the instance it is about, which is the
    # only thing that makes a panel answer usable.
    assert forecast.instance_keys == (("zone",) if case.instances > 1 else ())


def run_refusal(
    refusal: Refusal, *, descriptor: ModelDescriptor, provider: ProviderClient, store: Path
) -> None:
    """Assert the model refuses what it declared it cannot do, and says why."""
    recording = Recording(provider)
    client = client_for(descriptor, recording, store)

    with pytest.raises(refusal.error, match=refusal.match):
        client.fit(str(descriptor.ref), refusal.data(), horizon=refusal.horizon, plan=refusal.plan)

    # A refusal is a declaration meeting data, so it happens before the provider
    # is started rather than inside somebody's library.
    assert recording.fit_views == []


def client_for(
    descriptor: ModelDescriptor, provider: ProviderClient, store: Path
) -> of.OpenForecast:
    """A client that can execute exactly ``descriptor``, and nothing else."""
    return of.OpenForecast(
        store=store,
        catalog=ModelCatalog([descriptor]),
        providers=ProviderRegistry([provider]),
    )


# -- the data each case needs ------------------------------------------------


def _instance_counts(descriptor: ModelDescriptor) -> tuple[int, ...]:
    capabilities = descriptor.capabilities.instances
    return tuple(
        count
        for count, supported in ((1, capabilities.single), (PANEL_INSTANCES, capabilities.panel))
        if supported
    )


def _target_sets(descriptor: ModelDescriptor) -> tuple[tuple[str, ...], ...]:
    capabilities = descriptor.capabilities.targets
    return tuple(
        _target_names(count)
        for count, supported in ((1, capabilities.univariate), (2, capabilities.multivariate))
        if supported
    )


def _target_names(count: int) -> tuple[str, ...]:
    return ("load", "wind")[:count]


def _plan(contract: TrainingContract, origins: of.OriginSelection | None = None) -> of.FitPlan:
    """The plan the contract asks for: a window only where a window means something."""
    window = of.WindowPlan(context=CONTEXT) if contract.view is ViewKind.SEQUENCES else None
    if origins is None:
        return of.FitPlan(window=window)
    return of.FitPlan(window=window, origins=origins)


def _origins(contract: TrainingContract) -> of.OriginSelection:
    """Every vintage for a model that learns across them, one for a model that cannot."""
    if contract.learns_across_origins:
        return of.AllOrigins()
    return of.AtOrigin(datasets.at(CONTEXT - 1 + ORIGINS - 1))


def _roles(descriptor: ModelDescriptor) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
    """The observed, known and static features this model may be given.

    An observed feature has no value past the forecast origin, so a model that
    cannot see a missing value cannot be given one in a view that reaches past
    it — that is a property of the data, not a capability the model is missing.
    """
    capabilities = descriptor.capabilities.features
    tolerant = descriptor.capabilities.tolerates_missing_values
    observed = ("temp",) if capabilities.observed and tolerant else ()
    known = ("temp_fc",) if capabilities.known else ()
    return observed, known, capabilities.static


def _frame(
    descriptor: ModelDescriptor, *, instances: int, targets: Sequence[str]
) -> of.TimeSeriesFrame:
    observed, known, static = _roles(descriptor)
    return datasets.event_time(
        instances=instances,
        targets=targets,
        periods=PERIODS,
        observed=observed,
        known=known,
        static=static and instances > 1,
        future_periods=HORIZON,
    )


def _dataset(
    descriptor: ModelDescriptor, *, instances: int, targets: Sequence[str]
) -> of.ForecastDataset:
    """Real vintages, shaped for the view this model consumes.

    A series is one complete time series at one origin, so the vintage it is cut
    from has to describe every event time behind that origin; the other two
    views consume a window, so a window is what the vintages carry.
    """
    observed, known, static = _roles(descriptor)
    return datasets.point_in_time(
        instances=instances,
        targets=targets,
        origins=ORIGINS,
        context=CONTEXT,
        horizon=HORIZON,
        observed=observed,
        known=known,
        static=static and instances > 1,
        cumulative=descriptor.training.view is ViewKind.SERIES,
    )


def _frame_with_role(role: str) -> of.TimeSeriesFrame:
    """A frame carrying exactly the one feature role the model cannot consume."""
    return datasets.event_time(
        instances=1,
        targets=_target_names(1),
        periods=PERIODS,
        observed=("temp",) if role == "observed" else (),
        known=("temp_fc",) if role == "known" else (),
        static=role == "static",
    )


def _frame_with_gap(descriptor: ModelDescriptor) -> of.TimeSeriesFrame:
    observed, known, _ = _roles(descriptor)
    return datasets.event_time(
        instances=1,
        targets=_target_names(1),
        periods=PERIODS,
        observed=observed,
        known=known,
        gaps=(PERIODS // 2,),
    )


def _unsupported_role(descriptor: ModelDescriptor) -> str | None:
    """One feature role the model declares it cannot be given, if there is one.

    A single role is enough: the check is that an unsupported feature is refused
    rather than dropped, and refusing one of them is the same code path as
    refusing all three.
    """
    capabilities = descriptor.capabilities.features
    for role, supported in (
        ("known", capabilities.known),
        ("observed", capabilities.observed),
        ("static", capabilities.static),
    ):
        if not supported:
            return role
    return None


def _consumes_point_in_time(descriptor: ModelDescriptor) -> bool:
    """Whether real vintages can be offered to this model at all.

    A point-in-time frame holds at least one feature — an origin and an event
    time on their own carry no information — so a model that consumes no
    temporal feature has no point-in-time case to run. That is a limitation of
    the model, and :func:`refusals_for` does not manufacture one for it: the
    data would be refused for the feature, which is already covered.
    """
    capabilities = descriptor.capabilities.features
    return capabilities.known or (
        capabilities.observed and descriptor.capabilities.tolerates_missing_values
    )


# -- inference ---------------------------------------------------------------


def _inference_data(data: SemanticDataset, case: Case) -> object:
    """The one origin the forecast is made at.

    An event-time frame describes its own last origin. A point-in-time dataset
    holds many and is narrowed explicitly, because choosing one silently would
    forecast from information the caller never named.
    """
    if isinstance(data, of.ForecastDataset):
        return data.at_origin(_origin_of(data, case))
    return data


def _origin_of(data: SemanticDataset, case: Case) -> datetime:
    if isinstance(data, of.ForecastDataset):
        selected = case.plan.origins.select(data.origins)
        return selected[-1]
    moments: list[datetime] = datasets.column(data.history, data.schema.time)
    return max(moments)
