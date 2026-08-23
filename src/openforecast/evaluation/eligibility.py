"""Which models could be fitted on this data at all — the first half of `openforecast/auto`.

```python
for entry in of.eligible_models(data, horizon=24):
    print(entry)

builtin/seasonal-naive  eligible
nixtla/autoarima        eligible
nixtla/nhits            ineligible: this model learns from context -> horizon sequences ...
darts/tide              ineligible: darts/tide cannot be given the features ['wind_fc'] ...
```

``openforecast/auto`` — one reference that inspects the data, backtests what
could run on it, and fits the winner — needs five things, and four of them now
exist:

```text
inspect data semantics      the semantic layer, since Steps 2 and 3
determine eligible models    here
backtest them               of.backtest
rank the results             BacktestResult.leaderboard
fit the winner or ensemble   of.fit, on the recipe that won
persist the selected recipe  deferred: an artifact recording a *choice* is not
                             the same object as one recording a fit
```

So this module is deliberately the screening step and not the model. Registering
``openforecast/auto`` today would mean a descriptor whose contract cannot be
answered before the data is seen — a view it cannot name, a horizon it cannot
bind — and the honest version of that is a policy over these pieces rather than a
model reference. Naming the reference first is how a wrapper ends up with a
capability nothing behind it can support.

## What eligibility means

Exactly one thing: **the fit would not be refused.** So it is not a heuristic
about model families — it materializes the view the model's own contract asks
for and checks it against the capabilities the model declared, which is the same
sequence ``of.fit`` runs and the same two functions it runs it with. An answer of
"eligible" is therefore worth something, and an answer of "ineligible" comes with
the sentence the fit would have failed with.

No provider is started to answer any of it, for the same reason ``fit()`` needs
none in order to plan: a descriptor is complete enough to plan against. That also
means it is not free — every eligible model's training view is materialized to
find out — so this is a screening pass over a catalog, not a predicate to call in
a loop.

The rules the plan sketched fall out of that rather than being written down
again:

```text
AutoARIMA for multi-origin learning   OriginScopeError, from the planner
missing values it cannot consume      DataError, from the capability check
features it cannot be given           DataError, naming the features
a target dimensionality it lacks      DataError, naming the count
```

A pretrained model is therefore reported as ineligible, and the reason says why:
there is no fit to be refused, so ``amazon/chronos-2`` is not a model this
question is about. It is not a gap in the screen — a zero-shot model needs no
screening, since the thing eligibility is protecting against is spending a fit
to find out. Forecast with it directly, or put it in ``of.backtest`` beside the
fitted candidates, which is where the two lifecycles are actually compared.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from openforecast.client import OpenForecast, default_client
from openforecast.errors import OpenForecastError
from openforecast.models.descriptor import ModelDescriptor
from openforecast.models.ref import ModelRef
from openforecast.protocol.vocabulary import ViewKind
from openforecast.runtime.validation import validate_view
from openforecast.tasks.forecast import ForecastTask
from openforecast.tasks.plan import FitPlan
from openforecast.views.planner import ViewPlanner, ViewRequest

__all__ = ["Eligibility", "eligible_models"]


class Eligibility(BaseModel):
    """Whether one model could be fitted on the data at hand, and why not."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: ModelRef
    eligible: bool
    #: The sentence the fit would have failed with. ``None`` when it would not
    #: have failed — a reason is a refusal, not a remark.
    reason: str | None = None

    def __str__(self) -> str:
        return f"{self.model} {'eligible' if self.eligible else f'ineligible: {self.reason}'}"


def eligible_models(
    data: object,
    *,
    horizon: int | None = None,
    plan: FitPlan | None = None,
    models: Sequence[ModelRef | str] | None = None,
    client: OpenForecast | None = None,
) -> tuple[Eligibility, ...]:
    """Every model the client can reach, and whether this data could fit it.

    ``models`` narrows the question to the references named; leaving it out asks
    it of the whole catalog. ``horizon`` is required by every model that does not
    train on complete series, since the horizon is what bounds their samples.

    ``plan`` is adapted per model the way a backtest's is — a context window
    reaches the models that size samples with one and no others — so one plan can
    be asked of a whole catalog.
    """
    executor = default_client() if client is None else client
    descriptors = (
        executor.models.list()
        if models is None
        else tuple(executor.models.get(str(ref)) for ref in models)
    )
    planner = ViewPlanner()
    return tuple(
        _eligibility(descriptor, data, horizon, plan, planner) for descriptor in descriptors
    )


def _eligibility(
    descriptor: ModelDescriptor,
    data: object,
    horizon: int | None,
    plan: FitPlan | None,
    planner: ViewPlanner,
) -> Eligibility:
    """One model, answered by doing what a fit would do short of fitting."""
    try:
        request = ViewRequest.for_contract(
            descriptor.required_training,
            plan=_plan_for(descriptor, plan),
            task=None if horizon is None else ForecastTask(horizon),
        )
        validate_view(planner.fit_view(data, request), descriptor)
    except OpenForecastError as refusal:
        return Eligibility(model=descriptor.ref, eligible=False, reason=str(refusal))
    return Eligibility(model=descriptor.ref, eligible=True)


def _plan_for(descriptor: ModelDescriptor, plan: FitPlan | None) -> FitPlan | None:
    """The plan as this model's contract can receive it.

    The same adaptation :func:`~openforecast.evaluation.backtest.plan_for`
    makes, and for the same reason: a window is a field only a sequence model
    binds, so carrying it to the others would report every one of them as
    ineligible for the plan rather than for the data.
    """
    if plan is None or plan.window is None:
        return plan
    if descriptor.training is not None and descriptor.training.view is ViewKind.SEQUENCES:
        return plan
    return plan.model_copy(update={"window": None})
