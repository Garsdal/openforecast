"""``of.backtest``: the same models, the same origins, one comparable number.

```python
result = of.backtest(
    models=[
        "builtin/seasonal-naive",
        "nixtla/autoarima",
        "nixtla/nhits",
        "darts/nhits",
    ],
    data=data,
    validation=of.RollingOrigin(horizon=24, windows=5),
    metrics=[of.MAE(), of.Bias()],
)

result.leaderboard("mae")
```

Point-in-time data is the same call with the validation that fits it:

```python
result = of.backtest(
    models=["nixtla/nhits", "darts/nhits"],
    data=pit_dataset,
    validation=of.ForecastOriginValidation(
        origins=of.OriginsBetween(start, end, stride=24),
        horizon=72,
    ),
    metrics=[of.MAE()],
)
```

## Why there is no backtesting implementation in here

Everything below is a loop over ``client.fit`` and ``client.forecast``. There is
no Nixtla backtester, no Darts historical-forecasts call, no sktime evaluation
harness — and not because they were reimplemented, but because there is nothing
left for them to do. The semantic layer already answers every question a
backtest asks:

```text
which origins exist            the validation strategy, over the source data
what was knowable at one       up_to / at_origin, on the data itself
what to materialize            the ViewPlanner, from each model's contract
what happened                  the truth frame
```

So a backtest is provider-independent for the same reason a fit is: it never
learns who executes a model. That also means it works over any transport — a
backtest against ``of.OpenForecast(transport=of.HttpTransport(...))`` fits and
forecasts on the service and scores here.

## What it fits

One artifact per candidate and fold, published like any other, and its pinned
reference is recorded in the result. A backtest that scored models without
leaving anything behind would make its own winner unreproducible: the point of
``artifact`` in the table is that the number came from *that* revision, and you
can forecast with it.

## The plan a candidate is fitted with

``plan=`` is a template rather than a literal, because the candidates
deliberately do not share a contract. A ``WindowPlan`` is what a sequence model
sizes its samples with and a thing a series model cannot bind at all, so
``of.fit`` refuses to hand one to ARIMA — correctly, since somebody wrote it
expecting an effect. Comparing ARIMA against NHiTS therefore has to mean sizing
the window where it applies and dropping it where it cannot, which is what
:func:`plan_for` does and what this docstring exists to say out loud. A candidate
that needs something else states it:

```python
of.Candidate("nixtla/nhits", plan=of.FitPlan(window=of.WindowPlan(context=336)))
```
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from typing import Any

import pyarrow as pa
from pydantic import BaseModel, ConfigDict

from openforecast.artifacts.handle import ModelHandle
from openforecast.client import OpenForecast, default_client
from openforecast.data._arrow import build_table, column_values, is_missing, key_rows
from openforecast.errors import DataError, RecipeError
from openforecast.evaluation.metrics import Metric
from openforecast.evaluation.result import BacktestColumn, BacktestResult
from openforecast.evaluation.validation import Fold, Validation, truth_lookup
from openforecast.models.descriptor import ModelDescriptor
from openforecast.models.ref import ModelRef
from openforecast.protocol.vocabulary import ForecastColumn, ViewKind
from openforecast.recipes.nodes import Model, Recipe, estimator_refs
from openforecast.runtime.engine import ModelInput, normalize_recipe
from openforecast.runtime.forecast import Forecast
from openforecast.tasks.plan import FitPlan

__all__ = ["Candidate", "backtest", "plan_for"]

#: The prefix every artifact a backtest publishes is named under, so that a
#: store's aliases say where they came from.
ARTIFACT_PREFIX = "backtest"

_NOT_A_NAME = re.compile(r"[^a-z0-9]+")


class Candidate(BaseModel):
    """One entry of a backtest, when the model reference alone is not enough.

    ```python
    of.Candidate("nixtla/nhits", plan=of.FitPlan(window=of.WindowPlan(context=336)))
    of.Candidate(recipe, name="scaled-ensemble")
    ```

    Only ever needed for the two things a bare reference cannot say: the plan
    this model in particular should be fitted with, and what to call it in the
    result. Parameters belong on the ``of.Model`` inside, where every other part
    of OpenForecast reads them from.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: Recipe
    #: What this candidate is called in the result table. Defaults to the model
    #: reference, or to the references a composite recipe names.
    name: str | None = None
    #: Overrides the backtest's own plan entirely, template and all.
    plan: FitPlan | None = None

    def __init__(self, model: ModelInput | None = None, /, **data: Any) -> None:
        """``of.Candidate("nixtla/nhits")`` as well as ``Candidate(model=...)``."""
        if model is not None:
            if "model" in data:
                raise RecipeError("model was given both positionally and by keyword")
            data["model"] = normalize_recipe(model)
        super().__init__(**data)

    @property
    def label(self) -> str:
        """How this candidate appears in the result table."""
        if self.name is not None:
            return self.name
        if isinstance(self.model, Model):
            return str(self.model.ref)
        return "+".join(str(ref) for ref in estimator_refs(self.model))


def backtest(
    models: Sequence[ModelInput | Candidate],
    data: object,
    *,
    validation: Validation,
    metrics: Sequence[Metric],
    plan: FitPlan | None = None,
    client: OpenForecast | None = None,
) -> BacktestResult:
    """Fit and score every model at every origin ``validation`` selects.

    Leaving ``client`` out uses the same default client ``of.fit`` and
    ``of.forecast`` do, so a backtest writes its artifacts where everything else
    does. Passing one pointed at a service backtests there.
    """
    executor = default_client() if client is None else client
    if not metrics:
        raise RecipeError("a backtest needs at least one metric: of.backtest(metrics=[of.MAE()])")
    candidates = _candidates(models)
    folds = validation.folds(data)

    rows: list[_Row] = []
    for candidate in candidates:
        fitted = plan_for(candidate, executor, plan)
        for fold in folds:
            rows.extend(_measure(executor, candidate, fitted, fold, validation.horizon, metrics))
    return BacktestResult(_table(rows), metrics=metrics)


def plan_for(
    candidate: Candidate, client: OpenForecast, template: FitPlan | None
) -> FitPlan | None:
    """The plan this candidate is actually fitted with.

    A candidate's own plan wins outright. Otherwise the backtest's template is
    adapted to what the candidate's models can bind: a context window sizes the
    samples of a sequence model and is not a field a series or tabular model has,
    so it is dropped for a candidate holding none — which is what makes one
    ``plan=`` comparable across model families at all. Nothing else is adapted,
    because nothing else means different things to different contracts.
    """
    if candidate.plan is not None:
        return candidate.plan
    if template is None or template.window is None:
        return template
    views = {_descriptor(client, ref).training.view for ref in estimator_refs(candidate.model)}
    if ViewKind.SEQUENCES in views:
        return template
    return template.model_copy(update={"window": None})


# -- one measurement --------------------------------------------------------


@dataclass(frozen=True)
class _Row:
    """One measured value, and everything that makes it comparable."""

    model: str
    fold: int
    origin: datetime
    metric: str
    value: float
    pairs: int
    fit_seconds: float
    forecast_seconds: float
    origin_fidelity: str
    provider: str
    artifact: str


def _measure(
    client: OpenForecast,
    candidate: Candidate,
    plan: FitPlan | None,
    fold: Fold,
    horizon: int,
    metrics: Sequence[Metric],
) -> list[_Row]:
    """Fit this candidate at one origin, forecast from it, and score the answer."""
    started = perf_counter()
    handle: ModelHandle = client.fit(
        candidate.model,
        fold.train,
        horizon=horizon,
        plan=plan,
        name=_artifact_name(candidate, fold),
    )
    fit_seconds = perf_counter() - started

    started = perf_counter()
    forecast = client.forecast(handle.ref, fold.context, horizon=horizon)
    forecast_seconds = perf_counter() - started

    actual, predicted = _pairs(candidate, fold, forecast)
    return [
        _Row(
            model=candidate.label,
            fold=fold.index,
            origin=fold.origin,
            metric=metric.name,
            value=metric.compute(actual, predicted),
            pairs=len(actual),
            fit_seconds=fit_seconds,
            forecast_seconds=forecast_seconds,
            origin_fidelity=_fidelity(handle),
            provider=handle.manifest.provider,
            artifact=str(handle.ref),
        )
        for metric in metrics
    ]


def _pairs(candidate: Candidate, fold: Fold, forecast: Forecast) -> tuple[list[float], list[float]]:
    """The outcomes and the forecasts of them, aligned and both present.

    A forecast event time whose outcome was never published is dropped rather
    than scored as a zero error, and how many survived is recorded per row as
    ``pairs`` — a fold scored on a third of its horizon should be visible in the
    result rather than only in the metric.

    A *missing forecast* is dropped for the same reason and is the case worth
    naming: a model that answers a NaN has said it does not know, and letting
    that through would make the metric NaN — one unanswerable event time
    destroying the score of a whole fold, with nothing in the result saying which
    one it was.
    """
    known = truth_lookup(fold.truth, forecast.instance_keys)
    point = forecast.point()
    keys = key_rows(point, forecast.instance_keys)
    times: list[datetime] = column_values(point, ForecastColumn.EVENT_TIME.value)
    targets: list[str] = column_values(point, ForecastColumn.TARGET.value)
    values: list[float | None] = column_values(point, ForecastColumn.VALUE.value)

    actual: list[float] = []
    predicted: list[float] = []
    for cell, value in zip(zip(keys, times, targets, strict=True), values, strict=True):
        outcome = known.get(cell)
        # Null and NaN are the two spellings of "no value here", and both mean
        # this event time cannot be scored.
        if outcome is None or value is None or is_missing(outcome) or is_missing(value):
            continue
        actual.append(float(outcome))
        predicted.append(float(value))
    if not actual:
        raise DataError(
            f"nothing to score for {candidate.label} at origin {fold.origin.isoformat()}: the "
            f"forecast covers no event time the truth holds an outcome for. Select origins the "
            f"truth reaches past, or shorten the horizon"
        )
    return actual, predicted


def _fidelity(handle: ModelHandle) -> str:
    """What the artifact says its origins were, rather than what was intended.

    A composite artifact records one training record per leaf, and its members
    may have been materialized from different origins, so the distinct answers
    are reported instead of one of them being chosen.
    """
    found = {record.origin_fidelity.value for record in handle.training_records}
    return ",".join(sorted(found))


# -- candidates -------------------------------------------------------------


def _candidates(models: Sequence[ModelInput | Candidate]) -> tuple[Candidate, ...]:
    """Every entry as a candidate, with labels that identify one model each."""
    if not models:
        raise RecipeError("a backtest needs at least one model to compare")
    candidates = tuple(
        entry if isinstance(entry, Candidate) else Candidate(_fittable(entry)) for entry in models
    )
    for candidate in candidates:
        _reject_pinned(candidate)
    seen: dict[str, int] = {}
    for candidate in candidates:
        seen[candidate.label] = seen.get(candidate.label, 0) + 1
    repeated = sorted(label for label, count in seen.items() if count > 1)
    if repeated:
        raise RecipeError(
            f"{repeated} name more than one candidate, so their rows could not be told "
            f"apart; give each one a name with of.Candidate(model, name=...)"
        )
    return candidates


def _fittable(entry: ModelInput) -> ModelInput:
    """A backtest compares models, and a fitted artifact is not one.

    ``local/de-price@01K...`` is the *result* of a fit. Backtesting it would
    have to either refit the recipe it records — on data it was not fitted on,
    under a name that is not its own — or score one artifact against origins it
    never saw. Both are questions worth asking, and neither is the one this
    function was called to answer.
    """
    if isinstance(entry, ModelHandle):
        raise RecipeError(
            f"{entry.ref} is a fitted artifact, not a candidate; backtest the recipe it "
            f"records, and every fold will fit it on the data of that origin"
        )
    return entry


def _reject_pinned(candidate: Candidate) -> None:
    """A revision names one fit, and a candidate is fitted once per fold."""
    pinned = [str(ref) for ref in estimator_refs(candidate.model) if ref.is_pinned]
    if pinned:
        raise RecipeError(
            f"{pinned} pin fitted revisions; backtest the models they were fitted from, "
            f"and every fold will fit one of its own"
        )


def _artifact_name(candidate: Candidate, fold: Fold) -> str:
    """What to call the artifact one fold of one candidate publishes.

    A label is free text and an artifact name is one path segment, so this is a
    projection rather than the label itself. It carries the fold, because every
    fold is a real fit of its own and an alias that meant "the last fold that
    happened to finish" would be a lie the store then keeps.
    """
    slug = _NOT_A_NAME.sub("-", candidate.label.lower()).strip("-")
    return f"{ARTIFACT_PREFIX}-{slug or 'candidate'}-{fold.index}"


def _descriptor(client: OpenForecast, ref: ModelRef) -> ModelDescriptor:
    """What a reference resolves to, through whatever transport the client has."""
    return client.models.get(str(ref))


# -- the result table -------------------------------------------------------


def _table(rows: Sequence[_Row]) -> pa.Table:
    columns: dict[str, tuple[list[Any], pa.DataType]] = {
        BacktestColumn.MODEL.value: ([row.model for row in rows], pa.string()),
        BacktestColumn.FOLD.value: ([row.fold for row in rows], pa.int64()),
        BacktestColumn.ORIGIN.value: ([row.origin for row in rows], pa.timestamp("us")),
        BacktestColumn.METRIC.value: ([row.metric for row in rows], pa.string()),
        BacktestColumn.VALUE.value: ([row.value for row in rows], pa.float64()),
        BacktestColumn.PAIRS.value: ([row.pairs for row in rows], pa.int64()),
        BacktestColumn.FIT_SECONDS.value: ([row.fit_seconds for row in rows], pa.float64()),
        BacktestColumn.FORECAST_SECONDS.value: (
            [row.forecast_seconds for row in rows],
            pa.float64(),
        ),
        BacktestColumn.ORIGIN_FIDELITY.value: (
            [row.origin_fidelity for row in rows],
            pa.string(),
        ),
        BacktestColumn.PROVIDER.value: ([row.provider for row in rows], pa.string()),
        BacktestColumn.ARTIFACT.value: ([row.artifact for row in rows], pa.string()),
    }
    return build_table(columns)
