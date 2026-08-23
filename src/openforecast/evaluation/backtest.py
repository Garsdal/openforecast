"""``of.backtest``: evaluate models against history, at the origins it holds.

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

``models`` stays plural for a backtest of one: a single model over a stride of
origins passes a list of one, and a ``model=`` singular alias would be a second
door onto the same room.

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

## What it fits, and what it does not

One artifact per candidate and fold, published like any other, and its pinned
reference is recorded in the result. A backtest that scored models without
leaving anything behind would make its own winner unreproducible: the point of
``artifact`` in the table is that the number came from *that* revision, and you
can forecast with it.

A candidate that *is* already a revision — ``local/de-price@01K...``, or the
handle a fit returned — is evaluated rather than refitted, which is how you ask
whether the model in production has drifted:

```text
pinned revision      forecast at every origin, fit_seconds is null
recipe / bare ref    fit per fold
```

Read from what the candidate is rather than from a mode argument, since a
revision names one immutable fit and there is nothing else it could mean.

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
from openforecast.data._arrow import (
    InstanceKey,
    build_table,
    column_type,
    column_values,
    is_missing,
    key_rows,
)
from openforecast.errors import DataError, RecipeError
from openforecast.evaluation.metrics import Metric
from openforecast.evaluation.result import BacktestColumn, BacktestResult, PredictionColumn
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
            data["model"] = normalize_recipe(_as_reference(model))
        super().__init__(**data)

    @property
    def label(self) -> str:
        """How this candidate appears in the result table."""
        if self.name is not None:
            return self.name
        if isinstance(self.model, Model):
            return str(self.model.ref)
        return "+".join(str(ref) for ref in estimator_refs(self.model))

    @property
    def revision(self) -> ModelRef | None:
        """The frozen revision this candidate *is*, if it is one rather than a recipe."""
        if isinstance(self.model, Model) and self.model.ref.is_pinned:
            return self.model.ref
        return None


def backtest(
    models: Sequence[ModelInput | Candidate],
    data: object,
    *,
    validation: Validation,
    metrics: Sequence[Metric],
    plan: FitPlan | None = None,
    client: OpenForecast | None = None,
) -> BacktestResult:
    """Evaluate every model at every origin ``validation`` selects.

    A trainable candidate is fitted on the data of each origin and forecasts
    from it; a pinned revision skips the fit and forecasts as it stands. Both
    end up in one result, and one caveat comes with mixing them: a frozen
    artifact was fitted on data that may postdate the early origins, so its
    numbers are optimistic beside a candidate fitted per fold. That is reported
    — ``fit_seconds`` is null and ``artifact`` names the revision — rather than
    refused, the same way ``origin_fidelity`` is.

    Leaving ``client`` out uses the same default client ``of.fit`` and
    ``of.forecast`` do, so a backtest writes its artifacts where everything else
    does. Passing one pointed at a service backtests there.

    The result holds every point prediction as well as the metrics over them,
    which is what makes ``result.metrics_by("horizon_step")`` a projection
    rather than a second run. It is also the larger of the two tables by far:
    origins × horizon × instances × targets rows per model.
    """
    executor = default_client() if client is None else client
    if not metrics:
        raise RecipeError("a backtest needs at least one metric: of.backtest(metrics=[of.MAE()])")
    candidates = _candidates(models)
    folds = validation.folds(data)

    rows: list[_Row] = []
    predictions: list[_Prediction] = []
    keys: dict[str, pa.DataType] = {}
    for candidate in candidates:
        frozen = _frozen(candidate, executor)
        fit_plan = None if frozen is not None else plan_for(candidate, executor, plan)
        for fold in folds:
            measured = _measure(
                executor, candidate, fit_plan, fold, validation.horizon, metrics, frozen
            )
            rows.extend(measured.rows)
            predictions.extend(measured.predictions)
            keys.update(measured.key_types)
    return BacktestResult(_table(rows), _predictions_table(predictions, keys), scored_by=metrics)


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
    #: Null for a frozen revision: there was no fit, which is not a fit of zero
    #: seconds.
    fit_seconds: float | None
    forecast_seconds: float
    origin_fidelity: str
    provider: str
    artifact: str


@dataclass(frozen=True)
class _Prediction:
    """One forecast point, and the outcome it is about to be scored against."""

    model: str
    fold: int
    keys: InstanceKey
    origin_time: datetime
    event_time: datetime
    horizon_step: int
    target: str
    prediction: float | None
    actual: float | None

    @property
    def pair(self) -> tuple[float, float] | None:
        """``(outcome, forecast)`` when both are numbers, and ``None`` otherwise.

        Null and NaN are the two spellings of "no value here", and either on
        either side means this event time cannot be scored.
        """
        outcome, forecast = self.actual, self.prediction
        if outcome is None or forecast is None or is_missing(outcome) or is_missing(forecast):
            return None
        return outcome, forecast


@dataclass(frozen=True)
class _Measurement:
    """What one candidate at one origin produced: the scores and the predictions."""

    rows: tuple[_Row, ...]
    predictions: tuple[_Prediction, ...]
    #: The Arrow type of each instance key column, as the forecast carried it.
    key_types: dict[str, pa.DataType]


def _measure(
    client: OpenForecast,
    candidate: Candidate,
    plan: FitPlan | None,
    fold: Fold,
    horizon: int,
    metrics: Sequence[Metric],
    frozen: ModelHandle | None,
) -> _Measurement:
    """Forecast this candidate at one origin and score the answer.

    Fitting first, unless the candidate is a frozen revision — which is the
    whole of the difference between evaluating an artifact and backtesting a
    recipe.
    """
    if frozen is None:
        started = perf_counter()
        handle = client.fit(
            candidate.model,
            fold.train,
            horizon=horizon,
            plan=plan,
            name=_artifact_name(candidate, fold),
        )
        fit_seconds: float | None = perf_counter() - started
    else:
        handle, fit_seconds = frozen, None

    started = perf_counter()
    forecast = client.forecast(handle.ref, fold.context, horizon=horizon)
    forecast_seconds = perf_counter() - started

    predictions = _predictions(candidate, fold, forecast)
    scored = [pair for entry in predictions if (pair := entry.pair) is not None]
    actual = [outcome for outcome, _ in scored]
    predicted = [value for _, value in scored]
    rows = tuple(
        _Row(
            model=candidate.label,
            fold=fold.index,
            origin=fold.origin,
            metric=metric.name,
            value=metric.compute(actual, predicted),
            pairs=len(scored),
            fit_seconds=fit_seconds,
            forecast_seconds=forecast_seconds,
            origin_fidelity=_fidelity(handle),
            provider=handle.manifest.provider,
            artifact=str(handle.ref),
        )
        for metric in metrics
    )
    point = forecast.point()
    return _Measurement(
        rows=rows,
        predictions=predictions,
        key_types={name: column_type(point, name) for name in forecast.instance_keys},
    )


def _predictions(candidate: Candidate, fold: Fold, forecast: Forecast) -> tuple[_Prediction, ...]:
    """Every forecast point of one fold, beside what actually happened.

    A forecast event time whose outcome was never published is kept with a null
    ``actual`` rather than dropped or scored as a zero error, and how many
    survived into the metric is recorded per row as ``pairs`` — a fold scored on
    a third of its horizon should be visible in the result rather than only in
    the metric.

    A *missing forecast* is treated the same way and is the case worth naming: a
    model that answers a NaN has said it does not know, and letting that through
    would make the metric NaN — one unanswerable event time destroying the score
    of a whole fold, with nothing in the result saying which one it was.
    """
    known = truth_lookup(fold.truth, forecast.instance_keys)
    point = forecast.point()
    keys = key_rows(point, forecast.instance_keys)
    times: list[datetime] = column_values(point, ForecastColumn.EVENT_TIME.value)
    targets: list[str] = column_values(point, ForecastColumn.TARGET.value)
    values: list[float | None] = column_values(point, ForecastColumn.VALUE.value)
    steps = {moment: index + 1 for index, moment in enumerate(forecast.event_times)}

    found = tuple(
        _Prediction(
            model=candidate.label,
            fold=fold.index,
            keys=key,
            origin_time=fold.origin,
            event_time=moment,
            horizon_step=steps[moment],
            target=target,
            prediction=value,
            actual=known.get((key, moment, target)),
        )
        for key, moment, target, value in zip(keys, times, targets, values, strict=True)
    )
    if all(entry.pair is None for entry in found):
        raise DataError(
            f"nothing to score for {candidate.label} at origin {fold.origin.isoformat()}: the "
            f"forecast covers no event time the truth holds an outcome for. Select origins the "
            f"truth reaches past, or shorten the horizon"
        )
    return found


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
        raise RecipeError("a backtest needs at least one model to evaluate")
    candidates = tuple(
        entry if isinstance(entry, Candidate) else Candidate(entry) for entry in models
    )
    for candidate in candidates:
        _reject_pinned_members(candidate)
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


def _as_reference(entry: ModelInput) -> ModelInput:
    """A fitted handle, as the candidate it stands for: the revision it names.

    ``of.backtest(models=[handle])`` evaluates that artifact over history rather
    than refitting the recipe behind it, so the handle is narrowed to its pinned
    reference here and travels as one from then on.
    """
    if isinstance(entry, ModelHandle):
        return str(entry.ref)
    return entry


def _frozen(candidate: Candidate, client: OpenForecast) -> ModelHandle | None:
    """The artifact this candidate already is, or ``None`` if it is fitted per fold.

    A pinned revision names one immutable fit. Evaluating it over history is a
    real question — it is how you check whether the model in production has
    drifted — and it is the only thing a revision can mean here, since there is
    nothing left to train. What follows from that is that anything configuring a
    fit is refused rather than silently ignored: a plan that will never be bound
    and parameters that will never be read are a caller expecting an effect.
    """
    revision = candidate.revision
    if revision is None:
        return None
    if candidate.plan is not None:
        raise RecipeError(
            f"{revision} is a fitted revision, so it is evaluated rather than fitted and "
            f"the plan on this candidate would do nothing; backtest the model it was fitted "
            f"from to fit one per fold"
        )
    model = candidate.model
    if isinstance(model, Model) and model.params:
        raise RecipeError(
            f"{revision} is a fitted revision, so the parameters it was fitted with are "
            f"already part of it and {sorted(model.params)} would do nothing"
        )
    return client.artifact(str(revision))


def _reject_pinned_members(candidate: Candidate) -> None:
    """A revision is a whole fitted artifact, so it is not a step inside a recipe.

    A pinned candidate on its own is evaluated as it stands. Pinned *inside* a
    pipeline or an ensemble is a different thing: the recipe around it is fitted
    per fold, and there is no way to fit a step that is already fitted.
    """
    if candidate.revision is not None:
        return
    pinned = [str(ref) for ref in estimator_refs(candidate.model) if ref.is_pinned]
    if pinned:
        raise RecipeError(
            f"{pinned} pin fitted revisions inside a recipe that is fitted per fold; name "
            f"the models they were fitted from, or backtest the revision on its own to "
            f"evaluate it as it stands"
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


# -- the result tables ------------------------------------------------------


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


def _predictions_table(rows: Sequence[_Prediction], key_types: dict[str, pa.DataType]) -> pa.Table:
    """The predictions, with the instance keys under the caller's own names.

    Their Arrow types are the ones the forecasts came back with rather than
    inferred here, so a key that is an integer zone id stays one.
    """
    columns: dict[str, tuple[list[Any], pa.DataType]] = {
        PredictionColumn.MODEL.value: ([row.model for row in rows], pa.string()),
        PredictionColumn.FOLD.value: ([row.fold for row in rows], pa.int64()),
    }
    for position, name in enumerate(key_types):
        columns[name] = ([row.keys[position] for row in rows], key_types[name])
    columns[PredictionColumn.ORIGIN_TIME.value] = (
        [row.origin_time for row in rows],
        pa.timestamp("us"),
    )
    columns[PredictionColumn.EVENT_TIME.value] = (
        [row.event_time for row in rows],
        pa.timestamp("us"),
    )
    columns[PredictionColumn.HORIZON_STEP.value] = (
        [row.horizon_step for row in rows],
        pa.int64(),
    )
    columns[PredictionColumn.TARGET.value] = ([row.target for row in rows], pa.string())
    columns[PredictionColumn.PREDICTION.value] = (
        [row.prediction for row in rows],
        pa.float64(),
    )
    columns[PredictionColumn.ACTUAL.value] = ([row.actual for row in rows], pa.float64())
    return build_table(columns)
