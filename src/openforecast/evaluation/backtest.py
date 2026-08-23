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

Scoring a distribution is the same call with the output the metrics need:

```python
result = of.backtest(
    models=["nixtla/autoarima"],
    data=data,
    validation=of.RollingOrigin(horizon=24, windows=5),
    output=of.OutputSpec.quantiles([0.1, 0.5, 0.9]),
    metrics=[of.MAE(), of.PinballLoss(0.9), of.Coverage()],
)
```

``output`` is asked of every candidate, and the metrics are checked against it
before the first fit: what a metric needs is knowable from the request, and a
coverage of a point forecast is refused in the first line of the run rather than
after an hour of fitting.

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

A pretrained model — ``amazon/chronos-2`` — is the same shape of candidate for
the opposite reason: there was never anything to train, so it forecasts at every
origin as it stands. That is what makes the two lifecycles comparable on one
leaderboard:

```text
pinned revision      forecast at every origin, fit_seconds is null
pretrained model     forecast at every origin, fit_seconds is null, no artifact
recipe / bare ref    fit per fold
```

Read from what the candidate is rather than from a mode argument, since a
revision names one immutable fit, a pretrained reference names a model that
cannot be fitted, and there is nothing else either could mean.

The two null ``fit_seconds`` do not mean the same thing about the numbers beside
them, and ``origin_fidelity`` is where the difference shows: a frozen revision
reports the fidelity of the fit it came from and may have seen data that
postdates the early origins, while a pretrained model reports ``pretrained`` and
saw none of this data at any origin.

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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from typing import Any, cast

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, model_validator

from openforecast.artifacts.handle import ModelHandle
from openforecast.artifacts.manifest import LOCAL_NAMESPACE
from openforecast.client import OpenForecast, default_client
from openforecast.data._arrow import (
    InstanceKey,
    build_table,
    column_type,
    column_values,
    key_rows,
)
from openforecast.errors import DataError, RecipeError
from openforecast.evaluation.metrics import Metric
from openforecast.evaluation.predictions import PredictedValue, predictions_of
from openforecast.evaluation.result import BacktestColumn, BacktestResult, PredictionColumn
from openforecast.evaluation.validation import Fold, Validation, truth_lookup
from openforecast.models.descriptor import ModelDescriptor
from openforecast.models.ref import ModelRef
from openforecast.protocol.vocabulary import ForecastColumn, ViewKind
from openforecast.recipes.nodes import Model, Recipe, estimator_refs
from openforecast.runtime.engine import ModelInput, normalize_recipe
from openforecast.runtime.forecast import Forecast
from openforecast.tasks.forecast import OutputSpec
from openforecast.tasks.plan import FitPlan

__all__ = ["Candidate", "backtest", "plan_for"]

#: The prefix every artifact a backtest publishes is named under, so that a
#: store's aliases say where they came from.
ARTIFACT_PREFIX = "backtest"

_NOT_A_NAME = re.compile(r"[^a-z0-9]+")

#: What ``origin_fidelity`` says for a model that was never fitted here. The
#: column reports how the *training* origins were come by, and a pretrained model
#: has none of them in this run — so it is neither ``simulated`` nor ``observed``
#: rather than defaulting to one of the two.
PRETRAINED = "pretrained"

#: What a candidate already is, when it is not a recipe to fit per fold.
_Standing = ModelHandle | ModelDescriptor


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

    @model_validator(mode="before")
    @classmethod
    def _accept_a_reference(cls, value: object) -> object:
        """Let ``{"model": "nixtla/nhits"}`` mean what ``Candidate("nixtla/nhits")`` does.

        Deserialization goes through the core schema rather than through
        ``__init__``, so without this a candidate written down — in a CLI config
        file, in a saved experiment — would be the one place in OpenForecast
        where the short spelling of a model is not accepted. It is the same
        normalization :meth:`__init__` performs, and the same accommodation
        :class:`~openforecast.models.ref.ModelRef` makes for a plain string.
        """
        if not isinstance(value, Mapping):
            return value
        written = cast(Mapping[str, Any], value)
        reference = written.get("model")
        if isinstance(reference, str):
            return {**written, "model": normalize_recipe(reference)}
        return written

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
    output: OutputSpec | None = None,
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

    ``output`` is what every candidate is asked for, and it defaults to a point
    forecast. Asking for quantiles is what makes ``of.PinballLoss``,
    ``of.Coverage`` and ``of.IntervalWidth`` computable, and the metrics are
    checked against it before anything is fitted: a coverage of a point forecast
    is refused in the first line of the run rather than after an hour of fits.

    The result holds every prediction as well as the metrics over them, which is
    what makes ``result.metrics_by("horizon_step")`` a projection rather than a
    second run. It is also the larger of the two tables by far: origins ×
    horizon × instances × targets rows per model, once per quantile level or
    sample draw.
    """
    executor = default_client() if client is None else client
    if not metrics:
        raise RecipeError("a backtest needs at least one metric: of.backtest(metrics=[of.MAE()])")
    requested = OutputSpec.point() if output is None else output
    _check_metrics(metrics, requested)
    candidates = _candidates(models)
    folds = validation.folds(data)

    rows: list[_Row] = []
    predictions: list[_Prediction] = []
    keys: dict[str, pa.DataType] = {}
    for candidate in candidates:
        standing = _standing(candidate, executor)
        fit_plan = None if standing is not None else plan_for(candidate, executor, plan)
        for fold in folds:
            measured = _measure(
                executor,
                candidate,
                fit_plan,
                fold,
                validation.horizon,
                metrics,
                standing,
                requested,
            )
            rows.extend(measured.rows)
            predictions.extend(measured.predictions)
            keys.update(measured.key_types)
    return BacktestResult(_table(rows), _predictions_table(predictions, keys), scored_by=metrics)


def _check_metrics(metrics: Sequence[Metric], output: OutputSpec) -> None:
    """Every metric can score the forecast this backtest is going to ask for.

    Before the first fit, for the same reason the engine checks an output request
    against a model's declared capabilities before starting a provider: what a
    metric needs is knowable from the request, and discovering it afterwards
    means discovering it after the expensive part.
    """
    unanswerable = [reason for metric in metrics if (reason := metric.requirement(output))]
    if unanswerable:
        raise RecipeError("; ".join(unanswerable))


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
    contracts = [_descriptor(client, ref).training for ref in estimator_refs(candidate.model)]
    if any(contract is not None and contract.view is ViewKind.SEQUENCES for contract in contracts):
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
    #: Null where this metric could score nothing of this fold, beside a
    #: ``pairs`` of zero. A metric over nothing is not a zero score.
    value: float | None
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
    """One forecast row, and the outcome it is about to be scored against.

    One *row* rather than one number: a probabilistic forecast says several
    things about one outcome — a value per quantile level, or per sample draw —
    and ``kind``, ``quantile`` and ``sample`` are which of them this is, spelled
    exactly as a forecast spells them.
    """

    model: str
    fold: int
    keys: InstanceKey
    origin_time: datetime
    event_time: datetime
    horizon_step: int
    target: str
    kind: str
    quantile: float | None
    sample: int | None
    prediction: float | None
    actual: float | None

    @property
    def outcome(self) -> tuple[Any, ...]:
        """What identifies the thing forecast, without saying anything about it.

        The model and the fold are part of it: two candidates forecasting the
        same event time are two predictions of one outcome, and pooling their
        quantiles into one distribution would score a model that never existed.
        """
        return (
            self.model,
            self.fold,
            self.keys,
            self.origin_time,
            self.event_time,
            self.target,
        )

    @property
    def value(self) -> PredictedValue:
        """This row as a metric's input rather than as a table row."""
        return PredictedValue(
            outcome=self.outcome,
            kind=self.kind,
            level=self.quantile,
            draw=self.sample,
            predicted=self.prediction,
            actual=self.actual,
        )

    @property
    def is_scorable(self) -> bool:
        """Whether this row says something a metric can score.

        Null and NaN are the two spellings of "no value here", and either on
        either side means this row cannot be scored.
        """
        return self.value.is_scorable


@dataclass(frozen=True)
class _Executed:
    """The model one fold actually forecast with, as the result records it."""

    #: What ``client.forecast`` is handed: a pinned artifact, or a model that
    #: needs no fitting. Either way a reference, so a backtest over a service
    #: names it the same way a local one does.
    model: str
    fit_seconds: float | None
    provider: str
    artifact: str
    origin_fidelity: str


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
    standing: _Standing | None,
    output: OutputSpec,
) -> _Measurement:
    """Forecast this candidate at one origin and score the answer.

    Fitting first, unless the candidate already stands on its own — a frozen
    revision or a pretrained model — which is the whole of the difference
    between evaluating a model and backtesting a recipe.
    """
    executed = _fit_for(client, candidate, plan, fold, horizon, standing)

    started = perf_counter()
    forecast = client.forecast(executed.model, fold.context, horizon=horizon, output=output)
    forecast_seconds = perf_counter() - started

    predictions = _predictions(candidate, fold, forecast)
    scored = predictions_of(entry.value for entry in predictions)
    measured = [(metric, metric.measure(scored)) for metric in metrics]
    rows = tuple(
        _Row(
            model=candidate.label,
            fold=fold.index,
            origin=fold.origin,
            metric=metric.name,
            value=measurement.value,
            pairs=measurement.pairs,
            fit_seconds=executed.fit_seconds,
            forecast_seconds=forecast_seconds,
            origin_fidelity=executed.origin_fidelity,
            provider=executed.provider,
            artifact=executed.artifact,
        )
        for metric, measurement in measured
    )
    return _Measurement(
        rows=rows,
        predictions=predictions,
        key_types={name: column_type(forecast.table, name) for name in forecast.instance_keys},
    )


def _predictions(candidate: Candidate, fold: Fold, forecast: Forecast) -> tuple[_Prediction, ...]:
    """Every forecast row of one fold, beside what actually happened.

    Every row, whatever kind of answer it holds: a quantile forecast contributes
    one row per level and a sample forecast one per draw, which is what makes the
    prediction table the same table for every provider and every output kind.

    A forecast event time whose outcome was never published is kept with a null
    ``actual`` rather than dropped or scored as a zero error, and how many rows
    survived into each metric is recorded per row as ``pairs`` — a fold scored on
    a third of its horizon should be visible in the result rather than only in
    the metric.

    A *missing forecast* is treated the same way and is the case worth naming: a
    model that answers a NaN has said it does not know, and letting that through
    would make the metric NaN — one unanswerable event time destroying the score
    of a whole fold, with nothing in the result saying which one it was.
    """
    known = truth_lookup(fold.truth, forecast.instance_keys)
    table = forecast.table
    keys = key_rows(table, forecast.instance_keys)
    times: list[datetime] = column_values(table, ForecastColumn.EVENT_TIME.value)
    targets: list[str] = column_values(table, ForecastColumn.TARGET.value)
    kinds: list[str] = column_values(table, ForecastColumn.KIND.value)
    levels: list[float | None] = column_values(table, ForecastColumn.QUANTILE.value)
    draws: list[int | None] = column_values(table, ForecastColumn.SAMPLE.value)
    values: list[float | None] = column_values(table, ForecastColumn.VALUE.value)
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
            kind=kind,
            quantile=level,
            sample=draw,
            prediction=value,
            actual=known.get((key, moment, target)),
        )
        for key, moment, target, kind, level, draw, value in zip(
            keys, times, targets, kinds, levels, draws, values, strict=True
        )
    )
    if not any(entry.is_scorable for entry in found):
        raise DataError(
            f"nothing to score for {candidate.label} at origin {fold.origin.isoformat()}: the "
            f"forecast covers no event time the truth holds an outcome for. Select origins the "
            f"truth reaches past, or shorten the horizon"
        )
    return found


def _fit_for(
    client: OpenForecast,
    candidate: Candidate,
    plan: FitPlan | None,
    fold: Fold,
    horizon: int,
    standing: _Standing | None,
) -> _Executed:
    """What this fold forecasts with, and what the result table says it was."""
    if isinstance(standing, ModelDescriptor):
        return _Executed(
            model=str(standing.ref),
            fit_seconds=None,
            provider=standing.provider,
            artifact=str(standing.ref),
            origin_fidelity=PRETRAINED,
        )
    if standing is not None:
        return _described(standing, fit_seconds=None)
    started = perf_counter()
    handle = client.fit(
        candidate.model,
        fold.train,
        horizon=horizon,
        plan=plan,
        name=_artifact_name(candidate, fold),
    )
    return _described(handle, fit_seconds=perf_counter() - started)


def _described(handle: ModelHandle, *, fit_seconds: float | None) -> _Executed:
    return _Executed(
        model=str(handle.ref),
        fit_seconds=fit_seconds,
        provider=handle.manifest.provider,
        artifact=str(handle.ref),
        origin_fidelity=_fidelity(handle),
    )


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


def _standing(candidate: Candidate, client: OpenForecast) -> _Standing | None:
    """What this candidate already is, or ``None`` if it is fitted per fold.

    Two candidates need no fit, for opposite reasons, and they are found the
    same way — from what the candidate *is*, never from a mode argument:

    ```text
    local/de-price@01K...   a fitted revision: there is nothing left to train
    amazon/chronos-2        a pretrained model: there was never anything to train
    ```

    Both make the same thing true of the run, which is why they share a branch:
    ``fit_seconds`` is null because no fit happened, and anything configuring a
    fit is refused rather than silently ignored — a plan that will never be
    bound and parameters that will never be read are a caller expecting an
    effect.

    They differ in one way worth knowing when reading a leaderboard. A frozen
    revision was fitted on data that may postdate the early origins, so its
    numbers can be optimistic. A pretrained model never saw this data at all, so
    its numbers are the honest zero-shot ones, at every origin equally.
    """
    revision = candidate.revision
    if revision is not None:
        _reject_fit_settings(candidate, revision, "a fitted revision, so it is evaluated as it is")
        return client.artifact(str(revision))
    pretrained = _pretrained(candidate, client)
    if pretrained is None:
        return None
    _reject_fit_settings(
        candidate, pretrained.ref, "used zero-shot, so it forecasts as it was published"
    )
    return pretrained


def _pretrained(candidate: Candidate, client: OpenForecast) -> ModelDescriptor | None:
    """The pretrained model this candidate is, if there is nothing to fit.

    Only a bare provider reference can be one. A composite recipe is fitted as a
    whole — a member that cannot be fitted is refused by ``of.fit``, which is
    where that belongs — and a ``local/`` name is an artifact rather than a model
    the catalog describes.
    """
    model = candidate.model
    if not isinstance(model, Model) or model.ref.namespace == LOCAL_NAMESPACE:
        return None
    descriptor = _descriptor(client, model.ref)
    return None if descriptor.is_fittable else descriptor


def _reject_fit_settings(candidate: Candidate, ref: ModelRef, because: str) -> None:
    """Nothing configuring a fit may be set on a candidate that is never fitted."""
    if candidate.plan is not None:
        raise RecipeError(
            f"{ref} is {because} rather than fitted, and the plan on this candidate would "
            f"do nothing; backtest a model that is fitted per fold to use one"
        )
    model = candidate.model
    if isinstance(model, Model) and model.params:
        raise RecipeError(
            f"{ref} is {because} rather than fitted, so the parameters it runs with are "
            f"already part of it and {sorted(model.params)} would do nothing"
        )


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
    columns[PredictionColumn.KIND.value] = ([row.kind for row in rows], pa.string())
    columns[PredictionColumn.QUANTILE.value] = ([row.quantile for row in rows], pa.float64())
    columns[PredictionColumn.SAMPLE.value] = ([row.sample for row in rows], pa.int64())
    columns[PredictionColumn.PREDICTION.value] = (
        [row.prediction for row in rows],
        pa.float64(),
    )
    columns[PredictionColumn.ACTUAL.value] = ([row.actual for row in rows], pa.float64())
    return build_table(columns)
