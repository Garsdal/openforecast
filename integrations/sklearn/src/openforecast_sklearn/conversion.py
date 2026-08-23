"""The translation layer, and it is nearly nothing.

```text
TabularView   ->  X: ndarray, y: ndarray            already row-aligned
ForecastView  ->  X: ndarray, and what labels it    one row per instance and lead
predictions   ->  the canonical forecast columns, as Arrow
```

The other integrations have a real conversion module: Nixtla wants a long frame
with ``unique_id`` and ``ds``, Darts wants a ``TimeSeries`` object per series,
sktime wants a ``MultiIndex`` panel and an exogenous frame. Each of those is a
reshape, and each is somewhere a forecasting semantic can be lost.

scikit-learn wants a matrix and a vector. So there is no reshape here at all —
``TabularView.X`` is already the design matrix, in the column order its schema
declares, and this module casts it to ``float64`` and hands it over. That is the
claim Step 18 makes, and the length of this file is the evidence for it: what
would otherwise be a reduction — origins, leads, vintages, truth alignment — was
already done by the ``ViewPlanner``, once, for every provider.

Two asymmetries with the fit side are worth naming, because they are where the
work actually is.

**Inference assembles the matrix; training receives it.** A ``ForecastView`` is
not a table of rows, it is a history, a future and a static table. So the
horizon rows are built here, in one deterministic order — instance-major,
ascending event time — and the answer is labeled from that same order rather than
from anything the estimator returned. An estimator returns ``n`` numbers and no
idea what they are about.

**A static feature is a column, repeated.** The fit side got that for free: the
``ViewPlanner`` broadcast the static value onto every row of the instance it
belongs to. Here it is done by hand, from the view's static table, which is the
one place this integration duplicates something OpenForecast already knows how
to do — and it duplicates it against the recorded column order, so a mismatch is
a refusal rather than a shifted matrix.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from openforecast.errors import DataError, ProviderError
from openforecast.views import (
    EVENT_TIME,
    ForecastColumn,
    ForecastView,
    TabularView,
    forecast_columns,
)

__all__ = [
    "DesignMatrix",
    "InstanceKey",
    "InferenceMatrix",
    "answer",
    "design_matrix",
    "inference_matrix",
    "single_target",
]

#: An instance of the caller's data, as the tuple of its key columns.
InstanceKey = tuple[Any, ...]

#: What a scikit-learn estimator computes in. Casting once, here, is what makes
#: "the missing values reach the estimator unchanged" checkable: a null in an
#: Arrow column and a ``NaN`` in a float column are the two spellings of the same
#: absence, and ``float64`` holds both as ``NaN``.
DOUBLE = pa.float64()


@dataclass(frozen=True)
class DesignMatrix:
    """A ``TabularView`` as the two arrays ``estimator.fit`` takes."""

    X: np.ndarray[Any, Any]
    y: np.ndarray[Any, Any]
    #: The caller's name for the one target being modeled.
    target: str
    #: The columns of ``X``, in the order ``X`` holds them. The contract between
    #: this fit and every forecast made from it: an estimator has positions.
    features: tuple[str, ...]
    #: The two feature roles, so the forecast side knows which table to read each
    #: column from — a known feature is in ``future``, a static one in ``static``.
    known: tuple[str, ...]
    static: tuple[str, ...]


@dataclass(frozen=True)
class InferenceMatrix:
    """One origin's horizon rows, and what each of them is about."""

    X: np.ndarray[Any, Any]
    #: Row *i* of ``X`` asks about ``instances[i]`` at ``event_times[i]``.
    instances: tuple[InstanceKey, ...]
    event_times: tuple[datetime, ...]


def single_target(targets: Sequence[str]) -> str:
    """The one target being modeled, or a refusal counting what was given.

    A multi-target wrapper is deliberately not built here. scikit-learn has one,
    and reaching for it would mean the first estimator this integration exposes
    is a composite rather than an estimator.
    """
    if len(targets) != 1:
        raise ProviderError(
            f"this provider fits one target at a time and was given {len(targets)}: {list(targets)}"
        )
    return targets[0]


# -- training ----------------------------------------------------------------


def design_matrix(view: TabularView) -> DesignMatrix:
    """``X`` and ``y``, straight off the view.

    Nothing is grouped, deduplicated or sorted. Two rows describing the same
    event time from two origins are two rows, because their information vintages
    differ and they are therefore two distinct forecasting examples — and the
    label they share is repeated rather than reconciled. That the duplication
    looks like a mistake to a reader who has only seen event-time tables is
    exactly why it is stated here.

    A missing *feature* is fine and is the reason this estimator was chosen. A
    missing *label* is not: it is a row with nothing to learn from, and dropping
    it here would silently change the training set the manifest says was used.
    """
    schema = view.schema
    target = single_target(schema.targets)
    features = schema.x_columns
    if not features:  # pragma: no cover - the planner refuses to build one
        raise ProviderError("a supervised row needs at least one feature column")
    return DesignMatrix(
        X=_matrix(view.X, features, "X"),
        y=_labels(_matrix(view.y, (target,), "y")[:, 0], target),
        target=target,
        features=features,
        known=tuple(feature.name for feature in schema.known_features),
        static=schema.static_feature_names,
    )


# -- inference ---------------------------------------------------------------


def inference_matrix(
    view: ForecastView,
    *,
    features: Sequence[str],
    known: Sequence[str],
    static: Sequence[str],
) -> InferenceMatrix:
    """The horizon rows of one origin, in the column order the fit recorded.

    One row per instance and lead, which is the same row a training example was —
    the only difference being that its label has not happened yet. The known
    features come from the view's ``future`` table, because a tabular row
    describes an event time *after* the origin and that is the table which
    reaches there.
    """
    instances = view.instances
    event_times = view.event_times

    _require_columns(view.future.column_names, known, "known features")
    ordered = _horizon_rows(view, instances, event_times)
    columns: dict[str, pa.ChunkedArray[Any] | pa.Array[Any]] = {
        name: ordered.column(name) for name in known
    }
    if static:
        values = _static_values(view, static)
        for name in static:
            columns[name] = pa.array(
                [values[instance][name] for instance in instances for _ in event_times]
            )

    absent = sorted(set(features) - set(columns))
    if absent:  # pragma: no cover - the two roles cover every recorded column
        raise DataError(f"this model was fitted with the features {absent}, which are not here")
    rows = pa.table({name: columns[name] for name in features})
    return InferenceMatrix(
        X=_matrix(rows, tuple(features), "X"),
        instances=tuple(instance for instance in instances for _ in event_times),
        event_times=tuple(moment for _ in instances for moment in event_times),
    )


def answer(
    view: ForecastView,
    predicted: np.ndarray[Any, Any],
    *,
    rows: InferenceMatrix,
    target: str,
) -> pa.Table:
    """The canonical long forecast, from the numbers the estimator returned.

    An estimator answers with a bare array, so what each number is about is the
    order the matrix was built in and nothing else. That makes the length check
    below the whole of the mapping back: row *i* asks about ``instances[i]`` at
    ``event_times[i]``, so a prediction vector of the right length is labeled,
    and one of the wrong length is a provider that did not answer the question.
    """
    values = np.asarray(predicted, dtype=float).reshape(-1)
    if values.shape[0] != len(rows.instances):
        raise ProviderError(
            f"the fitted estimator was asked about {len(rows.instances)} instance/event-time "
            f"rows and answered {values.shape[0]} of them"
        )

    instance_keys = view.metadata.instance_keys
    columns: dict[str, pa.Array[Any]] = {
        name: pa.array(
            [instance[index] for instance in rows.instances], type=view.future.column(name).type
        )
        for index, name in enumerate(instance_keys)
    }
    count = len(values)
    columns[ForecastColumn.EVENT_TIME.value] = pa.array(
        list(rows.event_times), type=view.future.column(EVENT_TIME).type
    )
    columns[ForecastColumn.TARGET.value] = pa.array([target] * count, type=pa.string())
    columns[ForecastColumn.KIND.value] = pa.array(["point"] * count, type=pa.string())
    columns[ForecastColumn.QUANTILE.value] = pa.nulls(count, type=pa.float64())
    columns[ForecastColumn.SAMPLE.value] = pa.nulls(count, type=pa.int64())
    columns[ForecastColumn.VALUE.value] = pa.array(values.tolist(), type=pa.float64())
    return pa.table({name: columns[name] for name in forecast_columns(instance_keys)})


# -- the pieces --------------------------------------------------------------


def _matrix(table: pa.Table, columns: Sequence[str], label: str) -> np.ndarray[Any, Any]:
    """``columns`` of ``table`` as one ``float64`` array, missing values intact.

    A null and a ``NaN`` both arrive as ``NaN``, which is the one representation
    ``HistGradientBoostingRegressor`` reads as "no value here". Nothing is filled
    in: an imputation this integration performed would be a number the artifact
    does not record, and a caller who wanted one writes it down as an
    ``of.Impute`` step instead.
    """
    cast: list[np.ndarray[Any, Any]] = []
    for name in columns:
        column = table.column(name)
        try:
            numeric = pc.cast(column, DOUBLE)
        except pa.ArrowInvalid as error:
            raise ProviderError(
                f"a scikit-learn estimator takes numeric features and {label} column "
                f"{name!r} holds {column.type}: {error}. Encode it before it reaches "
                f"OpenForecast, or drop it from the data"
            ) from error
        except pa.ArrowNotImplementedError as error:
            raise ProviderError(
                f"a scikit-learn estimator takes numeric features and {label} column "
                f"{name!r} holds {column.type}, which has no numeric reading: {error}"
            ) from error
        cast.append(np.asarray(numeric.to_numpy(zero_copy_only=False), dtype=float))
    return np.column_stack(cast)


def _labels(y: np.ndarray[Any, Any], target: str) -> np.ndarray[Any, Any]:
    """``y``, once every row of it has an outcome to learn from.

    Refused rather than dropped: a manifest records how many supervised rows the
    fit was given, and quietly training on fewer of them would make that number
    wrong. The caller either removes those event times from the data or fills
    them in — both of which are visible.
    """
    missing = int(np.isnan(y).sum())
    if missing:
        raise ProviderError(
            f"{missing} of {y.shape[0]} supervised rows have no {target} to learn from. A "
            f"missing feature is information this estimator branches on; a missing label is "
            f"not an example. Drop those event times, or fill them in explicitly"
        )
    return y


def _horizon_rows(
    view: ForecastView,
    instances: Sequence[InstanceKey],
    event_times: Sequence[datetime],
) -> pa.Table:
    """``view.future``, reordered to one row per instance and lead.

    The order is this module's, not the table's: a forecast is labeled from it,
    so it may not depend on how a transport happened to lay the rows out.
    """
    keys = view.metadata.instance_keys
    positions = {
        cell: position
        for position, cell in enumerate(
            zip(_key_rows(view.future, keys), _column(view.future, EVENT_TIME), strict=True)
        )
    }
    order: list[int] = []
    for instance in instances:
        for moment in event_times:
            position = positions.get((instance, moment))
            if position is None:  # pragma: no cover - the view validates its own split
                raise DataError(
                    f"this forecast view does not describe {moment.isoformat()} for instance "
                    f"{instance}, and a supervised row cannot be built without it"
                )
            order.append(position)
    return view.future.take(pa.array(order, type=pa.int64()))


def _static_values(
    view: ForecastView, names: Sequence[str]
) -> Mapping[InstanceKey, Mapping[str, Any]]:
    """The static features of every instance in a forecast view."""
    if view.static is None:
        raise DataError(
            f"this model was fitted with the static features {list(names)} and the forecast "
            f"view carries none"
        )
    _require_columns(view.static.column_names, names, "static features")
    keys = view.metadata.instance_keys
    rows = _key_rows(view.static, keys)
    return {
        instance: {name: _column(view.static, name)[position] for name in names}
        for position, instance in enumerate(rows)
    }


def _require_columns(present: Sequence[str], wanted: Sequence[str], role: str) -> None:
    absent = sorted(set(wanted) - set(present))
    if absent:
        raise DataError(
            f"this model was fitted with the {role} {list(wanted)} and the forecast view "
            f"carries no {absent}"
        )


def _column(table: pa.Table, name: str) -> list[Any]:
    values: list[Any] = table.column(name).to_pylist()
    return values


def _key_rows(table: pa.Table, instance_keys: Sequence[str]) -> list[InstanceKey]:
    if not instance_keys:
        return [()] * table.num_rows
    columns = [_column(table, name) for name in instance_keys]
    return list(zip(*columns, strict=True))
