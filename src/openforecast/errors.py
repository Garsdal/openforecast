"""The OpenForecast exception hierarchy.

The innermost module in the package: it imports nothing from OpenForecast and
everything else may import it.

None of these inherit from :class:`ValueError`. That is deliberate — most of
them are raised from Pydantic model validators, and Pydantic converts
``ValueError`` into its own ``ValidationError`` while letting other exceptions
propagate unchanged. Keeping them outside the ``ValueError`` hierarchy means a
caller sees the OpenForecast error it can act on rather than a wrapped one.
"""

from __future__ import annotations

__all__ = [
    "DataError",
    "DuplicateModelError",
    "FrequencyError",
    "InconsistentTruthError",
    "ModelError",
    "ModelRefError",
    "OpenForecastError",
    "OriginScopeError",
    "SchemaError",
    "UnknownModelError",
]


class OpenForecastError(Exception):
    """Base class for every error OpenForecast raises deliberately."""


class SchemaError(OpenForecastError):
    """Declared semantics are internally inconsistent.

    Raised before any data is looked at: a target that is also a feature, a
    static feature carrying an availability, duplicate instance keys.
    """


class FrequencyError(SchemaError):
    """A frequency cannot be parsed, or has no fixed duration."""


class DataError(OpenForecastError):
    """Data does not satisfy the schema it was declared against.

    Raised instead of repairing the data. Duplicate rows are not deduplicated,
    off-grid timestamps are not snapped, missing values are not imputed.
    """


class OriginScopeError(OpenForecastError):
    """More forecast origins were selected than the requested view can express.

    A ``SeriesView`` is one complete time series, so it has room for exactly one
    origin. Asking for every historical vintage of a point-in-time dataset in
    that shape has no meaning, and picking one of them silently would train a
    model on data the caller never asked for.
    """


class ModelError(OpenForecastError):
    """A model could not be named or resolved.

    About the identifier and the catalog, not about the data: a model whose
    contract the data cannot satisfy raises a data or scope error instead.
    """


class ModelRefError(ModelError, SchemaError):
    """A model reference is not ``<namespace>/<name>[@revision]``.

    Inherits from :class:`SchemaError` as well, because a malformed reference is
    a declaration that is wrong before any data is looked at, and a caller
    catching declaration errors should see this one too.
    """


class UnknownModelError(ModelError):
    """Nothing is registered under that reference.

    Distinct from a malformed reference: the name is well-formed, and no
    provider advertises it.
    """


class DuplicateModelError(ModelError):
    """Two descriptors claim the same reference.

    A reference has to identify one model. Letting the second registration win
    would make which model you get depend on provider load order.
    """


class InconsistentTruthError(DataError):
    """Vintages of the same event time disagree about what happened.

    A point-in-time source table repeats its labels on every origin. If two of
    those copies hold different realizations, only one can be the outcome, and
    OpenForecast will not pick for you.
    """
