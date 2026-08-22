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
    "FrequencyError",
    "InconsistentTruthError",
    "OpenForecastError",
    "SchemaError",
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


class InconsistentTruthError(DataError):
    """Vintages of the same event time disagree about what happened.

    A point-in-time source table repeats its labels on every origin. If two of
    those copies hold different realizations, only one can be the outcome, and
    OpenForecast will not pick for you.
    """
