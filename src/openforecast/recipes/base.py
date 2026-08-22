"""Vocabulary shared by every recipe node.

A recipe is an AST, and every node in it carries a ``kind`` tag. The tag is a
field rather than a class attribute so that a serialized recipe says what it is,
which is what lets the same JSON travel to a provider subprocess in Step 9 and
over HTTP in Step 16 without a reader having to infer the shape.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from openforecast.data.schema import reject_duplicate_names
from openforecast.errors import RecipeError

__all__ = ["ColumnSelector", "ColumnSet", "ColumnTransform", "RecipeKind", "RecipeNode"]


class RecipeKind(StrEnum):
    """Every node a recipe can hold, and its wire spelling."""

    MODEL = "model"
    PIPELINE = "pipeline"
    ENSEMBLE = "ensemble"
    REDUCTION = "reduction"

    STANDARD_SCALER = "standard_scaler"
    MISSING_INDICATOR = "missing_indicator"
    IMPUTE = "impute"
    LEAD_TIME_FEATURE = "lead_time_feature"
    ORIGIN_CALENDAR_FEATURES = "origin_calendar_features"


class ColumnSet(StrEnum):
    """A role, rather than a list of names.

    ``columns="targets"`` keeps a recipe portable across datasets: the same
    pipeline can be fitted on German prices and on Danish load without being
    rewritten, because the role is resolved against whatever schema it meets.
    """

    TARGETS = "targets"
    FEATURES = "features"


#: Either a role or an explicit list of column names.
ColumnSelector = ColumnSet | tuple[str, ...]


class RecipeNode(BaseModel):
    """One node of a recipe.

    Frozen, so that a recipe handed to ``of.fit`` is the recipe recorded in the
    artifact manifest — an artifact that cannot say what produced it is not
    reproducible.

    ``kind`` is declared by each concrete node rather than here, because it is
    the discriminator of the recipe union: every node narrows it to exactly one
    value.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class ColumnTransform(RecipeNode):
    """A transform that applies to a set of columns."""

    columns: ColumnSelector

    @field_validator("columns", mode="before")
    @classmethod
    def _reject_a_bare_column_name(cls, value: object) -> object:
        """``columns="load"`` is a name, and names come as a list.

        Accepting a bare string would make ``columns="targets"`` ambiguous
        forever: a dataset is free to have a column actually called ``targets``.
        """
        if isinstance(value, str) and value not in ColumnSet.__members__.values():
            raise RecipeError(
                f"{value!r} is not a column role; pass a list of column names, or one of "
                f"{[member.value for member in ColumnSet]}"
            )
        return value

    @model_validator(mode="after")
    def _check_columns(self) -> Self:
        names = self.explicit_columns
        if names is None:
            return self
        if not names:
            raise RecipeError(
                f"{type(self).__name__} names no columns; pass a list of column names, "
                f"or one of {[member.value for member in ColumnSet]}"
            )
        reject_duplicate_names(names, "transform column")
        if any(not name.strip() for name in names):
            raise RecipeError(f"{type(self).__name__} names an empty column")
        return self

    @property
    def column_set(self) -> ColumnSet | None:
        """The role this transform applies to, if it names a role."""
        return self.columns if isinstance(self.columns, ColumnSet) else None

    @property
    def explicit_columns(self) -> tuple[str, ...] | None:
        """The column names this transform applies to, if it names them."""
        return None if isinstance(self.columns, ColumnSet) else self.columns

    def may_overlap(self, other: ColumnTransform) -> bool:
        """Whether the two transforms could touch the same column.

        Deliberately conservative. Two explicit lists can be compared exactly,
        but a role is resolved against a schema this recipe has not met yet, so
        "unknown" answers yes: the checks built on this reject orderings that are
        wrong, and a false positive costs the caller one explicit column list.
        """
        mine, theirs = self.explicit_columns, other.explicit_columns
        if mine is not None and theirs is not None:
            return bool(set(mine) & set(theirs))
        if mine is None and theirs is None:
            return self.columns == other.columns
        return True
