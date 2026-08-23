"""Step 27.3 and 27.4: one error protocol, and codes that do not move.

The table below is the contract. A code is what an agent branches on to decide
whether to install a provider, drop a feature, fit first or give up — so renaming
one is a breaking change to the protocol, and it should be as hard to do by
accident as changing a wire format. Editing this file is how you do it on
purpose.

The envelope is asserted here and re-asserted at each boundary that carries it:
``tests/unit/test_transports.py`` for HTTP, ``tests/unit/test_cli.py`` for the
CLI's ``--json``, ``tests/unit/test_subprocess_provider.py`` for a provider.
"""

from __future__ import annotations

from openforecast import errors

#: Every error OpenForecast raises deliberately, and the code it reports.
#: Frozen on purpose: a caller's recovery path depends on the right-hand column.
CODES: dict[type[errors.OpenForecastError], str] = {
    errors.OpenForecastError: "ERROR",
    errors.SchemaError: "INVALID_SCHEMA",
    errors.FrequencyError: "INVALID_FREQUENCY",
    errors.RecipeError: "INVALID_RECIPE",
    errors.InvalidModelParameters: "INVALID_MODEL_PARAMETERS",
    errors.UnsupportedPlanError: "UNSUPPORTED_PLAN",
    errors.DataError: "INVALID_DATA",
    errors.UnsupportedDataShape: "UNSUPPORTED_DATA_SHAPE",
    errors.UnsupportedFeature: "UNSUPPORTED_FEATURE",
    errors.UnsupportedOutput: "UNSUPPORTED_OUTPUT",
    errors.InconsistentTruthError: "INCONSISTENT_TRUTH",
    errors.OriginScopeError: "ORIGIN_SCOPE_ERROR",
    errors.ModelError: "MODEL_ERROR",
    errors.ModelRefError: "INVALID_MODEL_REF",
    errors.UnknownModelError: "MODEL_NOT_FOUND",
    errors.ModelRequiresFit: "MODEL_REQUIRES_FIT",
    errors.ModelDoesNotSupportFit: "MODEL_DOES_NOT_SUPPORT_FIT",
    errors.IncompatibleForecastTask: "INCOMPATIBLE_FORECAST_TASK",
    errors.DuplicateModelError: "DUPLICATE_MODEL",
    errors.ProviderError: "PROVIDER_EXECUTION_FAILED",
    errors.ProviderNotInstalled: "PROVIDER_NOT_INSTALLED",
    errors.ArtifactError: "INVALID_ARTIFACT",
}


def test_every_code_is_the_one_that_was_promised() -> None:
    """The whole point of declaring them: this test is the compatibility surface."""
    assert {kind: kind.code for kind in CODES} == CODES


def test_every_error_in_the_module_is_in_the_table() -> None:
    """A new error class arrives with a code, or it arrives with a failing test.

    Read off ``errors.__all__`` rather than off the class tree, because the
    exported names are what a caller can catch and therefore what needs a code.
    """
    exported = {getattr(errors, name) for name in errors.__all__}

    assert exported == set(CODES)


def test_codes_are_distinct_and_shaped_like_codes() -> None:
    """No two classes share one, so branching on a code is branching on a failure."""
    assert len(set(CODES.values())) == len(CODES)
    assert all(code.replace("_", "").isalpha() and code.isupper() for code in CODES.values())


def test_a_code_is_declared_rather_than_inherited() -> None:
    """A subclass that forgot its own code would silently answer its parent's.

    Which is worse than it sounds: ``UnsupportedFeature`` reporting
    ``INVALID_DATA`` would tell an agent to fix data that is not wrong.
    """
    missing = [kind.__name__ for kind in CODES if "code" not in vars(kind)]

    assert missing == [], "every error class declares its own code"


def test_the_envelope_is_the_three_fields_and_nothing_else() -> None:
    error = errors.UnsupportedFeature(
        "builtin/seasonal-naive cannot be given the features ['temp_fc']",
        model="builtin/seasonal-naive",
        features=["temp_fc"],
    )

    assert error.as_json() == {
        "code": "UNSUPPORTED_FEATURE",
        "message": "builtin/seasonal-naive cannot be given the features ['temp_fc']",
        "details": {"model": "builtin/seasonal-naive", "features": ["temp_fc"]},
    }


def test_the_message_is_what_str_says() -> None:
    """Two names for one thing, because ``str(error)`` is what a traceback prints."""
    error = errors.DataError("the history is empty")

    assert error.message == str(error) == "the history is empty"
    assert error.details == {}


def test_details_are_not_required() -> None:
    """Most failures are a sentence. An empty mapping is not a missing field."""
    assert errors.OpenForecastError().as_json() == {
        "code": "ERROR",
        "message": "",
        "details": {},
    }


def test_the_hierarchy_still_catches_what_it_used_to() -> None:
    """The finer codes of 27.4 are subclasses, so no ``except`` clause narrowed.

    A caller who wrote ``except of.DataError`` before Step 27 still catches an
    unsupported shape, an unsupported feature and an unsupported output; the code
    is what tells the three apart.
    """
    assert issubclass(errors.UnsupportedDataShape, errors.DataError)
    assert issubclass(errors.UnsupportedFeature, errors.DataError)
    assert issubclass(errors.UnsupportedOutput, errors.DataError)
    assert issubclass(errors.InvalidModelParameters, errors.RecipeError)
    assert issubclass(errors.ProviderNotInstalled, errors.ProviderError)
