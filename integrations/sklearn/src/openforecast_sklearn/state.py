"""What a fit leaves behind: the estimator, and what labels its columns.

```text
into/estimator.pkl   the fitted scikit-learn estimator, as scikit-learn saves it
into/metadata.json   which model, which columns, in which order
```

scikit-learn's own persistence is a pickle, and its documentation is explicit
that a pickle is only guaranteed to load in the environment that wrote it. That
is acceptable here and nowhere else: the pickle lives *inside the provider's own
directory* of an artifact, next to the environment record that says which version
of this distribution and which version of scikit-learn produced it. Nothing
outside the provider boundary reads it.

Everything that made the fit an *OpenForecast* fit — which caller column was the
target, which columns became the design matrix and in what order, which of them
were known features and which were static — is OpenForecast's to write down, in
JSON, because the forecast side has to rebuild exactly that matrix from a
different view. A column order recovered from a pickle would be a column order
only the pickle can explain.

The order is the part worth stating twice. A fitted estimator has no column
names: it has ``n_features_in_`` positions. So ``features`` is not a set, it is
the contract between the two calls, and a forecast that assembled the same
columns in a different order would predict confidently from a matrix where the
wind forecast is read as a capacity.
"""

from __future__ import annotations

import json
import pickle  # noqa: S403 - the fitted estimator, inside the provider's own directory
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from openforecast.errors import ProviderError

__all__ = [
    "ESTIMATOR_FILENAME",
    "METADATA_FILENAME",
    "read_estimator",
    "read_metadata",
    "write_estimator",
    "write_metadata",
]

#: The fitted estimator, as scikit-learn hands it over.
ESTIMATOR_FILENAME = "estimator.pkl"

#: Everything the forecast side needs that the estimator does not hold.
METADATA_FILENAME = "metadata.json"


def write_metadata(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        encoded = json.dumps(payload, indent=2)
    except TypeError as error:  # a column name no JSON document can hold
        raise ProviderError(f"this fit cannot be described in JSON: {error}") from error
    path.write_text(encoded + "\n", encoding="utf-8")


def read_metadata(path: Path, model: str) -> Mapping[str, Any]:
    """The metadata ``model`` wrote at fit time, or a refusal saying what is there."""
    try:
        persisted: Any = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ProviderError(f"{model} has no fitted state at {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ProviderError(
            f"the fitted state of {model} at {path} is not valid JSON: {error}"
        ) from error
    if not isinstance(persisted, dict) or persisted.get("model") != model:
        raise ProviderError(f"{path} does not hold the fitted state of {model}")
    return persisted


def write_estimator(path: Path, estimator: object) -> None:
    try:
        path.write_bytes(pickle.dumps(estimator, protocol=pickle.HIGHEST_PROTOCOL))
    except (OSError, pickle.PicklingError) as error:
        raise ProviderError(f"the fitted estimator could not be written to {path}: {error}") from (
            error
        )


def read_estimator(path: Path, model: str) -> Any:
    """The estimator a previous fit wrote, or a refusal saying what is there.

    A pickle that will not load is the one failure mode this persistence has, and
    it is reported as what it is: an artifact written by an environment this one
    cannot read, rather than an exception from inside ``pickle``.
    """
    try:
        return pickle.loads(path.read_bytes())  # noqa: S301 - written by this provider, in-artifact
    except OSError as error:
        raise ProviderError(f"{model} has no fitted estimator at {path}: {error}") from error
    except Exception as error:
        raise ProviderError(
            f"the fitted estimator of {model} at {path} could not be loaded: "
            f"{type(error).__name__}: {error}; it was written by another environment"
        ) from error
