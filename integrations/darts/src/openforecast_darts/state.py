"""What an adapter has to remember about a fit, beside the native model itself.

```text
into/state.json    which model, which columns, which windows
into/<native>      whatever Darts' own save produced
```

Darts persists its weights or its fitted coefficients and nothing else.
Everything that made the fit an *OpenForecast* fit — which caller column was the
target, which features went in as which kind of covariate, which instance each
of the fitted series belongs to — is OpenForecast's to write down, because the
forecast side has to reconstruct exactly that mapping to label its answer.

JSON rather than a pickle: an artifact outlives the process that wrote it, and a
fitted state that can only be read back by the same library version is a
migration problem waiting for the first upgrade.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from openforecast.errors import ProviderError

__all__ = ["STATE_FILENAME", "read_state", "write_state"]

#: Everything the forecast side needs that the native save does not hold.
STATE_FILENAME = "state.json"


def write_state(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        encoded = json.dumps(payload, indent=2)
    except TypeError as error:  # an instance key no JSON document can hold
        raise ProviderError(f"this instance key cannot be persisted: {error}") from error
    path.write_text(encoded + "\n", encoding="utf-8")


def read_state(path: Path, model: str) -> Mapping[str, Any]:
    """The state ``model`` wrote at fit time, or a refusal saying what is there."""
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
