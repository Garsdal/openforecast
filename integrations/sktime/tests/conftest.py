"""The OpenForecast checkout's conformance suite, made importable.

``tests/conformance/suite.py`` is the part integrations inherit: it generates a
provider's tests from what its descriptors declare, so that a capability is
never advertised without being exercised. It ships with the OpenForecast
repository rather than with the distribution, so these tests reach for the
checkout this integration is developed against — the same checkout
``[tool.uv.sources]`` installs ``openforecast`` from.

If it is not there, the conformance tests skip and the rest still run: a
published install has no checkout beside it, and that is not a broken test run.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: ``integrations/sktime/tests`` -> the repository root.
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

CONFORMANCE_SUITE = REPOSITORY_ROOT / "tests" / "conformance" / "suite.py"


def pytest_configure() -> None:
    if CONFORMANCE_SUITE.is_file() and str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
