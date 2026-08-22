"""The integration as the engine actually runs it: a child process.

```bash
python -m openforecast_nixtla
```

In production the command is the interpreter of this integration's own uv
environment; here it is the interpreter running the tests, which is the same
arrangement with one fewer moving part. What is asserted is Step 9's property
applied to Step 11's provider: the same model, fitted and forecast in this
process and over JSON Lines and Arrow bundles, produces the same numbers.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import golden
import pytest
from golden import AUTOARIMA, at

from openforecast.errors import RecipeError
from openforecast.runtime import SubprocessProvider

COMMAND = (sys.executable, "-m", "openforecast_nixtla")
HORIZON = 3


@pytest.fixture
def remote() -> Iterator[SubprocessProvider]:
    """This integration, in a child process speaking the wire protocol."""
    provider = SubprocessProvider(COMMAND, timeout=300)
    yield provider
    provider.close()


def test_the_handshake_reports_what_the_provider_reports_in_process(
    remote: SubprocessProvider,
) -> None:
    assert remote.name == "nixtla"
    assert remote.version == golden.PROVIDER.version
    assert remote.descriptors() == golden.PROVIDER.descriptors()


def test_the_same_model_forecasts_the_same_numbers_over_a_process_boundary(
    tmp_path: Path, remote: SubprocessProvider
) -> None:
    frame = golden.event_time_frame(instances=2, periods=24)
    local = golden.client(tmp_path / "local")
    over_the_wire = golden.client(tmp_path / "remote", remote)

    here = local.forecast(local.fit(AUTOARIMA, frame, name="zones"), frame, horizon=HORIZON)
    there = over_the_wire.forecast(
        over_the_wire.fit(AUTOARIMA, frame, name="zones"), frame, horizon=HORIZON
    )

    assert there.origin_time == at(23)
    assert there.table.equals(here.table)


def test_a_failure_inside_the_provider_arrives_as_the_error_it_is(
    tmp_path: Path, remote: SubprocessProvider
) -> None:
    """A model rejecting its own parameters is a recipe error, wherever it ran."""
    client = golden.client(tmp_path / "store", remote)

    with pytest.raises(RecipeError, match="season_length"):
        client.fit(AUTOARIMA, golden.event_time_frame(periods=24), params={"season_length": 0})

    assert not list((tmp_path / "store" / "models").glob("*")), "a failed fit left an artifact"
