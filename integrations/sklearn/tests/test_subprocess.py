"""The integration as the engine actually runs it: a child process.

```bash
python -m openforecast_sklearn
```

In production the command is the interpreter of this integration's own uv
environment; here it is the interpreter running the tests, which is the same
arrangement with one fewer moving part. What is asserted is Step 9's property
applied to the first ``TabularView`` consumer: the same model, fitted and
forecast in this process and over JSON Lines and Arrow bundles, produces the
same numbers — and the three tables of a tabular view all arrive, because a
design matrix that lost its keys is a set of rows nobody can label.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import golden
import pytest
from golden import FAST, HIST_GRADIENT_BOOSTING, at

import openforecast as of
from openforecast.errors import RecipeError
from openforecast.runtime import SubprocessProvider

COMMAND = (sys.executable, "-m", "openforecast_sklearn")
HORIZON = 3
ORIGINS = 6
INSTANCES = 2


@pytest.fixture
def remote() -> Iterator[SubprocessProvider]:
    """This integration, in a child process speaking the wire protocol."""
    provider = SubprocessProvider(COMMAND, timeout=300)
    yield provider
    provider.close()


def test_the_handshake_reports_what_the_provider_reports_in_process(
    remote: SubprocessProvider,
) -> None:
    assert remote.name == "sklearn"
    assert remote.version == golden.PROVIDER.version
    assert remote.descriptors() == golden.PROVIDER.descriptors()


def test_the_same_model_forecasts_the_same_numbers_over_a_process_boundary(
    tmp_path: Path, remote: SubprocessProvider
) -> None:
    """The seed is the fit plan's, so the same rows fit the same trees either side."""
    data = golden.point_in_time_dataset(instances=INSTANCES, origins=ORIGINS, static=True)
    origin = at(ORIGINS + 1)
    plan = of.FitPlan(seed=11)
    local = golden.client(tmp_path / "local")
    over_the_wire = golden.client(tmp_path / "remote", remote)

    here = local.forecast(
        local.fit(HIST_GRADIENT_BOOSTING, data, horizon=HORIZON, params=FAST, plan=plan),
        data.at_origin(origin),
        horizon=HORIZON,
    )
    there = over_the_wire.forecast(
        over_the_wire.fit(HIST_GRADIENT_BOOSTING, data, horizon=HORIZON, params=FAST, plan=plan),
        data.at_origin(origin),
        horizon=HORIZON,
    )

    assert there.origin_time == origin
    assert there.table.equals(here.table)


def test_a_tabular_view_survives_the_wire_with_its_keys_and_labels(
    tmp_path: Path, remote: SubprocessProvider
) -> None:
    """The bundle a tabular model needs is three tables, not one.

    ``X``, ``y`` and ``keys`` are row-aligned, and all three have to arrive: a
    design matrix without its labels is unfittable, and one without its keys is a
    pile of rows that cannot be mapped back to an instance or an origin. The keys
    never become features, which is why they can travel beside ``X`` rather than
    inside it.
    """
    data = golden.point_in_time_dataset(instances=INSTANCES, origins=ORIGINS, static=True)
    origin = at(ORIGINS + 1)
    client = golden.client(tmp_path / "remote", remote)

    handle = client.fit(HIST_GRADIENT_BOOSTING, data, horizon=HORIZON, params=FAST, name="de-price")
    forecast = client.forecast(handle, data.at_origin(origin), horizon=HORIZON)

    assert handle.training.samples == INSTANCES * ORIGINS * HORIZON
    assert handle.training.context is None
    assert forecast.origin_time == origin
    assert forecast.table.num_rows == INSTANCES * HORIZON
    assert forecast.instance_keys == (golden.ZONE,)


def test_a_failure_inside_the_provider_arrives_as_the_error_it_is(
    tmp_path: Path, remote: SubprocessProvider
) -> None:
    """An estimator rejecting its own parameters is a recipe error, wherever it ran."""
    client = golden.client(tmp_path / "store", remote)
    data = golden.point_in_time_dataset(instances=1, origins=ORIGINS)

    with pytest.raises(RecipeError, match="max_leaf_nodes"):
        client.fit(HIST_GRADIENT_BOOSTING, data, horizon=HORIZON, params={"max_leaf_nodes": 1})

    assert not list((tmp_path / "store" / "models").glob("*")), "a failed fit left an artifact"
