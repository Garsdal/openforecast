# OpenForecast

The unified interface for forecasting.

OpenForecast is a framework-agnostic forecasting API. You describe your data and
the model you want with OpenForecast's own vocabulary, and the library compiles
that into whatever the underlying forecasting library expects — statistical,
neural, or otherwise — behind one stable surface:

```python
import openforecast as of

model = of.fit(model="nixtla/nhits", data=train, horizon=24)
forecast = of.forecast(model=model, data=context, horizon=24)
```

> **Status: early development.** This repository currently contains the
> foundation from [Plan 1](PLAN.md) — packaging, layering rules, tooling and
> tests. The only public symbol is `openforecast.__version__`; the semantic
> types shown above land in later stages. See [PLAN.md](PLAN.md) for the full
> roadmap.

## The core architectural rule

> `openforecast` has no dependency on Nixtla, Darts, sktime, PyTorch or any
> other forecasting framework.

Integrations depend on OpenForecast, never the reverse. Each external provider
lives in its own distribution under `integrations/` with its own lockfile and
its own virtual environment, so providers with incompatible dependency graphs
(Torch vs. JAX, say) can coexist without ever meeting. The core install stays
small: `pydantic`, `pyarrow`, `platformdirs`.

This rule is enforced by tests, not by convention — `tests/unit/test_architecture.py`
scans the package for forbidden imports and rejects the declared dependencies of
any forecasting framework.

## Layering

Imports flow in one direction only:

```text
data/  models/  recipes/  tasks/
                ↓
runtime/  registry/  artifacts/
                ↓
            client.py
```

`protocol/` sits underneath everything and knows nothing about any provider.
The same architecture test enforces this direction.

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
uv sync                 # create .venv and install core + dev dependencies
uv run ruff check .     # lint
uv run ruff format .    # format
uv run pyright          # type check
uv run pytest           # test
```

Arrow is the canonical data-plane representation, so anything crossing a
process or language boundary is Arrow IPC rather than JSON.

## Repository layout

```text
src/openforecast/    the framework-agnostic core (this is what users install)
integrations/        provider distributions, each independently versioned
tests/unit/          unit and architecture tests
tests/contract/      canonical protocol fixtures
tests/conformance/   the reusable provider capability suite
tests/e2e/           end-to-end acceptance tests
spec/                protocol, Arrow and OpenAPI specifications
```

Directories that later stages fill are present but empty.
