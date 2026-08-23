# Installation

OpenForecast needs Python 3.11 or newer.

```bash
pip install openforecast
```

That is the whole install for using the library and for *calling* a remote
service: the core dependencies are `pydantic`, `pyarrow` and `platformdirs`, and
`HttpTransport` is `urllib`. No forecasting framework is pulled in, and CI greps
the dependency tree to keep it that way.

```bash
pip install 'openforecast[server]'    # to run a service with `openforecast serve`
```

Check the install by asking what this build can run:

```python
import openforecast as of

of.__version__
of.models.list()
```

A fresh install answers with one model, `builtin/seasonal-naive` — the reference
provider. It is a real model with a real contract, which is what lets the whole
workflow be run, and taught, before any forecasting library is installed.

## Provider environments

An integration is *not* installed into your OpenForecast environment. Nixtla
wants one version of `torch`, Darts wants another, and OpenForecast wants
neither, so each provider gets its own environment built with
[uv](https://docs.astral.sh/uv/) and is reached over a subprocess protocol:

```bash
openforecast providers install nixtla
openforecast providers install darts
openforecast providers install sktime
openforecast providers install sklearn
openforecast providers install amazon     # Chronos-2

openforecast providers list
openforecast providers inspect nixtla
openforecast providers remove nixtla
```

Installing one publishes an environment only once the provider inside it has
answered a handshake, and records what it said:

```text
~/.cache/openforecast/providers/
    nixtla/
        0.1.0/
            environment.json     what the provider said when it was installed
            .venv/
```

That recorded JSON is what `of.models.list()` reads, so discovery starts no
process. A process starts when a model is actually fitted or forecast with — and
the handshake is repeated then, so an environment whose contents changed
underneath its record is refused rather than executed as something it no longer
is. See [Providers](../concepts/providers.md) for what happens across that
boundary.

## Where things are stored

```text
~/.local/share/openforecast/models/     fitted artifacts, one directory per revision
~/.local/share/openforecast/aliases/    which revision a name follows
~/.cache/openforecast/providers/        provider environments
```

The exact paths follow the platform's conventions. A client can be pointed
somewhere else, which is what tests and containers do:

```python
client = of.OpenForecast(store="/srv/openforecast/models")
```

## Working on OpenForecast itself

```bash
git clone https://github.com/Garsdal/openforecast
cd openforecast

uv sync                    # core + dev dependencies, into .venv
uv run pytest              # unit, conformance and e2e suites
uv run ruff check .         # lint
uv run ruff format .        # format
uv run pyright             # type check
```

```bash
uv run generate-openapi    # regenerate spec/openapi/openapi.json
uv run generate-reference  # regenerate docs/reference/generated
uv sync --group docs && uv run mkdocs serve   # this site, locally
```

Both generators are diffed in CI, so a change to what a fit request means, or to
a docstring on the public surface, shows up as a diff rather than as
documentation that quietly disagrees with the code.
