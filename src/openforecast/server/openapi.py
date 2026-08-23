"""``spec/openapi/openapi.json``, generated from the models it describes.

```bash
uv run generate-openapi
git diff --exit-code spec/openapi/openapi.json
```

Rule 7 in one command. The document is not written by hand and then kept in step
with the code; it is *derived* from the Pydantic models in
:mod:`openforecast.server.wire` and the routes in
:mod:`openforecast.server.app`, committed, and checked in CI. A change to what a
fit request means therefore shows up as a diff in the spec, and a spec edited
directly is reverted by the next generation — which is the direction the
dependency has to run for generated SDKs in other languages to be trustworthy.

The document is a pure function of the models: no provider is started, no
artifact store is read, and the version in it is the package's own. Regenerating
it on a machine with different providers installed produces the same bytes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["SPEC_PATH", "document", "main", "write"]

#: Where the committed document lives, relative to the repository root.
SPEC_PATH = Path("spec") / "openapi" / "openapi.json"


def document() -> dict[str, Any]:
    """The OpenAPI document this build's routes and models describe."""
    from openforecast.server.app import create_app

    # The transport is never called: generating a document reads route
    # signatures and model schemas, and starting an engine to do it would make
    # the spec depend on what happens to be installed.
    return create_app(transport=_Unused()).openapi()


def render(spec: dict[str, Any]) -> str:
    """The exact bytes that are committed.

    Sorted keys and a trailing newline, so that regenerating on another machine
    or another Python produces the same file and ``git diff --exit-code`` means
    what it says.
    """
    return json.dumps(spec, indent=2, sort_keys=True) + "\n"


def write(root: Path | None = None) -> Path:
    """Write the document under ``root``, and return where it went."""
    path = (Path.cwd() if root is None else root) / SPEC_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(document()), encoding="utf-8")
    return path


def main() -> int:
    """The ``generate-openapi`` console script."""
    print(write())
    return 0


class _Unused:
    """A transport that would raise if a route were executed. None ever is.

    Generating the document reads route signatures and model schemas, so the
    routes are never called. Refusing rather than answering is what keeps that
    true: a generator that started asking an engine questions would produce a
    spec that depended on what happened to be installed.
    """

    def models(self) -> Any:
        raise AssertionError(_NOT_EXECUTED)

    def model(self, ref: str) -> Any:
        raise AssertionError(_NOT_EXECUTED)

    def fit(self, body: Any) -> Any:
        raise AssertionError(_NOT_EXECUTED)

    def forecast(self, body: Any) -> Any:
        raise AssertionError(_NOT_EXECUTED)

    def artifact(self, ref: str) -> Any:
        raise AssertionError(_NOT_EXECUTED)


_NOT_EXECUTED = "generating the OpenAPI document must not execute a route"


if __name__ == "__main__":  # pragma: no cover - exercised as a console script
    raise SystemExit(main())
