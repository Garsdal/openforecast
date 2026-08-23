"""Documentation as code: the reference pages under ``docs/reference/generated``.

Step 25's rule is the one Step 16 already made for OpenAPI, applied to prose:
a signature written by hand beside the code is a signature that drifts, so the
reference half of the documentation is *derived* from the public surface —
annotations, docstrings, Pydantic models, model descriptors — committed, and
diffed in CI.

```bash
uv run generate-reference
git diff --exit-code docs/reference/generated
```

The guides, tutorials and concept pages beside it are written by hand, because
what they answer — "how do I do X?", "why does OpenForecast work this way?" —
is not in the code. What *is* in the code is never retyped next to them.
"""

from openforecast.docs.reference import REFERENCE_ROOT, main, pages, write

__all__ = ["REFERENCE_ROOT", "main", "pages", "write"]
