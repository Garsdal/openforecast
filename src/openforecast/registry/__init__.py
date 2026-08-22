"""Model and provider registries.

Resolves a ``ModelRef`` string to either a provider-advertised model descriptor
or a locally fitted artifact:

```python
registry = ModelRegistry()

registry.for_fit("nixtla/nhits")     # ModelDescriptor  — plan a fit against this
registry.resolve("local/de-price")   # ModelHandle      — forecast with this
registry.resolve("nixtla/nhits")     # ModelRequiresFit — that is a model, not a fitted one
```

The catalog of Step 5 and the artifact store of Step 7 each answer half of "what
does this string mean", and this is where the halves meet, so that the engine of
Step 8 asks once. Provider environments join it in Step 9.
"""

from openforecast.registry.models import ModelRegistry, Resolution

__all__ = ["ModelRegistry", "Resolution"]
