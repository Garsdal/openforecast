"""The provider wire protocol.

Protocol version, request/response messages and the error envelope arrive with
the provider subprocess transport in Step 9. This is the innermost layer: it may
not import any other OpenForecast subpackage, and it knows nothing about any
specific provider.

What already lives here is the vocabulary that has to be spelled the same way on
both sides of a boundary. ``ViewKind`` is named by a model's training contract
in ``models/`` and by the execution views in ``views/``, and those two layers
cannot import each other.
"""

from openforecast.protocol.vocabulary import ViewKind

__all__ = ["ViewKind"]
