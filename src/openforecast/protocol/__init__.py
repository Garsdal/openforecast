"""The provider wire protocol.

Request/response messages and the error envelope arrive with the provider
subprocess transport in Step 9. This is the innermost layer: it may not import
any other OpenForecast subpackage, and it knows nothing about any specific
provider.

What already lives here is what has to be spelled the same way on both sides of
a boundary. ``ViewKind`` is named by a model's training contract in ``models/``
and by the execution views in ``views/``, and those two layers cannot import
each other. ``PROTOCOL_VERSION`` is stamped into an artifact manifest in Step 7
and negotiated over the handshake in Step 9, and those two have to be one
number.
"""

from openforecast.protocol.version import PROTOCOL_VERSION
from openforecast.protocol.vocabulary import ViewKind

__all__ = ["PROTOCOL_VERSION", "ViewKind"]
