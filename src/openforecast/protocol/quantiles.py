"""What "the 0.9 quantile of these draws" means, defined once.

Two places in OpenForecast reduce sample paths to a quantile: the engine, when a
model that draws samples was asked for quantiles, and a probabilistic metric,
when it scores a sample forecast. If the two disagreed about the estimator, a
backtest would score a number the forecast it came from never held — so the
definition lives here, in the layer both of them can name.

The estimator is linear interpolation between the order statistics, which is
what ``numpy.quantile`` and every library built on it means by a quantile:

```text
draws  [70, 80, 90, 100]     level 0.9

position   (4 - 1) * 0.9 = 2.7
value      90 + 0.7 * (100 - 90) = 97
```

Interpolating *between draws* is not the same thing as interpolating between
requested quantile levels, which OpenForecast refuses: the draws are what the
model actually said about the distribution, and a level between two of them is
still a reading of that distribution rather than an invented one.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from openforecast.errors import DataError

__all__ = ["quantile_of_samples"]


def quantile_of_samples(draws: Sequence[float], level: float) -> float:
    """The ``level`` quantile of ``draws``, by linear interpolation.

    Raised on rather than defaulted for an empty set of draws: the quantile of
    nothing is not a zero, and a caller that has no draws has no distribution to
    read a quantile out of.
    """
    if not draws:
        raise DataError(f"the quantile {level} of no draws is not a number")
    if not 0.0 < level < 1.0:
        raise DataError(
            f"a quantile level lies strictly between 0 and 1; got {level}. "
            f"0 and 1 name the bounds of the distribution, not quantiles of it"
        )
    ordered = sorted(draws)
    position = (len(ordered) - 1) * level
    below, above = math.floor(position), math.ceil(position)
    if below == above:
        return ordered[below]
    return ordered[below] + (position - below) * (ordered[above] - ordered[below])
