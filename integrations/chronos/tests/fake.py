"""A Chronos pipeline that answers without any weights.

The checkpoint is a download, and every generated conformance case would pay for
it — so the boundary is exercised against a stand-in that answers the shape of
the question and nothing else. What that leaves testable is exactly what the
suite asserts: which view reached the provider, how many rows came back, what
they were labeled with, and which requests were refused before anything ran.

What it deliberately does not test is whether Chronos forecasts well, and no
stand-in could. ``test_chronos2.py`` runs the real checkpoint under the
``weights`` marker for that.

The numbers are arbitrary and deliberately so. A stand-in that computed
something plausible would invite an assertion about the value, and a value this
file invented is the wrong thing to assert on.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

#: How far apart the fake quantiles are placed around the level-0.5 value, so
#: that an answer is visibly ordered by level rather than constant.
SPREAD = 10.0


class FakePipeline:
    """Answers ``predict_quantiles`` in the shape ``Chronos2Pipeline`` does."""

    def __init__(self, base: float = 1.0) -> None:
        self.base = base
        self.calls: list[dict[str, Any]] = []

    def predict_quantiles(
        self,
        *,
        inputs: Sequence[Mapping[str, Any]],
        prediction_length: int,
        quantile_levels: Sequence[float],
    ) -> tuple[list[Any], list[Any]]:
        """One ``(1, horizon, levels)`` tensor per input, plus the medians.

        Recorded as well as answered: what a covariate-carrying view turns into
        is the claim the conversion module makes, and a test that could only see
        the answer could not check it.
        """
        self.calls.append(
            {
                "inputs": list(inputs),
                "prediction_length": prediction_length,
                "quantile_levels": list(quantile_levels),
            }
        )
        quantiles: list[Any] = []
        medians: list[Any] = []
        for position in range(len(inputs)):
            value = self.base + position
            rows = [
                [value + (level - 0.5) * SPREAD for level in quantile_levels]
                for _ in range(prediction_length)
            ]
            quantiles.append(np.asarray([rows], dtype=float))
            medians.append(np.asarray([[value] * prediction_length], dtype=float))
        return quantiles, medians
