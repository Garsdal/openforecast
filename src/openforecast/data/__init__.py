"""Semantic source datasets.

Event-time primitives (``Frequency``, ``TimeSeriesSchema``, ``TimeSeriesFrame``)
arrive in Step 2; the point-in-time primitives (``PointInTimeFrame``,
``ForecastDataset``, ``ForecastContext``) arrive in Step 3.

Nothing here may import :mod:`openforecast.views` — views are materialized
*from* semantic datasets, never the other way around.
"""
