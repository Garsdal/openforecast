"""Provider-neutral execution views (Step 4).

``SeriesView``, ``WindowView``, ``TabularView`` and ``ForecastView`` are the
only data representations that cross the provider boundary. The ``ViewPlanner``
materializes them from semantic source datasets so that no provider ever has to
branch on whether the data originated as event-time or point-in-time.
"""
