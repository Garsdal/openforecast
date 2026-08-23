"""One module per execution model, because that is what Darts is split by.

```text
local_models    fitted per series, one model each  -> SeriesView
global_models   fitted across every sample at once -> SequenceView
```

Darts ships both behind one import path, but they are not one thing: a
``LocalForecastingModel`` has a single series in front of it and a
``TorchForecastingModel`` has a list of windows, so the view each consumes is
different and so is everything about persisting it. The split here is by that,
not by tidiness — and it is the same split the Nixtla integration makes between
its two libraries, which is the point: the boundary is drawn by how a model
learns rather than by whose package it came from.

The :mod:`openforecast_darts.catalog` is what makes them look the same from the
outside.
"""
