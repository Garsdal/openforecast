"""One module per Nixtla library, because they are different execution models.

```text
statsforecast   local statistical models, fitted per series  -> SeriesView
neuralforecast  global neural models, fitted across series   -> SequenceView
```

The split is by how a library trains, not by tidiness: a StatsForecast model has
one series in front of it and a NeuralForecast model has many windows, so the
view each consumes is different and so is everything about persisting it. The
:mod:`openforecast_nixtla.catalog` is what makes them look the same from the
outside.
"""
