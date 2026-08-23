"""One module per execution model, because that is what sktime is split by.

```text
local_models    fitted per series, one forecaster each  -> SeriesView
panel_models    fitted across every sample at once      -> SequenceView
```

sktime ships both behind one import path and one ``BaseForecaster``, and the
difference between them is a tag rather than a type: a forecaster handed a panel
is *vectorized* over its instances by default — many independent fits — unless it
pools across them, which is what makes it global. So the split here is by how a
model learns, which is the same split the Nixtla integration makes between its
two libraries and the Darts one makes between its two base classes.

Everything about persisting a fit differs with that split, and so does the view
consumed, which is why it is two modules rather than one with a flag.
:mod:`openforecast_sktime.catalog` is what makes them look the same from the
outside.
"""
