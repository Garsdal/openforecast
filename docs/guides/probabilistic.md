# Probabilistic forecasts

What kind of answer to produce is a request; what a model can answer is a
declaration. Both are explicit, and neither is inferred.

```python
import openforecast as of

of.OutputSpec.point()
of.OutputSpec.quantiles([0.1, 0.5, 0.9])
of.OutputSpec.samples(200)
```

A model declares the same three:

```yaml
outputs:
  point: true
  quantiles: true
  samples: false
```

A request the model does not declare is refused before the provider is started,
which is the rule every other capability follows. `builtin/seasonal-naive`
declares point output only, so the examples below name a provider model and are
not executed by the test suite; install one with
`openforecast providers install nixtla`.

## Asking for quantiles

<!-- docs-exec: skip — needs `openforecast providers install nixtla` -->

```python
client = of.OpenForecast()

model = client.fit("nixtla/autoarima", data=data, params={"season_length": 24})

forecast = client.forecast(
    model,
    data=data,
    horizon=72,
    output=of.OutputSpec.quantiles([0.1, 0.5, 0.9]),
)

forecast.quantile(0.9).to_pandas()
forecast.to_wide()      # zone, event_time, price_q0.1, price_q0.5, price_q0.9
```

There are no separate result classes for the three forms. `AutoARIMA`'s
prediction intervals, a neural model's sample paths and a naive point forecast all
arrive as one `Forecast` over one long table, so code downstream of it does not
learn which provider answered or which of the three forms that provider is native
in.

## The one conversion

```text
samples   -> quantiles     the draws are the distribution; read it
quantiles -> samples       refused: the paths would have to be invented
point     -> anything      refused: there is no distribution to read
```

And it is asked for rather than assumed, because how many draws a quantile was
estimated from is part of what it is:

<!-- docs-exec: skip — needs a provider whose native output is sample paths -->

```python
forecast = client.forecast(
    model,
    data=data,
    horizon=72,
    output=of.OutputSpec.quantiles([0.1, 0.9], from_samples=200),
)
```

The reduction happens in OpenForecast with one estimator, which is what makes two
providers' quantiles comparable rather than each library's own convention. A
deterministic model is never dressed up as a probabilistic one: a calibration
layer that turns point forecasts into distributions is something a caller can ask
for explicitly, and never something a request quietly triggers.

## Scoring a distribution

The same backtest call with the output it needs, and metrics that read it:

<!-- docs-exec: skip — needs `openforecast providers install nixtla` -->

```python
result = client.backtest(
    ["nixtla/autoarima"],
    data=data,
    validation=of.RollingOrigin(horizon=72, windows=5),
    output=of.OutputSpec.quantiles([0.1, 0.5, 0.9]),
    metrics=[of.MAE(), of.PinballLoss(0.9), of.Coverage(), of.IntervalWidth()],
)

result.leaderboard("pinball[0.9]").to_pandas()
```

| Metric | Asks |
| --- | --- |
| `of.PinballLoss(0.9)` | the loss the 0.9 quantile is the optimal answer to |
| `of.Coverage()` | how often the outcome fell inside the 0.1 to 0.9 interval |
| `of.IntervalWidth()` | how wide that interval was |
| `of.MAE()` | the error of the median, of quantiles or of draws alike |

`Coverage` and `IntervalWidth` are the calibration and sharpness halves of one
question, which is why the first is best *at* its nominal level rather than
highest, and why the second is only readable beside it. A pinball loss reads the
0.9 of a provider's native quantiles and of another provider's sample draws
identically; neither reading invents anything the model did not say.

The metrics are checked against the requested output before the first fit, so a
coverage of a point forecast is refused in the first line of the run rather than
after an hour — and a metric's name carries its parameter, `pinball[0.9]`,
`coverage[0.8]`, because two of them in one backtest are two rows of one table.
