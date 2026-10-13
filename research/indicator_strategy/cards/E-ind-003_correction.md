---
id: E-ind-003-correction
mechanism: breakout; trend-following
timeframe: 1d
target: pnl
model_class: rule
reopens: F-0054:R1
not_covered_by: F-0072=volume-confirmed Donchian on unseen assets, a different trigger and filter from this card; F-0070=cross-sectional momentum rotation of E-disc-004, a portfolio with turnover costs, not a single-asset breakout trade; F-0071=Bollinger band breakout with ADX gate on unseen assets, a different trigger from the Donchian channel of this card; F-0069=different scope, F-0069 removes ADX > 40 entries and tests the combined arm, E-ind-003 tests only the long-only daily Donchian with no ADX cut; F-0055=maker limit entry at EMA20 with scaling on 1h, this card is a daily Donchian breakout with market fills and a fixed ATR exit; F-0061=stop orders at PREDICTED candle extremes, this card uses an indicator trigger on the close; F-0062=regime filters on wick-capture fills, no regime filter here; F-0064=E-hedge-001 hedge-and-hold, no hedging here; F-0065=E-hedge-002 reversal hedge, no hedging here; F-0076=corrected copy of a bullet in E-ind-003, no new run and no new test - the original scope is the seven original coins; F-0077=corrected copy of a bullet in E-ind-003, no new run and no new test - the original scope is the seven original coins
---
# E-ind-003 correction: the year-by-year "shrinking edge" reading is retracted

Corrected copy of one bullet in the Result section of `cards/E-ind-003.md`. The original is kept; one marker line was added there.

## Original statement (kept in the original card)
"By year: 2024 +0.41R (n=96), 2025 +0.26R (n=57), 2026 +0.10R (n=52). The edge is shrinking, and the last year is weakest."

## Correction
The year means are as recorded, but the statement that the edge is shrinking is **not supported**: the differences are not
statistically significant. Recomputed from `E-ind-003_test_chosen_trades_annotated.csv` (205 trades, `net_r` by `year`):

| Year | n | Mean net R | SD |
|---|---|---|---|
| 2024 | 96 | +0.410 | 1.223 |
| 2025 | 57 | +0.264 | 1.219 |
| 2026 | 52 | +0.097 | 1.116 |

- Welch t-test, 2024 vs 2026: t = 1.58, p = 0.118 (reported as p = 0.12 in the README correction, commit `68d2cef`).
- Bootstrap 95 % interval of the 2026 mean (10 000 resamples, seed 0): [-0.21, +0.40].
- One-way ANOVA across the three years: F = 1.18, p = 0.31.

A difference of this size is compatible with sampling noise at these trade counts. The accurate wording is "the yearly means
decline from +0.41R to +0.10R, but the decline is not distinguishable from noise (p = 0.12)". No claim of a trend should be made.

## Status of the whole card
This correction does not change the verdict (accepted as a candidate for confirmation) and does not strengthen it. The card was
committed together with its results (`6b1acf7`), so its TEST run is exploration, not a verifiable single-shot confirmation; see
`research/strategy_discovery/PREREGISTRATION_NOTE.md`. The README of this study already carries the corrected wording.
