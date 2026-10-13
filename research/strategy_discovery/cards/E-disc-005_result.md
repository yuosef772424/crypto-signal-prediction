---
id: E-disc-005-result
mechanism: trend-following; breakout; volatility-structure; volume-flow
timeframe: 4h;1d
target: pnl
model_class: rule
reopens: F-0054:R1
not_covered_by: F-0074=this card is a source of the 4h non-survivors that F-0074 registers (its own 4h results); F-0072=the campaign card that selected this family, its unseen-asset failure is recorded from E-disc-006; F-0053=previous-candle range fade with limit orders, not a band or channel trigger; F-0061=stop orders at predicted candle extremes, this card uses indicator triggers on closes; F-0062=regime filters on wick-capture fills, the gates here are entry conditions on trend or volatility triggers; F-0063=chart-image CNN, no image model here; F-0068=4h breakout and pullback families, re-run only as part of the multiple-testing family, with the same gates reported separately; F-0069=daily ADX>40 and short-side refinements, a different test; F-0070=cross-sectional momentum rotation, a portfolio with turnover costs, not a single-asset trade; F-0071=Bollinger breakout with ADX gate on unseen assets, a different trigger; F-0076=result copy of E-disc-005 which includes S1 in its 60-configuration correction, no new run; F-0077=result copy of E-disc-005 which includes S3 in its 60-configuration correction, no new run
---
# E-disc-005 result (corrected copy of the empty Result section)

The Result section of `cards/E-disc-005.md` was left empty when the card was committed. This file adds the result, taken
mechanically from `results/E-disc-005_summary_campaign.csv` (columns `n`, `mean_net_r`, `pf`, `t_stat`, `coins_positive`,
`bh_q_campaign`; no number here was recomputed). The original card is not edited except for one marker line.

Run: `research/strategy_discovery/grid2.py` (DEV 2019-2021 and VAL 2022-2023, seven original coins, 4h and 1d).
Pre-registration caveat: this card and its results were committed together (`66e02d5`); see `PREREGISTRATION_NOTE.md`.
This is exploration, not confirmation.

## The 18 new configurations (3 families x 3 gates x 2 timeframes)
The BH q-value is computed over all 60 configurations of the campaign (42 of E-disc-001 and these 18).
Survivor rule from the card: DEV and VAL pooled mean net R > 0; BH q (VAL) <= 0.10; VAL pf >= 1.10; VAL net R positive on at least 5 of 7 coins.

| Configuration | n (DEV / VAL) | DEV net R | VAL net R | VAL pf | VAL t | BH q | VAL coins+ | Survivor |
|---|---|---|---|---|---|---|---|---|
| 4h supertrend, none | 812 / 656 | +0.038 | -0.025 | 0.95 | -0.56 | 0.854 | 3/7 | no |
| 4h supertrend, adx25 | 295 / 225 | -0.086 | -0.062 | 0.88 | -0.85 | 0.888 | 1/7 | no |
| 4h supertrend, ema200_aligned | 441 / 363 | +0.218 | -0.053 | 0.91 | -0.89 | 0.888 | 3/7 | no |
| 4h keltner_squeeze, none | 486 / 343 | +0.096 | -0.049 | 0.92 | -0.77 | 0.888 | 3/7 | no |
| 4h keltner_squeeze, adx25 | 156 / 101 | +0.047 | +0.042 | 1.08 | 0.35 | 0.604 | 4/7 | no |
| 4h keltner_squeeze, ema200_aligned | 371 / 252 | +0.152 | -0.031 | 0.95 | -0.42 | 0.854 | 2/7 | no |
| 4h volume_donchian, none | 1596 / 1216 | +0.142 | +0.074 | 1.14 | 2.20 | 0.117 | 5/7 | no |
| 4h volume_donchian, adx25 | 912 / 679 | +0.120 | +0.090 | 1.17 | 1.99 | 0.117 | 4/7 | no |
| 4h volume_donchian, ema200_aligned | 1318 / 984 | +0.198 | +0.086 | 1.17 | 2.31 | 0.106 | 6/7 | no |
| 1d supertrend, none | 125 / 105 | +0.212 | +0.224 | 1.53 | 2.02 | 0.117 | 4/7 | no |
| 1d supertrend, adx25 | 54 / 33 | -0.068 | +0.013 | 1.03 | 0.07 | 0.711 | 4/7 | no |
| 1d supertrend, ema200_aligned | 57 / 55 | +0.522 | +0.193 | 1.49 | 1.34 | 0.227 | 5/7 | no |
| 1d keltner_squeeze, none | 71 / 57 | +0.192 | +0.180 | 1.38 | 1.16 | 0.273 | 4/7 | no |
| 1d keltner_squeeze, adx25 | 36 / 22 | +0.093 | +0.173 | 1.36 | 0.67 | 0.484 | 4/7 | no |
| 1d keltner_squeeze, ema200_aligned | 49 / 43 | +0.440 | +0.202 | 1.45 | 1.14 | 0.273 | 4/7 | no |
| 1d volume_donchian, none | 269 / 181 | +0.232 | +0.220 | 1.49 | 2.52 | 0.070 | 5/7 | yes |
| 1d volume_donchian, adx25 | 185 / 100 | +0.186 | +0.312 | 1.78 | 2.70 | 0.070 | 4/7 | no |
| 1d volume_donchian, ema200_aligned | 219 / 137 | +0.276 | +0.187 | 1.42 | 1.91 | 0.119 | 5/7 | no |

## Reading
- Survivors among the new 18: **1** (`1d volume_donchian`, no gate): DEV +0.232, VAL +0.220, VAL pf 1.49, VAL t 2.52, q 0.070, 5 of 7 coins.
- Near misses by one condition each: `1d volume_donchian + adx25` (q 0.070, pf 1.78, but only 4 of 7 coins positive); `4h volume_donchian + ema200_aligned` (q 0.106 > 0.10); `1d supertrend` no gate (q 0.117).
- Supertrend: on 1d without a gate it is positive on DEV and VAL (+0.212 / +0.224) with fewer trades than Donchian, and fails the q rule (0.117); on 4h it is negative on VAL in all three gates (-0.025, -0.062, -0.053).
- Keltner squeeze: no survivor (4h VAL negative in two of three gates; 1d q >= 0.27).
- Under the same 60-configuration correction the three E-disc-001 survivors keep q between 0.062 and 0.070 (`1d donchian20` no gate 0.070, `1d donchian20 + adx25` 0.062, `1d bb_break + adx25` 0.070).
- Volume vs plain Donchian (card prediction): on 1d DEV, volume_donchian no gate is +0.232 against +0.252 for donchian20 no gate, so the volume gate did not improve DEV. The prediction that it would beat the plain Donchian on DEV is not supported.

Verdict: **screen complete; `1d|volume_donchian|none` goes to E-disc-006** (which then failed it on unseen assets, F-0072). It is not a finding.
