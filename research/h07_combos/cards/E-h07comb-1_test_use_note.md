---
id: E-h07comb-1-test-use-note
mechanism: trend-following
timeframe: 1d
target: pnl
model_class: rule
reopens:
not_covered_by: F-0079=this card is the experiment that produced F-0079 (its own result, not a different scope); F-0070=cross-sectional momentum rotation of 7 coins with turnover costs, not the single-coin 30d trend filter of H07 and its extensions; F-0072=volume-confirmed Donchian breakout with a fixed ATR stop on unseen assets, a different trigger and exit; F-0076=Donchian 20 breakout with ADX gate on unseen assets BCH TRX ZEC, a different rule (H07 is a 30d-return sign filter with volatility sizing and no stop); F-0077=Donchian 20 breakout without gate on unseen assets, a different rule from the H07 trend filter and its extensions
---
# E-h07comb-1 — TEST-use note (added 2026-10-13, review defect D1)

Companion to `E-h07comb-1.md`. The original card and its result section are unchanged; this note records where the run
departed from the card's own TEST rule and what that means for how the result may be used.

## What the card said
"TEST (once, accepted variants only): ΔSharpe > 0 and vol-matched max drawdown not worse." That is, TEST 2024-01-01 ..
2026-09-30 is opened once, and only for variants that already passed the VAL rule.

## What the run did
- `01_combos.py` loops over `(('val', VAL), ('test', TEST))` for every variant (lines ~112 onward), so TEST Sharpe and
  max drawdown were computed in the same run for all five series: V0, V1, V2, V3 and V4.
- No variant passed the VAL rule (every ΔSharpe interval at 98.75% includes 0), so under the card TEST should not have
  been opened at all. The result section of the card says TEST is "shown for transparency only", and that is true of
  the verdict (rejected, failure level `power`); it is not true of the next decision.
- The choice of V3 and V4 as shadow variants for the forward ledger was made partly on the argument "higher Sharpe and
  smaller raw drawdown in VAL and in TEST" (card result section; commit message of the run). That argument uses TEST, so
  TEST was used for selection, not only reported.

## Consequence
- TEST numbers for the five series (V0 0.89, V1 0.94, V2 0.86, V3 1.23, V4 1.44 Sharpe) are NOT an out-of-sample
  confirmation of V3 or V4. They were seen before the shadow choice, and V3/V4 were the two variants that looked best
  in them.
- V3 and V4 are therefore judged ONLY on the forward ledger (`research/h07_forward/ledger_v3_10coins.csv`,
  `ledger_v4_ens_10coins.csv`, bars from 2026-10-01), against the frozen H07 ledger, under the rule in
  `research/h07_forward/cards/E-h07fwd-1.md`. VAL and TEST are not pooled into that judgement.
- The registry row (F-0079 on this branch) is unaffected: it records a rejection on VAL and does not rely on TEST.
