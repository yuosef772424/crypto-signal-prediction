---
id: E-disc-006
mechanism: breakout; volume-flow; trend-following
timeframe: 1d
target: pnl
model_class: rule
reopens: F-0054:R1
not_covered_by: F-0072=this card is the source of F-0072 (its own result, not a different scope); F-0069=daily ADX>40 and short-side refinements of Bollinger and Donchian, a different test; F-0070=cross-sectional momentum rotation, a portfolio with turnover costs; F-0071=Bollinger breakout with ADX gate on unseen assets, a different trigger and gate; F-0061=stop orders at predicted candle extremes, no predicted level here
---
# Experiment card — E-disc-006: robustness, unseen assets and fresh bars for the new survivor `1d|volume_donchian|none`

Written and committed BEFORE the run (rules: `docs/research/RESEARCH_RULES.md`, `docs/research/MODEL_DEV_PROTOCOL.md`).

## The configuration (fixed)
- Signal: close above the prior 20-bar high (long) or below the prior 20-bar low (short), AND the bar's volume > 1.5 × the median
  volume of the prior 20 bars. Gate: none.
- Exit: stop 2 × ATR(14), target 3 × ATR(14), time exit 30 bars, cost 0.12% RT, entry at the next open.
- Selected by E-disc-005 with campaign-wide Benjamini-Hochberg over 60 configurations (VAL q = 0.070, DEV and VAL both positive).

## Honesty about the split
This family's VAL result is its first use: no earlier card looked at `volume_donchian` on VAL. Its VAL result is therefore a
first-time test, not a confirmation of something already seen. It is still the same kind of rule as the Donchian survivors, so
its evidence is correlated with theirs.

## Tests (the same rules as E-disc-002 and E-disc-003, written before the run)
**R. Robustness on DEV and VAL (seven original coins):**
1. cost × 2: pooled mean net R > 0 on DEV AND on VAL;
2. entry delayed one bar: pooled mean net R > 0 on VAL;
3. 36-setting exit grid (stop ∈ {1.5, 2, 2.5, 3}, target ∈ {2.5, 3, 3.5}, time ∈ {20, 30, 40}): at least 70% positive on VAL;
4. VAL mean net R positive in 2022 AND in 2023.

**U. Unseen assets BCH, TRX, ZEC, 2024-01-01 → 2026-09-30 (consumed period, unseen assets):** PASS only if pooled mean net R > 0,
profit factor > 1.0, and at least 2 of the 3 coins have a positive mean.

**F. Fresh bars 2026-10-01 → 2026-10-08 (all ten coins):** a verdict only with at least 10 trades; otherwise "insufficient sample".

A configuration that passes R and U is a candidate for the next data drop. It is not a verified strategy.

## Predicted diagnostic movement
- Cost × 2 should cost about 0.01R per trade on 1d.
- A one-bar delay may cut the mean; a large cut would point to fill sensitivity (as with S3 in E-disc-002).
- The volume threshold should lower the trade count against the plain Donchian (E-disc-001: 359 DEV trades for `donchian20|none`,
  269 for `volume_donchian|none`).

---
## Result (added after the run)
Run: `research/strategy_discovery/survivor_checks.py --family volume_donchian --gate none` → `results/E-disc-006_volume_donchian_none.csv`.

| Test | Result | Verdict |
|---|---|---|
| R: cost × 2 (DEV / VAL) | +0.222 / +0.208 | pass |
| R: one-bar delay (VAL) | +0.148 | pass |
| R: VAL exit-grid share positive | 100% of 36 | pass |
| R: VAL 2022 / 2023 | +0.050 / +0.351 | pass |
| **U: unseen assets, 2024 → 2026-09** | n = 112, mean +0.104, pf 1.20, **1 of 3 coins positive** (BCH −0.045, TRX −0.109, ZEC +0.360) | **FAIL** (rule needs at least 2 of 3) |
| F: fresh bars 2026-10 | n = 1 | insufficient sample |

Verdict: **closed for this grid.** The configuration is robust on the seven original coins, but its result on the unseen assets is carried by ZEC alone. Failure row F-0072 is recorded.

Note: the same pattern appears for the two Donchian survivors in E-disc-003 (their unseen-asset pass also depends on ZEC and TRX, with BCH negative). This is the main limitation of the whole campaign: the positive results concentrate in a few assets, so the family is not yet shown to generalise across coins.
