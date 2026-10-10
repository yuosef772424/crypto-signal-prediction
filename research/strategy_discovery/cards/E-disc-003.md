---
id: E-disc-003
mechanism: breakout; trend-following; regime-filter
timeframe: 1d
target: pnl
model_class: rule
reopens: F-0054:R1
not_covered_by: F-0072=volume-confirmed Donchian on unseen assets, a different trigger and filter from this card; F-0070=the momentum rotation of E-disc-004, this card applies breakout survivors to unseen assets; F-0071=this card is the source of F-0071 (its own result); F-0069=daily ADX>40 removal and short-side removal from E-ind-002, this card applies the E-disc-001 survivors unchanged to new assets; F-0061=stop orders at predicted candle extremes, no predicted level here; F-0062=regime filters on wick-capture fills, the ADX gate here is an entry condition on the breakout trigger
---
# Experiment card — E-disc-003: the three robust survivors on UNSEEN assets (BCH, TRX, ZEC) and on fresh bars

Written and committed BEFORE the run (rules: `docs/research/RESEARCH_RULES.md`, `docs/research/MODEL_DEV_PROTOCOL.md`).

## What is new here
- **Unseen assets:** BCHUSDT, TRXUSDT and ZECUSDT were never loaded by any earlier discovery, screen, robustness or TEST step.
  Their period, 2024-01-01 → 2026-09-30, was used before for the seven original coins, so this is an ASSET holdout inside a
  consumed period. It is weaker than a fresh period.
- **Fresh bars:** 2026-10-01 → 2026-10-08 (the last bar in the data is 2026-10-08 23:00 UTC). For all ten coins. This is a single
  short window: it is recorded as a sanity check, not as evidence, because it holds only about 8 daily bars per coin.

## Survivors (fixed; from E-disc-001 and checked in E-disc-002)
- S1 `1d|donchian20|adx25`; S2 `1d|bb_break|adx25`; S3 `1d|donchian20|none`.
- Exit: stop 2×ATR, target 3×ATR, 30 bars, cost 0.12% RT, entry at the next open. Unchanged.

## Accept rules (written before the run)
**A. Unseen assets, 2024-01-01 → 2026-09-30** (BCH, TRX, ZEC pooled, 3 coins):
- a survivor PASSES if pooled mean net R > 0 AND profit factor > 1.0 AND at least 2 of the 3 coins have mean net R > 0.

**B. Fresh bars, 2026-10-01 → 2026-10-08** (all ten coins, pooled):
- reported as a count and a mean. A verdict is given only if the window holds at least 10 trades; otherwise the result is
  "insufficient sample" and is not used to accept or reject anything.

A survivor that passes A and has enough trades in B is a candidate for an out-of-time test on the next data drop. It is not a
verified strategy: the evidence from the consumed period and the 8-day window is not enough to call an edge established.

## Predicted diagnostic movement
- The breakout survivors should keep a positive mean on new assets only if their edge is about trend and volatility, not about the
  specific coins. A failure on TRX or ZEC (both with different volatility) would be informative.
- The 8-day fresh window should contain few signals (a few per coin at most). A near-zero or negative result there is weak evidence.

---
## Result (added after the run)
Run: `research/strategy_discovery/oos.py` → `results/E-disc-003_results.csv`. Data: `/home/user/research/ohlc_full` (includes the bars up to 2026-10-08 23:00).

**A. Unseen assets BCH, TRX, ZEC, 2024-01-01 → 2026-09-30 (consumed period, unseen assets):**

| Survivor | n trades | Mean net R | pf | Coins positive | Per coin | Verdict |
|---|---|---|---|---|---|---|
| S1 donchian20 + adx25 | 98 | +0.116 | 1.23 | 2 of 3 | BCH −0.386, TRX +0.170, ZEC +0.335 | **PASS** |
| S2 bb_break + adx25 | 73 | −0.118 | 0.80 | 1 of 3 | BCH −0.421, TRX −0.123, ZEC +0.068 | **FAIL** |
| S3 donchian20, no gate | 158 | +0.135 | 1.27 | 2 of 3 | BCH −0.082, TRX +0.155, ZEC +0.288 | **PASS** |

**B. Fresh bars 2026-10-01 → 2026-10-08 (all ten coins):** one trade per configuration (n = 1 for S1, S2 and S3). Below the minimum of 10 trades, so the verdict is **insufficient sample**, as the card predicted. Nothing is concluded from it.

Reading of the result:
- The two Donchian survivors (S1, S3) keep a positive mean and profit factor above 1 on coins that were not in any earlier study. Both are carried by ZEC and TRX; BCH is negative in all three. So the result depends on which coins are in the sample.
- The Bollinger survivor (S2) fails on the new coins. This is consistent with its VAL result having been a selection effect, but three coins cannot show that by themselves.
- This is still a consumed period for the original seven coins, and the new coins are only three. The evidence is moderate, not established.
- Verdict: **S1 and S3 are candidates. S2 is closed (failure row F-0071).** A fresh-data verdict needs at least several weeks of bars after 2026-10-08.
