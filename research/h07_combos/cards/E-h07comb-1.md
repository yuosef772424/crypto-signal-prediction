---
id: E-h07comb-1
mechanism: trend-following
timeframe: 1d
target: pnl
model_class: rule
reopens:
not_covered_by:
---
# Experiment card — E-h07comb-1: can H07 be improved by combining it with other rules?

Pre-registered 2026-10-05, BEFORE any variant is computed (rules: `docs/research/RESEARCH_RULES.md`).

| Field | Value |
|---|---|
| Baseline V0 | H07 frozen: per coin, long when the 30-day log return > 0; weight = 0.02 / std30(daily log returns), cap 3, divided by the number of coins; BTC+ETH; daily close-to-close; 0.06%/side |
| Data | Binance spot daily closes (last 5m close of each UTC day) for BTC ETH SOL ADA BCH BNB DOGE TRX XRP ZEC, 2017-08..2026-09-30 (`tools/fetch_crypto_dataset.py`); a coin enters once it has 60 days of history |
| Splits | VAL (selection) 2018-01-01..2023-12-31 — used before only to *validate* the frozen V0, never to tune; TEST 2024-01-01..2026-09-30, opened once for accepted variants |
| Budget | 4 variants (Bonferroni α = 0.05/4) |

## Variants (exactly these, no tuning)
- **V1 multi-speed ensemble (BTC+ETH):** signal = mean of 1{N-day log return > 0} for N ∈ {10, 20, 30, 60} (a fraction
  0..1 instead of 0/1); same weight formula. Mechanism: averages over lookbacks → enters/exits gradually, less lag
  and fewer whipsaws (attacks the January-2026-type loss).
- **V2 fast exit (BTC+ETH):** long only when BOTH the 30-day and the 10-day log returns > 0. Mechanism: leaves a
  turning trend ~3 weeks earlier; cost: more trades.
- **V3 broader universe:** V0's rule on all 10 coins, weight divided by the number of coins with data (same total risk
  budget as V0). Mechanism: diversification across partly independent trends.
- **V4 = V1 on the 10 coins.**

## Mechanism checks (diagnostics that must move, VAL)
- V1/V2: average loss in the 20 days after an exit signal of V0 should shrink (less lag) — reported as
  "give-back" = mean return of V0's position over the 10 days before each V0 exit.
- V3/V4: average pairwise correlation of the coins' strategy returns < 0.7 (diversification exists).

## Accept / reject rule
- VAL: ΔSharpe(variant − V0) > 0 with a stationary-block-bootstrap (block 20 days, 2000 draws) interval at
  1 − 0.05/4 = 98.75% excluding 0, AND max drawdown — after scaling the variant to V0's realized volatility on VAL — not
  worse than V0 by more than 2 points.
- TEST (once, accepted variants only): ΔSharpe > 0 and vol-matched max drawdown not worse.
- Descriptive only (not a variant): idle cash earning a stablecoin yield (e.g. 4%/yr) adds roughly yield × average
  idle fraction; reported, not tested.

---
## Result (added after the run)
Run 2026-10-05 (`01_combos.py`, `combos_results.csv`). VAL decides; TEST was computed in the same run and is shown for
transparency only (no variant was accepted on VAL, so TEST is not a confirmation of anything).

| Variant | VAL 2018–23 Sharpe | ΔSharpe [98.75% CI] | maxDD (vol-matched) | TEST 2024–26/09 Sharpe | TEST maxDD |
|---|---|---|---|---|---|
| V0 H07 | 1.39 | — | −40% (−40%) | 0.89 | −31% |
| V1 ensemble 10/20/30/60 | 1.38 | −0.01 [−0.33, +0.29] | −37% (−39%) | 0.94 | −31% |
| V2 fast exit 30d∧10d | 1.43 | +0.04 [−0.34, +0.43] | −28% (−32%) | 0.86 | −29% |
| V3 10 coins | 1.51 | +0.12 [−0.56, +0.86] | −30% (−38%) | 1.23 | −19% |
| V4 ensemble on 10 coins | 1.40 | +0.01 [−0.63, +0.77] | −29% (−38%) | 1.44 | −17% |

**Mechanism checks — both moved as predicted.** Give-back in the 10 days before a V0 exit: V0 −16.9 bp/day, V1
−11.3, V2 −12.1 (less lag). Mean pairwise correlation of per-coin trend returns: 0.35 (real diversification).

**Verdict: not accepted (failure level: power).** No variant's VAL interval excludes 0: with daily data the Sharpe
difference of two highly correlated trend strategies has a ±0.3–0.8 uncertainty even over 6 years. The direction is
consistent for V3/V4 (higher Sharpe and smaller raw drawdown in VAL and in TEST), so they are tracked as **shadow
variants** in the forward paper trading instead of adopted. Registry row F-0065.
