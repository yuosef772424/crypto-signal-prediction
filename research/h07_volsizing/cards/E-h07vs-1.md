---
id: E-h07vs-1
mechanism: volatility-structure; trend-following
timeframe: 1d
target: magnitude
model_class: gbm; regression
reopens:
not_covered_by:
---
# Experiment card — E-h07vs-1: ML next-day volatility forecast for H07 position sizing

Pre-registered 2026-10-05, BEFORE any forecast or backtest is computed (rules: `docs/research/RESEARCH_RULES.md`).

| Field | Value |
|---|---|
| Baseline | H07 frozen (`research/edge_discovery/16_h07_oos.py`): long BTC/ETH when the 30-day log return > 0, weight = 0.02 / σ̂ clipped at 3, split equally, daily close-to-close, cost 0.06%/side |
| The ONE change | σ̂ in the weight: trailing 30-day std of daily log returns (baseline) → ML forecast of next-day realized volatility |
| Data | Binance spot 5m BTCUSDT/ETHUSDT (Speirsy11/crypto-dataset via `tools/fetch_crypto_dataset.py`, trimmed 2026-09-30); daily realized variance = Σ 5m log-return² per UTC day |
| Splits | forecast-model train 2018-01-01..2021-12-31; VAL 2022-01-01..2023-12-31 (selection); TEST 2024-01-01..2026-09-30 (opened once). The H07 rule itself is fixed (not re-selected) |
| Seeds / budget | GBM 3 seeds averaged; one session |

## Mechanism and prediction
- Why: the weight 0.02/σ̂ aims at constant risk. Volatility is the one quantity this project has shown to be predictable
  (H003 NATR, H20 magnitude tokens, F-0057 envelope size). A better σ̂ should make realized risk closer to the target
  and cut exposure before volatile days, which in crypto are skewed to losses (smaller drawdowns, higher Sharpe).
- Diagnostic that must move (val): forecast QLIKE of the ML σ̂ lower than BOTH the trailing-30d σ̂ and a HAR-RV
  (1/5/22-day) linear forecast; and the strategy's realized-vol / target-vol ratio closer to 1 (|log ratio| smaller).
- Output effect expected (val): Sharpe up and max drawdown not worse vs baseline H07.

## Models compared (all fitted on train only, frozen for val/test)
1. `trail30` — baseline (H07 as is).
2. `har` — HAR-RV: OLS of log RV(t+1) on log RV over 1, 5, 22 days.
3. `gbm` — HistGradientBoosting on: log RV 1/5/22d, log Parkinson range 1/5d, |return| 1/5d, 30-day return sign,
   the other coin's log RV 1d, weekday; target log RV(t+1). Monotone/simple defaults, no tuning on test.
Forecast → daily σ̂ = sqrt(exp(pred)) for the weight.

## Accept / reject rule
- Accept on VAL if: (a) gbm QLIKE < har and < trail30 (forecast diagnostic) AND (b) Sharpe(gbm) − Sharpe(trail30)
  > 0 with a 95% stationary-block-bootstrap interval (block 20 days, 2000 draws) excluding 0, AND (c) max drawdown
  not worse by more than 2 points.
- Then TEST once: confirmed if Sharpe difference > 0 and max drawdown not worse. Otherwise rejected.
- If (a) holds but (b) fails: "forecast better, sizing gain within noise" → rejected for sizing (cause: verified via (a)).
- `har` is reported as a second candidate under the same rule (2 variants count toward multiple testing).

---
## Result (added after the run)
Run 2026-10-05 (`01_volsizing.py`, `volsizing_results.csv`). The script computed VAL and TEST rows in one run; the
decision below uses VAL only, and TEST is reported for transparency (it agrees).

**(a) Forecast diagnostic — holds.** Next-day log-RV QLIKE on VAL (lower is better): BTC gbm 0.411 / har 0.448 /
trail30 0.617; ETH gbm 0.311 / har 0.326 / trail30 0.615. Risk-targeting error 0.333 (trail30) → 0.289 (gbm) / 0.270 (har).

**(b) Sharpe — fails (wrong sign).**

| VAL 2022–23 | Sharpe | maxDD | ΔSharpe vs trail30, 95% block-bootstrap |
|---|---|---|---|
| trail30 (H07 as is) | **0.908** | −24.4% | — |
| har | 0.720 | −28.7% | [−0.44, +0.09] |
| gbm | 0.627 | −27.4% | [−0.57, +0.05] |
| buy & hold 50/50 | 0.101 | −68.3% | |

TEST 2024-01..2026-09 (not used for the decision): trail30 0.873 (maxDD −30.6%), har 0.860, gbm 0.764; buy & hold 0.567
(maxDD −59.9%).

**Cause test (VAL, after the decision, descriptive).** Gross Sharpe (no costs) is also lower (trail30 0.96, har 0.82,
gbm 0.74) although turnover doubles (0.074 → 0.138/day), so costs are not the main cause. On held days the trend's
return rises with volatility: with gbm sizing, the lowest-weight (highest predicted vol) tercile earns +38 bp/day vs
+15 bp in the highest-weight tercile; with trail30 the weight keeps a positive correlation with the next day's return
(+0.049 vs −0.006 for gbm). Accurate next-day vol targeting cuts exposure on exactly the days that pay.

**Verdict: rejected for sizing** (failure level: economics — the forecast skill is real, its P&L use is negative).
Registry row F-0064. Side result: H07 itself re-confirmed on executable Binance spot closes (VAL Sharpe 0.91 vs 0.10 for
buy & hold; TEST 0.87 vs 0.57, max drawdown −31% vs −60%).

**Robustness (owner's request, 2026-10-05): realized variance from 1h and 4h bars instead of 5m** (`RV_BAR=1h|4h`,
`volsizing_results_rv1h.csv`, `volsizing_results_rv4h.csv`; 2 more variants for multiple testing, decision unchanged).

| RV bars | QLIKE val BTC / ETH (gbm · har · trail30) | Sharpe val (trail30 · har · gbm) | Sharpe test (trail30 · har · gbm) |
|---|---|---|---|
| 5m (pre-registered) | 0.41 / 0.31 · 0.45 / 0.33 · 0.62 / 0.62 | 0.91 · 0.72 · 0.63 | 0.87 · 0.86 · 0.76 |
| 1h | 0.63 / 0.45 · 0.63 / 0.49 · 0.65 / 0.64 | 0.91 · 0.77 · 0.70 | 0.87 · 0.84 · 0.64 |
| 4h | 1.03 / 0.82 · 1.00 / 0.73 · 0.85 / 0.76 | 0.91 · 0.82 · 0.88 | 0.87 · 0.79 · 0.59 |

Coarser bars make the forecast worse (with 4h bars, 6 per day, it no longer beats the trailing 30-day std), but the
sizing result does not depend on the bar size: in every case the trailing 30-day std gives the best Sharpe and the
smallest drawdown.
