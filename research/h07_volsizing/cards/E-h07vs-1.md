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
