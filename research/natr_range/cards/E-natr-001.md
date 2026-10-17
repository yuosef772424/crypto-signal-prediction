---
id: E-natr-001
mechanism: volatility-estimator; volatility-structure
timeframe: 1h; 1d
target: magnitude
model_class: regression
reopens: F-0021:R1
not_covered_by: F-0005=different task: no new indicator and no IC screen, one existing feature (NATR_14) calibrated against a magnitude target with a pre-registered coverage/pinball rule; F-0006=different task: Parkinson and other estimators are not tested, the estimator is NATR_14 only, and the target is the quantiles of the next range, not a rolling-window IC sign test; F-0057=different use: no direction, no asymmetry and no trades, only the calibration of the next-range quantiles; F-0078=different use and target: calibration of next-bar range quantiles from NATR_14, not an ML/HAR-RV next-day volatility forecast used to size H07 positions (no sizing, no trades); F-0081=different input and baseline: NATR_14 of ONE timeframe feeds a log-linear quantile regression judged on coverage, against trailing 90-bar empirical quantiles (F-0081 compared gate states + NATR with NATR-only ridge/GBM in NATR units)
---
# Experiment card — E-natr-001: calibrated next-range forecast from NATR_14 (phase R3)

Written and committed BEFORE any result (rules: `docs/research/RESEARCH_RULES.md`; roadmap `docs/research/plan/ROADMAP.md` R3).
Do not edit after the run; the result goes in `research/natr_range/README.md`. H003 is read as corrected in
`docs/research/h003_volatility_reversal_correction.md`: high `NATR_14` precedes a wider next range (volatility clustering), no direction.

| Field | Value |
|---|---|
| Date / author | 2026-10-17 / R3 builder session |
| Baseline | master 4249926; per-coin trailing 90-bar empirical quantiles of the same target |
| The ONE change | forecast the next-bar range quantiles with a log-log quantile regression on NATR_14 instead of trailing empirical quantiles |
| Failure being addressed | none (new use of the H003 survivor); F-0021 R1 (range-width input against a magnitude target) is the reopen condition used |
| Data / splits / holdout | 10 coins (ADA BCH BNB BTC DOGE ETH SOL TRX XRP ZEC), Binance spot 5m via `tools/fetch_crypto_dataset.py --end "2026-09-30 23:55"`, read through `tools/holdout_guard.py` (window end 2026-10-01). Train 2020-01-01..2024-12-31, val 2025-01-01..2026-09-30 (bars by open time of the forecast target bar). Bucket H2 is NOT opened in this PR. |
| Seeds / budget | deterministic (exact quantile regression, no randomness); <= 50 min, 4 CPU cores |

## Construction (fixed now)
- Bars: 5m resampled in UTC to 1h and 1d, label = bar open, a bar exists only if ALL its 5m members exist (12 per hour, 288 per day), a bar is used only after it closed. The decision is taken at the close of bar t; the target is bar t+1.
- `natr_t` = 100 * ATR14_t / close_t (percent), ATR14 = Wilder smoothing (ewm alpha 1/14, min 14 true ranges, gaps skipped) of the true range max(H-L, |H-C_prev|, |L-C_prev|) on the regular grid; a true range is missing if the previous grid bar is missing.
- Targets for row t (all in percent of the decision close C_t): `range` = 100 (H_{t+1} - L_{t+1}) / C_t; `up` = 100 max(H_{t+1} - C_t, 0) / C_t; `down` = 100 max(C_t - L_{t+1}, 0) / C_t. A target is floored at 1e-4 percent before the log. Rows need bar t, bar t+1 and `natr_t`.
- Model, one per timeframe, pooled over coins, NO tuning: `log y = a_tau + b_tau * log natr` for tau in {0.1, 0.5, 0.9}, exact pinball minimisation (profile over b, a = tau-quantile of the residual), fitted on train only. Forecast quantiles = exp(fit), sorted across tau per row (rearrangement; the number of crossed rows is reported). `up` and `down` are fitted at the same three taus; the accept rule uses `range` only.
- Baseline: per coin, the empirical tau-quantile (linear interpolation) of the target series over the last 90 bars known at the close of t (target-series values of bars t-89..t; the target-series value of bar s is the same formula with C_{s-1}); rows with fewer than 90 values are dropped from BOTH model and baseline in the comparison.
- NATR deciles: 9 cut points from the pooled TRAIN `natr` of the timeframe; val rows are assigned by these cut points (val decile sizes are not equal; the count of each cell is reported).
- Metrics on val, per timeframe: coverage = share of rows with p10 <= y <= p90; pinball loss in percent units = mean over tau in {0.1, 0.5, 0.9} of the pinball loss, per decile for model and baseline on the same rows; monthly rank IC = Spearman(natr_t, range y_{t+1}) per calendar month of val pooled over coins.
- Slices (reported, not rules): NATR decile x timeframe x {BTC/ETH, others} x year (2025, 2026) with n, coverage and pinball ratio model/baseline; train coverage per year 2020-2024 (drift); baseline coverage; `up`/`down` p50 pinball vs baseline.
- Configurations counted in the manifest: 2 timeframes x 3 targets x 3 tau = 18 quantile fits (+ 18 baselines); the decisions taken are 2 accept-rule evaluations (one per timeframe). Nothing is tuned and no alternative model is tried.

## Accept / reject rule (fixed now, identical to ROADMAP R3, applied per timeframe on val)
A timeframe is accepted on exploration data only if ALL hold:
1. overall [p10, p90] coverage within 80 +/- 3 pp (77.0..83.0);
2. coverage within 80 +/- 6 pp (74.0..86.0) in EVERY one of the 10 NATR deciles;
3. model pinball loss below the baseline in >= 8 of the 10 deciles;
4. monthly rank IC(natr, next range) > 0 in >= 90 % of the val months (>= ceil(0.9 * N months)).
A timeframe that fails any item is rejected: registry rows from F-0082 at the failing level (coverage drift train->val = generalization; pinball vs baseline = representation; a decile miss = data). Rejection of one timeframe does not rescue or sink the other; the bot2 tool spec lists only accepted timeframes. This is exploration: acceptance is provisional until the H2 confirmation (one run, a separate confirmation card with `confirmation: true` and `frozen_rules_sha256`; prepared as a Colab job, not queued, and only if at least one timeframe is accepted).
- Not claimed: no direction, no trading edge, no net-of-cost result (nothing is traded); the claim is calibration of the next-range quantiles only.

## Result
Added in `research/natr_range/README.md`.
