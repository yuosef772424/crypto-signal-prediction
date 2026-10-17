---
id: E-gate-001
mechanism: indicator-signal; trend-following; momentum; mean-reversion; volatility-structure
timeframe: 1h
target: direction; magnitude
model_class: logistic; gbm
reopens: F-0027:R1, F-0027:R2, F-0056:R2, F-0039:R2, F-0009:R2
not_covered_by: F-0036=different input: categorical multi-timeframe (15m/1h/4h/1d) derived states and their aggregate fed to a model, not the single 1h RSI_14 value and a 0.14% round-trip cost instead of 0.08%; F-0040=different target and inputs: direction and NATR-unit range from derived indicator states, not a LightGBM range-asymmetry score from pipeline features
---
# Experiment card — E-gate-001: gate-style derived technical states as model inputs (phase G2)

Written and committed BEFORE any result (rules: `docs/research/RESEARCH_RULES.md` 2.1 and 3; roadmap `docs/research/plan/ROADMAP.md` G2).
Do not edit after the run; the result goes in `research/gate_features/README.md`.

| Field | Value |
|---|---|
| Date / author | 2026-10-17 / G2 builder session |
| Baseline | master 84b866b; intercept (direction), NATR-only (magnitude) |
| The ONE change | feed `tools/gate_features.py` (`gate-style-v1, unverified`) categorical multi-timeframe states + aggregate to a model |
| Failure being addressed | F-0027 R2 (never compared old vs new set inside a model), F-0056 R2 / F-0039 R2 (capacity controls not run), F-0009 R2 (BOLL bandwidth vs future volatility) |
| Data / splits / holdout | 10 coins (ADA BCH BNB BTC DOGE ETH SOL TRX XRP ZEC), Binance spot 5m via `tools/fetch_crypto_dataset.py --end "2026-09-30 23:55"`, read through `tools/holdout_guard.py` (window end 2026-10-01). Train 2018-01..2022-12, val 2023-01..2024-12 (selects), 2025-01..2026-09 compares only. H5 confirmation is NOT run in this PR (Colab job prepared, not queued). |
| Seeds / budget | deterministic models (random_state 0) + 20 label-shift nulls; <= 50 min on 4 CPU cores |

## Construction
- Decision on every closed 1h bar t (bar open t, known at t+1h); entry at the open of the bar starting t+1h; exit at the open H hours later, H in {4, 24}. Resampling 5m -> 15m/1h/4h/1d is UTC, label = bar open, a bar is used only once it closed (open + length <= decision time); a bar with missing 5m members is dropped.
- States per timeframe (definitions in the module docstring, all `gate-style-v1, unverified`): ma_alignment A/B for SMA and EMA, rsi_zone (14, Wilder, 30/70), macd cross within 3 bars + histogram sign (12,26,9), KDJ (9,3,3) zone 20/80 + K/D cross within 3 bars, SAR (0.02,0.2) side, BOLL (20,2) %B zone + bandwidth percentile over 120 bars (bucketed <0.2 / mid / >0.8), ADX(14) bucket (<20, 20-40, >40) + DI dominance, WR(14) zone -80/-20, CCI(20) zone +-100. Aggregates: bullish minus bearish count per timeframe (trend group, oscillator group, all; trend states with their natural sign, oscillator zones contrarian: oversold = bullish) and over all timeframes; gate-like `signal` = sign of the overall aggregate. Missing history = NaN (aggregate is NaN if any member is NaN); NaN is never 0.
- Old set (the F-0027 R2 comparison): the raw indicator values of the same indicators on the same timeframes (RSI, MACD line/signal/hist over close, K/D/J, (close-SAR)/close, %B, bandwidth, ADX, +DI, -DI, WR, CCI, MA7/30/120/200 ratios). Same models, same k grid, same rows.
- Targets: direction = sign of the H-hour forward return from the entry open (4h and 24h); magnitude = (max high - min low over the H hours from entry) / entry open, divided by NATR_14 of the 1h bar at decision time (range in NATR units), modelled as log.
- Models (fixed hyper-parameters, no tuning): logistic (L2, C = 0.1, one-hot states with an explicit "missing" level; raw set standardized, train-median imputed + missing flag) and sklearn HistGradientBoosting (100 iterations, learning rate 0.05, depth 3, min leaf 500, L2 1.0; native NaN). Magnitude: ridge (alpha 10) on log range, quantiles from train residuals, and HistGradientBoosting quantile loss; both predict quantiles tau 0.25/0.5/0.75; loss = mean pinball on the NATR-unit range. Training rows every 4th hour (labels purged against the next split by H hours).
- Capacity controls (RESEARCH_RULES 2.1): effective sample size (non-overlapping 24h steps x independent coins from the mean cross-coin correlation of 1h returns) vs feature count; k-sweep k = 1,3,5,10,20,all of the columns ranked on train only (between-level R2 of the target, 10 quantile bins for continuous columns); learning curve on the most recent 25/50/100% of train; null = 20 label shifts (each coin's labels circularly shifted by a random 30-300 day offset, features kept) through the whole pipeline (ranking + fit + max over configurations); old vs new set at every (model, k).
- Configurations counted in the manifest: direction = 2 horizons x 2 sets x 2 models x 6 k = 48; magnitude = 48; plus the null refits.

## Accept / reject rule (fixed now, identical to ROADMAP G2)
- Direction, at a horizon (state set, best config chosen on val): val AUC >= 0.53 AND above the null's 99th percentile (max of the 20 null maxima, and mean + 2.33 sd) AND log-loss gain over the intercept in >= 4 of the 5 calendar years 2021-2025 (each year out-of-sample: model refit on all years before it, same config) AND the real k-sweep peak exceeds the peak of every null k-sweep. Then economics: top-decile (by |p - 0.5|) gross per-trade move > 0.14% round trip at 1x and > 0.28% at 2x on val; 2025-01..2026-09 is reported (1x/2x net, after removing the best 5%, per coin) and cannot rescue a failure.
- Magnitude, at a horizon: >= 2% relative pinball improvement of (state set + NATR columns) over the best NATR-only baseline on val, with the same sign in each of 2021-2025 (refit as above).
- Anything else -> registry rows from F-0080 with the level where it failed; a pass is "provisional" until the H5 confirmation (separate step).
- Reject-level mapping: AUC/null failure = generalization or data; economics failure = economics.

## Result
Added in `research/gate_features/README.md`.
