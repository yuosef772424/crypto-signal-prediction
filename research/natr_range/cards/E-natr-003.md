---
id: E-natr-003
mechanism: volatility-estimator; volatility-structure
timeframe: 1h
target: magnitude
model_class: regression
reopens: F-0021:R1, F-0083:R1
not_covered_by: F-0005=different task: no new indicator and no IC screen, one existing feature (NATR_14) with a train-only calibration judged on coverage and pinball; F-0006=different task: NATR_14 only, target is the quantiles of the next range; F-0057=different use: no direction, asymmetry or trades, only the calibration of next-range quantiles; F-0078=different use and target: calibration of next-bar range quantiles, not a next-day volatility forecast used to size H07 positions; F-0081=different input and baseline: log-log quantile regression on NATR_14 of ONE timeframe with a per-NATR-decile correction vs trailing 90-bar quantiles
---
# Experiment card — E-natr-003: high-NATR calibration of the 1h NATR range forecast (phase R3, reopening F-0083)

PREPARED, NOT RUN. Written and committed before any code and before any result. Parent: `cards/E-natr-001.md` (1h accepted on exploration data)
and `cards/E-natr-002.md` (H2 confirmation REJECTED on one rule: top NATR decile coverage 73.78 % < 74.0; failure row F-0083,
`closed-unverified-cause`, bucket H2 burned). Rules: `docs/research/RESEARCH_RULES.md` (localize, test the cause, fix, validate).
Do not edit after the run; the result goes in `research/natr_range/README.md`.

**Reopen condition used (F-0083 R1, verbatim):** "a new card with a high-NATR-specific calibration (e.g. decile-wise or vol-regime quantile
correction fitted on train only) validated on time-sealed forward data from >= 2026-10-01, since H2 is spent". This card is the new card and the
fit-on-train-only part; the forward validation is Stage B below and nothing here is claimed as confirmed before it.

| Field | Value |
|---|---|
| Date / author | 2026-10-19 / R3 builder session |
| Baseline | E-natr-001 model class (`tools/natr_range.py`, master cf395ff): pooled log-log quantile regression of the next 1h range on NATR_14, tau 0.1/0.5/0.9; per-coin trailing-90-bar quantiles for the pinball comparison; control arm = the same regression fitted on 2020-2023 with NO correction |
| The ONE change | a per-NATR-decile multiplicative (ratio-form split-conformal / CQR-style) correction of the p10 and p90 quantiles, fitted on a train-tail calibration slice (2024); the base regression is then fitted on 2020-2023 only (the calibration slice cannot also be in the base fit; this window change is part of the one change and is isolated by the control arm) |
| Failure being addressed | F-0083 (generalization; location: top NATR decile coverage, worse in 2026) |
| Data / splits / holdout | the 10 open coins of E-natr-001 (ADA BCH BNB BTC DOGE ETH SOL TRX XRP ZEC), Drive `history_15m` (trimmed tails, trim-tails-001) through `holdout_guard.load_unsealed`, window end 2026-10-01; 15m bars resampled to 1h (all 4 members required). Base fit: target open 2020-01-01..2023-12-31. Calibration slice: 2024-01-01..2024-12-31. Val: 2025-01-01..2026-09-30. No sealed bucket is read. Nothing dated 2025 or later is used for fitting (enforced in code, tested). |
| Seeds / budget | deterministic (exact quantile regression + order statistics, no randomness); one Colab job `research/natr_range/e_natr_003_job.json`, <= 30 min, CPU |

## Step 1 - localization (pre-registered; train data only, run in the same job)
Question: is the top-decile (decile 9, edges from pooled NATR 2020-2024) under-coverage (a) a STATIC miscalibration already present in train
2020-2024 out of fold, or (b) a DRIFT visible only in 2025-2026?
Test: forward-chain out-of-fold coverage. For each fold year Y in 2021, 2022, 2023, 2024: fit the regression and the decile edges on target
years 2020..Y-1 only and measure the [p10, p90] coverage of year Y in every decile (n reported). Pooled OOF top-decile coverage = row-weighted over the
four folds. Control val top-decile coverage = the UNCORRECTED 2020-2023 model on val, separately for 2025 and 2026.
Decision rule (constants of the card):
- **static**: pooled OOF top-decile coverage <= 78.0 % AND at least 3 of 4 folds <= 79.0 %.
- **drift**: pooled OOF top-decile coverage >= 79.0 % AND the control's top decile is < 78.0 % in 2025 or in 2026.
- **not_reproduced**: pooled OOF >= 79.0 % and the control is >= 78.0 % in both val years (nothing left to explain).
- **mixed**: anything else.
Consequence: only **static** supports the mechanism of the one change (a train-fitted correction can remove a miscalibration that is already in train).
With drift, mixed or not_reproduced, Stage A is still computed and reported but the card can end at best "unexplained" (not adopted) and a Stage A pass
is not forwarded to Stage B; a new card would then have to test an adaptive (rolling) calibration instead. In-sample train coverage per decile is
reported, not ruled. Context known before pre-registration (committed `e_natr_001_run1/slices.csv`, base fitted 2020-2024): top decile of the
open coins 2025 others 76.7 % (n 3877), BTC/ETH 70.2 % (n 104); 2026 others 79.3 % (n 1189), BTC/ETH 77.6 % (n 58). So on the open coins the miss is
mostly 2025 and the 2026 slump seen on H2 (68.6 %) is not expected to reproduce here; Stage B is the real test.

## The change (fixed now)
Base regression `log y = a_tau + b_tau log natr` fitted on rows with target open in 2020-2023 (as E-natr-001, `fit_models`). NATR decile edges = 9 quantiles of the
pooled NATR of 2020-2024 (the E-natr-001 edges). On the calibration rows (2024) of decile d (n_d rows; if n_d < 200 the factors stay 1 and are reported):
`c90_d` = the ceil((n_d+1)*0.9)-th smallest of `y / q90_raw`, `c10_d` = the floor((n_d+1)*0.1)-th smallest of `y / q10_raw` (q_raw = base regression
quantiles, before correction). Forecast = `[c10_d q10, q50, c90_d q90]` for a row in decile d, then sorted across tau as before (crossed rows counted).
Why a 2024 slice with a ratio (multiplicative) form: the quantiles are positive and scale with NATR, so a ratio keeps the band proportional; one
year is the most recent train year (closest regime to val, 12 months covers seasonality) with about a tenth of its rows in the top decile; a time-ordered
slice, not a random one, because the rows are not exchangeable and the correction must be fitted on the past of what it will face. The finite-sample
conformal indices give each one-sided tail its nominal level in the calibration slice, so any remaining miss on val is drift, not fit.
`q50`, `up` and `down` forecasts are not corrected. Listed alternatives, NOT run here: regime-dependent slope (b depending on a volatility regime), quantile
regression on log NATR with a spline, rolling (adaptive) decile recalibration, conformal on a rolling window.

## Stage A - exploration accept rule (open coins, val 2025-01..2026-09, the corrected arm; run now)
The E-natr-001/002 rule set, unchanged, plus the year rule and the localization. ALL must hold:
1. overall [p10, p90] coverage 80 +/- 3 pp (77.0..83.0);
2. coverage 80 +/- 6 pp (74.0..86.0) in every evaluable decile (n >= 200 rows with the train edges; sparse deciles are reported, not ruled; at least 6 evaluable, as in the E-natr-002 amendment `sparse_rule` min_decile_n 200 / min_evaluable_deciles 6);
3. pinball below the per-coin trailing-90-bar baseline in >= ceil(0.8 * n_evaluable) evaluable deciles (8 of 10 when all are evaluable);
4. monthly rank IC(natr, next range) > 0 in >= 90 % of the months with rows;
5. **top decile coverage within 80 +/- 6 pp (74.0..86.0) in 2025 AND in 2026 separately**, each with at least 200 top-decile rows (otherwise the rule fails as "not evaluable");
6. predicted diagnostic moving: the top-decile coverage of the corrected arm is closer to 80 % than the control's in BOTH 2025 and 2026;
7. at least 8 of the 10 coins contribute rows in base fit, calibration slice and val; and the localization label is **static** (Step 1).
Reported, not ruled: deciles by year, BTC/ETH vs others, control arm table, correction factors and calibration n per decile, pinball totals of control and corrected, crossed rows, row counts per coin, stale-bar rows dropped.
Passing Stage A only ALLOWS Stage B ("accepted on exploration data, provisional"). Rejection: registry row F-0084 at the failing level (a miss in a train-fitted quantity such as the calibration slice itself = representation/data; a val miss while train OOF is calibrated = generalization/drift); nothing is relaxed after the run.

## Stage B - confirmation on time-sealed forward data (LATER; not part of this PR's run)
- Data: bars with open time >= 2026-10-01 (the time seal), first evaluable when at least 3 months of forward bars exist, i.e. not before 2027-01-01. Coins: the 10 open coins. A separate confirmation card
  (`confirmation: true`, `frozen_rules_sha256` of the frozen file the Stage A job writes: base coefficients, decile edges and the 10x3 correction factors) is written before any forward bar is read.
- Frozen: coefficients, edges and corrections from Stage A; nothing is refitted on forward data; the E-natr-003 code at the Stage A commit.
- Rule: the Stage A rule set 1-4 on the pooled forward window plus top-decile coverage 74.0..86.0 over the forward window; the top decile needs >= 200 rows or there is no verdict yet (wait for more bars); at least 8 coins.
- Lock rules (`docs/research/plan/lock/README.md` sections 6-7, D-009, D-010): `load_unsealed` refuses any window reaching the time seal, so reading forward bars needs a new reviewed guard path (`time_seal_read`, ledgered); that
  tool is NOT built here. The sealed buckets are not used: H2 is burned (one open only; the guard refuses a second), H1 and H3-H6 belong to other phases. Any extra coins (open-pool coins of level A/B, D-009) must be listed in the Stage B card before reading; their reads make them touched.

## Data hygiene
Drive `history_15m` has its dead tails trimmed (trim-tails-001). Mid-history padded gaps: the loader does NOT require volume > 0, so this job applies its own mask `natr_range.drop_stale_runs`:
source bars inside runs of >= 168 consecutive hours of flat (high == low) zero-volume 15m bars are dropped (every 1h bar touching one is then not built, every row needing it is absent); shorter zero-volume stretches stay. The number of dropped source rows is reported per coin.

## Result
Added in `research/natr_range/README.md` after the run.

## Status (added after the run; the sections above are unchanged)
**closed - Stage A REJECTED by rule 7** (2026-10-19). Rules 1-6 pass; the localization label is **drift** (pooled OOF top-decile coverage 81.37 %, folds 2023 75.7 % and 2024 76.1 %), so
accepted = False and the result is not forwarded to Stage B. Failure row F-0084 (`docs/research/failure_registry.csv`). Bundle and numbers:
`research/natr_range/results/e_natr_003_explore/` (`RESULT.md`). Follow-up: `cards/E-natr-004.md` (rolling recalibration).
