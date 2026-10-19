---
id: E-natr-004
mechanism: volatility-estimator; volatility-structure
timeframe: 1h
target: magnitude
model_class: regression
reopens: F-0021:R1, F-0083:R1, F-0084:R2
not_covered_by: F-0005=different task: no new indicator and no IC screen, one existing feature (NATR_14) with a rolling calibration judged on coverage and pinball; F-0006=different task: NATR_14 only, target is the quantiles of the next range; F-0057=different use: no direction, asymmetry or trades, only the calibration of next-range quantiles; F-0078=different use and target: calibration of next-bar range quantiles, not a next-day volatility forecast used to size H07 positions; F-0081=different input and baseline: log-log quantile regression on NATR_14 of ONE timeframe with a per-NATR-decile rolling correction vs trailing 90-bar quantiles
---
# Experiment card — E-natr-004: rolling recalibration of the high-NATR band of the 1h NATR range forecast (phase R3, reopening F-0084)

PREPARED, NOT RUN. Written and committed before any code and before any result. Parent: `cards/E-natr-003.md` (Stage A rejected by rule 7: rules 1-6 pass, localization
label = drift, F-0084 `closed-unverified-cause`), grandparent `cards/E-natr-001.md` / `E-natr-002.md` (F-0083, H2 burned). Rules: `docs/research/RESEARCH_RULES.md`
(localize, test the cause, fix, validate). Do not edit after the run; the result goes in `research/natr_range/README.md`.

**Reopen conditions used.** F-0084 R2 (verbatim): "a new card with an adaptive calibration of the decile factors (rolling window, refit at fixed dates, fitted only on realized
rows strictly before each refit) since the label is drift, validated on exploration data and then on forward data >= 2026-10-01". F-0083 R1 asks for a high-NATR-specific
decile-wise calibration validated on time-sealed forward data from >= 2026-10-01; this card generalizes its "fitted on train only" to "fitted only on realized rows strictly before
each refit", which is the only change and is stated below. F-0021:R1 as cited by E-natr-003. Nothing here is claimed as confirmed before Stage B.

| Field | Value |
|---|---|
| Date / author | 2026-10-19 / R3 builder session |
| Baseline | E-natr-003 (`tools/natr_range.py`, branch `claude/r3-highvol-calib`): base regression `log y = a_tau + b_tau log natr` fitted ONCE on target open 2020-01-01..2023-12-31; decile edges = 9 quantiles of pooled NATR 2020-2024; per-coin trailing-90-bar quantiles for the pinball comparison |
| The ONE change | the per-NATR-decile multiplicative p10/p90 correction is recomputed ROLLING instead of once: at the start of each calendar month m, fit the factors on the trailing 12 months of realized rows strictly before m. Everything else (base fit, edges, ratio/conformal form, data, rules 1-6) is unchanged |
| Failure being addressed | F-0084 (generalization; location: top NATR decile coverage drifts from 2023 on, a fixed factor set goes stale) |
| Arms | **control** = base regression, no correction. **static** = E-natr-003 factors (2024 slice, fixed). **rolling** = the arm under test. Control and static are reported for separability; only rolling is ruled |
| Data / splits / holdout | the 10 open coins of E-natr-001 (ADA BCH BNB BTC DOGE ETH SOL TRX XRP ZEC), Drive `history_15m` (trimmed tails) through `holdout_guard.load_unsealed`, window end 2026-10-01; 15m resampled to 1h (all 4 members), stale runs >= 168 h dropped as in E-natr-003. Val (Stage A): target open 2025-01-01..2026-09-30 (exploration data, already touched by E-natr-001 and E-natr-003). No sealed bucket is read |
| Seeds / budget | deterministic (exact quantile regression + order statistics); one Colab job `research/natr_range/e_natr_004_job.json`, <= 30 min, CPU |

## The change (fixed now)
For each calendar month m that has val rows (first refit 2025-01-01 00:00 UTC):
- Window: rows with target open in `[m - 12 months, m)` AND target bar end `tgt_open + 1h <= m` (the target horizon must have ended before m starts; no overlap leak). Rows are pooled over coins.
- Factors per decile d (edges FIXED, the E-natr-001/003 edges), base regression FIXED (the 2020-2023 fit): `c90_d` = the ceil((n_d+1)*0.9)-th smallest of `y / q90_raw`, `c10_d` = the floor((n_d+1)*0.1)-th smallest of `y / q10_raw` over the window rows of decile d (identical formula to E-natr-003). `q50` is not corrected. Forecast for a row of month m in decile d = `[c10_d q10, q50, c90_d q90]`, sorted across tau.
- Hyper-parameters, fixed here and not tuned: **window 12 months; monthly refit; minimum rows per decile per refit 200**. If a decile has fewer than 200 window rows its factors are the POOLED factors (the same two formulas over all window rows of all deciles); this fallback is counted and reported per month and decile. (E-natr-003 used factor 1 for a sparse decile; here the pooled factor is used because after a drift 1 is not neutral.) If the whole window has fewer than 200 rows the run raises.
- Strictly-past guard (code, tested): the refit function raises if any window row has `tgt_open + 1h > m`, and the window selector can only return rows that satisfy it. The refit for month m never sees a row of m.
- Consistency check (reported): the January 2025 refit window is exactly the E-natr-003 calibration slice, so the rolling factors for 2025-01 equal the static factors.
Listed alternatives, NOT run here: 6- or 24-month windows, weekly refit, exponentially weighted conformal, a regime-conditional slope, per-coin factors, quantile-tracking (adaptive conformal inference) with a learning rate. No other tuning is allowed after the run.

## Stage A - exploration accept rule (rolling arm; val 2025-01..2026-09; the run now)
ALL must hold (rules 1-7 are the E-natr-003 rules, with 7 reduced to the coin count because localization was settled as drift by F-0084; 8-9 replace the localization rule):
1. overall [p10, p90] coverage 80 +/- 3 pp (77.0..83.0);
2. coverage 80 +/- 6 pp (74.0..86.0) in every evaluable decile (n >= 200; sparse deciles reported; >= 6 evaluable; E-natr-002 `sparse_rule` unchanged);
3. pinball below the per-coin trailing-90-bar baseline in >= ceil(0.8 * n_evaluable) evaluable deciles;
4. monthly rank IC(natr, next range) > 0 in >= 90 % of the months with rows;
5. top-decile coverage within 74.0..86.0 in 2025 AND in 2026 separately, each with >= 200 top-decile rows;
6. the rolling top-decile coverage is closer to 80 % than the control's in BOTH 2025 and 2026;
7. at least 8 of the 10 coins contribute rows in base fit, the first window and val;
8. **quarterly top-decile coverage**: in EVERY calendar quarter of val (2025Q1..2026Q3) with >= 200 top-decile rows, rolling top-decile coverage within 74.0..86.0; at least 5 of the 7 quarters must be evaluable (>= 200 rows) or the rule fails as "not evaluable";
9. **stability (not chasing noise)**: between consecutive refits, for every decile and for both c10 and c90 (effective factors, fallback included), `|ln(c_m / c_{m-1})|` is at most **0.05** (5 %) and the mean of all these changes is at most **0.02**. The static factors of E-natr-003 lie within 0.948..1.020, so a 5 % one-month move is a large move for a 12-month window that replaces one twelfth of its rows.
Reported, not ruled: static and control arms (same tables), pinball totals of the three arms, per-month factors, window n and fallback flags per decile and month, the number of fallbacks, top-decile coverage by month and quarter and year for all arms, crossed rows, BTC/ETH vs others, row counts per coin, stale source rows dropped.
Passing Stage A only ALLOWS Stage B ("accepted on exploration data, provisional"). On rejection: registry row F-0085 at the failing level (a miss of rules 1-8 on val = generalization; a miss of rule 9 only = representation, the refit is noisy); nothing is relaxed after the run. Because val is already touched, no result of this stage is a confirmation, and it would not be one even if every rule passes.

## Stage B - confirmation on time-sealed forward data (LATER; not built or run here)
- Data: bars with open time >= 2026-10-01 (the time seal) on the same 10 coins; first evaluable when at least 3 months of forward bars exist, i.e. not before 2027-01-01. Each forward month m is forecast with factors refit at the start of m on the trailing 12 months of realized rows before m (the first forward months use the open val rows 2025-10..2026-09 plus forward rows realized so far), the same code, the same base regression and edges, the same hyper-parameters.
- Frozen BEFORE any forward bar is read: the Stage A code commit, the base coefficients, the edges, the hyper-parameters (12 / monthly / 200 / fallback), the rules below. A confirmation card (`confirmation: true`) records the sha256 of the frozen config file (`frozen_stage_b.json` written by the Stage A job) first.
- Rules on the pooled forward window: Stage A rules 1-4 and 7, top-decile coverage within 74.0..86.0 with >= 200 top-decile rows (otherwise there is no verdict yet; wait for more bars), rule 8 for each forward quarter with >= 200 top-decile rows, rule 9 over the forward refits.
- PREREQUISITE (not built here): `holdout_guard.load_unsealed` refuses any window reaching the time seal (lock rules `docs/research/plan/lock/README.md` sections 6-7, D-009, D-010). Reading forward bars needs a new REVIEWED forward-read guard path (`time_seal_read`, ledgered), which is a separate PR. Until it exists Stage B cannot run. The sealed buckets are not used: H2 is burned, H1 and H3-H6 belong to other phases.

## Data hygiene
As E-natr-003: Drive `history_15m` with trimmed tails, `natr_range.drop_stale_runs` with 168 h, nothing dated at or after the time seal is read in Stage A (window end 2026-10-01).

## Result
Added in `research/natr_range/README.md` after the run.

**Status (appended 2026-10-19, rules above not edited): Stage A ACCEPTED on exploration data (provisional).** Rules 1-9 pass; bundle `research/natr_range/results/e_natr_004_explore/` (`RESULT.md`). Rolling and static are nearly identical on val, so Stage A does not separate them; Stage B (`cards/E-natr-004-confirm.md`) is the real test.
