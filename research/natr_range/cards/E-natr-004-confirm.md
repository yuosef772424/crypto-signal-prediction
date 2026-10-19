---
id: E-natr-004-confirm
mechanism: volatility-estimator; volatility-structure
timeframe: 1h
target: magnitude
model_class: regression
confirmation: true
forward_confirmation: true
frozen_config_file: research/natr_range/frozen/frozen_stage_b.json
frozen_config_sha256: 7493a719559b636962273c1211060fa472cc46e1307db74ac06805b2b6b6c8e7
frozen_code_file: tools/natr_range.py
frozen_code_sha256: 7c5889abc6cef50baf1ef2cc1aeb4729b72ce347b895c29e978f1f4ff3e3a0de
forward_start: 2026-10-01
parent: E-natr-004
reopens: F-0021:R1, F-0083:R1, F-0084:R2
not_covered_by: F-0005=different task: no new indicator and no IC screen, one existing feature (NATR_14) with a rolling calibration judged on coverage and pinball; F-0006=different task: NATR_14 only, target is the quantiles of the next range; F-0057=different use: no direction, asymmetry or trades, only the calibration of next-range quantiles; F-0078=different use and target: calibration of next-bar range quantiles, not a next-day volatility forecast used to size H07 positions; F-0081=different input and baseline: log-log quantile regression on NATR_14 of ONE timeframe with a per-NATR-decile rolling correction vs trailing 90-bar quantiles
---
# Experiment card - E-natr-004-confirm: Stage B forward confirmation of the rolling NATR range calibration (phase R3)

PREPARED, NOT RUN. Written and committed before any forward bar is read and before the runner exists as a queued job. Parent: `cards/E-natr-004.md` (Stage A ACCEPTED on
exploration data 2026-10-19, `results/e_natr_004_explore/RESULT.md`). Rules: `docs/research/RESEARCH_RULES.md`. Do not edit after the run; the result goes in
`research/natr_range/README.md`. Reopens F-0084 R2 (the "then on forward data >= 2026-10-01" part) and F-0083 R1.

**Honest scope.** On the Stage A validation window the rolling and static arms were nearly identical (no further drift inside 2025-26). This card therefore tests the rolling
rule on data nobody has seen; if the forward regime does not move, rolling and static will again be indistinguishable and a pass will mean "calibrated and stable", not "rolling is
needed". The static and control arms are reported next to rolling for that reason; only rolling is ruled.

| Field | Value |
|---|---|
| Frozen config | `research/natr_range/frozen/frozen_stage_b.json` (byte-identical copy of the Stage A job output), sha256 `7493a719559b636962273c1211060fa472cc46e1307db74ac06805b2b6b6c8e7` (front matter). It holds the base coefficients (fit 2020-2023), the 9 decile edges (pooled NATR 2020-2024), the hyper-parameters (window 12 months, monthly refit, min 200 rows per decile, pooled-factor fallback) and the rule constants |
| Frozen code | `tools/natr_range.py` (sha256 in the front matter, equal to its Stage A state at repo_ref `e32d8a6d9da2fd54153a754585485894b5ef6c3d`); the runner refuses to start if the file hash differs |
| The rolling rule on forward data | unchanged from E-natr-004: at the start of each calendar month m, the p10/p90 decile factors are refit on realized rows with target open in `[m - 12 months, m)` AND target bar ended (`tgt_open + 1h <= m`), pooled over coins, with the strictly-past guard of `natr_range.fit_month_factors` (raises on any look-ahead). Rows are the open-coin rows 2025-09 onward plus the forward rows realized so far; forward month m never sees a row of m. Nothing is refitted except these factors |
| Data | the same 10 open coins (ADA BCH BNB BTC DOGE ETH SOL TRX XRP ZEC), Drive `history_15m`. Evaluated rows: target bar open time >= 2026-10-01 (the time seal) and < the evaluation end. Forward bars are read ONLY through `holdout_guard.load_forward` (sha256-checked card, logged, refuses sealed coins and any bar before 2026-10-01). Bars before 2026-10-01 (from 2025-06-01) are read through `load_unsealed` ONLY as history for NATR_14, the 90-bar baseline and the first rolling windows; they are never evaluated. 15m resampled to 1h, stale runs >= 168 h dropped as in Stage A |
| Evaluation date | first evaluation on target opens in `[2026-10-01, 2027-01-01)`; the run is not allowed before 2027-01-01 (the runner refuses `forward_end` later than today) |
| Seeds / budget | deterministic, no fitting beyond the monthly factors; one Colab job `research/natr_range/e_natr_004_confirm_job.json`, <= 30 min, CPU. NOT queued by this card |

## Accept rules (applied once to the pooled forward window; fixed now)
Notation: "evaluable decile" = n >= 200 rows (the E-natr-002 sparse rule, unchanged). Minimum sample sizes are part of the rules:
- **Sufficiency (checked first).** The window is *sufficient* iff the top NATR decile has >= 200 pooled rows AND >= 6 deciles are evaluable. If it is not sufficient the verdict is **"insufficient, extend by 3 months"**: ONE extension only, to the window `[2026-10-01, 2027-04-01)`, evaluated not before 2027-04-01, same rules, same frozen config, all rows pooled. If the extended window is still not sufficient the verdict is **"insufficient (final)"**, the confirmation is NOT obtained, and no further extension, change of coins, window or rule is allowed. No other extension is permitted, and no extension is allowed for a window that is sufficient and fails a rule.
- Then ALL of the following must hold for a *confirmed* verdict (otherwise **rejected**):
1. overall [p10, p90] coverage 80 +/- 3 pp (77.0..83.0);
2. coverage 80 +/- 6 pp (74.0..86.0) in every evaluable decile; the others are reported;
3. pinball below the per-coin trailing-90-bar baseline in >= ceil(0.8 * n_evaluable) evaluable deciles;
4. monthly rank IC(natr, next range) > 0 in >= 90 % of the forward months with rows (integer test positives*100 >= 90*months; 3 months = all 3);
5. top-decile coverage within 74.0..86.0 on the pooled forward window (>= 200 rows, by sufficiency);
6. the rolling top-decile coverage is closer to 80 % than the control's on the pooled forward window;
7. at least 8 of the 10 coins contribute forward rows;
8. in EVERY forward calendar quarter with >= 200 top-decile rows (2026Q4; plus 2027Q1 after the extension) the rolling top-decile coverage is within 74.0..86.0, and at least 1 forward quarter is evaluable;
9. stability: between consecutive refits from 2026-09 (computed from pre-seal rows only, not evaluated) through the last forward month, for every decile and both c10 and c90 (effective factors, fallback included) `|ln(c_m / c_{m-1})|` <= 0.05, and the mean of all these changes <= 0.02.
Reported, not ruled: static and control arms (same tables), monthly factors with n and fallback flags, pinball totals of the three arms, top-decile coverage by month and quarter, crossed rows, BTC/ETH vs others, row counts per coin, stale rows dropped, the rolling-vs-static top-decile gap (if it is below 1 pp the report says rolling is not separated from static).
On a **confirmed** verdict R3 1h is confirmed on time-sealed forward data for the 10 open coins only (not the H2 coins; the registry keeps F-0083 for H2). On **rejected**: registry row at the failing level (rules 1-8 = generalization, rule 9 only = representation); nothing is relaxed after the run.

## Guard path and ledger
The run needs the reviewed forward-read path `holdout_guard.load_forward(data_dir, coins, start, end, card_path)`: it requires this card (`confirmation: true`, `forward_confirmation: true`, a frozen config file whose sha256 equals `frozen_config_sha256`), refuses sealed coins and bars before 2026-10-01, and appends exactly one row (bucket `FWD`) to `docs/research/plan/lock/holdout_access_log.csv` per call. The reviewer commits the row copied to the bundle as `access_log_row.csv`. No sealed bucket (H1-H6) is opened.

## Result
Added in `research/natr_range/README.md` after the run.
