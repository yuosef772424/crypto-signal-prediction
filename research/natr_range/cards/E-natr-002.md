---
id: E-natr-002
mechanism: volatility-estimator; volatility-structure
timeframe: 1h
target: magnitude
model_class: regression
confirmation: true
holdout_bucket: H2
frozen_rules_sha256: a441aec550f8323a9bcf87a423dfdced1ec84bf5d5e00cc96e5add9c0a7d65da
frozen_rules_file: research/natr_range/frozen_rules.json
reopens: F-0021:R1
not_covered_by: F-0005=different task: no new indicator and no IC screen, one existing feature (NATR_14) with frozen coefficients judged on coverage and pinball; F-0006=different task: NATR_14 only, target is the quantiles of the next range; F-0057=different use: no direction, asymmetry or trades, only the calibration of next-range quantiles; F-0078=different use and target: calibration of next-bar range quantiles, not a next-day volatility forecast used to size H07 positions; F-0081=different input and baseline: frozen log-log quantile regression on NATR_14 of ONE timeframe vs trailing 90-bar quantiles
---
# Experiment card — E-natr-002: H2 confirmation of the 1h NATR range forecast (phase R3, confirmation step)

PREPARED, NOT RUN. Written and committed before the single H2 run. Parent: `research/natr_range/cards/E-natr-001.md` (1h accepted on
exploration data; 1d failed the per-decile coverage rule and is NOT confirmed here). Do not edit after the run; the result goes in
`research/natr_range/README.md`.

| Field | Value |
|---|---|
| Frozen rules | `research/natr_range/frozen_rules.json` (sha256 above): the 1h coefficients and NATR decile edges fitted on the 10 train coins 2020-2024, the accept-rule constants, the window and `min_coins`. Nothing is refitted on H2. |
| Data | bucket H2 only (sealed coins of `docs/research/plan/lock/holdout.json`), Drive `history_15m`, opened ONCE through `holdout_guard.open_holdout(card, 'H2')` by `research/natr_range/run_natr_confirm.py`; 15m bars resampled to 1h (all 4 members required). Target bars with open time in [2025-01-01, 2026-10-01) (time seal respected). |
| Run | one Colab job `research/natr_range/h2_job.json` (<= 20 min); the bundle also carries `access_log_row.csv` (the log row appended in the clone) to be committed to `holdout_access_log.csv` by the reviewer. |
| Seeds / budget | deterministic, no fitting; one run. |

## Accept / reject rule (E-natr-001 rule, amended for sparse deciles; applied once to the pooled H2 coins, 1h)
Constants `min_decile_n` = 200, `min_evaluable_deciles` = 6, `pinball_win_frac` = 0.8 are in `frozen_rules.json` (`sparse_rule`). An *evaluable* decile (train edges) has n >= 200 rows; the others are REPORTED (n, coverage, pinball) but NOT ruled.
1. overall [p10, p90] coverage 80 +/- 3 pp;
2. coverage 80 +/- 6 pp in every evaluable decile;
3. pinball loss below the per-coin trailing-90-bar baseline in >= ceil(0.8 * n_evaluable) evaluable deciles (8 of 10 when all 10 are evaluable, as before);
4. monthly rank IC(natr, next range) > 0 in >= 90 % of the months with rows;
5. coin count: at least `min_coins` (20) H2 coins contribute rows; otherwise the run is "not evaluable" and the bucket stays burned (reported, no verdict);
6. at least 6 evaluable deciles; fewer fails the run (reason "too few evaluable deciles").
Accepted = all six. A failure is recorded in the failure registry at the failing level (a train->H2 coverage shift in an evaluable decile is generalization; too few evaluable deciles is data/universe). Nothing is relaxed after the run.

## Amendment (2026-10-19, before any H2 access)
H2 holds lower-cap, higher-NATR coins, so some NATR deciles (train edges) may be nearly empty. An empty or tiny decile is a universe shift, not a model failure, and under the original rule 2 it alone would have failed the run (NaN coverage). Amended before any H2 access (holdout_access_log.csv has no H2 row): sparse deciles are reported, not ruled; a floor of 6 evaluable deciles keeps the test meaningful; the pinball count scales to ceil(0.8 * n_evaluable). Coefficients and edges are unchanged (only `sparse_rule` was added to the frozen file, hence the new sha256). With all 10 deciles evaluable the verdict equals the original rule.

## Result
Added in `research/natr_range/README.md` after the single run.
