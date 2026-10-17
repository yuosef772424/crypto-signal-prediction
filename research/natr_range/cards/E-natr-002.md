---
id: E-natr-002
mechanism: volatility-estimator; volatility-structure
timeframe: 1h
target: magnitude
model_class: regression
confirmation: true
holdout_bucket: H2
frozen_rules_sha256: 9787549bf263669fafa445aadc9a24397f25b08c1f85cfcf97572b386764d61e
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

## Accept / reject rule (identical to E-natr-001, applied once to the pooled H2 coins, 1h)
1. overall [p10, p90] coverage 80 +/- 3 pp;
2. coverage 80 +/- 6 pp in each of the 10 NATR deciles (train edges; a decile with no row is NaN and fails);
3. pinball loss below the per-coin trailing-90-bar baseline in >= 8 of the 10 deciles;
4. monthly rank IC(natr, next range) > 0 in >= 90 % of the months with rows;
5. coin count: at least `min_coins` (20) H2 coins contribute rows; otherwise the run is "not evaluable" and the bucket stays burned (reported, no verdict).
Accepted = all five. A failure is recorded in the failure registry at the failing level (a train->H2 coverage shift is generalization; a sparse or empty
decile because H2 coins sit in other NATR deciles is data/universe). Sparse deciles are NOT relaxed after the fact. Note for the reviewer before queueing:
H2 contains lower-cap coins, so the lower NATR deciles may be nearly empty and rule 2 would fail for that reason alone; if that is unacceptable, amend this card
(and re-freeze) BEFORE the run, not after.

## Result
Added in `research/natr_range/README.md` after the single run.
