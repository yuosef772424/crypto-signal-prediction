# natr_range — E-natr-001 (phase R3): calibrated next-range forecast from NATR_14

Card: `cards/E-natr-001.md` (committed before any result, accept rule from ROADMAP R3). Code: `tools/natr_range.py` (library) and
`run_natr_range.py` + `config_e_natr_001.json` (run); bundle `results/e_natr_001_run1/` (manifest.json, 4 CSVs + results.json, 32 KB, verbatim).
H003 is read as corrected (`docs/research/h003_volatility_reversal_correction.md`): high NATR_14 precedes a **wider** next range, no direction.

## Verdict (exploration data, val 2025-01..2026-09, 10 coins): 1h ACCEPTED (provisional; NOT confirmed on H2, see below), 1d REJECTED (F-0082)
Model: `log(next range %) = a + b log(NATR_14 %)`, pooled over 10 coins, tau 0.1/0.5/0.9, exact quantile fit on train 2020-01..2024-12, no tuning.
Baseline: per-coin empirical quantiles of the last 90 bars. Run: 9 s, 18 quantile fits (2 timeframes x 3 targets x 3 tau), 2 accept-rule decisions, not truncated.

| Rule (per timeframe) | Threshold | 1h | 1d |
|---|---|---|---|
| overall [p10, p90] coverage | 77.0..83.0 % | 80.00 % PASS | 82.55 % PASS |
| every NATR decile coverage | 74.0..86.0 % | 77.2..80.4 % PASS | 76.6..86.13 % FAIL (decile 4 = 86.13 %, 0.13 pp over) |
| pinball below baseline | >= 8 of 10 deciles | 10/10 PASS (0.1339 vs 0.1488, -10.0 %) | 10/10 PASS (0.7704 vs 0.7996, -3.6 %) |
| monthly rank IC > 0 | >= 90 % of months | 21/21 PASS (mean 0.71, min 0.66) | 19/19 PASS (mean 0.58, min 0.45) |
| accepted | all four | YES (provisional until H2) | NO |

- 1h decile coverage (0..9): 80.3 79.5 80.3 80.0 80.2 80.3 80.3 80.4 79.3 77.2 (n 30196 .. 5228). Baseline coverage of its own p10-p90 is only 75.4 % (1d: 77.5 %).
- 1d decile coverage: 84.8 83.0 83.8 80.8 86.1 82.3 81.7 81.0 76.6 78.3 (n 1028 .. 207). Train coverage 79.99 %, so val drifted up 2.6 pp; the low deciles over-cover.
- Data note: the 5m source has a gap in May 2026 (about 180 missing bars per day on 21-29 May, all coins), so the strict "all members present" 1d bars are missing there and the 90-day baseline is
  NaN after it: 934 of 6240 1d val rows are dropped, and June and July 2026 have no 1d rows (the IC month count is 19, not 21). 1h loses 2068 of 151424 rows. No quantile crossing in train or val (0 rows).
- Reported, not rules: BTC/ETH undercover in 2026 at 1h (75.1 % vs 80.6 % for the other eight coins; 2025: 77.8 % vs 81.0 %); `up`/`down` p50 pinball beats the baseline at both timeframes
  (1h 0.1726/0.1741 vs 0.1795/0.1791; 1d 0.960/0.938 vs 0.984/0.970). Slices (decile x group x year, 78 cells) are in `slices.csv`.
- Cause of the 1d miss is NOT verified (F-0082, `closed-unverified-cause`): a 0.13 pp miss on 591 rows is inside sampling noise of the band, and the model beats the baseline in every decile; the honest
  reading is "not accepted", not "no information". No claim of direction or edge: nothing was traded.

## Deviations and process notes
- Months with no rows are not counted in the IC share (1d: 19 of 19); the card said "months of val". It does not change the verdict (1d fails on the decile rule).
- Commit `a0ccdde` (`test`) holds the failing tests; the `feat` commit also fixes a lambda-`drop` bug of one test written there. The H2 preparation has its own `test` then `feat` commit.
- Reproduce: `python tools/fetch_crypto_dataset.py --out /tmp/r3data --interval 5m --coins ADAUSDT BCHUSDT BNBUSDT BTCUSDT DOGEUSDT ETHUSDT SOLUSDT TRXUSDT XRPUSDT ZECUSDT --end "2026-09-30 23:55"`, then
  `python research/natr_range/run_natr_range.py --config research/natr_range/config_e_natr_001.json --out <dir>`. Manifest `git_sha` 05963754dcc22af15b1b244a07cba7b36bd44154; output sha256 values are in the manifest.

## H2 confirmation (E-natr-002): REJECTED 2026-10-19 (F-0083), H2 burned
Result bundle `results/e_natr_002_h2/`. 36/36 H2 coins (after trim-tails-001), 440,834 rows. Overall coverage 79.37 % PASS, pinball 10/10 deciles (0.2483 vs 0.2703) PASS, monthly IC > 0 21/21 PASS; per-decile coverage FAIL: top NATR decile 9 = 73.78 % (< 74.0 by 0.22 pp; 76.6 % in 2025, 68.6 % in 2026; decile 8 = 75.8 % in 2026). R3 1h is NOT confirmed; the 1h band under-covers at the highest NATR, as flagged on exploration data. 1d stays rejected.

Preparation notes (written before the run):
`cards/E-natr-002.md` (`confirmation: true`, `holdout_bucket: H2`, `frozen_rules_sha256` of `frozen_rules.json`, which `freeze_rules.py` writes from run1: 1h coefficients and decile edges), `run_natr_confirm.py` (opens H2 once via
`open_holdout`, reads Drive `history_15m`, resamples 15m to 1h, no refit, same rule plus >= 20 coins) and `h2_job.json` (id carries DO-NOT-QUEUE; `repo_ref` to be filled). 1d is not confirmed.
Sparse deciles: settled by the 2026-10-19 amendment of the card (deciles with n < 200 are reported, not ruled; >= 6 evaluable deciles required; pinball wins in ceil(0.8 * n_evaluable)), re-frozen before any H2 access.

## bot2
`bot2_tool_spec.md`: `get_range_forecast(symbol, interval)` for `1h` only, percent units, `as_of` on every number. It is a spec; nothing was changed in bot2.
