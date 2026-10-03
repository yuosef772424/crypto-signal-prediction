# Audit round 2 — pinned at b14b9bb (branch claude/charming-sagan-kswo2r)

Auditor: independent (round-1 report, agent reports and builder rationale not read). Scope: the full Colab path for
1h / window 32 / stride 8 / horizon 1 (pipeline -> main -> cross_asset). All runs are on a synthetic Drive
(`_r2_synth.py`: 1h + 15m klines, funding, OI, futures_metrics in the vision-fetch formats) driving the real notebook
functions through `_nbload.py`. Nothing is a real-data measurement. No training beyond 2-epoch smoke runs.

Result: no finding that blocks the run; no look-ahead or split contamination found. Seven proven defects (3 bias results,
4 minor), six suspicions.

Repro convention: each `r2_0N_*.py` exits 1 (assert) at b14b9bb and must exit 0 once fixed. `r2_00`, `r2_08`, `r2_09` are
clean checks and exit 0 now (they must stay green after any fix).

---

## Findings

### R2-01 — stale per-asset pipeline checkpoint is reused after the raw data grew  [biases results]
- Repro: `python docs/research/audit/r2_01_stale_pipeline_checkpoint.py`
- Output:
  ```
  samples  old build 1014 | rebuild with same checkpoint_dir 1014 | no-cache 1572
  last ts  old build 2025-02-28 16:00 | rebuild 2025-02-28 16:00 | no-cache 2025-03-31 16:00
  AssertionError: DEFECT: stale per-asset checkpoint reused although the raw data changed
  ```
- Expected: a checkpoint written from shorter raw data is detected and recomputed. Actual: returned as is; newer samples missing.
- Cause: `_checkpoint_fingerprint` hashes configuration only; nothing about the raw file (rows, last timestamp) or the phase-2
  archives (funding/OI/metrics/15m content). Also absent: `reg_target_clip`, `custom_settings` parameters.
- Fix: store in each `<coin>.meta.json` a data signature (n_rows, first/last index, hash of the last ~256 closes, and mtime+size
  of every phase-2 archive file used for that coin) and compare on load; a mismatch recomputes that coin. Loading the raw file
  first is cheap next to indicators.

### R2-05 — no sealed holdout: the last data months are part of test and are exported  [biases results / protocol]
- Repro: `python docs/research/audit/r2_05_no_sealed_holdout.py`
- Output:
  ```
  data ends 2026-09-20; last test sample 2026-09-20; test samples in the final 60 days: 540 of 2715
  AssertionError: DEFECT: no sealed holdout ...
  ```
- Expected (PROTOCOL.md): the last 2-3 months are seen by neither side. Actual: with exactly the documented config
  (`split_dates={train_end:2025-06-24, val_end:2025-11-21}`), test = everything after val_end+33h to the last candle; the panel run
  writes `signals_test*.csv.gz` for all of it and hourly_1h.md sends them to the evaluators.
- Fix: a default or documented-config `holdout_days` (>= 60) that `split_data` removes from test before the test dict is built
  (kept in a separate `holdout` output that no notebook cell reads); `split_asset_names`/panel paths must use the same cut.
  The repro passes for any implementation that makes the documented config exclude the final 60 days.

### R2-06 — a split without the `reg_target_scale` stamp is silently read as scale 1.0 by the panel path  [biases results, conditional]
- Repro: `python docs/research/audit/r2_06_panel_scale_unstamped.py`
- Output: `target_scale used = 1.0 | mu_high[0] = -0.4364 vs realised -0.0044 | pred_high[0] = 56.65 vs fut_high 100.09`
- Expected: error, or scale recovered from the data. Actual: exported `mu_*` are 100x the return and `pred_*` prices are wrong;
  AUC/IC are scale-free so nothing looks broken. `asym_score` (val_ic_asym, ic_asym) is also computed with the wrong scale.
- Path to this state: `split_data` drops metadata; only main.ipynb cell 8 stamps it. main cell 2 (`REPO_SOURCE="auto"`) uses a
  notebook copy under `/content/drive/MyDrive/crypto` when present, while cell 38 clones `cross_asset` from the branch when it is
  missing there: an older main + current cross_asset gives exactly this. Also any direct `panel_split_from(split_data(...))` use.
- Fix: in `PanelSplit.__init__`, derive the scale from `last_candles` (median of `yreg[:,i] / (future_i/last_i - 1)` on rows with
  |return| in (1e-4, 0.5); must be ~1 or a round number) and raise when it disagrees with `target_scale`; or refuse construction when
  the key is absent and `yreg` std > 0.3.

### R2-03 — three of the 43 features are dead availability flags  [minor]
- Repro: `python docs/research/audit/r2_03_dead_availability_flags.py`
- Output:
  ```
  FUND_available / OI_available / ITD_available / MET_available  covered +1.00 | uncovered -1.00  ok
  EFF_RATIO_available          kind=unit_0_1  covered +0.00 | uncovered +0.00  DEAD
  VWAP_DEVIATION_available     kind=unit_0_1  covered +0.00 | uncovered +0.00  DEAD
  VOL_CONC_HHI_available       kind=unit_0_1  covered +0.00 | uncovered +0.00  DEAD
  ```
- Cause: `FEATURE_KINDS` maps them to UNIT_0_1 (`EFF_RATIO_`, `VOL_CONC_`, explicit `VWAP_DEVIATION_available`); `process_windows`
  zeroes constant windows for every kind not in `NEVER_ZEROED_KINDS`, and a flag is constant in nearly every window.
- Fix: classify the three as BINARY_FLAG (explicit entries; the explicit `VWAP_DEVIATION_available` line is what keeps it UNIT_0_1).
  They are duplicates of `ITD_available` in coverage terms; dropping them from the feature list is equally valid (43 -> 40).

### R2-04 — day-level feature `ITD_TRADES_LOG` (CUMULATIVE) loses the size of the step and saturates  [minor]
- Repro: `python docs/research/audit/r2_04_step_feature_saturation.py`
- Output:
  ```
  step on 9/32 rows: max|z| tiny step (1e-3) = 1.00 | big step (1.0) = 1.00
  step on 5/32 rows: max|z| tiny step (1e-3) = 5.00 | big step (1.0) = 5.00
  dataset: fraction of ITD_TRADES_LOG values at the +-5 clip = 0.051; windows with any clipped value = 0.29
  ```
- Cause: within-window IQR standardisation of a step function (median distinct values per window = 2): IQR = the step (output exactly
  1) or ~0 (1e-8 floor, output at the clip). `VWAP_DEVIATION` (SIGN_ROBUST, day-level) has the milder version (2.8% clipped).
- Fix: drop `ITD_TRADES_LOG` (`ITD_TRADES_Z` already carries it against the coin's own history) or classify it as an absolute kind
  (e.g. ZSCORE of log-count minus a 30-day mean computed in the archive step). Day-level features should not use window-relative kinds.

### R2-02 — input windows may straddle a data hole  [minor]
- Repro: `python docs/research/audit/r2_02_window_spans_data_hole.py`
- Output: `raw hole: 3 h; frame hole: 23h; samples 165; windows spanning the hole: 3` (e.g. ts 2025-02-03 08:00, window starts
  2025-02-01 03:00: spans 53 h instead of 31).
- Expected: every kept sample's window is 32 consecutive hours. Actual: R1-03 made the target contiguous only; the window is cut by
  row position, and indicator warm-up deletes ~20-50 rows after a hole, so the first grid samples after it mix pre- and post-hole rows.
- Fix: in `prepare_single_asset` add to the existing `keep` test `base_index[end_idx] - base_index[end_idx - win + 1] == (win-1)*bar`.
  Cost: ~4 samples per hole (also removes them from cross-sectional groups, which is fine).

### R2-07 — Colab instructions point to a results folder the notebook does not create  [minor]
- Repro: `python docs/research/audit/r2_07_docs_panel_dir_mismatch.py`
- Output: `notebook: .../training_runs/crypto_model_v1_s100_am_panel` vs `docs: .../training_runs/crypto_model_v1_am_panel`
- Effect: the `cd ... && zip` step in hourly_1h.md fails (loud), and "delete the old `..._panel` folder" can be applied to the wrong name.
- Fix: change the docs (hourly_1h.md lines 224 and 226, «ما يُرسَل للتقييم المحلي») to `crypto_model_v1_s100_am_panel`.

---

## Checked and clean (runnable: `r2_00`, `r2_08`, `r2_09`)

1. Look-ahead, all 43 features (`r2_00` part 1): variant B replaces every source after the knowable boundary T0 (1h and 15m klines of all
   coins incl. the BTC reference and the breadth/orth-momentum universe, funding, OI, futures_metrics). X of the 1296 samples with
   ts <= T0 (9 at T0 itself) is bit-identical; 711 later samples differ in every feature family, so the check reaches each family.
   Harness power confirmed by injecting a 1 h OI/metrics availability leak (`infer_period -> 0`): caught on OI_chg_1, LSR_*, TAKER_LSR_1d.
   Holds for the ISO-text timestamp format of the fetch script.
2. Availability masks (`r2_00` part 2): coin with no archives -> all four BINARY_FLAGs at -1 and every archive value exactly 0; coin
   whose futures_metrics start later flips MET_available -1 -> +1 at the start. (Exception: the 3 slot flags of R2-03.)
3. Scale round trip (`r2_00` part 3): pipeline dataset (scale 100) -> split_data -> main-style stamp -> PanelSplit -> oracle output ->
   `export_signals`: `pred_high/pred_low == realised next high/low`, `mu_high == fut_high/last_high - 1`, `summarize` AUC(high/low) = 1.0.
   Divided exactly once on this path (`PanelTrainer.predict` multiplies by `reg_scale`; `export_signals`/`asym_score` divide by `target_scale`).
4. Labels: target candle is ts+1h (R1-03 re-verified, `repro_03` PASS); `y_*_class` 1/0 read as `> 0` by PanelSplit; `reg`: clip(+-1) then x100.
5. Grid/groups: all samples on 8h grid, `group_ns_for("1h", 8)` = 8 h, `check()` ok, no mixed-timestamp groups (R1-05 re-verified).
6. Purge: split_dates 33 h gap leaves the first val window strictly after the last train target candle (window start > train_end + 2 h).
7. `min_coins` partition covers every sample once per epoch (R1-04, `repro_04` PASS); resume refuses other data (R1-02, `repro_02` PASS);
   AUC labels (R1-01, `repro_01` PASS).
8. Early stopping / best-epoch restore (`r2_09`): recorded best epoch = argmin val_loss (not last); after `load_best()` the live weights
   reproduce the recorded best val_loss (2.2154 = 2.2154); `predict` does not swap EMA a second time; selection uses val only; test is
   read only after `load_best`.
9. Integration smoke (`r2_08`): 43-feature dataset -> real encoder (`build_model_fn` + ANTI_MEMORIZATION_CONFIG, heads high/low) ->
   `run_panel_experiment(A_ic_k, k_eval=(5, None))` runs; exported columns complete, no close head anywhere, resume with identical
   data reuses state.
10. Existing suites: `python -m unittest discover -s tests` -> 38 tests OK; `repro_00..05` all PASS at b14b9bb.
11. `SUSPENDED_TARGETS=("close",)`: IC-loss target moves consistently to `low` (`_ic_index`, `PanelSplit.yrank`, `evaluate`); no close key
    is required anywhere in the panel path; `report.summarize` returns AUC rows only (other panel_compare rows are NaN by design).
12. Read but not executed (main path, not used by the panel run): `wst_`/`mu_` division in `collect_signals`, chicks `entry / reg_scale`
    inverse, `retarget_splits` scaling: each divides/multiplies once.

## Suspicions (no failing repro)

- S1: `ic_asym` contains the last candle's wick asymmetry (`log(H/C-1+1e-3) - log(1-L/C+1e-3)` plus small `mu` terms). No baseline
  with `mu=0` (or a last-step GBM) is reported for `ic_asym`; the pre-registered criteria (`A_ic - B_ic > 0.01`) could be met or missed by
  the shape term alone. Suggested: add column `ic_asym_mu0` to `evaluate_k_coins` and require model > mu0. Needs real data to prove.
- S2: multiple testing: 3 variants x 2 seeds x k in (5,10,20,all) x targets x metrics, plus rounds, with no trial counter or correction; the
  0.01 threshold in the criteria is ~2.7 standard errors of a mean IC over ~900 groups (sd ~0.11/sqrt(n)), before autocorrelation.
- S3: IC t-stats (`report._t`) assume independent groups; groups 8 h apart share 75% of input windows and volatility regimes (no Newey-West).
- S4: real-archive semantics (OI/metrics stamped at hour start carrying the last 5m snapshot; funding stamped at settlement; kline `datetime_utc`
  = open time) are taken from `tools/intraday_features.py` and the fetch script; verified only against my synthetic files in those formats.
- S5: 83 coins are the currently listed universe (`registry status TRADING`): survivorship in the cross-section and in `MKT_BREADTH_24`.
  Also the metrics archive starts 2024-01-01 (module docstring), so most of train (2020-2023) has LSR_*/TAKER_LSR_1d/MET_available at
  0 while val/test have them: train/val/test distribution shift on those columns.
- S6: operational: (a) `split_dates` is not in the repo's cell 5, only in docs; forgetting it silently falls back to `kept_share` 80/10/10
  (the printed cut dates are the only sign); (b) if `tools/intraday_features.py` is not found or the archive folders are empty the build
  silently drops the phase-2 columns (23 features when both toggles resolve to off; only a printed line); suggest `assert len(feature_order) == 43` in the Colab cell; (c)
  `save_data_to_drive` writes the same ~2.5 GB gzip twice (versioned + latest, level 9, measured 24 MB/s -> ~3.5 min per copy, ratio 0.93) and
  `estimate_hourly_feasibility` reads every CSV once more from Drive; ~8 min of avoidable time, not a failure.

## Files
`_r2_synth.py` (helper), `r2_00_clean_checks.py`, `r2_01..r2_07_*.py` (defects), `r2_08_panel_integration_smoke.py`, `r2_09_best_epoch_restore.py`.
