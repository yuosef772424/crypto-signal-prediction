# Open code / pipeline defects (read-only sweep, 2026-11-01)
Sources: module PITFALLS and TODOs, `docs/research/hardening/STATUS.md` leftovers, `docs/research/audit/ledger.md` (status open), `plan/STATUS.md`,
`tools/pattern_baseline.json`, `data_prep_audit.md`. Research verdicts and closed failures are NOT listed. sev: H/M/L (guess); size: S <1 day, M 1-3 days, L > 3.
Checked in code on this commit unless marked (ledger) = taken from the ledger without re-running. 21 items.

| # | Where | Defect | sev | size |
|---|---|---|---|---|
| 1 | `data/normalize.py` `DEFAULT_KIND` / `classify_feature` | an unclassified feature name silently becomes `CUMULATIVE` (per-window scaling); cause of the past VWAP_DEVIATION / SUPERT_DIR / MOM_RANK mis-scalings | M | S (raise on unknown prefix, list them) |
| 2 | `data/normalize.py` `process_windows` clip `clip_abs=5` | clips `close` in 2.1% of windows (breakouts: the largest moves lose their size), `volume` 0.5%, `BBB` 0.4% (data_prep_audit M5) | M | M (new option + card, default stays) |
| 3 | `data/normalizers.py`, `data/level_channel.py`; `workflow/run.py` `make_datasets` | an UNFITTED dataset (logret_std, level channel) is accepted by the trainer: nothing refuses X that is not stamped | M | S (check the stamp in `make_datasets` / `trainer`) |
| 4 | `data/defaults.py` `stale_bars: None` | exact-zero horizon returns stay in training (1.3% train on 10 liquid coins, 7.1% / 17.1% train / val on the 222-coin set) | M | S (default flip needs its own card) |
| 5 | `data/split.py` `split_data` | val is selection AND calibration unless `split_data_with_calibration` is used; the confidence head trains on in-sample errors | M | M |
| 6 | `data/split.py` `per_asset` mode | leaks when coins' histories differ in length (documented, still selectable) | L | S (raise unless lengths equal) |
| 7 | `workflow/run.py:200` `make_splits`, `main.ipynb` cell 3 | raises when `calibration_split` is set; the notebook still calls it | L | S |
| 8 | `workflow/selective_eval.py:31` `collect_signals`, `cross_asset/train.py:404` `export_signals` | not adapters over `PredictionTable`: a second signal-frame producer with no point-in-time columns | M | M |
| 9 | `data/live.py:59` | raw `"1d"` string reaches pandas (the Pandas4Warning class; `pandas_tf` exists) | L | S |
| 10 | `tools/natr_range.py` | own metric definitions (pattern ratchet `metric-def-outside-stats` 12); blocked by the sealed E-natr-004-confirm sha | L | S (owner re-seal first) |
| 11 | `trainer/tasks.py:142-185` | IC loss pooled over coins and time (batches carry no timestamp); selection IC is Pearson, rank-IC needs epoch-end computation (TODO line 169) | M | M |
| 12 | `tools/pattern_baseline.json` | frozen anti-patterns: config-get-default 140 (silent defaults), target-mode-branch 51, output-name-literal 32, seed-literal 30, hlc-target-tuple 27 | M | L (PR13 TargetCodec, then config PR) |
| 13 | `tools/implicit_deps_allowlist.txt` | 96 allowed implicit couplings of the shared-namespace packages (60 cross-package, 37 forward, 4 notebook-global, 3 duplicate defs) | L | L |
| 14 | `data/cross_sectional_norm.py` | the inverse needs the peers' realized targets at the same timestamp: not usable for a live prediction (read, not run) | M | M |
| 15 | model NIG head (`plan/STATUS.md` T3) | variance floor hit up to 96% on `y_high` (heavy-tailed standardized target); waiting for the owner's Arm-0 numbers | M | M |
| 16 | `docs/research/hourly_1h.md` panel claims (ledger R1-05b) | stride-32 1h panel results invalid; aligned 1h_s8 rerun pending | M | S (rerun) |
| 17 | `data/phase2.py`, ledger R2-S4 / R2-S6 (ledger) | real archive timestamp semantics unverified on real files; silent neutral fallback of phase-2 features | M | S-M |
| 18 | `data/windows.py` `x_storage_dtype` (ledger R3-S1) | float16 storage has no guard when `clip_abs` is falsy (values > 65504 become inf) | L | S |
| 19 | `data/live.py` (ledger R3-S2) | live path with the dummy zero-volume tail never exercised with `higher_tf_mode='closed'` | M | M |
| 20 | `data/funding_oi.py` `*_available` flags (ledger R3-S3) | flag look-ahead not exercised (flags constant 1 in the synthetic universe) | L | S |
| 21 | `workflow/capacity.py` | `tools/paired_compare.py` (dependence-aware comparison) not wired into the capacity step | L | S |

Top by severity x size: 3, 1, 4, 8, 16, 17, 2, 5, 19, 14.
