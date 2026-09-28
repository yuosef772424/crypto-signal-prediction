# Audit ledger

Evidence only (see [PROTOCOL.md](PROTOCOL.md)). Status: open / proven / rejected / fixed.

| id | claim | status | evidence |
|---|---|---|---|
| R1-00 | truncation, 8h grid, purge, last-step features are clean | proven | [repro_00](repro_00_clean_checks.py) |
| R1-01 | panel_compare AUC scored against a label other than the trained one | fixed (cda8d61) | [repro_01](repro_01_report_auc_label_mismatch.py), `tests/test_cross_asset.py::AuditRound1Tests::test_report_auc_uses_trained_label` |
| R1-02 | resume fingerprint ignores data content | fixed (cda8d61) | [repro_02](repro_02_resume_fingerprint_ignores_data.py), `AuditRound1Tests::test_resume_refuses_different_data` |
| R1-03 | 1h label taken from a later candle after a data gap | fixed (cda8d61) | [repro_03](repro_03_label_across_data_gap.py), pipeline self-test `t_label_target_candle_is_ts_plus_horizon_across_gap` |
| R1-04 | A_ic_k trains on ~56% of samples per epoch | fixed (cda8d61) | [repro_04](repro_04_a_ic_k_sees_half_the_data.py), `AuditRound1Tests::test_min_coins_covers_every_sample_each_epoch` |
| R1-05 | floor groups over unaligned timestamps leak future via attention; checks pass | fixed (cda8d61) | [repro_05](repro_05_floor_groups_leak_future_via_attention.py), `AuditRound1Tests::test_build_panel_splits_rejects_mixed_timestamps` |
| R1-05b | stride-32 1h panel results (hourly_1h.md §٤, panel direction claim) | open (invalidated, pending aligned 1h_s8 rerun) | [repro_05](repro_05_floor_groups_leak_future_via_attention.py), [hourly_1h.md §٤](../hourly_1h.md) |
| R2-00 | no look-ahead on all 43 features (15m, funding, OI, metrics, breadth, orth-mom, BTC ctx); masks work; scale round trip exact | proven | [r2_00](r2_00_clean_checks.py) |
| R2-01 | per-asset pipeline checkpoint reused after raw data grew (fingerprint ignores data) | proven | [r2_01](r2_01_stale_pipeline_checkpoint.py) |
| R2-02 | input window can straddle a data hole (only the target was made contiguous) | proven | [r2_02](r2_02_window_spans_data_hole.py) |
| R2-03 | EFF_RATIO_available, VWAP_DEVIATION_available, VOL_CONC_HHI_available are dead (UNIT_0_1 zeroed when constant) | proven | [r2_03](r2_03_dead_availability_flags.py) |
| R2-04 | ITD_TRADES_LOG (CUMULATIVE, day-level) loses step size, 5% of values at clip | proven | [r2_04](r2_04_step_feature_saturation.py) |
| R2-05 | no sealed holdout: last 60 days inside test and exported | proven | [r2_05](r2_05_no_sealed_holdout.py) |
| R2-06 | split without reg_target_scale stamp read as 1.0 by panel path (mu/pred 100x off, silent) | proven | [r2_06](r2_06_panel_scale_unstamped.py) |
| R2-07 | docs results folder (crypto_model_v1_am_panel) != notebook (crypto_model_v1_s100_am_panel) | proven | [r2_07](r2_07_docs_panel_dir_mismatch.py) |
| R2-08 | 43-feature pipeline output -> real encoder -> run_panel_experiment (A_ic_k, k-eval, resume) runs and exports correct columns | proven | [r2_08](r2_08_panel_integration_smoke.py) |
| R2-09 | best-epoch restore reproduces recorded best val_loss; best != last | proven | [r2_09](r2_09_best_epoch_restore.py) |
| R2-S1 | ic_asym may be reproduced by last-candle wick asymmetry (no mu=0 baseline) | open | [round2.md S1](round2.md) |
| R2-S2 | no multiple-testing counter; 0.01 criterion ~2.7 SE | open | [round2.md S2](round2.md) |
| R2-S3 | IC t-stat assumes independent groups (75% input overlap, vol regimes) | open | [round2.md S3](round2.md) |
| R2-S4 | real archive timestamp semantics unverified on real files | open | [round2.md S4](round2.md) |
| R2-S5 | survivorship in 83-coin universe; most of train has metrics features at 0 | open | [round2.md S5](round2.md) |
| R2-S6 | operational: split_dates only in docs, silent phase-2 feature fallback, duplicate 2.5 GB gzip | open | [round2.md S6](round2.md) |
