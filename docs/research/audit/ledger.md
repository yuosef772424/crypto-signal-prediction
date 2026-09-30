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
| R2-01 | per-asset pipeline checkpoint reused after raw data grew (fingerprint ignores data) | fixed (a23e02e) | [r2_01](r2_01_stale_pipeline_checkpoint.py), `tests/test_audit_round2.py::StaleCheckpointTests`, pipeline self-test `t_build_dataset_from_loader_checkpoint_resume_end_to_end` |
| R2-02 | input window can straddle a data hole (only the target was made contiguous) | fixed (a23e02e) | [r2_02](r2_02_window_spans_data_hole.py), `WindowContiguityTests`, pipeline self-test `t_label_target_candle_is_ts_plus_horizon_across_gap` |
| R2-03 | EFF_RATIO_available, VWAP_DEVIATION_available, VOL_CONC_HHI_available are dead (UNIT_0_1 zeroed when constant) | fixed (a23e02e) | [r2_03](r2_03_dead_availability_flags.py), `NormalizationKindTests::test_availability_flags_separate_covered_from_uncovered` |
| R2-04 | ITD_TRADES_LOG (CUMULATIVE, day-level) loses step size, 5% of values at clip | fixed (a23e02e) | [r2_04](r2_04_step_feature_saturation.py), `NormalizationKindTests::test_step_feature_keeps_step_size` |
| R2-05 | no sealed holdout: last 60 days inside test and exported | fixed (a23e02e) | [r2_05](r2_05_no_sealed_holdout.py), `SealedHoldoutTests` |
| R2-06 | split without reg_target_scale stamp read as 1.0 by panel path (mu/pred 100x off, silent) | fixed (a23e02e) | [r2_06](r2_06_panel_scale_unstamped.py), `tests/test_cross_asset.py::AuditRound2Tests` |
| R2-07 | docs results folder (crypto_model_v1_am_panel) != notebook (crypto_model_v1_s100_am_panel) | fixed (a23e02e) | [r2_07](r2_07_docs_panel_dir_mismatch.py), `DocsPanelDirTests` |
| R2-08 | 43-feature pipeline output -> real encoder -> run_panel_experiment (A_ic_k, k-eval, resume) runs and exports correct columns | proven | [r2_08](r2_08_panel_integration_smoke.py) |
| R2-09 | best-epoch restore reproduces recorded best val_loss; best != last | proven | [r2_09](r2_09_best_epoch_restore.py) |
| R2-S1 | ic_asym may be reproduced by last-candle wick asymmetry (no mu=0 baseline) | open | [round2.md S1](round2.md) |
| R2-S2 | no multiple-testing counter; 0.01 criterion ~2.7 SE | open | [round2.md S2](round2.md) |
| R2-S3 | IC t-stat assumes independent groups (75% input overlap, vol regimes) | open | [round2.md S3](round2.md) |
| R2-S4 | real archive timestamp semantics unverified on real files | open | [round2.md S4](round2.md) |
| R2-S5 | survivorship in 83-coin universe; most of train has metrics features at 0 | open | [round2.md S5](round2.md) |
| R2-S6 | operational: split_dates only in docs, silent phase-2 feature fallback, duplicate 2.5 GB gzip | open | [round2.md S6](round2.md) |
| R3-00 | 1h+4h at f2442be: no look-ahead through 4h (every phase of t mod 4h, on/between 4h and 8h boundaries, mid-bar; 4h indicators, MKT_*, breadth, MOM_ORTH_NATR, 15m/funding/OI/metrics all replaced from T on); 4h window = 32 consecutive closed bars with close <= t <= t+1h; 4h normalised by its own window; 129h purge clean on the shipped stride-8 grid; float16 finite, <=5, err 1.95e-3 (also with volume x1e12, trades x1e9, OI x1e9) | proven | [r3_00](r3_00_clean_checks.py), [r3_00b](r3_00b_no_lookahead_perturbation.py) |
| R3-00c | 1h and 4h gathered with the same indices in train batches, eval/collect_signals, pooled chicks batch, PanelSplit (shuffled, k-coin, chunk_groups, subset); float16 upcast by the model boundary (0.0 diff) | proven | [r3_00c](r3_00c_gather_alignment.py) |
| R3-00d | single-TF and legacy 1h+4h output byte-identical to 4cb2d7b (4 configs, dataset + split + embargo); tests/golden_pre_multi_tf.json digests match | proven | [r3_00d](r3_00d_single_tf_unchanged.py) |
| R3-01 | 129h purge assumes t mod 4h = 0: at stride 1/5 (not a multiple of 4) val/test windows contain train/val target candles | proven (shipped stride 8 grid is clean) | [r3_01](r3_01_purge_short_off_4h_phase.py) |
| R3-02 | legacy 1h+4h dataset (accepted by main with a print-only warning) is split with a 33h purge while the 4h window spans 128h: 12 of 80 val windows contain train targets | proven | [r3_02](r3_02_legacy_two_tf_embargo_33h.py) |
| R3-03 | audit_normalization on float16 X returns std=inf for 23/43 features and flips 23 verdicts vs the float32 audit | proven | [r3_03](r3_03_audit_normalization_float16_std_inf.py) |
| R3-S1 | float16 storage has no guard if clip_abs is falsy (values > 65504 become inf silently) | open (unproven; shipped clip_abs=5) | [round3.md S1](round3.md) |
| R3-S2 | live path (build_dataset_live, dummy zero-volume tail) not exercised with higher_tf_mode='closed' | open | [round3.md S2](round3.md) |
| R3-S3 | availability flags (*_available) constant 1 in the synthetic universe, so flag look-ahead not exercised in R3-00b | open | [round3.md S3](round3.md) |
