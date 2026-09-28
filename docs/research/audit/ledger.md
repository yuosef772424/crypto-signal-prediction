# Audit ledger

Evidence only (see [PROTOCOL.md](PROTOCOL.md)). Status: open / proven / rejected / fixed.

| id | claim | status | evidence |
|---|---|---|---|
| R1-00 | truncation, 8h grid, purge, last-step features are clean | proven | [repro_00](repro_00_clean_checks.py) |
| R1-01 | panel_compare AUC scored against a label other than the trained one | fixed (COMMIT) | [repro_01](repro_01_report_auc_label_mismatch.py), `tests/test_cross_asset.py::AuditRound1Tests::test_report_auc_uses_trained_label` |
| R1-02 | resume fingerprint ignores data content | fixed (COMMIT) | [repro_02](repro_02_resume_fingerprint_ignores_data.py), `AuditRound1Tests::test_resume_refuses_different_data` |
| R1-03 | 1h label taken from a later candle after a data gap | fixed (COMMIT) | [repro_03](repro_03_label_across_data_gap.py), pipeline self-test `t_label_target_candle_is_ts_plus_horizon_across_gap` |
| R1-04 | A_ic_k trains on ~56% of samples per epoch | fixed (COMMIT) | [repro_04](repro_04_a_ic_k_sees_half_the_data.py), `AuditRound1Tests::test_min_coins_covers_every_sample_each_epoch` |
| R1-05 | floor groups over unaligned timestamps leak future via attention; checks pass | fixed (COMMIT) | [repro_05](repro_05_floor_groups_leak_future_via_attention.py), `AuditRound1Tests::test_build_panel_splits_rejects_mixed_timestamps` |
| R1-05b | stride-32 1h panel results (hourly_1h.md §٤, panel direction claim) | open (invalidated, pending aligned 1h_s8 rerun) | [repro_05](repro_05_floor_groups_leak_future_via_attention.py), [hourly_1h.md §٤](../hourly_1h.md) |
