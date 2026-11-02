# Run profile 2026-11 (S03 evidence, S04 candidates)

Source: Colab job `profile-003` (T4, 2 CPUs, 12.7 GB RAM, `RunSettings` default except `train.epochs=2`, `batch_size=512`, high/low targets only;
`colab-results/runs/profile-003/`) and local measurements on synthetic data (4 CPUs). `profile-004` repeats it with the new defaults
(close target, prediction batch 2048). Numbers below are `profile-003` unless marked *local*.

## Where the time went (1886 s of run steps)
| step | s | share | note |
|---|---:|---:|---|
| report:verification | 456 | 24% | |
| report:chicks | 446 | 24% | |
| report:candle | 428 | 23% | |
| train_model (2 epochs) | 412 | 22% | 34 ms/step at batch 512 (29 steps/s); input pipeline 0.96 ms/batch (`input_bound_ratio` 0.028) |
| report:signals | 72 | 4% | |
| make_splits | 35.5 | 2% | RSS peak 9.4 GB of 12.7 GB |
| everything else | < 40 | 2% | |

The reports are 75% of a run; training is 22%.

## Done in this PR
- **Prediction batch 256 -> 2048** (`core.schema.PREDICT_BATCH_SIZE`, used by every `evaluation/` default and the candle baseline). *Local*:
  7120 rows through `model_outputs_batched`: 256 -> 7.06 s, 512 -> 4.20 s, 2048 -> 1.81 s (x3.9), 4096 -> 1.24 s; outputs identical (max abs diff 0.0).
  The training/validation batch (`TrainSettings.batch_size`) is untouched: `make_eval_dataset` uses `drop_remainder=True`, so changing it moves which
  validation rows are dropped and with them the best-epoch selection.

## Candidates for S04 (one PR each, with an equivalence proof)
Ranked by evidence. Share of a report that is model prediction, *local*, 40 synthetic coins: gap 52%, chicks 51%, signals 50%, verification 17%,
candle 10%, market_neutral 9% - so the 2048 batch helps gap/chicks/signals most and the three large reports (verification, candle, market_neutral) hardly.

1. **`_day_bootstrap_auc_diff` (workflow/candle_baseline.py:37)** - used by `verification` and `candle` (together 47% of a run). 300 bootstrap draws x 2
   `sklearn.roc_auc_score` calls per call, 4 calls per report; *local* cProfile: 18-20 s of a 23-26 s report, mostly sklearn input validation
   (`type_of_target`, `check_array`, `label_binarize`) per call. A rank-based AUC (`scipy.stats.rankdata`) gives the same value (*local*:
   |diff| <= 1.1e-16 with ties) at x15 (n=700) to x4 (n=20 000) the speed. Fix: one AUC function in the metrics component (S05 `stats/metrics.py`,
   audit class C7) and use it here; proof = the bootstrap output is allclose on the fixture. Expected: verification + candle 884 s -> roughly 250-400 s.
2. **`report:chicks` (446 s): pattern discovery from the model outputs** (`evaluation/patterns.py`, owner pointed here). *Local*, 342 604 rows on 2 CPUs
   (Colab): the 200-tree RandomForest used only to rank feature importance = 106 s per head (x2 heads = 212 s of the 446 s); decision tree 4 s,
   K-Means (n_init 10) 2.3 s, IsolationForest 2.1 s, PCA scatter 5.5 s. **Fixed in this PR**: `rf_max_samples` (default `PATTERN_RF_MAX_SAMPLES` = 50 000;
   `None` = old behaviour; tables up to the cap train the old forest exactly). *Local* at 342 k rows: x5 faster (11 s), max |d importance| 0.015, importances
   sum to 1; test `tests/test_pattern_rf_cap.py` (60 k rows, cap 20 k: max |d| 0.0089, same top-3 informative features). The rest of chicks:
   per-asset predict loops (*local*: 56 model calls for 40 coins) and a joblib pool started/stopped repeatedly (`_terminate_and_reset` 4.7 s of 49 s) - pool the
   assets (`predict_pooled_raw` exists) and reuse one pool.
3. **`_mn_period_returns` / `_mn_rank_weighted` (workflow/market_neutral.py)** - *local*: 24 of 32 s of the report in pandas per-period Series
   arithmetic (150 calls, 70 000 `DataFrame.__getitem__`). Vectorise by period. In `profile-003` this report ended after 14.5 s with
   `produced_output=False` (failed, reason lost from the 200 KB log tail): `profile-004` stores the output tail of a failed report in `profile.json`.
4. **Training step 34 ms is per-step overhead, not GPU or input** (same 31 ms on 4 CPUs at batch 64): the model has 93 586 parameters. Fewer, larger
   steps are the lever (batch 1024-2048 with the learning rate scaled) - this changes training, so it needs an experiment card and
   MODEL_DEV_PROTOCOL, not a refactor. Also: first step 9.2 s (tracing), epoch-2 "other" 11.2 s vs 0.8 s (epoch-end callbacks, probably the
   checkpoint save to Drive), validation 8-9 s per epoch.
5. **Memory: `make_splits` peaks at 9.4 GB (74% of RAM)** from 2.8 GB before it: transient copies of 2.74 M x 32 x 3 windows. Candidate for the
   `rolling_splits` `_take` full-copy item of the plan; a memory-mapped or index-based split removes it. S18 (slot panel data) needs this number.
6. **Environment**: the runner installs `tensorflow-cpu`, whose files win over the CUDA build (`gpu-diag-001`); `profile_run.install_cuda_tensorflow` fixes
   it per job (~2 min). A runner-level `gpu` job key is the permanent fix.
7. **`verification` printed "15 decode failures"** (profile-003 log): a correctness item, not speed; look at it before trusting the report.

## Baseline with the three targets: `profile-004` @ `d4cc437` (before the 2048 prediction batch and the forest cap)
T4, batch 512, 2 epochs, high+low+close (102 328 parameters). Run steps 2693 s (profile-003, high/low only: 1886 s).
| step | s | share |
|---|---:|---:|
| report:chicks | 776 | 28.8% |
| train_model (2 epochs) | 611 | 22.7% |
| report:verification | 544 | 20.2% |
| report:candle | 484 | 18.0% |
| report:market_neutral | 116 | 4.3% |
| report:signals | 98 | 3.6% |
| make_splits | 39 (RSS peak 9.4 GB) | 1.5% |
| report:gap | 12 | 0.4% |
- Training step 52.4 ms (19 steps/s) vs 34.3 ms with two targets: +53% for +9% parameters. The step is overhead-bound, so cost follows the number of
  heads/ops (hypothesis; same T4 type, not re-measured on one machine). `input_bound_ratio` 0.024.
- `chicks` grew 446 -> 776 s with the third head: the pattern-discovery forest runs once per head (3 x ~110 s on 342 k rows), consistent with the
  fix above. `market_neutral` now produces its output (116 s): it scores with `p_up_close` / `y_close_class` (workflow/market_neutral.py:225), so it
  needs the close head; with close suspended (profile-003) it failed after 14.5 s. It also costs 116 s now: the pandas per-period loops (candidate 3).
- `profile-005` (same config; adds the confidence/uncertainty check, tools/signal_diagnostics.py) runs the fixed code: prediction batch 2048, forest cap, output tails of failed reports; the before/after table goes here.
