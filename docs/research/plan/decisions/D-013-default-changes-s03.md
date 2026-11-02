# D-013: three default changes from the S03 profiling (close is a default target, prediction batch 2048, pattern-forest cap)

**Status:** Accepted (owner, 2026-11-02).

**Context.** S03 (`tools/profile_run.py`, PR #62; evidence in `docs/research/hardening/profile_2026-11.md`) measured a main run on Colab
(T4) and found where the time goes. Three defaults changed in the same PR by owner decision.

**Decision.**
1. **`close` is a trained target by default.** `TargetSettings.suspended_targets` defaults to `()` (was `None` = the old rule: close suspended unless `target_mode` starts with
   `entry_range`). `None` still means exactly the old rule and the old folder name; the new default adds the suffix `_susp-none` to the training folder.
   Cost measured: `profile-004` (high+low+close) 2693 s of run steps vs 1886 s for `profile-003` (high/low only); training step 52.4 ms vs 34.3 ms.
   `market_neutral` needs the close head (`p_up_close`); with close suspended it failed after 14.5 s.
2. **`core.schema.PREDICT_BATCH_SIZE = 2048`** (was 256) for every `evaluation/` default and the candle baseline. Outputs are identical on CPU
   (max abs diff 0.0 on 7120 rows); 3.9x faster on the model call. The training/validation batch (`TrainSettings.batch_size`) is untouched.
3. **`rf_max_samples = 50 000`** in `evaluation/patterns.py` (`PATTERN_RF_MAX_SAMPLES`; `None` = old behaviour). Tables up to the cap train the
   old forest exactly; above it the feature-importance forest is fitted on a 50 000-row sample: max |d importance| 0.015 at 342 k rows
   (x5 faster, 11 s instead of 106 s per head), 0.0089 and the same top-3 features in `tests/test_pattern_rf_cap.py`.

**What this breaks.**
- Resuming or evaluating a model trained under the old default (the `entry_range` model with high/low heads only, folder without `_susp-none`)
  needs `suspended_targets=None` in `RunSettings`; otherwise the target set (and the folder name) differs and the checkpoint does not load.
- Pattern-importance tables of runs with more than 50 000 rows differ from past reports by up to 0.015 per feature (rankings of close features
  can swap); recorded past results are not edited. Pass `rf_max_samples=None` to reproduce an old report.
- Past run timings are not comparable with new ones without noting the three-target / batch / cap state (`profile-003` vs `-004` vs `-005`).

**Revisit if.** Step M3 compares `high`/`low` with and without the `close` head (same seeds, `docs/research/MODEL_DEV_PROTOCOL.md`); if `high`/`low`
get worse with `close` present, reconsider decision 1 (default back to `None`, the old rule, or a per-experiment choice).
Decisions 2 and 3: revisit if a report needs the exact old importance values or a GPU run shows batch 2048 changes outputs beyond float noise.
