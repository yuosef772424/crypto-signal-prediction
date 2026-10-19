---
id: E-norm-001
mechanism: representation
timeframe: 1h
target: direction;high-low
model_class: nig-timenet
reopens:
not_covered_by: F-0029=this card changes the DATA-side input normalization (price_norm_mode) and leaves the model-side InstanceNorm untouched, whereas F-0029 closed a model-side InstanceNorm change
---
# Experiment card — E-norm-001: train NIG-TimeNet with price_norm_mode=logret_std vs window_scale

Pre-registered 2026-10-19, BEFORE any run (NOT run). Rules: `docs/research/RESEARCH_RULES.md`,
`docs/research/MODEL_DEV_PROTOCOL.md`. Follows the static probe `research/norm_probe` (E-normprobe-1).

| Field | Value |
|---|---|
| Date / author | 2026-10-19 / builder session (phase N1) |
| Baseline (commit, checkpoint, dataset fingerprint) | master at the N1 branch point, `price_norm_mode='window_scale'` (byte-identical to the old pipeline, pinned by `tests/test_price_norm_mode.py`); same trimmed 1h dataset in both arms (`tools/trim_dead_tails.py` output), dataset fingerprint to be recorded in the run bundle |
| The ONE change | `CONFIG['price_norm_mode']`: `'window_scale'` (arm A, baseline) vs `'logret_std'` (arm B). In B the open/high/low/close input channels are per-bar log returns vs the previous close, standardized with TRAIN-only mean/std, clip +-5; the regression targets `y_{high,low,close}_reg` are `log(future_price / last_close)` standardized with TRAIN-only per-target mean/std; evaluation inverts them with `denormalize_price_targets`. Everything else (model, heads, losses, seeds, epochs, splits) identical |
| Failure being addressed (level, location, registry id) | representation level, model input: E-normprobe-1 measured input effective rank 0.057 under window_scale vs 0.318 for logret_std (V4) and +0.04 high/low IC for a linear probe; layer health reports show 2-12% effective rank. No registry id |
| Data / splits / holdout | trimmed data (dead tails removed) with `CONFIG['stale_bars'] = {'min_run_hours': 168}`; rows with `y_close_reg == 0` are NOT dropped; `global_time` split as the baseline; sealed holdout stays sealed |
| Seeds / budget | >= 3 seeds per arm (same seed list in both arms); epochs and hardware as the baseline's latest run, recorded in the bundle |

## Verified cause and mechanism
- Verified cause this change attacks: E-normprobe-1 (merged) localized part of the collapse at the INPUT, before any
  layer: effective rank 0.057 of the flattened window under window_scale (robust per-window centre/IQR on strongly
  co-moving O/H/L/C). That probe is linear and says nothing about the network; this experiment tests whether the network
  uses the extra rank.
- Why the change should fix it: log returns vs the previous close are stationary and nearly white, so windows spread over
  many directions (probe: rank 0.318, near-duplicate neighbourhoods fewer). Targets in the same log-return space remove
  the per-window reference shift between inputs and targets.
- Diagnostic that must move: first-layer / input effective rank ratio from the model health report
  (`model/diagnostics.py`, definition of E-normprobe-1): window_scale ~0.06 -> logret_std >= 0.2 (null: the same statistic
  across the 3 seeds of arm A and on a randomly initialised copy). Secondary: the gap of the model's high/low IC to the
  linear reference (probe IC 0.557 high / 0.549 low under window_scale) shrinks.
- Output effect expected on val: high/low IC (Spearman of predicted vs realised return) up by about 0.02-0.04; close
  direction stays at chance (AUC 0.51-0.53, F-0080 consistent).

## Accept / reject rule (val)
- Accept if: mean high AND low IC of B exceeds A by more than the larger seed spread (max-min over seeds of either
  arm) AND the input/first-layer effective rank rose to >= 0.2 (above the null above).
- Reject if: IC difference within seed spread or negative while the rank rose (then the rank was not the bottleneck: close
  the failure with a registry row), or the rank did not rise (implementation fault: check the normalization audit).
- Otherwise (IC up but the rank diagnostic did not move, or vice versa): "unexplained" (not adopted).
- Every price-level metric of B is computed on predictions passed through `denormalize_price_targets`; A is untouched.
- Unsupported in B (raises by design, so not part of this experiment): `price_norm_mode='logret_sigma'` with regression
  heads, `target_mode` other than the default, `reg_target_scale != 1`.

---
## Result (added after the run)
- Output metrics (mean +- seed spread):
- Diagnostic movement vs prediction (with null):
- Failure level and location, if failed:
- Verdict: accepted / rejected / unexplained — artifact links:
- If rejected: registry row `F-____` added (claim, scope tags, verified cause, invariants, reopen_if).

## Implementation note (N1, added after the feat commit; the pre-registered fields above are unchanged)
- Code: `data/normalize.py` (fit/apply/`denormalize_price_targets`), module card PURPOSE/PITFALLS describe the option. Stats are fitted on train in `split_data` / `rolling_splits` and stamped in `split['price_norm_meta']`; `split_holdout` needs `price_norm_meta=train['price_norm_meta']`.
- Evaluation: `workflow/run.py:prepare_chicks` builds specs with `TargetSpec.with_log_norm(mean, std)` when the test split carries `price_norm_meta`; `workflow/reports.py` passes the meta to `invert_reg_predictions`, which delegates to `denormalize_price_targets`.
- Raises in the new modes: `logret_sigma` with any `_reg` head, `logret_sigma` with more than one timeframe, `reg_target_mode != 'return'`, `reg_target_scale != 1`, `x_storage_dtype` with `logret_std`, `retarget_splits`, `cross_sectional_normalize`, `build_dataset_live`, `invert_reg_predictions` without meta.
- Colab job: none written. No existing `tools/*_colab_job.py` entry point trains a model; training runs in `main.ipynb` through `RunSettings`, so a job spec waits for a training entry point.
