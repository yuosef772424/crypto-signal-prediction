---
id: E-norm-002
mechanism: representation
timeframe: 4h
target: high-low
model_class: nig-timenet
reopens:
not_covered_by: F-0029=this card REMOVES the per-window InstanceNorm on inputs that the data side already standardizes globally (price_norm_mode=logret_std), whereas F-0029 closed the default encoder that KEEPS InstanceNorm on window_scale inputs
---
# Experiment card — E-norm-002: logret_std inputs without the model's per-window InstanceNorm

Pre-registered 2026-10-20, BEFORE any run (NOT run). Rules: `docs/research/RESEARCH_RULES.md`,
`docs/research/MODEL_DEV_PROTOCOL.md`. Follows E-norm-001 (arm B = `logret_std`, one seed seen at the time of writing).

| Field | Value |
|---|---|
| Date / author | 2026-10-20 / planner session |
| Baseline (commit, checkpoint, dataset fingerprint) | master 10e749b; arm A = E-norm-001 arm B exactly (`price_norm_mode='logret_std'`, `RunSettings` default `ModelSettings.anti_memorization=True`, i.e. `MODEL_CONFIG` + `ANTI_MEMORIZATION_CONFIG`: GRU encoder, `stats_mode='symlog'`, `input_clip=4`, `level_passthrough=True`, `level_norm='batch'`, `use_instance_norm=True`); its >= 3 seed runs are reused as arm A, not re-run, if their bundles record the same settings as below |
| The ONE change | `MODEL_CONFIG['use_instance_norm']`: `True` (arm A) -> `False` (arm B). Existing model option; main.ipynb passes it through the new `ModelSettings.config_overrides` (default `{}` = identical run, separate feat PR), applied after `ANTI_MEMORIZATION_CONFIG`. The encoder then reads the globally standardized log returns directly (`z = inputs`) and the stats path (`stats_proj`, window mean / log std) is absent because the inputs already carry the window volatility. Verified on CPU at 10e749b: the model builds and runs with 3 inputs x 32 steps (866,206 vs 883,492 parameters). Everything else (data, splits, targets, heads, losses, trainer, epochs, seeds) identical to E-norm-001 arm B |
| Failure being addressed (level, location, registry id) | representation level, model input layer: in E-norm-001 arm B (seed 1) the linear probe of `y_high_reg` / `y_low_reg` at the `instance_norm` output was 0.058 / 0.104, vs 0.516 / 0.584 under `window_scale`; regression predictions far too narrow; NIG floor share 25% (high). No registry id yet (E-norm-001 not closed) |
| Data / splits / holdout | as E-norm-001 run by the owner: 4h, window 32, stride 32, `align_windows_to_grid`, targets high/low/close, horizon 1, indicators off (inputs H/L/C), trimmed data + `stale_bars={'min_run_hours':168}`, rows with `y_close_reg == 0` kept, `reg_target_mode='return'`, `reg_target_scale=1`, `global_time` split; sealed holdout stays sealed |
| Seeds / budget | the same >= 3 seeds as arm A; same epochs and hardware (Colab), recorded in the bundle |

## Verified cause and mechanism
- Cause this change attacks (hypothesis, from one seed): `logret_std` makes each bar's return comparable across coins and
  time, so the window's volatility level is in the input itself. `InstanceNorm` (`model/input_norm.py`, built in
  `model/builder.py` `_branch`) then divides every window by its own std per feature, which removes exactly that level
  (probe at its output 0.52 -> 0.06 for high). The level survives only through the side stats path, which the probe
  and the narrow regression outputs say the heads do not use enough. F-0029 found the same mechanism (InstanceNorm
  erases the feature level) on other inputs.
- Caveat known before the run: arm A already sends the raw inputs to the encoder through `level_passthrough`
  (SymLog + BatchNorm, concatenated at `shape_and_level`), so the volatility level may already reach it there.
  Pre-run gate (no training): read the probe of `y_high_reg` / `y_low_reg` at `shape_and_level` in the existing arm A
  health reports. If it is already >= 0.40 for both, the cause above is refuted (the level reaches the encoder and is
  not used) and this card is withdrawn before any run; record that in the Result section.
- Why the change should fix it: with the data already standardized on train statistics, the per-window rescaling is
  redundant; removing it lets the window volatility reach the decomposition, encoder and linear path directly.
- Diagnostic that must move: linear probe R of `y_high_reg` and `y_low_reg` on the `trend_seasonal` (decomp) output,
  the first layer present in both arms (model health report, same probe as E-norm-001): arm A value (to be recorded from
  its seeds, expected near the 0.06 / 0.10 seen at `instance_norm`) -> arm B >= 0.40 for both. Null: the same probe on a
  randomly initialised copy of each arm. Secondary: regression prediction std / realised target std rises toward 1
  (arm A seed 1: far below), NIG alpha-floor share of the high head falls below 25%.
- Output effect expected on val: high/low IC (Spearman of the denormalized prediction vs the realised return) up by
  about 0.02-0.05 vs arm A; close direction stays at chance (AUC 0.50-0.53, F-0080 consistent).

## Accept / reject rule (val)
- Accept if: mean val high AND low IC of B exceeds A by more than the larger seed spread (max-min over seeds of either
  arm) AND the `trend_seasonal` probe rose to >= 0.40 for both targets (above the random-init null).
- Reject if: IC difference within seed spread or negative while the probe rose (then the per-window rescaling was not
  the bottleneck: close with a registry row), or the probe did not rise (then the volatility level does not reach the
  encoder for another reason: localize before any new change).
- Otherwise (IC up but the probe did not move, or vice versa): "unexplained" (not adopted).
- All price-level metrics of both arms go through `denormalize_price_targets` (`test['price_norm_meta']`).
- Out of scope (later one-change cards, not mixed in here): InstanceNorm demean-only (keeps the window mean removal),
  stride 32 -> 4 (effective samples ~50 with the grid-aligned 4h windows), scaling the linear path before the add.

## Owner run settings (main.ipynb)
```
update_config(price_norm_mode="logret_std", stale_bars={"min_run_hours": 168},
              reg_target_mode="return", reg_target_scale=1.0, x_storage_dtype=None)
# model: the only difference from E-norm-001 arm B
RunSettings(..., model=ModelSettings(config_overrides={"use_instance_norm": False}))  # after the feat PR merges
```
Report per seed: val high/low IC, the model health report (probe at `trend_seasonal`, effective ranks, NIG floor,
prediction std), and the run bundle id.

---
## Result (added after the run)
- Output metrics (mean +- seed spread):
- Diagnostic movement vs prediction (with null):
- Failure level and location, if failed:
- Verdict: accepted / rejected / unexplained — artifact links:
- If rejected: registry row `F-____` added (claim, scope tags, verified cause, invariants, reopen_if).
