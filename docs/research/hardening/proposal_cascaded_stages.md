# Proposal: cascaded / stacked training stages

Status: **PROPOSED, not adopted.** No cost (code, compute, cards) is to be spent on it until the owner adopts it and records a
decision. This file only states what the system must already allow so that the idea can later be added as NEW files, and
which of those seams the repo lacks today (audit of 2026-10-31, see `audit_2026-10-31.md`).

## The owner's idea (translated, 2026-10-31)

- **Stage 1** takes the ~40 input features and forecasts all ~40 features (next step).
- **Stage 2** takes the original inputs plus the Stage-1 forecasts and forecasts the same features again (a refinement).
- **Stage 3** (the more stable one) takes all of that and predicts only a few final outcomes (for example 3).

## What must already hold for "new files only"

A new stage must be addable as: one new module that registers itself, one new config section naming it, and no edit to
`data/`, `model/builder.py`, `trainer/trainer.py`, `workflow/run.py` or `evaluation/`. Five requirements, each with the repo's
current state (checked against code at commit `17e2b7b`, branch `claude/crypto-project-hardening-yptgm9`).

### (a) Heads and targets declared by config; "forecast all features" is just a target set

Required: a model's outputs are a named dict, and the set of heads, their labels and their losses come from one declaration, so
a target set of 40 feature-forecast heads is a config value.

Current state: **partly present, mostly missing.**
- Present: `trainer/` is already target-agnostic. `config["targets"]` maps any name to `true_key`, `task_type`, `output_keys`
  (`trainer/trainer.py:31-46`, `trainer/tasks.py:68` registry), and the model returns a dict (`model/builder.py:241`).
- Missing: the head/label vocabulary is hard-wired to price targets. `data/windows.py` builds only `y_{high|low|close}_{reg,class}`
  (`core/schema.py:26`); `model/builder.py:210-241` builds heads only for `price_targets` with `nig_regression` special-cased;
  `workflow/training_config.py:6-37` and `workflow/run.py:277-280` hard-code the two head kinds. There is no multi-output
  regression head type in `HEAD_REGISTRY` (`model/heads.py`: only the two classification heads), no way to say "the label of
  head X is column group Y of the feature tensor shifted by +h", and 40 heads would need 40 `y_<name>_nu/_alpha/_beta/...` keys.
  Audit class C1 (one head/target spec) closes this.

### (b) A stage contract: read the previous stage's prediction TABLE, never its internals

Required: a stage consumes a table keyed by `(timestamp, coin)` with a column `available_at` (when the value was known), and
produces such a table. Nothing else crosses the boundary.

Current state: **missing.**
- The only prediction table is `collect_signals` (`workflow/selective_eval.py:~29-88`, a second producer `export_signals` in
  `cross_asset/train.py:403`). It has `timestamp` (the window end), no `available_at`, no horizon, no stage/run id, no schema
  version; its columns are per-target ad-hoc names (`mu_<t>`, `wst_<t>`, `p_up_<t>`).
- There is no way to feed an external per-(timestamp, coin) column into the model input: input columns come only from
  `feature_order`, produced inside the pipeline from price data (`data/features.py`, `data/custom.py`).
  Audit class C5 (prediction/decision log schema) covers the table; a feature-source seam (a registered "external column
  provider" that is checked against `available_at <= window end`) is a follow-up PR inside C5, not a new class.

### (c) Out-of-fold predictions inside the walk-forward split; purge and embargo for every stage

Required: Stage 2 must never train on Stage-1 predictions made in-sample (stacking leakage). Stage-1 predictions used to train
Stage 2 must be out-of-fold, and the split gaps (purge for the label horizon, embargo for the window span) must hold between
every pair of (train, predict) sets of every stage.

Current state: **missing, and one trap.**
- `data/split.py` has correct time-based purge+embargo (`embargo_candles`, lines 52-96) and a walk-forward builder
  (`rolling_splits`, line 831), but nothing runs a model per fold and collects out-of-fold predictions.
- `trainer/kfold.py:16` `purged_walk_forward_splits` is the only fold-training loop; it ignores its `embargo` argument
  (never read), purges a fixed 50 sample indices (not a time), and knows nothing about window span, stride, horizon or the
  multi-asset pooled order. It has no caller (only a "not in namespace" test). Do not build stacking on it.
- Samples carry no `label_end` / `available_at`; the gap is derived from config. A stage whose label horizon differs from
  Stage 1's (forecasting features vs a 3-outcome stage) cannot state its own gap. Audit class C6 covers this.

### (d) Chained checkpoints and fingerprints: retraining Stage 1 invalidates Stage 2 and 3

Required: each stage's identity includes its parents' identity, and a stage refuses to run or resume if a parent changed.

Current state: **missing.** `config_fingerprint` (`trainer/config.py:238`) is used only for an in-session cache
(`trainer/system.py:149-157`); it is not written next to the checkpoint and not compared on disk resume (verified: a run resumed
with a changed `loss_weight`, no error). There is no parent link, no data fingerprint in the main path (only `cross_asset`
keeps one, `cross_asset/train.py:264-290`). Audit class C3 (run identity) covers this; the parent hash is one extra field
once a persisted identity exists.

### (e) Evaluate each stage independently against baselines

Required: score Stage 1, Stage 2 and Stage 3 on the same out-of-fold rows, each against a no-cascade model and trivial
baselines, with a dependence-aware comparison.

Current state: **mostly present once (b) exists.** `tools/paired_compare.py` (T2, per-timestamp paired differential, block
bootstrap) and the baselines of `workflow/generalization.py` / `workflow/capacity.py` already work on tables with a timestamp
column. What is missing is the common table (b) and metric definitions with a single owner (audit class C7: `_auc` and
rank-IC are defined 4+ times), so that "Stage 2 beats Stage 1" is computed by one code path.

## Summary of seams, mapped to audit classes

| Seam | Present? | Audit class that closes it |
|---|---|---|
| (a) config-declared heads/targets, multi-output feature-forecast head | partly (trainer yes, data/model no) | C1 |
| (b) stage table keyed (timestamp, coin) with `available_at`; external column provider | no | C5 |
| (c) OOF generation in walk-forward with purge/embargo per stage, per-sample `label_end` | no (kfold is a trap) | C6 |
| (d) chained, persisted fingerprints | no | C3 |
| (e) one metric code path, per-stage scoring | partly (`paired_compare`) | C7 + C5 |

## Not decided here

How many stages, whether Stage 2 sees raw inputs, the label for "forecast all features" (raw vs differenced vs normalized with
the Normalizer's `inverse`), and whether it beats a single model are research questions for an experiment card
(`docs/research/RESEARCH_RULES.md`, `failure_registry.csv`), not for this proposal.
