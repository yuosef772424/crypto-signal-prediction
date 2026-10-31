# Proposal: rolling-retrain study (does the market change enough to make the model random?)

Status: **PROPOSED, not adopted.** Nothing is built or run on the strength of this file. It records the owner's question
(2026-10-31) and what the code lacks to answer it by SETTINGS ONLY, so the missing seams are built once, as part of the
hardening classes, not as a one-off script.

## The owner's question (translated)

Models look excellent on short test windows but may become random over long windows. Prove whether the market changes enough
to make the model random: train on period A, test on B; then train on A+B (or slide the window), test on C; and so on.

## What the experiment needs

1. A retrain loop over rolling folds (expanding or sliding train), each fold with its own purge and embargo.
2. Per-fold prediction tables (one table per fold, same schema), so any metric can be recomputed later.
3. A staleness curve: metric as a function of distance between the prediction time and the end of the training data.
4. Train-once versus retrain-every-k: the same test period scored by a model frozen at the first cut and by models retrained
   every k periods.
5. Seed noise and a null: >= 3 seeds per arm (MODEL_DEV_PROTOCOL), a label-permutation control per fold, and a dependence-aware
   comparison (`tools/paired_compare.py` exists).
6. All of it driven by `RunSettings` plus one new section, with the run identity of every fold recorded.

## What exists (checked at `17e2b7b`)

| Piece | State |
|---|---|
| Fold schedule with purge+embargo | **Present**: `data/split.py:738 rolling_split_schedule`, `:831 rolling_splits` (expanding via `initial_train_span`, sliding via `train_span`, `step`, `max_windows`; embargo from `embargo_duration`; the price Normalizer is refit on each window's train, `data/split.py:896`) |
| Contract for a per-window "fit then predict" function | **Present for non-learned signals**: `signal_eval/integration.py:157 evaluate_hypothesis_over_rolling_windows(windows, predict_fn)` with `evaluate_windows` (`signal_eval/windows.py:22`) |
| Per-window metrics with null | partly: `signal_eval/windows.py` (IC with shuffles); `workflow/permutation_control.py` for the NN (single split only) |
| Dependence-aware paired comparison | **Present**: `tools/paired_compare.py` (T2) |

## What is missing (each folded into an audit class, `audit_2026-10-31.md`)

| Missing seam | Why it blocks the study | Class |
|---|---|---|
| A retrain loop that trains the neural model per fold. `trainer/kfold.py:16` is index-based, ignores `embargo`, has no caller; `workflow/run.py` steps are single-split (`make_splits` once); `main.ipynb` runs one split | no way to produce models per fold except by hand | C6 (one `SplitPlan` / fold object) + C3 |
| Run identity per fold (seed, fold id, data cut, parent) and a `run_dir` derived from it. `run_dir_for` (`workflow/settings.py:379`) has no fold or seed; resume is unchecked (verified) so a re-run of fold k can silently continue a different objective | folds would overwrite or mix, and multi-seed is not controllable (`TrainSettings` has no seed; the training dataset shuffle is unseeded, verified) | C3 |
| A standard prediction table with `prediction_made_at`, `label_available_at`, fold id, model id (`collect_signals`, `workflow/selective_eval.py`, has one `timestamp`) | the staleness curve needs "distance from train end" per row; `train_end` is not in any table | C5 |
| Out-of-fold / per-fold table writer with `*.meta.json` (commit, settings, data fingerprint, seed per PHILOSOPHY 5.1) | per-fold results cannot be re-scored later | C5 |
| Staleness-curve and train-once-vs-retrain-every-k evaluators | not present (no module computes a metric by distance from train end) | new file on top of C5/C7: one module consuming prediction tables, no edit elsewhere |
| One metric implementation (`_auc` defined 2x, rank-IC 5x) | a staleness curve compares the same metric across folds and arms: definitions must not differ per consumer | C7 |
| Batched evaluation predict (per-asset loop, `evaluation/all_assets.py:178-198`) | a study with F folds x S seeds x arms multiplies the per-asset predict cost (measured about 50x slower than one pooled call) | C5 (evaluation data flow) |
| Memory: `rolling_splits` materialises full (train, val, test) copies per window (`_take`) | expanding windows on the 800-coin Drive data may not fit in Colab RAM | C6 (index-based `SplitPlan`, views not copies) |
| Settings: no `RollingSettings` section in `RunSettings` | cannot run by settings only | C2 (a new strict section, defaults leave current behaviour untouched) |

## Design constraints for when it is adopted

- One new settings section and new modules only; `data/split.py` is wrapped, not rewritten (its embargo logic is the
  reference, covered by `tests/test_data_package.py` and the multi-timeframe golden).
- Each fold is a normal run (same `workflow/run.py` steps) with a fold-specific `RunIdentity`; the study driver only loops and
  collects tables.
- The sealed holdout and the holdout lock are never inside any fold (`tools/holdout_guard.py`); a time seal applies to the
  last cut.
- The study is an experiment: card and registry check per `docs/research/RESEARCH_RULES.md` and `failure_registry.csv`
  (F-0084 closed with localization = drift; check the registry for others before writing the card), predicted diagnostic and accept rule before the run.
- Not decided here: fold lengths, k, expanding vs sliding, the metric set, the number of seeds.
