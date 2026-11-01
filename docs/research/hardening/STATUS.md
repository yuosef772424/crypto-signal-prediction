# Hardening — STATUS (handoff for a fresh session)

Updated 2026-11-01 (end of session; handoff). A new session starts here: read CLAUDE.md + this file only, then open other files via MAP.md and line ranges.
Plan of record: `audit_2026-10-31.md` (8 root-cause classes C1-C8, PR sequence §5). Reply to the owner in Arabic.

## Done
| Batch | PR | What | State |
|---|---|---|---|
| 1 | #57 | docs (audit, proposals, external prompts); PR1 `core/registry.py` + `tools/check_patterns.py` ratchet; PR2 pooled evaluation predict (O(1) model calls, ~16x); PR3 strict `trainer.build_config`/`validate_config`; PR4 run identity (`identity.json`, resume refuses a changed objective), `TrainSettings.seed`, `core/seeds.py` (framework seeder injected, core stays stdlib) | merged |
| 2 | #58 | PR5 TASK/HEAD/NORMALIZERS/SAMPLE_FILTER registries -> `Registry` (duplicate raises); PR6 `core/roles.py` split roles + role-checked `Calibrator` (`evaluation/calibrators.py`, refuses train/holdout/test/unstamped); PR7 `core/heads.py` HeadSpec/HeadSet (one head vocabulary, views derived); PR8 output-name literals migrated; PR9 `train.loss` plan (`core/loss_plan.py`: LOSS_BALANCERS static/kendall, SELECTION_METRICS, schedules) | open, branch `claude/crypto-project-hardening-yptgm9`; check CI then owner merges |
| 4 | #58 (same branch) | E-tgt-003 code: TargetCodec registry (return/volnorm/volret/relative_median; volnorm now decodes), target_mode `volret`, BestModelTracker EMA fix (ema_weights snapshot re-validated on val; was restoring unvalidated weights under the anti_memorization preset), LOSS_BALANCERS `ema_norm`, SAMPLE_WEIGHTINGS curriculum_snr/curriculum_random/jump_downweight (`a+b` products), utility aux term (`loss.utility`, ramped), `beta_nll` head term, TCP confidence head (`tcp_head` + `loss.tcp`, Brier on stop_gradient), `mu_skip="unc_gated"`, SIGMA_CALIBRATORS `sigma_scale`, event features (`custom_settings.event_features`), return-space IC/AUC/MAE + persistence MAE in evaluation/integrity.py, anti-fear diagnostics + crps_gaussian + confidence_quality in stats/metrics.py; `tools/bracket_path.py` (path-resolved 1:1 TP/SL backtest, finer-bar touch order, random-direction null, confidence deciles); level side-channel `data/level_channel.py` (level_v1, train-only fit); docs `data_prep_audit.md`, `open_defects.md` (21 items); card `docs/research/cards/E-tgt-003.md` (arms S0-S6, 5 seeds) | pushed; CI pending at handoff |

Guards now in CI (all inside `python tools/check_deps.py`): pattern ratchet (counts only shrink), golden files
`tests/golden_trainer_configs.json`, `golden_head_vocabulary.json`, `golden_registry_contents.json`, `golden_calibrator_outputs.json`,
`golden_head_consumers.json`. Ratchet now: output-name-literal 32 (only trainer/example.py, smoke_test.py — generic examples),
hlc-target-tuple 27, config-get-default 140, target-mode-branch 51, seed-literal 30, metric-def-outside-stats 19, per-asset loop 1 (false positive).
Last full local suite: 963 passed (before PR4); CI green on batch 1 head and on PR5 head.

| 3 | #58 (same branch) | PR10 `stats/metrics.py` (one impl per metric, explicit NaN/tie policy; 14 consumers migrated byte-identical, ratchet metric-def 19 -> 12; `evaluate_heads` owner policy; trading metrics QSS/coverage/Winkler/profit factor/net Sharpe/threshold_trades); E-tgt-002 code (HEAD_LOSS_TERMS ic_weight/var_weight/huber via `train.loss.head_terms`, `val_<head>_ic`, `TargetSettings.clip`, SAMPLE_WEIGHTINGS abs_target_clip) + card; PR11 PredictionTable (`core/prediction_schema.py`, `stats/prediction_table.py`, opt-in `prediction_table=` in evaluation); PR12 `label_end` on splits, `SplitPlan` (data/split_plan.py, golden `tests/golden_split_arrays.json`), opt-in calibration split (`data.calibration_split={"source":"train_tail"|"val_head","share":..}` + `evaluation.calibration_source="calibration"`, purity tests), `trainer/kfold.py` removed (unused, coin blocks not periods) | open; CI green up to f112394 |

## Next (start of the NEXT session) — FIX, do not just document
Owner rule (2026-11-01): a known code defect is FIXED with a test, never only recorded. Owner cannot audit the repo himself.
1. Fix the 21 items in `open_defects.md` (top: unfitted dataset accepted by trainer — check the stamp; `data/normalize.py` DEFAULT_KIND silently CUMULATIVE -> raise; `stale_bars` default off keeps exact-zero returns; collect_signals/export_signals outside PredictionTable; val used for both selection and calibration; cross_sectional_normalize has no inverse; per_asset split leak with unequal histories). One `fix` commit each, defaults that change behaviour get an explicit note.
2. Wire what exists but is not wired into the evaluation run: with/without-jump metrics, anti-fear diagnostics (pure functions in stats/metrics.py), per-epoch weight snapshots + return-space comparison (owner: best vs worst epoch must differ — price-level IC hid it).
3. Dedicated Gaussian/Student-t heads (only beta_nll term exists); main.ipynb still calls `make_splits` (switch to `make_splits_with_calibration` when calibration_split is set).
4. Owner experiments on Colab (sync main.ipynb from this branch first; old notebook copies miss `make_datasets(..., settings=settings)`):
   - E-tgt-002 (A-D x 3 seeds), snippet in `docs/research/cards/E-tgt-002.md`.
   - E-tgt-003 (S0-S6, 10 arms x 5 seeds = 50 runs), snippet in `docs/research/cards/E-tgt-003.md`; evaluate every run also with `tools/bracket_path.py` (1:1 TP/SL, hold 6 bars, sizing chosen on val, confidence deciles).
5. Later (planned, not started): convert the shared-namespace packages to normal imports (95 implicit refs in tools/implicit_deps_allowlist.txt; order model/trainer -> workflow/evaluation -> data; 3-5 sessions); a fast test subset (~2 min) for agents; chart view of `yuosef772424/bot` trades; PR13 remainder (other target-mode branches, ratchet target-mode-branch); config-get-default (140).
Owner critique to keep in mind: past "no edge" verdicts were reached with defective tooling (unsupervised confidence, NIG floors binding ~96 %, EMA/best mismatch, price-level IC ~0.997 for every checkpoint, per-window normalization erasing level; 51/85 failures closed with unverified cause). Do not blame the data before the pipeline is proven.

## Open questions / critiques
- "Confidence/uncertainty trained on train data": calibrators can no longer be fit on train (PR6). Still open: no separate calibration split (PR12), and the model's confidence head trains on in-sample errors — needs an experiment, not a refactor.
- Proposals recorded, NOT adopted (no cost until adopted): `proposal_cascaded_stages.md`, `proposal_rolling_retrain.md`. Seams they need: PR4 (parent_hashes reserved), PR11, PR12.
- Adaptive-trainer research (owner, incomplete) and a second multi-output research prompt are coming: treat as suggestions; seams exist in LOSS_BALANCERS / SELECTION_METRICS / HeadSpec roles.

## External research (Perplexity)
- Prompts: `external_research_prompts.md` (+ `external_excluded_symbols.txt`, 183 sealed coins; data must end 2026-09-30).
- Prompt 1 results (repo `yuosef772424/crypto-decay`, commit 5614cfa) audited in `external_decay_audit.md`: sealed coins NOT excluded (126/183 present), its sealed period used before selection, "sign-flip profits" = static long basket (stablecoins + majors), dollar-neutral negative; monthly retrain IC +0.013 real but ~0 net; staleness not shown (expanding beats train-once, sliding does not). No experiment card justified.
- OWNER DECISION 2026-10-31: do NOT record anything in `holdout.json` for now. Still: do not base any decision on these results.
- Next prompts must demand visible exclusion code + a hard-fail check printing the excluded count.

## Working rules agreed with the owner
- Manager role: delegate to `model: "sonnet"` builders in worktrees; narrow prompts (files, line ranges, proofs, report format); integrate by cherry-pick; regenerate maps on map conflicts; run targeted tests locally, full suite via CI.
- Batches of PRs on the single branch above (one change type per commit) to avoid MAP.md conflicts; after a merge restart the branch from master.
- Cost: per-agent cache writes are not shared; prefer resuming the same agent for adjacent work, fewer/larger tasks, and fresh sessions from this file over long conversations.
- Cost (2026-11-01): resume of a paused agent re-sends its whole context (300-380k tokens) after the cache expires — prefer a FRESH agent pointed at the old worktree + its git log; tell agents: targeted tests only, Grep+Read ranges, no exploratory refactors. Batch the owner's new requirements for the next agent instead of messaging a running one.
