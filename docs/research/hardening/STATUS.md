# Hardening — STATUS (handoff for a fresh session)

Updated 2026-10-31. A new session starts here: read CLAUDE.md + this file only, then open other files via MAP.md and line ranges.
Plan of record: `audit_2026-10-31.md` (8 root-cause classes C1-C8, PR sequence §5). Reply to the owner in Arabic.

## Done
| Batch | PR | What | State |
|---|---|---|---|
| 1 | #57 | docs (audit, proposals, external prompts); PR1 `core/registry.py` + `tools/check_patterns.py` ratchet; PR2 pooled evaluation predict (O(1) model calls, ~16x); PR3 strict `trainer.build_config`/`validate_config`; PR4 run identity (`identity.json`, resume refuses a changed objective), `TrainSettings.seed`, `core/seeds.py` (framework seeder injected, core stays stdlib) | merged |
| 2 | #58 | PR5 TASK/HEAD/NORMALIZERS/SAMPLE_FILTER registries -> `Registry` (duplicate raises); PR6 `core/roles.py` split roles + role-checked `Calibrator` (`evaluation/calibrators.py`, refuses train/holdout/test/unstamped); PR7 `core/heads.py` HeadSpec/HeadSet (one head vocabulary, views derived); PR8 output-name literals migrated; PR9 `train.loss` plan (`core/loss_plan.py`: LOSS_BALANCERS static/kendall, SELECTION_METRICS, schedules) | open, branch `claude/crypto-project-hardening-yptgm9`; check CI then owner merges |

Guards now in CI (all inside `python tools/check_deps.py`): pattern ratchet (counts only shrink), golden files
`tests/golden_trainer_configs.json`, `golden_head_vocabulary.json`, `golden_registry_contents.json`, `golden_calibrator_outputs.json`,
`golden_head_consumers.json`. Ratchet now: output-name-literal 32 (only trainer/example.py, smoke_test.py — generic examples),
hlc-target-tuple 27, config-get-default 140, target-mode-branch 51, seed-literal 30, metric-def-outside-stats 19, per-asset loop 1 (false positive).
Last full local suite: 963 passed (before PR4); CI green on batch 1 head and on PR5 head.

## Next (audit §5)
1. PR10 `stats/metrics.py` — one implementation per metric (rank-IC x5, auc x2), migrate one `refactor` commit per consumer (C7; shrinks metric-def ratchet). Add QSS / quantile coverage / net Sharpe with turnover here (suggested by the cryptoformer digest).
2. PR11 `PredictionTable` schema + writer/validator (`prediction_made_at`, `label_available_at`, `train_end`, fold, identity) (C5).
3. PR12 `label_end` on samples + `SplitPlan` over data/split.py; a real CALIBRATION split carved with embargo (today `calibration_source="calibration"` raises NotImplementedError); purity test; remove/rebuild `trainer/kfold.py` (ignores `embargo`) (C6).
4. PR13 `TargetCodec` registry (C4; shrinks target-mode-branch). PR14 optional seams (sampler, callbacks).
5. config-get-default (140) is untouched: candidate for a PR after PR12.
Owner metric policy (2026-10-31, sent to the PR10 builder): direction/sign accuracy is computed ONLY for the close head (if present);
high/low (and close) are judged by rank-IC + closeness to truth (MAE/RMSE + skill score vs a naive no-change baseline) + interval
coverage (share of true targets inside the predicted [low, high] / quantile interval, reported with mean width, Winkler score).
Backlog (owner, later): chart view of the trades of the `yuosef772424/bot` project.
Small known items: Pandas4Warning `'d'` in `data/split.py:89`, `data/sources.py:291`, `data/align.py` (use `'D'`); STATUS pending cards T3/T5 touch loss/target code (land before PR13 or rebase).

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
