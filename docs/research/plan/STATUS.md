# STATUS: research plan hand-off

> Read this first in every session (planner and build). Update it in the same PR that changes the state.
> Keep it under 60 lines.

**Phase:** T0 done; R0, G2, R3 merged. R3 1h REJECTED on H2 (E-natr-002, F-0083), H2 burned. trim-tails-001 done (D-010).
**Next:** stale_bars PR #30 pending, then stale-bars-002 (queued); G waits for the owner's raw gate-cli JSON; then R1 confluence map, R1b, R2, R5. R6 has no survivor now.

## Scope (owner, 2026-10-13)
- Goal (D-001): find or reject, with honest evidence, information that adds something real; a survivor feeds a
  model (feature/target → full model, R6) and/or a bot2 code-computed tool. New docs in English.
- The planner session plans, reviews and merges; builders are subagents it launches (one phase = one branch = one PR).
  Cost rules: `WORKING_METHOD.md`.
- ≤ 1 h compute per phase (D-003). Data: Drive `history_15m` 800+ coins (D-004, Colab; sealed buckets come from it);
  10 touched coins at 1m–1w via `tools/fetch_crypto_dataset.py`, usable in the cloud session (D-007).
- The planner may merge a phase PR after review (D-002).

## Honest state (from master, 2026-10-19)
- 82 closed failures (`docs/research/failure_registry.csv`); no tradeable edge found.
- Merged: #13 restructuring, #19 H07 (F-0078/79), #21 + #23 Colab runner, #24 R0 lock, #25 G2, #26 docs, #27 R3.
- Survivors: H003 (NATR_14 forecasts next range **width**; R3 1h not confirmed on H2, F-0083); H07 (BTC/ETH 30-day trend filter, a drawdown reducer,
  medium confidence); MOM_ORTH_NATR (relative low, weak).
- Untested: funding carry / basis; chart-structure confluence as overlapping zones.
- Consumed: the edge_discovery 2026 holdout; MOM_ORTH_NATR's 38 OOS windows. No holdout bucket opened yet (access log empty).
- Data issue in progress: `y_close_reg` exact-zero share 7.1 % train / 17.1 % val (close[t+h] == close[t]); branch
  `claude/zero-target-fix` (`CONFIG['stale_bars']`, default off) awaits the Colab job `stale-bars-001` (first run interrupted, re-queued).

## Phases
| Phase | State | PR | Verdict |
|---|---|---|---|
| T0 triage open PRs | done | #13, #15, #16, #19 | #6, #14 closed |
| R0 lock (+M) | merged | #24 | salt b4f3790; holdout.json sha256 70322ba699d5068e4fbe7f8b014f85492aeebf15d3c743551fbf1062cd889b2c; 860 coins, 459 eligible (14 A / 445 B), 183 sealed (H1 36, H2 36, H3 3, H4 36, H5 36, H6 36); time seal ≥ 2026-10-01; D-009 two-level rule |
| G2 gate-style features | merged, FAILED | #25 | direction beat nulls (AUC 0.545/0.534) but net negative after costs F-0080; magnitude worse than NATR-only F-0081; H5 not opened |
| G gate-cli reverse + QC | waits for the owner's raw JSON | — | — |
| R3 NATR range tool | merged | #27 | 1h ACCEPTED on exploration (coverage 80.00 %, deciles 77.2-80.4 %, pinball 10/10, monthly IC > 0 21/21; flag: BTC/ETH coverage 75.1 % at 1h in 2026); 1d REJECTED F-0082; bot2 spec `research/natr_range/bot2_tool_spec.md` |
| R3 H2 confirmation E-natr-002 | run 2026-10-19, H2 opened once (36/36 coins, 440,834 rows) | this PR | REJECTED F-0083: coverage 79.37 %, pinball 10/10, IC 21/21 pass; top NATR decile 73.78 % < 74.0 (76.6 % 2025, 68.6 % 2026). R3 1h NOT confirmed |
| R3 high-NATR calibration E-natr-003 | Stage A run 2026-10-19 (open coins), card closed | #32 | REJECTED by rule 7: rules 1-6 pass (coverage 80.18 %, deciles 78.9-81.8 %, pinball 10/10, IC 21/21, top decile 79.17 % 2025 / 81.11 % 2026 vs control 75.60 / 78.15) but localization = drift (pooled OOF top 81.37 %; folds 2023 75.7, 2024 76.1), F-0084; not forwarded to Stage B. Result `research/natr_range/results/e_natr_003_explore/` |
| R3 rolling recalibration E-natr-004 | pre-registered (card + code + job spec, NOT queued) | `claude/natr-004` | rolling 12-month monthly refit of the decile factors; Stage A on open coins, Stage B on forward data from 2027-01-01 (needs a reviewed forward-read guard path, not built) |
| R1 confluence map 🔒 | not started | — | — |
| R1b confluence confirm 🔒 | not started | — | — |
| R2 funding/basis 🔒 | not started (data first; H3 has only 3 coins → time seal) | — | — |
| R5 E-disc S1/S3 confirm | optional | — | — |
| R4 H07 forward | parked | — | — |
| R6 model integration 🔒 | no survivor now (R3 1h not confirmed on H2) | — | — |

## Holdout
- Locked (R0, #24). Spec `docs/research/plan/lock/README.md`; salt = `b4f3790287a2e12d6e5cb3fa20b2ea81998566e5`
  (merge PRs with a merge commit, never squash). Guard `tools/holdout_guard.py`. Buckets H1 (R1b), H2 (R3), H3 (R2), H4 (R5),
  H5 (G2, not opened), H6 (reserve); time seal ≥ 2026-10-01. Access log: H2 opened 2026-10-19 (E-natr-002, burned); others unopened.
- D-010: 42 of the 183 sealed coins are dead coins (zero-volume tails trimmed by trim-tails-001, 2026-10-19; tails kept in `history_15m_tails`). Future bucket runs use live data only (trimmed files + `CONFIG['stale_bars']` once #30 merges).
- D-009 (two-level holdout): level A untouched / level B daily-only touched both eligible, 365-day history, H3 takes no
  funding/OI-touched coin.

## Owner decisions (`decisions/`)
- All accepted (D-010 dead-coin lock amendment added): D-001 goal (revised) · D-002 merge after review · D-003 ≤ 1 h/phase · D-004 Drive history_15m
- D-005 phase order (amended by D-008) · D-006 open-PR triage · D-007 5m fetch source · D-008 G2 gate-style features · D-009 two-level holdout

## Open questions for the owner 🔒
1. G stage 0: save the raw gate-cli JSON straight to files (no `head -c`, no LLM summary) — the nine calls of #16 §3
   plus `kline --with-indicators` for BTC/ETH/SOL at 15m/1h/4h/1d with the largest `--size`, and gate's history depth.

## Known issues
- Colab runner merged (#21, #23; `colab_runner.ipynb`, `COLAB_RUNNER.md`): owner one-time setup = token file in Drive.
- stale-bars-002 result: after trim + `min_run_hours=168` the zero-target share at 1h is train 2.48 % -> 1.87 % after masking, val 2.61 % (18.2 % before trim), test 2.59 %; masking touches train only. Decision: use trimmed data + `stale_bars`, stop dropping `y_close_reg == 0` rows.
- N1 (`price_norm_mode`, branch `claude/n1-price-norm`) is being built.
- N2 (owner-approved, after N1 merges): `refactor` — consolidate normalization into one registered `Normalizer` interface (fit/transform/inverse_targets/state; window_scale, logret_std, logret_sigma as components), byte-identical outputs in every mode; then run E-norm-001 on it.
- Zero-target data issue above: PR #30 (`CONFIG['stale_bars']`) pending; `stale-bars-002` queued.
