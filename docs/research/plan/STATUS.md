# STATUS: research plan hand-off

> Read this first in every session (planner and build). Update it in the same PR that changes the state.
> Keep it under 60 lines.

**Phase:** plan merged (#17). T0 review of #13/#15/#16 running; R0 kickoff given (does not wait for T0).
**Next:** R0 build → review → G2.

## Scope (owner, 2026-10-13)
- Goal (D-001): find or reject, with honest evidence, information that adds something real; a survivor feeds a
  model (feature/target → full model, R6) and/or a bot2 code-computed tool. New docs in English.
- Planner/reviewer session plans and reviews; a separate build session implements one phase per PR.
- ≤ 1 h compute per phase (D-003). Data: Drive `history_15m` 800+ coins (D-004, Colab; sealed buckets come from it);
  10 touched coins at 1m–1w via `tools/fetch_crypto_dataset.py`, usable in the cloud session (D-007).
- The planner may merge a phase PR after review (D-002).

## Honest state (from master, 2026-10-13)
- 63 closed failures (`docs/research/failure_registry.csv`); no tradeable edge found.
- Survivors: H003 (NATR_14 forecasts next range **width**, IC ≈ +0.21 high / −0.20 low, 38/38 OOS windows; not
  direction); H07 (BTC/ETH 30-day trend filter, a drawdown reducer, medium confidence); MOM_ORTH_NATR (relative low, weak).
- Untested: funding carry / basis; chart-structure confluence as overlapping zones.
- Consumed: the edge_discovery 2026 holdout; MOM_ORTH_NATR's 38 OOS windows.
- Open drafts not on master: #6, #13, #14, #15 (F-0064/65 collide: #15 keeps them, #13 renumbers to F-0076/77).

## Phases
| Phase | State | PR | Verdict |
|---|---|---|---|
| T0 triage open PRs | #16 merged (tool defects → G row); #15 fix first (S1/S3 not significant); #6 is a real loader fix (needs master merge); #13 split; #14 close after #15 | #16 | in progress |
| R0 lock (+M) | kickoff given | — | — |
| G2 gate-style features | next | — | — |
| G gate-cli reverse + QC | waits for the owner's raw JSON | — | — |
| R3 NATR range tool | not started | — | — |
| R1 confluence map 🔒 | not started | — | — |
| R1b confluence confirm 🔒 | not started | — | — |
| R2 funding/basis 🔒 | not started (data first) | — | — |
| R5 E-disc S1/S3 confirm | optional | — | — |
| R4 H07 forward | parked | — | — |
| R6 model integration 🔒 | waits for a survivor | — | — |

## Holdout
- Not locked yet (R0). Buckets H1 (R1b), H2 (R3), H3 (R2), H4 (R5), H5 (G2), H6 (reserve); time seal ≥ 2026-10-01. Access log: none.

## Owner decisions (`decisions/`)
- All accepted: D-001 goal (revised) · D-002 merge after review · D-003 ≤ 1 h/phase · D-004 Drive history_15m
- D-005 phase order (amended by D-008) · D-006 open-PR triage · D-007 5m fetch source · D-008 G2 gate-style features

## Open questions for the owner 🔒
1. G stage 0: save the raw gate-cli JSON straight to files (no `head -c`, no LLM summary) — the nine calls of #16 §3
   plus `kline --with-indicators` for BTC/ETH/SOL at 15m/1h/4h/1d with the largest `--size`, and gate's history depth.

## Known issues
- Build branch name (`<BUILD_BRANCH>`): unknown until the build session's first report.
- Baseline checks on master: build_map --check OK, check_deps 0 errors, registry validate OK (63 rows); pytest 190 passed, 0 failed (12 min, CPU).
