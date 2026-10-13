# STATUS: research plan hand-off

> Read this first in every session (planner and build). Update it in the same PR that changes the state.
> Keep it under 60 lines.

**Phase:** plan approved by the owner (2026-10-13); no phase started. **Next:** T0 (reviewer reviews #15, #16, #13;
closes #14/#6 after telling the owner), then the R0 kickoff.

## Scope (owner, 2026-10-13)
- Goal (D-001): find or reject, with honest evidence, information that adds something real; a survivor feeds a
  model (feature/target → full model, R6) and/or a bot2 code-computed tool. New docs in English.
- Planner/reviewer session plans and reviews; a separate build session implements one phase per PR.
- Heavy runs on Colab, ≤ 1 h per phase (D-003); data on Drive = `history_15m`, 800+ coins, compressed (D-004).
- The planner may merge a phase PR after review (D-002).

## Honest state (from master, 2026-10-13)
- 63 closed failures (`docs/research/failure_registry.csv`); no tradeable edge found.
- Survivors: H003 (NATR_14 forecasts next range **width**, IC ≈ +0.21 high / −0.20 low, 38/38 OOS windows; not
  direction); H07 (BTC/ETH 30-day trend filter, a drawdown reducer, medium confidence); MOM_ORTH_NATR (relative low, weak).
- Untested: funding carry / basis; chart-structure confluence as overlapping zones.
- Consumed: the edge_discovery 2026 holdout; MOM_ORTH_NATR's 38 OOS windows.
- Open drafts not on master: #6, #13, #14, #15, #16 (registry rows F-0064.. collide between #13 and #15).

## Phases
| Phase | State | PR | Verdict |
|---|---|---|---|
| T0 triage open PRs | waiting for D-006 | — | — |
| R0 lock (+M) | not started | — | — |
| G gate-cli QC | not started (stage 0 = owner) | — | — |
| R3 NATR range tool | not started | — | — |
| R1 confluence map 🔒 | not started | — | — |
| R1b confluence confirm 🔒 | not started | — | — |
| R2 funding/basis 🔒 | not started (data first) | — | — |
| R5 E-disc S1/S3 confirm | optional | — | — |
| R4 H07 forward | parked | — | — |
| R6 model integration 🔒 | waits for a survivor | — | — |

## Holdout
- Not locked yet (R0). Buckets H1 (R1b), H2 (R3), H3 (R2), H4 (R5); time seal ≥ 2026-10-01. Access log: none.

## Owner decisions (`decisions/`)
- All accepted: D-001 goal (revised) · D-002 merge after review · D-003 Colab ≤ 1 h/phase
- D-004 data = history_15m 800+ coins · D-005 phase order · D-006 open-PR triage

## Open questions for the owner 🔒
1. Is BTC/ETH/SOL 5m data also on Drive? Is anything else there (1d history, hourly futures metrics)?
2. G stage 0: save the raw gate-cli JSON straight to files (no `head -c`, no LLM summary) — the nine calls of #16 §3
   plus `kline --with-indicators` for BTC/ETH/SOL at 15m/1h/4h/1d with the largest `--size`, and gate's history depth.

## Known issues
- Build branch name (`<BUILD_BRANCH>`): unknown until the build session's first report.
- Baseline checks on master: build_map --check OK, check_deps 0 errors, registry validate OK (63 rows); pytest: see the plan PR.
