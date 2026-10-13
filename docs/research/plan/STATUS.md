# STATUS: research plan hand-off

> Read this first in every session (planner and build). Update it in the same PR that changes the state.
> Keep it under 60 lines.

**Phase:** plan written (this PR, draft); no phase started. **Next:** owner approves the plan and confirms the order
(D-005) and the open-PR triage (D-006); then T0 (reviewer) and the R0 kickoff.

## Scope (owner, 2026-10-13)
- Goal (D-001, to confirm): find or reject, with honest evidence, signals worth exposing to bot2 as code-computed
  tools; survivors get a bot2 hand-off note.
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

## Holdout
- Not locked yet (R0). Buckets H1 (R1b), H2 (R3), H3 (R2), H4 (R5); time seal ≥ 2026-10-01. Access log: none.

## Owner decisions (`decisions/`)
- D-001 goal (proposed) · D-002 merge after review (accepted) · D-003 Colab ≤ 1 h/phase (accepted)
- D-004 data = history_15m 800+ coins (accepted) · D-005 phase order (proposed) · D-006 open-PR triage (proposed)

## Open questions for the owner 🔒
1. Confirm the goal (D-001) and the order T0 → R0 → G → R3 → R1 → R1b → R2, R5/R4 optional (D-005).
2. Approve the triage of #6/#13/#14/#15/#16 (D-006).
3. Is BTC/ETH/SOL 5m data also on Drive? Is anything else there (1d history, hourly futures metrics)?
4. G stage 0: can you re-run the nine gate-cli calls without `head -c` and paste the JSON?

## Known issues
- Build branch name (`<BUILD_BRANCH>`): unknown until the build session's first report.
- Baseline checks on master: build_map --check OK, check_deps 0 errors, registry validate OK (63 rows); pytest: see the plan PR.
