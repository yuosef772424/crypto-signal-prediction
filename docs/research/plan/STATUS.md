# STATUS: research plan hand-off

> Read this first in every session (planner and build). Update it in the same PR that changes the state.
> Keep it under 60 lines.

**Phase:** plan merged (#17). T0 nearly done (only #13 restructuring left); R0 kickoff given (does not wait for T0).
**Next:** R0 build → review → G2.

## Scope (owner, 2026-10-13)
- Goal (D-001): find or reject, with honest evidence, information that adds something real; a survivor feeds a
  model (feature/target → full model, R6) and/or a bot2 code-computed tool. New docs in English.
- The planner session plans, reviews and merges; builders are subagents it launches (one phase = one branch = one PR).
  Cost rules: `WORKING_METHOD.md`.
- ≤ 1 h compute per phase (D-003). Data: Drive `history_15m` 800+ coins (D-004, Colab; sealed buckets come from it);
  10 touched coins at 1m–1w via `tools/fetch_crypto_dataset.py`, usable in the cloud session (D-007).
- The planner may merge a phase PR after review (D-002).

## Honest state (from master, 2026-10-13)
- 79 closed failures (`docs/research/failure_registry.csv`); no tradeable edge found.
- Survivors: H003 (NATR_14 forecasts next range **width**, IC ≈ +0.21 high / −0.20 low, 38/38 OOS windows; not
  direction); H07 (BTC/ETH 30-day trend filter, a drawdown reducer, medium confidence); MOM_ORTH_NATR (relative low, weak).
- Untested: funding carry / basis; chart-structure confluence as overlapping zones.
- Consumed: the edge_discovery 2026 holdout; MOM_ORTH_NATR's 38 OOS windows.
- Registry on master: 79 rows (F-0064..F-0077 from #15, S1/S3 not significant on unseen coins → F-0076/77; F-0078/79 H07 vol-sizing
  rejected and H07 combos `power`, from #19). #13 (restructuring only) being fixed.

## Phases
| Phase | State | PR | Verdict |
|---|---|---|---|
| T0 triage open PRs | #16, #15, #19 (H07 studies split from #13, F-0078/79) merged; #6, #14 closed; #13 restructuring: merge master + rtol fix + history note | #15, #16, #19 | in progress |
| R0 lock (+M) | builder agent on `claude/r0-lock` | — | — |
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
- Baseline checks on master: build_map --check OK, check_deps 0 errors, registry validate OK (79 rows); pytest 227 passed, 0 failed (13 min, CPU).
