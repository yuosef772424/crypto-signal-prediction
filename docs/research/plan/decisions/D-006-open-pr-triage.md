# D-006: Triage of the open draft PRs

**Status:** Accepted (owner approved the plan, 2026-10-13). Closing #14 and #6 is announced to the owner first.

**Context.** Five drafts are open: #6 (loader path, old), #13 (H07 vol-sizing/forward/combos + notebooks → `workflow/`,
236 files), #14 (hedge recovery), #15 (strategy discovery, contains #14's commits, adds `tools/trade_engine.py`),
#16 (gate-cli QC plan + `tools/indicator_reverse.py`). #13 and #15 both add registry rows `F-0064`/`F-0065`.

**Decision (proposed).** The planner reviews each PR with the same evidence checks as a phase PR, in this order:
#15 (merge; then close #14 as included), #16, #13 (largest; ask for a split if the restructuring breaks the suite),
#6 (close if stale). The second of #13/#15 to merge renumbers its colliding rows and their references in a `fix`
commit on its own branch before merge.

**Consequences.** R0's touched-data ledger includes everything merged here, so the sealed buckets exclude every coin
any study has read.
