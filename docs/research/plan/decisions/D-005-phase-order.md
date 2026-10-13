# D-005: Phase order

**Status:** Proposed (2026-10-13), pending the owner's confirmation.

**Context.** Asked for the order, the owner pointed to `research/gate_data_qc/README.md` (open PR #16): the gate-cli
data-quality plan. Open draft PRs (#13, #15) also change what is on master and which data was already touched.

**Decision (proposed).** T0 (triage open PRs) → R0 lock (+M correction) → G gate-cli QC → R3 NATR range tool → R1
confluence map 🔒 → R1b confirm 🔒 → R2 funding/basis 🔒. R5 (confirm #15's S1/S3 on sealed coins) is optional and
cheap; R4 (H07 forward) is parked until forward data exists.

**Why.** T0 and R0 must precede any confirmation (a complete touched-data ledger, then a lock). G is what the owner
pointed to and protects every later bot2 tool from broken inputs. R3 is the survivor closest to a usable tool. R1 is
exploration with a pre-registered stop. R2 needs data that is not on Drive yet.

**Consequences.** If the owner reorders, only this file is superseded; ROADMAP phase bodies stay valid.
