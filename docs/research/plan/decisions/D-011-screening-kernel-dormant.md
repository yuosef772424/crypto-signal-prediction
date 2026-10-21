# D-011: screening kernel (`LAB_DESIGN.md` rev. 2), accepted in principle and DORMANT

**Status:** Accepted in principle, **dormant** (owner, 2026-10-21). Not in force until activated (below).

**Context.** `LAB_DESIGN.md` proposes one importable package (working name `lab/`) that absorbs legacy screening code, with a complexity
budget, a retirement clause and a one-in-one-out rule. The owner accepted its five recommendations but judged that building it now would add
expansion before the project has a plan for how it ends.

**Decision.**
1. The five recommendations of `LAB_DESIGN.md` are accepted as the design to use *when it is activated*: the absorbing kernel (option C) and the feature
   freeze in shared-namespace packages; 4,000 lines / 16 modules / 6 weeks / 12 configs per family; R1 implemented on the kernel; first data = 10 coins at
   1h/4h in the cloud (exploration only); E-rel-001 parked until L1.
2. **Nothing is active now.** No `lab/` code, no freeze, no absorption or shims, no budget or time-box running, no new CI rules. The document is a stored
   design, not a work item, and no phase (L0-L5) is scheduled. E-rel-001 stays not-run; its own open decision (absolute vs relative target) is unchanged.
3. **Activation condition.** The owner first draws a roadmap and a plan for completing the project (what is finished, what is stopped, what the end state is).
   Only then does the owner record an activation decision (D-nnn) naming the phase to start and the date. At that time `LAB_DESIGN.md` is re-read against the
   plan and revised if it no longer fits; the measurements in its §1 are re-taken.
4. Until activation, nobody relies on the document, cites its budget or rules in a PR review, or starts work "because the design exists".

**Revisit if.** The owner activates it; or the project plan makes the screening question moot (then supersede with a note and archive the document).
