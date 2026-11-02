# D-012: development friction is a bottleneck too; shared namespace goes gradually; models plug in by contract

**Status:** Accepted (owner, 2026-11-02).

**Context.** The owner asked for the 2026-11 execution plan (`docs/research/hardening/PLAN_2026-11.md`) to measure not only
runtime bottlenecks but what slows agents down or makes them fail when they update or add something, so that the system gets
easier to change over time instead of decaying.

**Decision.**
1. Development friction is measured and reduced like a runtime bottleneck: every step session appends real obstacles to
   `docs/research/hardening/friction_log.csv`; S03b audits friction from the record; S25 reviews the trend periodically.
2. The shared-namespace loading (`<pkg>/_loader.py`, `load_into`) is removed gradually, package by package, where it slows
   agents (S24, method of `docs/architecture_review.md` S9). This overrides the "S9 out of this round" line of
   `audit_2026-10-31.md` §6. Notebooks keep working throughout.
3. Flexibility is kept by contract: a radically different model, or any single part, plugs in by matching the input/output
   contract of the approved model, then trains with the same trainer and is evaluated by the same evaluation code (S23).

**Revisit if.** S25 shows the friction trend does not fall, or a package conversion costs more than it saves.
