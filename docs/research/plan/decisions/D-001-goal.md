# D-001: Goal of the research plan

**Status:** Proposed (2026-10-13); the owner's answer did not contradict it, explicit confirmation pending.

**Context.** 63 closed failures, no tradeable edge; three weak survivors (H003, H07, MOM_ORTH_NATR). The sibling repo
bot2 runs an LLM trading agent whose tools are computed by code (bot2 ADR 0003: no LLM direction from OHLCV).

**Decision.** The plan's goal is to find or reject, with honest evidence, signals worth exposing to the bot2 agent as
code-computed tools. A phase that survives ends with a bot2 hand-off note (tool name, inputs, outputs with units,
validated limits). A phase that fails ends with failure-registry rows.

**Consequences.** Calibration and risk tools (range forecast, exposure filter, data QC) count as deliverables even
when they carry no direction edge. No phase is judged by in-sample results.
