# D-001: Goal of the research plan

**Status:** Accepted (owner, 2026-10-13), wording revised by the owner.

**Context.** 63 closed failures, no tradeable edge; three weak survivors (H003, H07, MOM_ORTH_NATR). The sibling repo
bot2 runs an LLM trading agent whose tools are computed by code (bot2 ADR 0003: no LLM direction from OHLCV).

**Decision.** Find or reject, with honest evidence, information that adds something real. Supporting bot2 is one
outlet, not the only one. A survivor goes to (a) a model: a feature or target trained under `MODEL_DEV_PROTOCOL.md`,
and the model is shipped in full if it beats seed noise (phase R6); and/or (b) bot2, as a code-computed tool with a
hand-off note (tool name, inputs, outputs with units, validated limits). A phase that fails ends with
failure-registry rows. Plans and documentation of anything new are written in English.

**Consequences.** Calibration, risk and data-quality tools count as deliverables even without a direction edge.
No phase is judged by in-sample results.
