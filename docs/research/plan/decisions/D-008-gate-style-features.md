# D-008: Build gate-style derived features and test them as model inputs

**Status:** Accepted (owner, 2026-10-13). Amends D-005's order: G2 runs right after R0, before G.

**Context.** The owner: indicators computed before need not be recomputed, but the gate agent's construction method
and derived features (states per timeframe, multi-timeframe aggregation) may be useful because a known exchange's
agent relies on them. Single raw indicators are closed in the registry; their multi-feature, multi-timeframe,
categorical use inside a model has not been tested (F-0027 R2, F-0056 R2, F-0039 R2).

**Decision.** Phase G2 builds the states correctly (causal, NaN never 0, versioned `gate-style-v1, unverified`) in
`tools/gate_features.py` and tests them with the capacity controls of RESEARCH_RULES §2.1, a fixed accept rule and one
sealed confirmation (bucket H5). G later recovers gate's exact definitions from the owner's raw JSON and changes one
definition per card.

**Consequences.** Survivors go to R6 (model) and, for direction, to bot2 only after the economics check. If nothing
survives, bot2 gets a note that a gate-like `signal` carries no measurable information.
