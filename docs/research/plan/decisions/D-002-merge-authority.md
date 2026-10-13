# D-002: The planner may merge a phase PR after review

**Status:** Accepted (owner, 2026-10-13).

**Decision.** When a phase PR passes the review in KICKOFF/the planner prompt (evidence checks, suite, registry check,
protected files unchanged, DoD), the planner marks it ready and merges it with the full 40-char `expectedHeadSha`, then
unsubscribes. Any doubt (a relaxed accept rule, a holdout question, an owner-level trade-off) is escalated instead.

**Consequences.** Closing another session's PR or changing a decision file still needs the owner.
