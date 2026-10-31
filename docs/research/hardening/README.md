# hardening/: root-cause audits of the codebase before a new experiment round

Purpose: record design-level findings (root-cause classes of fragility, not bug lists), the guard that stops each class from
coming back, and the PR sequence that fixes them. Nothing here changes code. A refactor that follows an audit is its own PR
(change type `refactor`, byte-identical outputs proven) and links the audit class it closes.

| File | What |
|---|---|
| `audit_2026-10-31.md` | Auditor report: baseline health, ranked root-cause classes, PR sequence, out-of-scope list |
| `proposal_cascaded_stages.md` | PROPOSED (not adopted) owner idea: cascaded / stacked training stages, and the seams it needs |

Rules: recorded audits are not edited (add a dated successor). Proposals stay `PROPOSED` until the owner records a decision
in `docs/research/plan/decisions/`.
