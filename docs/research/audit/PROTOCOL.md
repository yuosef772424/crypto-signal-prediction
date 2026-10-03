# Audit / Improvement Protocol (Auditor ↔ Builder)

Purpose: improve the model and the data pipeline through rounds of auditing and fixing without either side being swayed
by the other's arguments. Mutual influence produces agreement, not improvement, and hides shared defects instead of
exposing them. **The auditor's independence is therefore the most important rule here**, and everything below serves it.

## Roles

| Role | Sees | Does not see | Produces |
|---|---|---|---|
| **Auditor** | The code at a pinned commit, the data, and raw evaluation outputs (CSV/JSON) | Builder reports, discussions, rationale. Docs and code comments are treated as claims to test, never as explanations to adopt | One reproduction script per defect (`repro_*.py`) plus evidence: command, output, expected vs actual |
| **Builder** | The reproduction scripts and their outputs only | The auditor's arguments or analysis | A fix that makes the repro pass, plus a pre-registered improvement hypothesis (what changes, by how much, how it is measured) |
| **Judge** | — | — | Not an agent: a fixed evaluation harness with pre-registered criteria. No improvement claim is accepted without its output |
| **Orchestrator** | Everything | — | Relays **evidence only** between the two sides — never its own opinions or theirs |

## Independence rules

1. **The only channel between the sides is runnable evidence.** The auditor hands over a failing script; the builder
   hands back code that makes it pass. No argumentative text crosses over.
2. **Expectation before reading.** Before reading the rationale for any component, the auditor writes down the correct
   behaviour from first principles (no look-ahead, correct alignment, clean split), then checks the code against it.
3. **Past fixes are not exempt.** The list of fixed bugs is context, not a boundary; a fix may be incomplete.
4. **Auditor diversity.** The same model shares the builder's blind spots. When possible, run the auditor on a different
   model, or with a different checklist, each round.
5. **No debate.** If the builder disputes a finding, the only acceptable reply is a script showing the behaviour is
   correct. The run is the judge.

## Evaluation integrity

- **val** selects, **test** compares rounds, and a **sealed holdout** (the last 2–3 months) is seen by neither side and
  opened once, at release.
- Every variant tried counts toward the multiple-testing correction, including inversions and rejected settings.
- Rounds stop when a round yields neither a proven defect nor an improvement beyond seed noise.

## Shared ledger

`ledger.md` in this folder. Each row: id, claim, status (open / proven / rejected / fixed), evidence link.
**No arguments in the ledger** — evidence only, so it cannot become an indirect channel of influence.
