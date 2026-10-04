---
id: E-<study>-<n>
mechanism: <tags, e.g. momentum; direction>        # reuse spellings from docs/research/failure_registry.csv
timeframe: <e.g. 1h>
target: <e.g. direction; next-8-bar-return>
model_class: <e.g. cnn; logistic; nig-timenet; rule>
reopens:                                            # e.g. F-0003:R1 — a reopen_if condition that now holds
not_covered_by:                                     # e.g. F-0004=different target: volatility, not direction
---
# Experiment card — <id>: <one-line title>

Written and committed BEFORE the run (rules: `docs/research/RESEARCH_RULES.md`, `docs/research/MODEL_DEV_PROTOCOL.md`).
Copy this file to `docs/research/cards/` or `research/<study>/cards/`, then run
`python tools/experiment_registry.py check <card>` — it must print `card OK`. Do not edit after the run; add the
"Result" section below the line instead.

| Field | Value |
|---|---|
| Date / author | |
| Baseline (commit, checkpoint, dataset fingerprint) | |
| The ONE change | |
| Failure being addressed (level, location, registry id) | |
| Data / splits / holdout | same as baseline (state any difference = new experiment) |
| Seeds / budget | ≥ 3 seeds; epochs; hardware |

## Verified cause and mechanism
- Verified cause this change attacks (test + result):
- Why the change should fix it:
- Diagnostic that must move (layer / signal, direction, size, null):
- Output effect expected on val (metric, size):

## Accept / reject rule (val)
- Accept if: output improves beyond seed spread AND the predicted diagnostic moves beyond its null.
- Reject if:
- Otherwise: "unexplained" (not adopted).

---
## Result (added after the run)
- Output metrics (mean ± seed spread):
- Diagnostic movement vs prediction (with null):
- Failure level and location, if failed:
- Verdict: accepted / rejected / unexplained — artifact links:
- If rejected: registry row `F-____` added (claim, scope tags, verified cause, invariants, reopen_if).
