# Experiment card — <id>: <one-line title>

Written and committed BEFORE the run (see `docs/research/MODEL_DEV_PROTOCOL.md`). Do not edit after the run; add a
"Result" section below the line instead.

| Field | Value |
|---|---|
| Date / author | |
| Baseline (commit, checkpoint, dataset fingerprint) | |
| The ONE change | |
| Data / splits / holdout | same as baseline (state any difference = new experiment) |
| Seeds / budget | ≥ 3 seeds; epochs; hardware |

## Mechanism and prediction
- Why it should help:
- Diagnostic that must move (layer / signal, direction, size):
- Output effect expected on val (metric, size):

## Accept / reject rule (val)
- Accept if: output improves beyond seed spread AND the predicted diagnostic moves beyond its null.
- Reject if:
- Otherwise: "unexplained" (not adopted).

---
## Result (added after the run)
- Output metrics (mean ± seed spread):
- Diagnostic movement vs prediction (with null):
- Failure level (protocol §3) and layer, if failed:
- Verdict: accepted / rejected / unexplained — artifact links:
