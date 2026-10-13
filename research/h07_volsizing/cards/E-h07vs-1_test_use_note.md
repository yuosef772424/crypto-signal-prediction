---
id: E-h07vs-1-test-use-note
mechanism: volatility-structure; trend-following
timeframe: 1d
target: magnitude
model_class: gbm; regression
reopens:
not_covered_by: F-0078=this card is the experiment that produced F-0078 (its own result, not a different scope)
---
# E-h07vs-1 — TEST-use note (added 2026-10-13, review defect D2)

Companion to `E-h07vs-1.md`. The original card and its result section are unchanged; this note records how many times
TEST was read, against the card's "TEST once".

## What the card said
"Then TEST once: confirmed if Sharpe difference > 0 and max drawdown not worse." TEST 2024-01-01 .. 2026-09-30 is
opened once, after the VAL decision.

## What the study did
`01_volsizing.py` computes VAL and TEST rows together in every run (loops over `(('val', VAL), ('test', TEST))`). TEST
was therefore read in three runs, one per realized-variance bar size:

| # | Run | Output file | Date |
|---|---|---|---|
| 1 | `RV_BAR=5min` (the pre-registered run) | `volsizing_results.csv` | 2026-10-05 |
| 2 | `RV_BAR=1h` (robustness, owner's request) | `volsizing_results_rv1h.csv` | 2026-10-05 |
| 3 | `RV_BAR=4h` (robustness, owner's request) | `volsizing_results_rv4h.csv` | 2026-10-05 |

## Consequence
- The verdict (rejected for sizing) was decided on VAL in run 1 and did not change in runs 2 and 3, which were
  robustness checks added after TEST of run 1 had been seen. In all three runs the trailing 30-day std is best on VAL
  and TEST is reported, not used, but "TEST once" is not literally true: TEST was read three times (5m, 1h, 4h).
- The TEST column of the baseline (trail30, 0.87) is the same in the three runs because the baseline does not depend on
  the realized-variance bar size. The 1h and 4h variants count as 2 extra variants for multiple testing, as the card
  result already states; since all variants were rejected, no TEST-based selection followed from any of the reads.
- TEST values of this study must not be reused as confirmation of anything: they were read three times and a
  rejection does not need them.
