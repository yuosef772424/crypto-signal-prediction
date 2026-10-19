# E-natr-003 Stage A result (exploration, 10 open coins, val 2025-01..2026-09) - REJECTED by rule 7 (localization = drift), F-0084

Files are copied unchanged from branch `colab-results`, `runs/e-natr-003-explore/` (job `e-natr-003-explore`). `run_log.txt`, `env.json`, `bundle_summary.md` stay on that branch.
Card: `research/natr_range/cards/E-natr-003.md` (pre-registered, not edited). 152,160 val rows, 10/10 coins in base, calibration and val, 0 fit rows at or after 2025, 0 crossed rows, 0 stale source rows dropped, run 0.4 min.

| Rule | Corrected arm | Result |
|---|---|---|
| 1 overall coverage 77..83 | 80.18 % (control 80.23 %) | PASS |
| 2 every decile 74..86 | 10/10 evaluable, all in 78.9..81.8 % (control 76.3..80.6 %) | PASS |
| 3 pinball below baseline | 10/10 deciles (0.13559 vs 0.15039) | PASS |
| 4 monthly rank IC > 0 | 21/21 months | PASS |
| 5 top decile 74..86 in 2025 and 2026, n >= 200 | 79.17 % (n 3922) and 81.11 % (n 1382) | PASS |
| 6 top decile closer to 80 than control in both years | control 75.60 % / 78.15 % | PASS |
| 7 >= 8 coins and localization == static | 10 coins; localization label = **drift** | **FAIL** |

accepted = False. Not forwarded to Stage B.

Localization (Step 1, train data only): pooled forward-chain OOF top-decile coverage 81.37 % (>= 79.0), control top decile < 78.0 % in 2025 (75.60 %), so the label is drift.
Per fold top decile: 2021 82.6 %, 2022 81.8 %, 2023 75.7 %, 2024 76.1 %; 2 of 4 folds <= 79.0 (static needs pooled <= 78.0 and >= 3 folds).
Correction factors (2024 slice, ratio form) are close to 1 (c90 0.965..1.020, top decile c10 0.948, c90 1.020; n per decile 5158..12267).

Interpretation: the top-decile under-coverage is absent in 2021-2022 and appears from 2023 on (a regime drift), not a static misfit of the log-log line. A correction fitted on the most
recent year (2024) fixes val on the open coins, but the card ruled that a drift label ends at best "unexplained"; a stale factor set would be expected to decay as the regime moves.
Next step: E-natr-004 (rolling recalibration, `cards/E-natr-004.md`). No claim of edge: nothing was traded.
