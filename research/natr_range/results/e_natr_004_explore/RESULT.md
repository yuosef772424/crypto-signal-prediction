# E-natr-004 Stage A result (exploration, 10 open coins, val 2025-01..2026-09) - ACCEPTED (provisional, exploration data only)

Files are copied unchanged from branch `colab-results`, `runs/e-natr-004-explore/` (job `e-natr-004-explore`, repo_ref `e32d8a6d9da2fd54153a754585485894b5ef6c3d`). `run_log.txt`, `env.json`, `bundle_summary.md` stay on that branch.
Card: `research/natr_range/cards/E-natr-004.md` (pre-registered, rules not edited). 152,160 val rows, 10/10 coins in base, calibration and val, 0 crossed rows, 0 stale source rows dropped, 21 refits (2025-01..2026-09), **0 pooled-factor fallbacks**, run 0.35 min. The January 2025 rolling factors equal the E-natr-003 static factors (max abs diff 0.0).

| Rule | Rolling arm | Result |
|---|---|---|
| 1 overall coverage 77..83 | 79.75 % (control 80.23 %, static 80.18 %) | PASS |
| 2 every evaluable decile 74..86 | 10/10 evaluable, all in 79.0..80.7 % | PASS |
| 3 pinball below baseline | 10/10 deciles (0.135605 vs 0.150386) | PASS |
| 4 monthly rank IC > 0 | 21/21 months | PASS |
| 5 top decile 74..86 in 2025 and 2026, n >= 200 | 79.93 % (n 3922) and 81.26 % (n 1382) | PASS |
| 6 top decile closer to 80 than control in both years | control 75.60 % / 78.15 % | PASS |
| 7 >= 8 coins | 10 coins | PASS |
| 8 every quarter with n >= 200 in 74..86; >= 5 evaluable | 6 evaluable: 2025Q1 77.6, Q2 82.7, Q4 80.9, 2026Q1 81.0, Q2 85.6, Q3 76.3 (2025Q3 n = 141, not evaluable) | PASS |
| 9 stability: max abs ln change <= 0.05, mean <= 0.02 | max 0.0217, mean 0.0042 (400 changes) | PASS |

accepted = True. This ALLOWS Stage B; it is not a confirmation (val was already touched by E-natr-001 and E-natr-003).

Reported arms, top decile (static / control for comparison): 2025 79.93 % rolling, 79.17 % static, 75.60 % control; 2026 81.26 % rolling, 81.11 % static, 78.15 % control.
Pinball totals: rolling 0.135605, static 0.135589, control 0.135601 (baseline 0.150386); the three are equal to four digits. Per-quarter values of all three arms are in `quarters.csv`; rolling and static differ by at most 1.2 pp in any quarter.

## Honest note (read before citing this result)
**On this validation window the rolling and static arms are nearly identical.** The rolling factors move by at most 2.2 % between refits (mean 0.4 %), the pinball totals agree to 4 digits, and the top-decile coverage differs by 0.8 pp (2025) and 0.15 pp (2026) in favour of rolling. There is no further drift inside 2025-26 that the monthly refit has to follow, so **Stage A does not separate "rolling" from "static"**; it only shows that rolling does no harm, stays stable, and fixes the control's top-decile under-coverage (the same fix the static factors already gave). Note also that the control arm passes the absolute rules 1-5 on its own (its top decile 75.6 % / 78.1 % is inside 74..86); only rule 6 (closer to 80 than the control) separates the corrected arms from it. Whether the monthly refit is needed at all is the question of **Stage B on forward data (>= 2026-10-01)**, where a regime change would show a static set going stale. No claim of edge: nothing was traded.
