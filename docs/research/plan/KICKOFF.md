# Phase kickoff: how the build session starts a phase

One build session = one phase = one PR. The planner gives the owner a paste-ready prompt built from the template
below and the phase's row. Between phases the owner runs `/clear` (not `/compact`) in the build session; add-on
messages for the same phase are pasted without `/clear`. If the build session is cleared mid-phase, the planner
resends the full phase task with "do NOT reset the branch again".

## Kickoff prompt (template)

```
<FIRST PHASE>  Work on your designated branch.
<LATER PHASES> PR #<N> is merged. Restart your branch from origin/master
               (git fetch origin master && git checkout -B <BUILD_BRANCH> origin/master),
               unsubscribe from PR #<N>, and open a NEW draft PR. Never reuse a merged PR.

Read CLAUDE.md, then docs/research/plan/STATUS.md. Your task is phase <Pn> only
(docs/research/plan/ROADMAP.md §<Pn>). Read only: <read list from the row below>.

Goal: <one line>.  Reviewer decisions: <from the row below>.

Strict process:
1. The experiment card and its pre-registration (accept rule, slices, configuration count) are committed BEFORE any
   result; `python tools/experiment_registry.py check <card>` passes on that commit.
2. A failing `test` commit before each `feat`/`exp` commit. One change type per commit; never commit on master.
3. Add, don't modify: recorded results are never edited (a correction is a new file); failure_registry.csv is append-only;
   the lock files under docs/research/plan/lock/ are never changed (only the access log is appended, by open_holdout).
4. A new option's default reproduces the old behaviour; unknown config keys raise.
5. Data is read only through tools/holdout_guard.py; the holdout is never read except in the single frozen confirmation step.
6. Colab scripts follow the run contract in ROADMAP.md (manifest.json, --max-minutes, outputs <= 300 KB, paste bundle).
   You cannot reach Drive or Binance: test on small fixtures; I run the scripts on Colab and paste the bundle back.
7. Before every commit: python tools/build_map.py, python tools/check_deps.py, python -m pytest tests/ -q.

Definition of done: the <Pn> DoD in ROADMAP.md, and docs/research/plan/STATUS.md updated in the last (`docs`) commit.
Open a draft PR, do NOT merge, and report in Arabic: what was built, test counts, the exact Colab cell(s) I must run.
```

## Per-phase inputs

| Phase | Read | Reviewer decisions | As-built notes | Hand-offs |
|---|---|---|---|---|
| T0 | (reviewer) the five PR descriptions, their `git log`, registry diffs | merge #15 then close #14; #16; #13 last; renumber the second `F-0064/65` set before merge; close #6 if stale (D-006) | | touched-data sources for R0: #15 README (10 coins 1h + BCH/TRX/ZEC), #13 H07 data |
| R0 (+M) | RESEARCH_RULES.md §3, audit/PROTOCOL.md (evaluation integrity), failure_registry.csv `universe`/`period` columns, `research/*/README.md` data sections (master and open PR branches #13–#16), `tools/colab_bridge.py` and `tools/fetch_crypto_dataset.py` cards, h003_volatility_reversal.md lines 1–40 | salt = SHA of the card commit; 40 % sealed; buckets H1–H6 by hash; the 10 Speirsy11 coins are touched; time seal 2026-10-01; eligible ≥ 2 y history, last bar ≥ 2026-09-01; M is one `docs` commit | | holdout.json bucket lists for every later phase |
| G2 | ROADMAP §G2 and §Common contracts, RESEARCH_RULES §2.1 and §3, model_diagnostics.md (sweep, learning curve), registry rows F-0009, F-0027, F-0036, F-0039, F-0054, F-0056, `tools/fetch_crypto_dataset.py` card | candidate definitions `gate-style-v1` as listed in ROADMAP; NaN never 0; decision on closed 1h bars, entry next open; accept rule fixed in ROADMAP; runs in the build session, H5 once on Colab | | R6 input if accepted; bot2: verdict on a gate-like `signal`, `get_tech_states` spec only after the economics check |
| G | ROADMAP §G, research/gate_data_qc/README.md §3–§6, `tools/indicator_reverse.py` card, raw fixtures (never an LLM summary) | T0 review of #16 — fix `tools/indicator_reverse.py` FIRST (`fix` commits, failing test first): D1 relative error is absolute below 1 (`clip(lower=1.0)`, L114) → scale-aware denominator + a 5e-5 price test that must reject the wrong family; D2 no minimum `n_compared`, silent zero-masking → `min_compared` (20) and report dropped zeros; D3 no length/time alignment guard → raise or align by time; D4 MACD histogram: test `macd-dea`, `2*(macd-dea)` and a field-label permutation (BTC row: macd_difference = macd + dea); D5 add RSI SMA/EMA smoothing, RSI 7/10, MACD (8,17,9), EMA seeding variants, KDJ (9,3,3), SAR (0.02,0.2), DI+/DI−, BOLL upper/lower with ddof 0 and 1, WILLR sign; D6 docstring of `insufficient_history`; non-circular round-trip tests (an independent implementation per convention), a newest-first end-to-end test; README line "no code yet" corrected by an added note; recovered = ≥ 95 % of bars within 1e-4 rel. error on the provider's own OHLCV; `signal` rule must reproduce every observed value or is "not reproducible"; no card per QC rule; thresholds in contract.json | | rebuilt features → R6 candidates (no new information by themselves); non-OHLCV fields + history depth → R2/R6; bot2 note: refuse `missing`/`invalid`, `as_of` on every number |
| R3 | h003_volatility_reversal.md + the M correction, h003_multiple_testing_review.md (summary), EXPERIMENT_CARD template, bot2 `ai_agent/docs/TOOLS.md` (format only) | quantile regression on log NATR; baseline trailing 90-bar quantiles; deciles fixed on train; bucket H2 once | | bot2: `get_range_forecast` spec, percent units |
| R1 | ROADMAP §R1, registry rows F-0019, F-0053, F-0054, F-0060–F-0062 (+#15 rows), `tools/bracket_eval.py` card (stop-first) | 6 detectors, 164 cells, random-zone null ×20, continue bar fixed in ROADMAP; 20 unsealed coins; P1/P2 | | candidate list → owner 🔒 |
| R1b | R1 map + card, ROADMAP §R1b | ≤ 3 rules frozen with sha256; bucket H1 once; Holm over rules | | bot2: `get_zones` spec if any rule passes |
| R2 | ROADMAP §R2, registry F-0042–F-0044, edge_discovery REPORT "what could change the verdict" | two hypotheses only; acquisition script first; stop on unreachable data | T0 hand-off from closed #6 (f9f1eaf): master already has the cross-sectional pre-pass (`_load_cross_asset_frames`, c62b4f0) but not `drop_unavailable_funding_oi` — a funding/OI kind with no archive for any asset must leave `feature_order` (never constant columns); port it with self-tests before any funding feature reaches the pipeline; | bot2: compare with its `get_derivatives` (P8a) |
| R5 | #15 README + E-disc-002/003/006 cards | freeze S1/S3 verbatim; bucket H4 once | | bot2 note if it passes |
| R4 | REPORT H07 + Addendum C, #13 H07 docs | frozen rule only; waits for forward data | | bot2: exposure-filter note |
| R6 | MODEL_DEV_PROTOCOL.md, model_diagnostics.md, the survivors' cards | one change per card, ≥ 3 seeds, sweep + learning curve + linear baseline; holdout bucket assigned at kickoff | | full model or bot2 tool wrapping its output |
