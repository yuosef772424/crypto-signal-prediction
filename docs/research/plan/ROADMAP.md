# Research roadmap (2026-10 →)

Goal (D-001): find or reject, with honest evidence, information that adds something real. A survivor has two
possible outlets, chosen per phase: (a) a **model input or target** (pipeline feature with a card and a test, then
trained under `MODEL_DEV_PROTOCOL.md` and, if it beats seed noise, shipped as a full model), and/or (b) a
**code-computed tool for the bot2 agent** (hand-off note in bot2's `ai_agent/docs/TOOLS.md` format).
A rejected phase is a valid result: it is closed with `failure_registry.csv` rows (verified cause + `reopen_if`).
All new plans, cards and docs are written in English.

Rules that apply to every phase: `CLAUDE.md`, `docs/research/RESEARCH_RULES.md`, `docs/research/MODEL_DEV_PROTOCOL.md`
(model phases only), `docs/research/audit/PROTOCOL.md`. Each phase is **one branch → one PR**, built by the build
session, reviewed by the planner (KICKOFF.md), merged only when its DoD holds and STATUS.md is updated.
Phases marked 🔒 need an explicit owner decision before they start.

```
T0 triage open PRs (in parallel) ‖ R0 lock (+M) ─▶ G2 gate-style features ─▶ G gate-cli reverse + QC ─▶ R3 NATR range tool
   ─▶ R1 confluence map 🔒─▶ R1b confirm
                                  └─▶ R5 confirm E-disc S1/S3 (optional, cheap)            R2 funding/basis 🔒 (data first)
                                                                                           R4 H07 forward (parked)
Survivors of G2 / G / R2 / R3 / R1b / R5 ─▶ R6 model integration (🔒 starts when ≥ 1 new input survives)
```
Order accepted by the owner (D-005).

## Common contracts

**Data (D-004, D-007).** Two sources. (1) Drive `history_15m`: 800+ coins, compressed, read on Colab — the only source
of untouched coins, so every sealed bucket comes from it; intrabar order inside a 15m bar is unknown, so brackets on it
use **stop-first** on ambiguity (the honest lower bound, as in `tools/bracket_eval.py`). (2) `tools/fetch_crypto_dataset.py`
(Speirsy11, Binance spot): 10 coins (ADA BCH BNB BTC DOGE ETH SOL TRX XRP ZEC), every interval from 1m to 1w,
2017-08..now, ~9 s per coin per interval, reachable from the cloud session. All 10 coins are already touched, so they
serve exploration and selection only, never confirmation; fetch with `--end "2026-09-30 23:55"` (time seal).
Coarser bars are resampled inside the scripts (UTC, label = bar open, a bar is used only after it closes).
No funding/basis history is available yet (R2 starts with an acquisition step).

**Run contract (D-003, ≤ 1 h per phase).** Runs on the 10-coin dataset may execute in the build session itself (it can
fetch them); the planner re-runs the same command and compares the output sha256 values (reproduction check). Runs on
Drive data (sealed buckets, 800-coin universes) execute on Colab: one cell per run, by the owner. Every run script:
- takes `--config <json>` (unknown keys raise) and `--out <dir>`, and stops cleanly at `--max-minutes` (default 50);
- reads data only through `tools/holdout_guard.py` (after R0);
- writes `manifest.json` (git SHA, config, input fingerprint = sha256 of the sorted input file list + sizes, row counts,
  runtime, **number of configurations evaluated**, sha256 of every output) and outputs ≤ 300 KB in total;
- prints a paste bundle (manifest + small CSVs) the owner pastes into the build session; the builder commits it
  verbatim as an `exp` commit under `research/<study>/results/<run_id>/`. The reviewer re-checks the sha256 values.
- One-click Colab runner: instead of pasting a cell, queue a `job.json` on branch `colab-queue`; the owner clicks Run all in `colab_runner.ipynb`, and the bundle lands on branch `colab-results` (`docs/research/plan/COLAB_RUNNER.md`).

**Costs.** Taker 0.05 % + 0.02 % slippage per side (0.14 % round trip) unless the card says otherwise; every trade
result is also reported at 2× cost. Costs in R = round-trip cost / stop distance, per trade.

**Evidence bar (every trading claim).** Net of cost at 1× and 2×; day-clustered t; result after removing the best 5 %
of trades; per-coin consistency (share of coins with ≥ 20 trades whose mean has the same sign); every configuration
tried is counted (BH-FDR or Holm as the card states; Deflated Sharpe where a Sharpe is claimed).

**Holdout (R0 onwards).** Sealed coins are split into six disjoint buckets H1–H6, one per confirmation phase. A bucket
is read once, by the single frozen confirmation step of its phase, through `holdout_guard.open_holdout(card)`, which
appends a row to `docs/research/plan/lock/holdout_access_log.csv`. Bars with open time ≥ 2026-10-01 are time-sealed
for every coin (forward data, for R4 and later re-checks). The edge_discovery 2026 holdout is consumed and not reused.

---

## T0 — Triage the open PRs (reviewer-led; no new build phase unless a fix is needed)
- **Goal:** master carries every finished study before anything is locked, so the touched-data ledger and the failure
  registry are complete. Open drafts: #6 (old loader path), #13 (H07 vol-sizing/forward/combos + notebooks → `workflow/`,
  236 files), #14 (hedge recovery, contained in #15), #15 (strategy discovery + `tools/trade_engine.py`), #16 (gate-cli
  QC plan + `tools/indicator_reverse.py`).
- **Known conflict:** #13 and #15 both add `F-0064`/`F-0065`. Whichever merges second renumbers its rows (and every
  reference) in a `fix` commit on its own branch before merge; rows already on master are never edited.
  *Resolved:* #15 merged first; the H07 rows (once `F-0064`/`F-0065` on #13) are `F-0078`/`F-0079` on master via PR #19, and
  #13 carries no registry rows of its own.
- **Proposed (D-006, pending owner):** review and merge #15, close #14 as included; review #16 (docs + tool); review #13
  last (largest; split request if the restructuring breaks the suite); close #6 if stale.
- **DoD:** each PR merged or closed with a one-line reason; `experiment_registry.py validate` passes on master; STATUS updated.
- **Card / registry / holdout:** none (review only). **Colab:** none.

## M — H003 wording correction (folded into R0's PR as a `docs` commit)
- **Goal:** add `docs/research/h003_volatility_reversal_correction.md` (add, don't modify): the original H003 text says
  high `NATR_14` precedes a *contraction* of the next range; with the corrected signs (IC ≈ +0.21 on next high, −0.20 on
  next low, 38/38 OOS windows) high NATR precedes a **wider** next range — volatility clustering, not reversal or
  contraction. Link it from `docs/research/README.md` (one added line).
- **DoD:** note cites the exact lines/numbers it corrects and the evidence file; original file unchanged.

## R0 — Lock: sealed-holdout registry + loader guard
- **Goal:** decide, before any new result, which coins and periods nobody may read, and make code refuse them.
- **Inputs:** a coin manifest of `history_15m` produced on Colab by the builder's `tools/data_manifest.py`
  (symbol, first/last bar, rows, file sha256); a touched-data ledger built from the repo (registry `universe`/`period`
  columns, `research/*/README.md`) **and from the open PR branches** (#13, #14, #15, #16; `git show <branch>:<path>`),
  so R0 does not wait for T0: `docs/research/plan/lock/touched.csv` (symbol or rule, timeframe, period, source). A
  universe described only as a rule ("top-50 by OI", "195 perps with metrics", "222 assets history_1d") is resolved to
  symbols where the repo lists them; where it cannot be resolved, the rule's whole universe counts as touched for the
  periods it names (strict side).
- **Reviewer decisions:** eligible = ≥ 2 years of 15m history, last bar ≥ 2026-09-01, not a stablecoin / leveraged token,
  **not touched at any timeframe**. Sealed = 40 % of eligible coins ranked by `sha256(salt + symbol)`, where `salt` = the
  40-char SHA of the commit that adds the R0 card (committed before the manifest is pasted). Sealed coins → buckets
  H1 (R1b), H2 (R3), H3 (R2), H4 (R5), H5 (G2), H6 (reserve, R6) by the same hash, round-robin. Time seal: open time ≥ 2026-10-01, all coins.
- **Deliverables:** `tools/holdout_guard.py` (card + test): `load_unsealed(...)`, `assert_unsealed(symbols, start, end)`,
  `open_holdout(card_path, bucket)` (refuses a second open of a bucket, refuses a card without `confirmation: true` and
  a frozen-rules hash); `docs/research/plan/lock/holdout.json` + `holdout.sha256`; empty `holdout_access_log.csv` with a header.
- **DoD:** tests show a sealed symbol / a time-sealed bar raises through every public loader, and a second
  `open_holdout` on the same bucket raises; lock files committed; checks pass. After merge the lock files never change
  except appended access-log rows.
- **Card:** none (infrastructure). **Registry:** none. **Colab:** one run of `data_manifest.py` (~5 min).

## G2 — Gate-style derived technical states as model inputs (owner direction, D-008)
- **Goal:** the gate agent (a known exchange's agent) feeds its users derived *states*, not raw indicators: per
  timeframe `ma_alignment`, `rsi_zone`, `macd_signal`, KDJ, SAR side, BOLL position, ADX/DI, WR/CCI zones, and one
  multi-timeframe `signal`. Raw single indicators are closed (F-0002, F-0004, F-0009, F-0026, F-0027) and are not
  recomputed. What is new and tested here is the construction: categorical states, multi-timeframe (15m/1h/4h/1d) at
  one decision time, and their aggregation — as a feature set for a model and as a verdict on whether a gate-like
  `signal` deserves bot2's trust.
- **Build (`tools/gate_features.py`, card + tests):** causal states from closed bars only; missing or insufficient
  history = NaN, never 0 (the gate bugs); each state definition is versioned `gate-style-v1, unverified` until G
  recovers gate's exact definition (then a new card changes one definition at a time). Candidate definitions fixed in
  the card: `ma_alignment` A = MA7 > MA30 > MA120 bullish / reverse bearish / else neutral, B = A plus MA200; same for
  EMA; `rsi_zone` 30/70 (RSI 14, Wilder); `macd_signal` = golden/dead cross of (12,26,9) within the last 3 bars, else
  none, plus histogram sign; KDJ (9,3,3) zone 20/80 and K/D cross; SAR (0.02, 0.2) side; BOLL (20, 2) %B zone (< 0,
  0–1, > 1) and bandwidth percentile over 120 bars; ADX(14) bucket (< 20, 20–40, > 40) × DI dominance; WR(14) zone
  −80/−20; CCI(20) zone ±100. Aggregate = bullish count − bearish count per timeframe and over all; gate-like `signal`
  = sign of the aggregate.
- **Test (card `research/gate_features/cards/E-gate-001.md`, before any result):** decision every closed 1h bar on the
  10-coin 5m dataset; entry at the next bar's open. Targets: direction of the 4h and 24h forward return; magnitude =
  next 4h and 24h range in NATR units. Models: logistic (one-hot) and a GBM. Capacity controls of RESEARCH_RULES §2.1
  are mandatory: effective sample size, k-sweep k = 1, 3, 5, 10, 20, all (ranked on train only), learning curve
  25/50/100 %, shuffled-label null (≥ 20 shuffles), and the comparison old vs new set (raw indicator values vs the
  state set, same model) that F-0027 R2 asks for. Baselines: intercept (direction), NATR-only (magnitude). Splits:
  train 2018-01..2022-12, val 2023-01..2024-12 (selects), 2025-01..2026-09 compares only (already touched by earlier
  studies); confirmation once on sealed bucket H5 (Drive 15m, Colab).
- **Accept rule (fixed now):** direction — val AUC ≥ 0.53 at a horizon AND above the shuffled null's 99th percentile
  AND gain over the intercept in ≥ 4 of the 5 calendar years 2021–2025 AND the k-sweep peak is not matched by the
  shuffled-label fit; then, economics level: top-decile gross per-trade move > the 0.14 % round trip at 1× and 2×
  (F-0036's invariant: 1h direction information existed but cost ate it). Magnitude — ≥ 2 % relative improvement in
  pinball loss over NATR-only on val, same sign every year. Anything else → registry rows with the level where it failed.
- **Registry:** expected `reopens` — F-0027:R1 (the 15m archive of 800+ coins, used for the H5 confirmation) and
  F-0027:R2 (old vs new set inside a model), F-0056:R2 (capacity controls run), F-0039:R2 (capacity sweep),
  F-0009:R2 (BOLL bandwidth against a future-volatility target); `not_covered_by` for F-0036 / F-0054 and every other
  row `covering` lists must name a concrete scope difference. Tags from the vocabulary (`indicator-signal;
  trend-following; momentum; mean-reversion; volatility-structure`, `timeframe: 1h`, `target: direction; magnitude`,
  `model_class: logistic; gbm`).
- **DoD:** `tools/gate_features.py` with look-ahead perturbation tests (changing bars after t leaves states at t
  unchanged) and NaN-not-0 tests; card committed first; run bundle (in the build session) reproduced by the planner;
  verdict per target; if a set is accepted, it is registered as an R6 model input and (for direction only after the
  economics check) a bot2 tool spec `get_tech_states(symbol)`; the bot2 note says whether a gate-like `signal` carried
  information (with the AUC and its null).
- **Compute:** build session (10 coins, ≤ 40 min) + one Colab confirmation (H5, ≤ 20 min).

## G — gate-cli: reverse-engineer its derived fields, rebuild them, QC the rest (from PR #16; needs the owner's raw JSON)
- **Goal:** the gate agent's data quotes indicators and summary fields (`technical_view` per timeframe: `ma_alignment`,
  `macd_signal`, `rsi_zone`, the aggregate `signal`; kline indicators: MA/EMA 7/30/120/200, BOLL, MACD, KDJ, RSI, WR,
  CCI, ADX/DI, SAR). (1) Recover their exact definitions with `tools/indicator_reverse.py` (PR #16): parameters,
  windows, and the rule that maps indicators to `signal`. (2) Rebuild them as our own code-computed features from our
  OHLCV (`tools/gate_features.py`, created in G2: G replaces its `gate-style-v1` candidates by recovered definitions), so neither a model nor bot2 depends on gate's delivery bugs (zero-as-missing,
  wrong symbol match, frozen series). (3) For fields we cannot rebuild (funding, OI, long/short ratios, taker volume,
  exchange flows/reserves, ETF flows), a field-level QC verdict (`ok/suspect/invalid/unverifiable`, #16 rules R01–R19).
- **Inputs (stage 0, owner):** raw gate-cli JSON saved to files directly by the CLI (`> file.json`, no `head -c`, no
  LLM summary — a pasted model summary is not evidence): the nine calls of #16 §3, plus `kline --with-indicators` for
  BTC, ETH, SOL at 15m/1h/4h/1d with the largest `--size` allowed, and the matching `technical_view` calls at the same
  minute. The build session commits them under `research/gate_data_qc/fixtures/raw/`.
- **Reviewer decisions:** a definition is "recovered" only if it reproduces the provider's values on ≥ 95 % of
  non-missing bars within 1e-4 relative error (the provider's own bars as OHLCV, so no exchange mismatch); the
  `signal` rule counts as recovered only if it reproduces every observed `signal` value — otherwise it is recorded as
  "not reproducible" and never rebuilt or used. QC rules are not experiments (no card per rule); thresholds live in
  `contract.json`. Agent-side rules (#16 §5 stage 6) go to bot2 as a note.
- **What this does and does not add:** rebuilt OHLCV indicators are a clean tool, not new information — single standard
  indicators are already closed as signals (F-0002, F-0004, F-0009, F-0026, F-0027). The non-OHLCV fields are new
  information; they become R2-style candidates only once enough point-in-time history exists (gate history depth, or
  bot2's recorder), each with its own card.
- **DoD:** fit report per indicator × timeframe (recovered / not reproducible, with error); `tools/gate_features.py`
  (card + round-trip tests against the raw fixtures); `contract.json` (unknown keys raise) and `tools/gate_qc.py`
  R01–R19 as registered functions (unregistered name raises); tests per #16 §6 (positive + negative per rule, recall
  100 % on confirmed items, truncated JSON raises); false-positive rate per rule < 1 % on a clean fixture; a list of the
  non-OHLCV fields with their available history depth.
- **Card / registry / holdout:** none (no signal claim). **Colab:** none (gate-cli runs on the owner's machine).

## R3 — NATR range forecast with a calibration report → model feature/target and bot2 tool spec
- **Goal:** turn the one robust survivor (H003: NATR_14 forecasts next range *width*, not direction) into a calibrated
  forecast: p10/p50/p90 of the next bar's range (percent of close), plus separate up- and down-excursion, at 1h and 1d.
- **Method (fixed in the card):** quantile regression of log next-range on log NATR_14 (one model per timeframe, pooled
  over coins); train 2020-01..2024-12, val 2025-01..2026-09 on unsealed coins; baseline = per-coin trailing 90-bar
  empirical quantiles. NATR deciles fixed on train. Slices fixed in advance: NATR decile × timeframe × {BTC/ETH, others} × year.
- **Accept rule (holdout H2, one run):** overall [p10, p90] coverage 80 ± 3 pp; every NATR decile 80 ± 6 pp; pinball loss
  below the baseline in ≥ 8/10 deciles; monthly rank IC(NATR, next range) > 0 in ≥ 90 % of months.
- **Card:** `research/natr_range/cards/E-natr-001.md`, scope `mechanism: volatility-estimator; volatility-structure`,
  `target: magnitude`, `model_class: regression`. **Registry:** cite every row `covering` reports (expect F-0005, F-0006,
  F-0021, F-0057): not a re-run because those rows fail on direction/high-low IC screens or on P&L, while this phase
  claims only magnitude calibration and trades nothing. The reviewer verifies each declaration.
- **DoD:** card committed before results; val calibration report; one holdout run logged in the access log; verdict in
  the card; if accepted, `research/natr_range/BOT2_HANDOFF.md`: tool `get_range_forecast(symbol, interval)` →
  `{p10, p50, p90, up_p50, down_p50 (percent), natr_pct, natr_decile, model_version, calibration_asof}`, units in percent
  as bot2 uses them, and how it differs from bot2's existing `range_pct` / `natr` (a forecast with stated coverage, not a
  current measure). Not a direction signal (bot2 ADR 0003). The same forecast is registered as a candidate model
  feature / sizing target for R6 (pipeline column with a card and a test).
- **Colab:** 2 runs (val; holdout once), ≤ 20 min each.

## R1 — Chart-structure confluence, stage 1: exploration map (🔒 stop for the owner after it)
- **Goal:** test, as exploration, whether **overlapping price zones** from different chart-structure detectors behave
  differently from single zones and from random zones. Output a relationship map. Exploration is not evidence.
- **Detectors (causal; each zone carries `valid_from` = the close of the bar that confirms it):** S/R from swing pivots
  (fractal w = 5, confirmed 5 bars later; zone = pivot ± 0.25 ATR); FVG (valid after the 3rd candle closes, dies when
  filled); order block (last opposite candle before a ≥ 1.5 ATR displacement, valid after the displacement closes);
  Fibonacci 0.5–0.618 of the last leg whose both pivots are confirmed; liquidity pool (two confirmed pivots within
  0.1 ATR; sweep = wick through and close back); Ichimoku cloud as drawn at t (spans built from data ≤ t; Chikou excluded).
  Zone timeframes 1h and 4h; ATR(14) of the zone timeframe.
- **Event and outcomes:** a touch = the first 15m bar that trades into a valid zone; side from the approach (from above
  → long, from below → short); entry = that bar's close; R = 1 ATR. Over 96 × 15m bars: MFE/MAE in ATR, P(+1R before −1R),
  P(+2R before −1R), net R of a +1.5R / −1R / time-exit bracket (stop-first).
- **Cells:** 6 singles + 15 pairs + 20 triples (overlap = intervals intersect, both valid at the touch) × 2 sides × 2
  zone timeframes = 164 cells. Lift = cell metric − best member's metric on touches where that member is alone.
- **Null:** random zones with the same count, widths and `valid_from`, placed uniformly within ±3 ATR of price at
  `valid_from` (20 seeded repetitions).
- **Universe / periods:** the 10-coin dataset with the exact 5m path (1m where a 5m bar touches both levels); P1
  2019-01..2022-12, P2 2023-01..2026-09. Runs in the build session. R1b confirms on sealed Drive coins (15m, stop-first),
  so the R1 report also gives every candidate's metrics under stop-first on 15m bars, to show the confirmation's handicap.
- **Pre-registered "worth continuing" bar (fixed now):** a cell is a candidate iff n ≥ 300 touches on ≥ 8 coins; gross
  mean R above the null's 95th percentile by ≥ 0.10R; net mean R > 0 at 1× cost; lift ≥ +0.05R; same sign in P1 and P2;
  and it survives BH-FDR q = 0.10 over all 164 cells on P(+1R first) vs the null. No candidate → the phase closes with
  registry rows. Candidates → the owner decides whether R1b runs.
- **Card:** `research/confluence/cards/E-conf-001.md`, scope tags from the vocabulary (`chart-pattern; range-fade;
  breakout; mean-reversion`, `timeframe: 1h; 4h`, `target: high-low; pnl`, `model_class: rule`), never a new spelling to
  escape coverage. **Registry:** expect F-0054 (classic setups, cost ≥ 0.10R at 15m/1h — the card reports cost in R per
  zone timeframe and does not claim to satisfy its `reopen_if`; it declares the concrete scope difference: multi-detector
  overlapping zones on 1h/4h ATR, which F-0054 did not test), F-0053 (prev-candle extremes are coin flips — different
  level definition), F-0062, F-0060/F-0061, F-0019 (`reopens: F-0019:R1` only if covered: structure from the intraday
  path), plus #15's F-0068/F-0073/F-0074 once merged. The reviewer rejects any declaration that is not concrete.
- **DoD:** detector look-ahead tests (perturbing bars after `valid_from` changes nothing up to `valid_from`), touch/outcome
  tests on hand-built fixtures, the map (CSV + markdown heatmap) from one run, the candidate list mechanically
  checked against the bar. **Compute:** build session, ≤ 50 min; reproduced by the planner.

## R1b — Confluence: refine and confirm (🔒 starts only if R1 has candidates and the owner says go)
- **Goal:** at most 3 rules chosen from R1 candidates on P1/P2 only, frozen in the card (parameters + sha256 of the rule
  file), then one run on sealed bucket H1.
- **Accept rule (fixed before the run):** per rule, net mean R > 0 at 1× and ≥ 0 at 2× cost; day-clustered t ≥ 2 (Holm
  over the ≤ 3 rules); still > 0 after removing the best 5 % of trades; same sign in ≥ 60 % of coins with ≥ 20 trades.
- **DoD:** verdict per rule; pass → bot2 hand-off spec (`get_zones(symbol, tf)` → zones, confluence tag, frozen
  historical stats); fail → registry rows. **Colab:** 1 run.

## R2 — Funding carry / basis (🔒 data acquisition first)
- **Goal:** test the one untested information source not derivable from OHLCV. Two pre-registered hypotheses only:
  (a) delta-neutral carry — long spot / short perp while trailing 7-day annualised funding > a threshold fixed in the
  card, net of four legs' costs; (b) cross-sectional funding rank → next 7-day perp return, market-neutral.
- **Data step:** `tools/fetch_funding.py` run on Colab (Binance USDT-M funding history and premium-index klines from
  `data.binance.vision`). If unreachable from Colab, the phase stops with a blocker; no proxy series.
- **Splits:** DEV 2021–2023, VAL 2024-01..2025-06 (unsealed perps); confirmation once on H3 ∩ perps with funding.
- **Accept rule:** net > 0 at 1× and 2×; day-clustered t ≥ 2; best-5 % removal; ≥ 60 % per-coin consistency; Deflated
  Sharpe with the count of every configuration.
- **Card:** `research/funding_carry/cards/E-fund-001.md`. **Registry:** `reopens: F-0042:R1` (a source not derivable
  from hourly OI/price/long-short ratio) and every other row `covering` lists (expect F-0043/F-0044 positioning).
- **Colab:** fetch (~20 min) + DEV/VAL run + one confirmation run.

## R5 — Confirm E-disc S1/S3 on sealed coins (optional; after T0 merges #15)
- **Goal:** #15 left two daily breakout candidates (S1, S3) that passed on BCH/TRX/ZEC. Freeze them exactly as in
  #15's cards and run once on bucket H4 with the evidence bar above. Pass → bot2 hand-off note; fail → registry rows.
- **Card:** `research/strategy_discovery/cards/E-disc-007.md` citing #15's rows. **Colab:** 1 run.

## R4 — H07 on longer history and forward data (parked)
- **Goal:** the frozen H07 rule (TSMOM30, BTC/ETH, long-only) on (a) the earliest 15m BTC/ETH history on Drive
  (pre-2018 if present) and (b) forward bars ≥ 2026-10-01 once ≥ 6 months exist. No parameter changes.
- **Waits for:** T0 decision on #13 (its H07 forward/vol-sizing work) and enough forward data.
- **DoD:** report vs buy-and-hold (Sharpe, max drawdown, 2× cost); bot2 hand-off note as an exposure filter (bot2
  already has a `tsmom` indicator — the note states the validated parameters and limits).

## R6 — Model integration of survivors (🔒 starts when at least one new input survives)
- **Goal:** add every surviving input (e.g. R3 range forecast as a feature or target, R2 funding features, G's
  non-OHLCV fields once they have history) to the model pipeline and test whether the model improves.
- **Method:** `MODEL_DEV_PROTOCOL.md`: one card per change, ≥ 3 seeds, feature-count sweep, learning curve, simple
  (logistic/ridge) baseline, the per-layer health report; the predicted diagnostic must move. Training runs on Colab
  within D-003 (a small model; larger runs need an owner decision).
- **Holdout:** a bucket not used by the input's own confirmation (assigned at R6 kickoff from unopened buckets or the
  time seal).
- **DoD:** accepted / rejected / unexplained per card; accepted → the model ships with its card, or a bot2 tool spec
  wrapping its output; rejected → registry rows.
