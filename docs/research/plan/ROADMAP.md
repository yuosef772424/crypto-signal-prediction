# Research roadmap (2026-10 →)

Goal (D-001): find or reject, with honest evidence, signals worth exposing to the bot2 trading agent as
code-computed tools. Anything that survives gets a hand-off note for bot2 (`ai_agent/docs/TOOLS.md` format there).
A rejected phase is a valid result: it is closed with `failure_registry.csv` rows (verified cause + `reopen_if`).

Rules that apply to every phase: `CLAUDE.md`, `docs/research/RESEARCH_RULES.md`, `docs/research/MODEL_DEV_PROTOCOL.md`
(model phases only), `docs/research/audit/PROTOCOL.md`. Each phase is **one branch → one PR**, built by the build
session, reviewed by the planner (KICKOFF.md), merged only when its DoD holds and STATUS.md is updated.
Phases marked 🔒 need an explicit owner decision before they start.

```
T0 triage open PRs ─▶ R0 lock (+M) ─▶ G gate-cli QC ─▶ R3 NATR range tool ─▶ R1 confluence map 🔒─▶ R1b confirm
                                  └─▶ R5 confirm E-disc S1/S3 (optional, cheap)            R2 funding/basis 🔒 (data first)
                                                                                           R4 H07 forward (parked)
```
Order is the planner's proposal (D-005, pending owner confirmation).

## Common contracts

**Data (D-004).** Drive holds `history_15m` (800+ coins, compressed). Coarser bars (1h/4h/1d) are resampled from 15m
inside the scripts (UTC, label = bar open, a bar is used only after it closes). There is no 5m path for most coins:
intrabar order inside a 15m bar is unknown, so every bracket uses **stop-first** on ambiguity (the honest lower bound,
as in `tools/bracket_eval.py`). No funding/basis history is on Drive (R2 starts with an acquisition step).

**Colab run contract (D-003, ≤ 1 h per phase).** The builder delivers scripts + tests on small fixtures (CPU, no Drive).
The owner runs one Colab cell per run. Every run script:
- takes `--config <json>` (unknown keys raise) and `--out <dir>`, and stops cleanly at `--max-minutes` (default 50);
- reads data only through `tools/holdout_guard.py` (after R0);
- writes `manifest.json` (git SHA, config, input fingerprint = sha256 of the sorted input file list + sizes, row counts,
  runtime, **number of configurations evaluated**, sha256 of every output) and outputs ≤ 300 KB in total;
- prints a paste bundle (manifest + small CSVs) the owner pastes into the build session; the builder commits it
  verbatim as an `exp` commit under `research/<study>/results/<run_id>/`. The reviewer re-checks the sha256 values.

**Costs.** Taker 0.05 % + 0.02 % slippage per side (0.14 % round trip) unless the card says otherwise; every trade
result is also reported at 2× cost. Costs in R = round-trip cost / stop distance, per trade.

**Evidence bar (every trading claim).** Net of cost at 1× and 2×; day-clustered t; result after removing the best 5 %
of trades; per-coin consistency (share of coins with ≥ 20 trades whose mean has the same sign); every configuration
tried is counted (BH-FDR or Holm as the card states; Deflated Sharpe where a Sharpe is claimed).

**Holdout (R0 onwards).** Sealed coins are split into four disjoint buckets H1–H4, one per confirmation phase. A bucket
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
  columns, `research/*/README.md`, merged T0 PRs): `docs/research/plan/lock/touched.csv` (symbol or rule, timeframe,
  period, source).
- **Reviewer decisions:** eligible = ≥ 2 years of 15m history, last bar ≥ 2026-09-01, not a stablecoin / leveraged token,
  **not touched at any timeframe**. Sealed = 40 % of eligible coins ranked by `sha256(salt + symbol)`, where `salt` = the
  40-char SHA of the commit that adds the R0 card (committed before the manifest is pasted). Sealed coins → buckets
  H1 (R1b), H2 (R3), H3 (R2), H4 (R5) by the same hash, round-robin. Time seal: open time ≥ 2026-10-01, all coins.
- **Deliverables:** `tools/holdout_guard.py` (card + test): `load_unsealed(...)`, `assert_unsealed(symbols, start, end)`,
  `open_holdout(card_path, bucket)` (refuses a second open of a bucket, refuses a card without `confirmation: true` and
  a frozen-rules hash); `docs/research/plan/lock/holdout.json` + `holdout.sha256`; empty `holdout_access_log.csv` with a header.
- **DoD:** tests show a sealed symbol / a time-sealed bar raises through every public loader, and a second
  `open_holdout` on the same bucket raises; lock files committed; checks pass. After merge the lock files never change
  except appended access-log rows.
- **Card:** none (infrastructure). **Registry:** none. **Colab:** one run of `data_manifest.py` (~5 min).

## G — gate-cli data QC (from PR #16's plan)
- **Goal:** a field-level QC verdict (`ok/suspect/invalid/unverifiable`) for gate-cli responses, so no bot2 tool or
  study consumes a broken field (zero-as-missing indicators, wrong symbol match, `avg_mid_price_1d` = depth/2, stale data).
- **Inputs:** `research/gate_data_qc/README.md` (§3–§6). Stage 0 is the owner's: re-run the nine calls without `head -c`
  and paste the raw JSON (the build session commits them under `research/gate_data_qc/fixtures/raw/`).
- **Reviewer decisions:** QC rules are not experiments — no experiment cards per rule (the #16 plan's §5 stage 2 card
  step is dropped); thresholds measured on fixtures are recorded in `contract.json`; registry rows only if a study later
  fails because of a field. Agent-side rules (#16 §5 stage 6) go to bot2 as a hand-off note, not into this repo.
- **DoD:** `contract.json` (unknown keys raise), `tools/gate_qc.py` R01–R19 as registered functions (unregistered name
  raises), tests per #16 §6 (positive + negative per rule, recall 100 % on confirmed items, truncated JSON raises);
  false-positive rate per rule < 1 % on a clean 30-day BTC/ETH/SOL fixture; bot2 hand-off note.
- **Card / registry / holdout:** none. **Colab:** none (gate-cli runs on the owner's machine).

## R3 — NATR range forecast with a calibration report → bot2 tool spec
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
  current measure). Not a direction signal (bot2 ADR 0003).
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
- **Universe / periods:** the 20 unsealed coins with the highest median quote volume; P1 2022-01..2024-04, P2 2024-05..2026-09.
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
  tests on hand-built fixtures, the map (CSV + markdown heatmap) from one Colab run, the candidate list mechanically
  checked against the bar. **Colab:** 1 run, ≤ 50 min.

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
