---
id: E-ind-001
mechanism: indicator-signal; breakout; trend-following; mean-reversion; regime-filter
timeframe: 4h;1d
target: pnl
model_class: rule
reopens: F-0054:R1
not_covered_by: F-0072=volume-confirmed Donchian on unseen assets, a different trigger and filter from this card; F-0070=scope differs, the cross-sectional momentum rotation, a portfolio with turnover costs, not a single-asset trade; F-0071=scope differs, the Bollinger breakout with ADX gate on unseen assets, a different trigger; F-0066=this card is the source of F-0066 (its own RSI result); F-0067=this card is the source of F-0067 (its own MACD result); F-0068=this card is the source of F-0068 (its own 4h result); F-0069=E-ind-002, added after its run as a compliance declaration only (it removes shorts and ADX > 40 entries from the daily breakouts, a later refinement of this screen); F-0055=maker limit entry at EMA20 with scaling on 1h, this screen uses market fills at the next open, a fixed ATR stop and target, 4h and 1d only, and tests six signal families, not one pullback; F-0061=stop orders placed at PREDICTED candle extremes, this screen uses indicator triggers on closes and no predicted level; F-0062=regime filters on wick-capture fills, the ADX gate here is an entry filter on trigger signals, not a fill-side filter; F-0064=E-hedge-001 hedge-and-hold, no hedging here, every trade has a stop; F-0065=E-hedge-002 reversal hedge, no hedging here
---
# Experiment card — E-ind-001: indicator screen on 4h and 1d, fixed ATR stop, target and time exit

Written and committed BEFORE the run (rules: `docs/research/RESEARCH_RULES.md`, `docs/research/MODEL_DEV_PROTOCOL.md`).
This is a SCREEN: it chooses which signal families deserve a follow-up, it does not declare a strategy. It is one change in
the sense that every configuration uses the same exit and the same cost model; only the signal differs.

| Field | Value |
|---|---|
| Date / author | 2026-10-10 / Claude (for the project owner) |
| Baseline (commit, dataset fingerprint) | this card's commit; Speirsy11/crypto-dataset 1h spot OHLCV through 2026-09-30 23:00, resampled to 4h and 1d (UTC, left-labelled), BTC ETH SOL BNB XRP ADA DOGE |
| The ONE change | the signal family (trigger, optionally one gate); the exit is fixed for all configurations |
| Failure being addressed | none yet. Reopens F-0054 R1: the measured median round-trip cost is 0.027R on 4h and 0.010R on 1d (≤ 0.03R on both), which is the condition the registry names for reopening the chart-setup family on those timeframes. Measured on 2026-10-10 from the same data, before this card was committed |
| Data / splits / holdout | DEV 2019-01-01→2021-12-31 and VAL 2022-01-01→2023-12-31 are used for selection. TEST 2024-01-01→2026-09-30 is NOT touched by this card. TEST is consumed at most once, by a later card for the chosen configuration only |
| Seeds / budget | deterministic; 6 triggers × 2 gate settings × 2 timeframes = 24 configurations × 7 coins × 2 splits; CPU, minutes |

## Definitions (fixed before the run)
- **Triggers** (event at the close of bar t, using bars ≤ t):
  - `rsi_mr`: RSI(14) crosses up through 30 (long), down through 70 (short). Mean reversion.
  - `macd_cross`: MACD histogram (12, 26, 9) crosses 0 up (long) or down (short). Momentum.
  - `bb_break`: close crosses above the upper Bollinger band (20, 2) (long) or below the lower band (short). Breakout.
  - `adx_trend`: ADX(14) crosses up through 25 with +DI > −DI (long) or −DI > +DI (short). Trend start.
  - `donchian`: close above the prior 20-bar high (long) or below the prior 20-bar low (short). Trend breakout.
  - `ema_pullback`: EMA20 > EMA50, low ≤ EMA20, close > EMA20 (long); mirror for short. Trend pullback.
- **Gates:** none, or `adx25` (ADX ≥ 25 at the signal bar). The gate is applied to both directions.
- **Entry:** market at the OPEN of the bar after the signal. One position per coin at a time.
- **Exit (fixed):** stop = 2 × ATR(14) measured at the signal bar; target = 3 × ATR(14); time exit at the close 30 bars after
  entry. If a bar touches both stop and target, the stop is taken first (conservative). A gap through the stop fills at the open.
- **Cost:** 0.12% round trip per trade, expressed in R as cost / (stop distance as a fraction of price).
- **Unit of outcome:** net R per trade = gross R − cost in R.

## Predicted diagnostic movement
- Mean-reversion and momentum triggers (`rsi_mr`, `macd_cross`) should show negative or near-zero net R on both splits, because
  the literature the screen draws on reports that plain RSI mean reversion lost after costs. If they are positive on both splits,
  that contradicts the prediction and is reported.
- Breakout and trend triggers (`bb_break`, `donchian`, `adx_trend`) should show gross R above cost, with the ADX gate raising
  the profit factor and lowering the trade count. BTC on 4h over the full sample already shows net R above zero for these, which
  is a prediction of this card, not evidence for it.
- The gap between gross and net R should be close to the measured cost of about 0.01R (1d) and 0.03R (4h).

## Accept / reject rule (screen, written before the run)
A configuration is a **candidate for a follow-up card** only if ALL of:
1. pooled mean net R > 0 on DEV AND on VAL (pooled over the 7 coins);
2. profit factor > 1.0 on VAL;
3. VAL net R is positive on at least 5 of the 7 coins;
4. VAL t-statistic of the trade-level net R ≥ 2.0 (trades treated as independent; this is optimistic, so the threshold is kept).

The screen selects from the 24 configurations by these rules, not by TEST. Because 24 configurations are compared, a passing
configuration is a candidate, not a finding. Every configuration's numbers are reported, including those that fail.
Nothing in this card is a strategy until a later card passes its own rule on TEST.

## Verified cause and mechanism
- The screen tests whether any signal family has net R above zero after costs on a timeframe where the cost is small relative to
  the stop. The question it answers is which families are worth a follow-up, not whether a strategy makes money.

---
## Result (added after the run)
Run: `research/indicator_strategy/screen.py` → `E-ind-001_summary.csv` and `E-ind-001_trades.csv` (24 configurations × 7 coins × DEV and VAL; TEST not loaded).
Pooled net R per trade after cost. Cost measured on 2026-10-10: median 0.027R on 4h and 0.010R on 1d (reopen condition F-0054 R1 holds).

Configurations that pass the card's rule (all on 1d):
| Configuration | DEV net R | VAL net R | VAL pf | VAL t | VAL coins positive |
|---|---|---|---|---|---|
| bb_break + adx25 | +0.172 | +0.312 | 1.70 | 2.52 | 6/7 |
| donchian + adx25 | +0.161 | +0.306 | 1.76 | 3.08 | 5/7 |
| donchian (no gate) | +0.252 | +0.196 | 1.43 | 2.57 | 7/7 |
| bb_break (no gate) | +0.259 | +0.179 | 1.36 | 2.09 | 6/7 |

Failing, and why:
- `rsi_mr` on 4h and 1d: negative on DEV (4h −0.130R, 1d −0.300R); 0 of 7 coins positive on 4h DEV. Predicted, and confirmed. Registry F-0066.
- `macd_cross`: 4h negative on both splits; 1d negative on DEV. Registry F-0067.
- `bb_break`, `donchian`, `ema_pullback` on 4h: DEV positive, but VAL falls to 0.03 to 0.05R and VAL t below 2 (1.58 and 1.30). Registry F-0068.
- `adx_trend` on 1d: DEV +0.450R, VAL +0.068R, VAL t 0.54 (fails rule 4).
- `ema_pullback` on 1d: VAL −0.074R.
- On `adx_trend` the adx25 gate has no effect, because the trigger already requires ADX to cross 25.

- Failure level and location: economics for the mean-reversion and momentum families (gross R is negative or near zero, so cost only deepens it); generalization for the 4h breakouts (DEV edge halves in VAL).
- Verdict: **screen complete.** Four configurations are candidates for follow-up, all 1d breakouts. The screen does not establish an edge.
- Registry rows added: F-0066, F-0067, F-0068. F-0069 belongs to E-ind-002. Coverage declarations for this card are in its front matter (compliance, after the run; no result changed).
