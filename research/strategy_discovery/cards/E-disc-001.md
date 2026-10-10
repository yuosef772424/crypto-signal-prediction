---
id: E-disc-001
mechanism: trend-following; breakout; mean-reversion; volatility-structure; indicator-signal; regime-filter
timeframe: 4h;1d
target: pnl
model_class: rule
reopens: F-0054:R1; F-0068:R1
not_covered_by: F-0072=volume-confirmed Donchian on unseen assets, a different trigger and filter from this card; F-0070=the momentum rotation of E-disc-004, a portfolio of daily returns, not a trade grid on breakout signals; F-0071=S2 of this grid (1d bb_break with ADX gate), failed later on unseen assets in E-disc-003, the grid selected it on VAL before that test, so it is the source of the row through its survivor status; F-0069=daily ADX>40 removal and short-side removal, E-disc-001 has no ADX>40 cut and keeps both sides in the grid, the daily breakouts here are a different test on the same data family; F-0066=RSI(14) 30/70 crossings as a reversal trigger, this grid uses RSI(2) 10/90 crossings with an EMA200 alignment gate (a different trigger, and the trend-bias conditioning of F-0066 R1 is judged here by the rule below); F-0067=MACD histogram zero-cross, family removed from the grid so that no closed failure is re-run; F-0055=maker limit entry at EMA20 on 1h, this grid uses market fills at the next open on 4h and 1d; F-0061=stop orders at predicted candle extremes, no predicted level here; F-0062=regime filters on wick-capture fills, the ADX and EMA200 gates here are entry conditions on indicator triggers; F-0064=hedge-and-hold, no hedging here; F-0065=reversal hedge, no hedging here
---
# Experiment card — E-disc-001: discovery grid on 4h and 1d, DEV and VAL, with multiple-testing control

Written and committed BEFORE the run (rules: `docs/research/RESEARCH_RULES.md`, `docs/research/MODEL_DEV_PROTOCOL.md`).
This is a DISCOVERY screen: it chooses which configurations deserve the robustness and unseen-asset tests. It does not declare a strategy.

| Field | Value |
|---|---|
| Date / author | 2026-10-10 / Claude, on branch `claude/strategy-discovery` |
| Baseline | data: Speirsy11/crypto-dataset 1h spot, the seven original coins (BTC ETH SOL BNB XRP ADA DOGE), resampled to 4h and 1d (UTC, left-labelled), `/home/user/research/ohlc`, through 2026-09-30 23:00. The engine is `tools/trade_engine.py` (tests in `tests/test_trade_engine.py`) |
| The ONE change | the signal family and gate; the exit is fixed for every configuration |
| Failure being addressed | none yet. Reopens F-0054 R1 (cost on 1d and 4h within 0.03R, as measured in E-ind-001). F-0068 R1 is reopened for the 4h breakout and pullback families: its condition is a pre-registered regime gate (ADX ≥ 25 and EMA200 alignment, both fixed here) judged on VAL first; TEST is held for survivors |
| Data / splits | DEV 2019-01-01→2021-12-31 and VAL 2022-01-01→2023-12-31 only. TEST (2024→2026-09) is NOT loaded here. BCH, TRX and ZEC are NOT loaded here (reserved for the unseen-asset test). The 2026-10 bars are NOT loaded here (reserved for the fresh holdout) |
| Seeds / budget | deterministic; 7 families × 3 gates × 2 timeframes = 42 configurations × 7 coins × 2 splits; CPU, minutes |

## Families (fixed before the run)
- `ema_pullback`: EMA20 > EMA50, low ≤ EMA20, close > EMA20 (long); mirror for short.
- `donchian20`: close above the prior 20-bar high (long) / below the prior 20-bar low (short).
- `donchian55`: the same with 55 bars.
- `bb_break`: close crosses the Bollinger (20, 2) band outward.
- `adx_trend`: ADX(14) crosses 25 with +DI or −DI leading.
- `rsi2_pullback`: RSI(2) crosses up through 10 (long) / down through 90 (short).
- `squeeze_break`: the previous bar's Bollinger bandwidth is in the lowest 20% of its last 120 bars; then close breaks the band outward.
- `macd_cross` is deliberately NOT in the grid (closed failure F-0067).

## Gates (fixed before the run)
- `none`; `adx25` (ADX ≥ 25 at the signal bar); `ema200_aligned` (long only if close > EMA200, short only if close < EMA200).

## Exit (fixed for every configuration)
Stop 2 × ATR(14) at the signal bar; target 3 × ATR(14); time exit 30 bars; stop first inside a bar; gap through stop fills at the open.
Entry: market at the open of the bar after the signal. One position per coin. Cost 0.12% round trip, in R.

## Accept / reject rule (discovery, written before the run)
A configuration is a **survivor** for the robustness card only if ALL of:
1. pooled mean net R > 0 on DEV AND on VAL;
2. Benjamini-Hochberg q-value of the one-sided VAL t-statistic, over the 42 configurations, ≤ 0.10;
3. VAL profit factor ≥ 1.10;
4. VAL net R is positive on at least 5 of the 7 coins.
Every configuration's numbers are reported, including those that fail. A survivor is a candidate for the next card, not a finding.

## Predicted diagnostic movement
- The 1d configurations should show the largest net R, because the cost in R is about 0.01 on 1d and about 0.03 on 4h.
- Mean-reversion (`rsi2_pullback` without a gate) should be weaker than trend or breakout families. A positive result there would contradict the expectation and is reported.
- The number of survivors should be small, perhaps 0 to 3. Zero survivors is a valid outcome.

---
## Result (added after the run)
Run: `research/strategy_discovery/grid.py` → `results/E-disc-001_summary.csv` (42 configurations × 7 coins × DEV and VAL; the trade log is kept outside the repo, `/home/user/research/strategy_discovery/`).

Survivors under the rule (all four conditions): **3 of 42**, all daily Donchian or Bollinger breakouts.

| Configuration | DEV net R | VAL net R | VAL pf | VAL t | BH q (VAL) | VAL coins+ | Survivor |
|---|---|---|---|---|---|---|---|
| 1d donchian20 + adx25 | +0.161 | +0.306 | 1.76 | 3.08 | 0.043 | 5/7 | yes |
| 1d bb_break + adx25 | +0.172 | +0.312 | 1.70 | 2.52 | 0.081 | 6/7 | yes |
| 1d donchian20, no gate | +0.252 | +0.196 | 1.43 | 2.57 | 0.081 | 7/7 | yes |
| 1d bb_break, no gate | +0.259 | +0.179 | 1.36 | 2.09 | 0.151 | 6/7 | no (q > 0.10) |

Other notable results:
- The 4h breakouts and the 4h pullbacks are positive on DEV and near zero on VAL (for example 4h bb_break no gate: +0.100 → +0.053), with BH q between 0.16 and 0.23. They do not pass. This is the F-0068 pattern again, so F-0068 R1 is not satisfied.
- RSI(2) pullbacks are negative on both splits on both timeframes (1d no gate: DEV −0.234, VAL −0.119). Registry row not added: they are a different trigger from F-0066, and the failure is clear.
- ADX trend and squeeze breaks have few trades and VAL t below 1.5.

Important caveat: **VAL was already used in E-ind-001**, which chose the same daily breakouts. These survivors are a re-discovery with a multiple-testing correction, not new evidence. The new evidence is in E-disc-002 (stability) and E-disc-003 (unseen assets and fresh bars).

- Verdict: **screen complete; 3 survivors go to E-disc-002.** No new registry row (no new failure beyond those already recorded; the 4h and RSI(2) results are consistent with F-0066 and F-0068).
