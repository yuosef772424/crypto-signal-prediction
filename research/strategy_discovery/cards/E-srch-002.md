---
id: E-srch-002
mechanism: momentum; indicator-signal
timeframe: 15m;1h
target: pnl
model_class: rule
reopens:
not_covered_by: F-0073=this card is the source of F-0073 (its own intraday result); F-0067=MACD histogram zero-cross on 4h and 1d (closed there), this card tests the same trigger on 15m and 1h, which F-0067 does not cover by timeframe; F-0054=chart-setup study at 15m and 1h closed by round-trip cost of 0.21 to 0.38R (15m) and 0.10 to 0.18R (1h): this card is a momentum trigger that is not a chart setup, and its cost in R is measured and reported, so the cost barrier is tested rather than assumed; F-0055=maker limit pullback, no limit orders here
---
# Experiment card — E-srch-002: MACD momentum on 15m and 1h (first intraday test)

Written and committed BEFORE the run (rules: `docs/research/RESEARCH_RULES.md`, `docs/research/MODEL_DEV_PROTOCOL.md`).

## Why only this family
The registry blocks most families on 15m and 1h: their closed failure F-0054 (chart setups, cost in R 0.21–0.38R on 15m and 0.10–0.18R on 1h)
covers breakout and trend-pullback triggers at those timeframes, and its reopen condition (round-trip cost ≤ 0.03R) does not hold.
MACD is not covered by F-0054 and F-0067 is closed only on 4h and 1d, so MACD on 15m and 1h is the one intraday family that the
registry does not block. Other intraday families are NOT run here; see README.

| Field | Value |
|---|---|
| Date / author | 2026-10-10 / Claude, on branch `claude/strategy-discovery` |
| Data | 15m and 1h spot OHLCV, ten coins, `/home/user/research/ohlc15` (15m) and `/home/user/research/ohlc_full` (1h), through 2026-10-08 |
| The ONE change | the timeframe (15m or 1h) for the MACD trigger; the gates and exits are fixed |
| Splits | DEV 2019-01-01→2021-12-31, VAL 2022-01-01→2023-12-31 on the original seven coins. Unseen assets BCH, TRX, ZEC and the fresh bars 2026-10-01→2026-10-08 are reserved for a later card |
| Budget | 2 timeframes × 3 gates = 6 configurations; BH over these 6 |

## Definitions (fixed before the run)
- Trigger: MACD(12, 26, 9) histogram crosses zero up (long) or down (short), as in the shared engine.
- Gates: `none`, `adx25` (ADX ≥ 25), `ema200_aligned` (long above EMA200, short below).
- Entry: market at the next bar's open. One position per coin.
- Exit: stop 2 × ATR(14), target 3 × ATR(14), time exit 96 bars on 15m (one day) and 48 bars on 1h (two days). Stop first inside a bar.
- Cost: 0.12% round trip, in R (the card reports the measured median cost in R for each timeframe).

## Accept rule (written before the run)
A configuration is a survivor only if ALL of:
1. pooled mean net R > 0 on DEV AND on VAL;
2. Benjamini-Hochberg q of the one-sided VAL t-statistic over the 6 configurations ≤ 0.10;
3. VAL profit factor ≥ 1.10;
4. VAL net R positive on at least 5 of the seven coins.
A survivor is not a finding; it still needs the unseen-asset test and the fresh window in a later card.

## Predicted diagnostic movement
- The measured cost in R on 15m should be roughly 0.1–0.3R, so a positive gross edge of similar size would be needed. A negative result on 15m
  with a small gross is the expected outcome, and the card reports gross R next to net R to show which barrier applies.
- On 1h the cost in R should be smaller; a result that is positive gross but negative net would confirm the F-0054 cost barrier.

---
## Result (added after the run)
Run: `research/strategy_discovery/intraday_macd.py` → `results/E-srch-002_summary.csv` (six configurations, DEV and VAL, seven original coins).

| Config | DEV net R (gross) | VAL net R (gross) | Median cost in R (VAL) | VAL t | BH q | Passes |
|---|---|---|---|---|---|---|
| 15m, no gate | −0.101 (−0.000) | −0.157 (−0.008) | 0.120 | −16.7 | 1.0 | no |
| 15m, adx25 | −0.121 (−0.026) | −0.163 (−0.025) | 0.112 | −14.2 | 1.0 | no |
| 15m, ema200_aligned | −0.106 (+0.003) | −0.147 (+0.015) | 0.133 | −13.5 | 1.0 | no |
| 1h, no gate | −0.035 (+0.015) | −0.046 (+0.023) | 0.058 | −2.7 | 1.0 | no |
| 1h, adx25 | −0.083 (−0.035) | −0.044 (+0.021) | 0.055 | −2.1 | 1.0 | no |
| 1h, ema200_aligned | −0.013 (+0.039) | −0.071 (+0.003) | 0.060 | −3.4 | 1.0 | no |

No configuration passes. Gross R is about zero on 15m for every gate, and on 1h it is positive but smaller than the median cost in R (0.05–0.06R).
This matches the cost barrier recorded for F-0054: the signal has no gross edge large enough to survive the cost at these timeframes.
Verdict: **rejected.** Registry row F-0073 is recorded.
