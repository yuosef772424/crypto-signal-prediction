---
id: E-srch-003
mechanism: seasonality
timeframe: 15m
target: pnl
model_class: rule
reopens:
not_covered_by: F-0075=this card is the source of F-0075 (its own result); F-0046=hour-of-day seasonality on BTC and ETH at 1h with a 20-22 UTC long window and a return target, closed on HOLDOUT 2026, this card is a different timeframe (15m), all ten coins, a fixed-length hold for each hour in both directions, with the selection rule above; F-0054=chart setups closed on 15m by round-trip cost (0.21 to 0.38R), this card has no chart setup, it enters at the hour boundary, and its cost in basis points is measured and reported next to the gross return
---
# Experiment card — E-srch-003: time-of-day seasonality on 15m (24 hours × 3 holds × 2 directions)

Written and committed BEFORE the run (rules: `docs/research/RESEARCH_RULES.md`, `docs/research/MODEL_DEV_PROTOCOL.md`).

| Field | Value |
|---|---|
| Date / author | 2026-10-10 / Claude, on branch `claude/strategy-discovery` |
| Data | 15m spot OHLCV, seven original coins (DEV and VAL), `/home/user/research/ohlc15`, through 2026-10-08 |
| The ONE change | the hour of the day (UTC) at which the position opens, the holding length and the direction; the cost and the metric are fixed |
| Splits | DEV 2019-01-01→2021-12-31 (selection). VAL 2022-01-01→2023-12-31 (judgement). Unseen assets and fresh bars are reserved |
| Budget | 24 hours × 3 holds (4, 8, 16 bars = 1, 2, 4 hours) × 2 directions = 144 configurations; BH on the VAL t-statistics over all 144 |

## Rule (fixed before the run)
- Entry: at the open of the 15m bar whose start hour is h (UTC) and minute 0. Exit: at the close of the bar `hold` bars later.
- Direction: long (`+1`) or short (`-1`) as one configuration each.
- Return per trade: `dir × (close_exit / open_entry − 1) − 0.0012` (cost 0.12% round trip). No stop, no target: the hold is the only exit.
- Trades overlap by design (each hour opens a trade); pooled returns are averaged over all trades.

## Accept rule (written before the run)
A configuration is a survivor only if ALL of:
1. mean net return per trade > 0 on DEV AND on VAL;
2. Benjamini-Hochberg q of the one-sided VAL t-statistic over the 144 configurations ≤ 0.10;
3. VAL positive on at least 5 of the seven coins;
4. VAL trades: at least 500.
A survivor is a candidate for the unseen-asset test and the fresh window, not a finding. Note that an hour-of-day effect that is positive on DEV and on VAL is
still exposed to the multiple-testing count of 144 and to the cost of 12 bp per trade.

## Predicted diagnostic movement
- Gross moves of 15m bars over 1 to 4 hours are usually a few basis points; the cost of 12 bp per trade should make almost every
  configuration negative. The card reports gross return next to net, so the cost barrier is measured, not assumed.
- Long and short configurations should be close to mirror images if the effect is market-wide drift, and different if it is directional.

---
## Result (added after the run)
Run: `research/strategy_discovery/seasonality15m.py` → `results/E-srch-003_summary.csv` (144 configurations; DEV and VAL; seven original coins).

- **0 of 144** configurations pass the rule. None is positive on VAL net return. The VAL net return lies between −20.4 and −3.6 bp per trade.
- The largest VAL gross return of any configuration is 8.4 bp, below the 12 bp round-trip cost. The smallest BH q is 1.0.
- Verdict: **rejected.** Registry row F-0075 is recorded. The cost barrier of F-0054 applies to seasonality as well: at 15m the hour-of-day move is smaller than the cost.
