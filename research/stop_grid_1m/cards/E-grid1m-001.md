---
id: E-grid1m-001
mechanism: breakout; grid-hedge; regime-filter; limit-order-entry
timeframe: 1m
target: pnl
model_class: rule
reopens:
not_covered_by: F-0038=timeframe: F-0038 is a 1h grid at predicted range boundaries (fade side), this card is a 1-minute stop-entry grid on both sides with a trailing exit that activates after a favourable move and with rebuild triggers - 1m is outside every registry timeframe; F-0061=timeframe and construction: F-0061 is breakout stops at predicted candle extremes exited at the close on 15m/1h/4h, this card has N stop levels per side around a reference price with SL and trailing or TP exits and cost-to-volatility coin selection on 1m - the prior (gross about +1 bp at candle extremes) is NOT assumed away, it is the null expectation; F-0062=timeframe and use: F-0062 filtered fade and breakout at candle extremes on 15m/1h/4h with ADX/NATR/ATR on a 2022 selection, here the same family of gates (ADX on 1m or 5m, fast/slow ATR ratio, cost-viability) gates a grid build on 1m; F-0054=timeframe and setup: F-0054 is chart patterns at 15m/1h with R-multiple exits, not a symmetric stop grid; F-0068=timeframe: F-0068 is 4h breakout triggers judged on VAL decay, not 1m grids and not trailing activation
---
# Experiment card — E-grid1m-001: owner's stop-order grid with trailing activation on 1-minute bars

Written and committed BEFORE the runner and any result (rules: `docs/research/RESEARCH_RULES.md`). Do not edit after the run; the
result goes in `research/stop_grid_1m/README.md` (a "Result" section added later) and, if rejected, a new `failure_registry.csv` row.

| Field | Value |
|---|---|
| Date / author | 2026-10-24 / builder session (code + tests on SYNTHETIC data only; no real-data run exists yet) |
| Baseline | the two null baselines in `tools/stop_grid_sim.py` (sign-flip of whole days, pure random walk) run with the same config on the same windows |
| The ONE change | a grid of N buy-stop-market orders above and N sell-stop-market orders below a reference price on 1m bars, each fill with an initial stop-loss and either a fixed TP (limit), a trailing exit that activates only after a favourable move A (trail gap G), or half/half; grid rebuilt by ONE trigger at a time (breakout, after-flat, time) |
| Failure being addressed | none (new construction on a new timeframe). Closest closed rows: F-0038 (1h grid), F-0061 (breakout stops at candle extremes, gross about +1 bp vs 9-14 bp cost), F-0062 (regime filters 0/84). Not claimed as reopening any of them |
| Data / splits / holdout | owner's 300 Binance coins of 1m data (`history_1m/<SYMBOL>.csv.gz`). Common date range split by time: SELECT first 40 percent (coin choice only), DEV next 30 percent (parameter grid chosen here), TEST last 30 percent (used ONCE per frozen config). Coin choice: top 5 by median 1m (high-low)/close in bp divided by round-trip cost in bp on SELECT, subject to median quote volume per minute >= 20000 USD and >= 90 percent bar coverage |
| Seeds / budget | deterministic simulator; nulls use fixed seeds (20 sign-flip draws + 20 random-walk draws per test coin); CPU only |

## Verified cause and mechanism
- Verified cause this change attacks: none. The registry says stop entries at candle extremes pay the cost without an edge (F-0061: fade and breakout gross sum about -2x slippage). This experiment tests whether a different construction on 1m (several levels, trailing that only starts after +A, rebuild rules, volatility/cost gates) changes that, so the expected result is NO.
- Why the change could work, stated as hypothesis not fact: trailing activation lets winners run while the SL caps losers (convex payoff), and gates select minutes where realised 1m range is large relative to the round-trip cost. A convex payoff on a martingale has zero gross expectancy; it can only help if real 1m returns have short-horizon momentum after range expansion.
- Diagnostic that must move: gross (after slippage, before fees) bp per trade on TEST, lower-bound ordering. Null: sign-flip and random-walk runs of the same config, gross about 0 and net about -(fees + slippage). Gross must exceed the sign-flip null 95th percentile.
- Output effect expected on TEST: gross per trade about 0 on both nulls and on real data unless a regime gate selects real momentum; net negative (about -8 to -12 bp per trade at 5 bp taker per side, 2 bp maker, 1 bp slippage).

## Accept / reject rule (TEST, frozen config chosen on DEV)
- Accept ONLY if ALL hold on TEST for the single frozen config, lower-bound intrabar ordering (O-L-H-C or O-H-L-C, the worse): (1) pooled net bp per trade > 0; (2) day-clustered t > 3; (3) at least 3 of the 5 coins have positive net; (4) pooled LB net above the 95th percentile of the sign-flip null LB net; (5) pooled net still > 0 at 2x costs (fees and slippage doubled). Multiple-testing note on DEV: number of configs is counted and Bonferroni/BH-adjusted p-values are reported; the DEV pick does not count as evidence.
- Reject if any condition fails (including a pass only on the upper bound). A rejection closes the idea with a new `failure_registry.csv` row (scope tags as in this card, timeframe 1m).
- Otherwise: "unexplained" (not adopted). Funding is ignored (document); no live-trading claim in either case.

---
## Result (added after the run)
- Not run on real data yet.
