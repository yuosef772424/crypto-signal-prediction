---
id: E-fwd-001
mechanism: breakout; trend-following
timeframe: 1d
target: pnl
model_class: rule
reopens:
not_covered_by: F-0069=daily ADX>40 and short-side refinements of Bollinger and Donchian, this card keeps both sides and has no ADX cut; F-0071=Bollinger breakout with ADX gate on unseen assets, a different trigger and gate; F-0072=Donchian with a volume filter, this card has no volume condition; F-0070=cross-sectional rotation, a portfolio, not single-asset trades
---
# Experiment card — E-fwd-001: forward test of ONE frozen hypothesis (daily Donchian 20, no gate, long and short)

Written and committed BEFORE any forward bar is seen (rules: `docs/research/RESEARCH_RULES.md`, `docs/research/MODEL_DEV_PROTOCOL.md`).
This card freezes the hypothesis and the decision before new data arrives. Nothing in it may be changed after the first forward trade.

## The hypothesis (frozen)
- Signal: close above the prior 20-bar high (long) or below the prior 20-bar low (short). No gate. The ADX≥25 variant (S1) is NOT
  a separate hypothesis: it is a subset of this one and is not tested separately here.
- Exit: stop 2 × ATR(14) from the entry, target 3 × ATR(14), time exit 30 bars. Stop first inside a bar. Entry at the next open.
- Cost: 0.12% round trip, in R.
- Universe: BTC, ETH, SOL, BNB, XRP, ADA, DOGE, BCH, TRX, ZEC (the ten coins in the dataset). The universe is not changed.

## Why this hypothesis (stated before the test)
- Selected on DEV and VAL (E-disc-001, E-disc-002) and on the unseen-asset test (E-disc-003). Its earlier results are therefore NOT
  evidence for the forward test: they are the reason it was chosen. The forward window is the first test with no selection exposure.
- Expected effect: smaller than the earlier estimates, which are selected and overstate the edge. The power analysis below says that
  only an effect of about 0.2R or more can be resolved.

## Forward window and data
- Forward window: entries on or after 2026-10-09 00:00 UTC. The last historical bar is 2026-10-08 23:00 UTC.
- A position open at the data end is reported as open and is not counted.
- Data: the Speirsy11/crypto-dataset mirror through `tools/fetch_crypto_dataset.py`, downloaded into a NEW directory on every run (the fetcher skips existing files).
- Delisted or halted coin: a coin with no new bars after its last bar leaves the sample from that date. No imputation.
- Data revision: if a bar before 2026-10-08 23:00 differs in a later download, the run is reported as void and the cause is recorded.

## Metric and decision (frozen)
- Primary metric: mean net R per closed trade over the forward window.
- Standard error: clustered by entry month (sum of deviations within each month, squared, over n squared). The trade-level standard error
  is NOT used for the decision, because trades overlap across coins and cluster in time. The naive value is reported for reference only.
- Decision, computed ONCE, at the end condition:
  - **consistent with an edge**: mean > 0 AND clustered z ≥ 2;
  - **rejected**: clustered z ≤ −2;
  - **inconclusive**: anything else (a mean above zero with z below 2 is inconclusive, not positive).
- End condition, fixed now: at least 300 closed trades, OR 2028-08-10, whichever comes first. If the date comes first with fewer than
  300 trades, the verdict is "insufficient sample".
- No rule changes during the test. No peeking: the decision is not read before the end condition is reached.

## Power (stated before the test, from the VAL trade log)
- Trade-level SD of net R on VAL is about 1.16R. With the month-clustered standard error, the SE at 300 trades is about 0.10 to 0.105R.
- An effect of 0.20R is resolved at z ≥ 2 about half the time. An effect of 0.10R is resolved about 16% of the time.
- The expected trade count at the end date is roughly 300 (about 14 to 16 per month across ten coins). The end date of 2028-08-10 is
  chosen so that the 300-trade condition is reached before the date in the expected case.

## Predicted diagnostic movement
- Forward mean net R below 0.2R is expected; a result near 0.2R would be consistent with the earlier estimates but not proof of them.
- A negative forward mean at the end condition would show that the earlier positive results were selection effects.

---
## Result (added after the forward window ends; the card is not edited after the end condition)
(to be filled at the end condition; running results are written to `results/E-fwd-001_summary.csv` and are not decisions)
