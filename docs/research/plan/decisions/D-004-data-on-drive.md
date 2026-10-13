# D-004: Data on Drive

**Status:** Accepted (owner, 2026-10-13).

**Context.** The owner reports a `history_15m` folder on Drive with 15-minute bars for 800+ coins, compressed.
No funding/basis history was reported.

**Decision.** `history_15m` is the base dataset. 1h/4h/1d bars are resampled from it inside the scripts. Outcomes on
an intrabar path use 15m bars with stop-first on ambiguity. R2 starts with a funding/premium-index acquisition step
run on Colab; if it cannot reach the source, R2 stops.

**Open.** Whether the BTC/ETH/SOL 5m data and the 1d/hourly-futures sets used before are also on Drive.
