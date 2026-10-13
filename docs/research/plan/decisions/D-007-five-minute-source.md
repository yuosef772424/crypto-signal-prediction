# D-007: 5m (and 1m–1w) data from the repo's fetch tool

**Status:** Accepted (owner, 2026-10-13).

**Context.** The owner pointed out that `tools/fetch_crypto_dataset.py` downloads any interval. Checked by the planner
on 2026-10-13: the source (Speirsy11/crypto-dataset, Binance spot) has 10 coins — ADA, BCH, BNB, BTC, DOGE, ETH, SOL,
TRX, XRP, ZEC — at 1m, 5m, 15m, 1h, 4h, 1d, 1w, 1mo, from 2017-08 (BTC) to 2026-10; BTC 5m = 956,100 bars to
2026-09-30 23:55, fetched in ~9 s from the cloud session.

**Decision.** Exploration and selection that need an exact intrabar path use these 10 coins at 5m (1m to break a tie
inside a 5m bar), and may run in the build session; the planner reproduces them. All 10 coins are already touched, so
no confirmation runs on them. Confirmation stays on sealed coins of Drive `history_15m` (stop-first on 15m bars).
Fetches use `--end "2026-09-30 23:55"` so the time seal (≥ 2026-10-01) is never read.

**Consequences.** Less Colab load (D-003); confirmation runs carry a known handicap (15m stop-first vs 5m exact),
reported next to every candidate.
