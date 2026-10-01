# Edge Discovery — Pre-registration (written BEFORE any test was run)

Date: 2026-10-01

## Data actually available in this session
- Binance USDT-M futures metrics (hourly, 2024-01-01 → 2026-09-26) for 195 perpetuals listed before
  2023-10-01 **including later-delisted ones** (survivorship-safe):
  `sum_open_interest`, `sum_open_interest_value`, top-trader account & position long/short ratios,
  global account long/short ratio, taker buy/sell volume ratio.
- Hourly price proxy: `px = sum_open_interest_value / sum_open_interest` (mark price at snapshot).
  Verified against BTC (42.5k on 2024-01-01, 84.4k on 2026-09-26).
- The 15m OHLCV database (2020→2026) could NOT be loaded: the Drive connector refuses files > ~2 MB
  and Binance hosts are blocked by this environment's network policy. Results here are therefore
  limited to 2024-01 → 2026-09 and to close-to-close prices (no intrabar high/low, no volume).

## Time split (fixed now, never changed)
| Segment | Period | Use |
|---|---|---|
| DISCOVERY | 2024-01-01 → 2025-03-31 | free exploration, all screening |
| VALIDATION | 2025-04-01 → 2025-12-31 | confirm sign & size of candidates chosen on DISCOVERY; parameters frozen |
| HOLDOUT | 2026-01-01 → 2026-09-26 | touched ONCE, only for candidates that pass VALIDATION |

Walk-forward inside DISCOVERY+VALIDATION uses expanding windows, quarterly refits.

## Execution / cost model
- Signal computed from data stamped ≤ t; entry at px(t+1h) (one full hour of latency, conservative
  vs. the ~5-min publication delay of Binance metrics). Exit at px of the exit hour.
- Costs per round trip: base 0.12% (taker 0.05%×2 + 0.02% slippage) for coins in the top-50 by OI
  value, 0.20% for others; stress test at 2× costs. Funding not modelled (flagged where holds > 1 day).
- Tradable universe at time t: coins with OI value ≥ $10M at t−1h.

## Acceptance criteria for a candidate edge
1. DISCOVERY: rank-IC / event mean with |t| > 3 using non-overlapping observations, sign consistent
   in ≥ 4 of 5 quarters.
2. VALIDATION (frozen params): same sign, net-of-cost expectancy > 0, t > 2.
3. HOLDOUT: same sign, net expectancy > 0. Deflated Sharpe computed with the TOTAL number of
   configurations tried (logged in `hypothesis_log.csv`).
4. Robustness: survives 2× costs, survives removal of best 5% of trades, survives large/small-cap split.

Anything failing a step is recorded as rejected with the reason; no criterion is changed after
seeing results.

---
## Addendum A (2026-10-01, written BEFORE downloading listing data): H-NL post-listing drift

Motivation: hourly-panel screens (cross-sectional, time-series, events) produced no mean-return edge;
the robust stylised fact found was extreme right-skew of individual coin returns. New-listing drift
is a different mechanism (low float / high FDV / unlock supply) with many *independent* events.

- Events: every USDT-M perpetual whose first metrics bar is ≥ 2024-01-15, with ≥ 30 days of data.
- Entry: price 24h after the first metrics bar (avoid listing-hour chaos). Exits: +7, +30, +60 days.
- Primary statistic: log return of coin minus log return of BTC over the same window; trade = SHORT coin
  / LONG BTC (beta 1). Mean across events, t-stat with listing-month clustering.
- Split by listing time: DISCOVERY 2024-01-15→2025-03-31, VALIDATION 2025-04-01→2025-12-31,
  HOLDOUT 2026-01-01→(last date allowing the exit).
- Cost: 0.30% round trip (thin new books). **Funding is NOT in the dataset** — the result is a price-only
  edge and funding drag must be measured before any real use (new listings often carry negative funding,
  which shorts pay).
- Accept only if DISC and VAL both show the same sign with t > 2 and HOLDOUT sign agrees.

### Addendum A.1 (before any listing return was looked at)
Traditional-finance perps (stocks, ETFs, pre-IPO, metals, energy, gold tokens) are excluded from H12 because the
hypothesis concerns token supply/unlock dynamics. Exclusion list fixed by name only: `data_tools/tradfi_exclude.txt`.

### Addendum A.2 (after seeing DISC/VAL fixed-horizon results, BEFORE any stop-loss test or HOLDOUT look)
Interim fact: median 30–60d short return is large (+25–42% vs BTC, win 71–83%) but the right tail is real
(COAIUSDT +4600% in 30d). A naked short is not tradable. Pre-specified risk-controlled variant:
- Short at entry (listing+24h), hedge long BTC same notional. Exit at day H ∈ {30, 60} or when coin price
  ≥ entry × (1+S), S ∈ {none, 0.5, 1.0}; checked on hourly closes, stop fill = the breaching hourly price
  + 2% adverse slippage. 6 combos total.
- Selection on DISC only (max mean net return per trade with t_month > 2); VAL must agree in sign with t > 2;
  then HOLDOUT once. Cost 0.3% RT. Funding still not modelled (to be estimated from a sample afterwards).
