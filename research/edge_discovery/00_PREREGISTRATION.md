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

---
## Addendum B (2026-10-01, written BEFORE any HOLDOUT data is looked at)

**H12 final (full data, 413 listings):** the A.2 rule selects H=60, S=0.5 on DISC (mean +17.3%/trade,
t_month 2.74 — the only configs with t>2 are S=0.5). VAL for that config: mean −2.4%, t −0.53 → **fails**.
H12 is REJECTED; per the rule, its HOLDOUT is **not** spent. (Interim DISC t=3.3 on the alphabetically
partial sample was a sample-completeness artefact.)

**H06 and H07 did not meet the formal acceptance criteria** (DISC |t|>3 and VAL t>2 were never reached).
They can therefore NOT be accepted by any HOLDOUT outcome. The single HOLDOUT look below is
**descriptive only**: it decides whether they deserve a longer-history test (2020→2026 15m data), nothing more.
Specs frozen here:
- **H06-BTC / H06-ETH:** every day, long at the 20:00 UTC hourly price, exit at the 22:00 UTC price
  (= the two hourly returns stamped 21h and 22h). Report mean gross bp/trade, t (daily), win rate,
  net at maker 0.04% RT and taker 0.12% RT. "Agrees" = gross mean > 0.
- **H07:** `TSMOM30_LO_BTC+ETH` exactly as in `07_trend.py` (sign of 30d log return, long-only,
  2% vol target per coin, 0.12% RT), compared with `BUYHOLD_BTC+ETH` on the same window.
  "Agrees" = Sharpe > 0 AND Sharpe ≥ buy&hold Sharpe. Deflated Sharpe reported with N = 44 trend
  configs (family) and N ≈ 750 (all configs tried in this project).

---
## Addendum C (2026-10-01, written BEFORE computing any of the results below)

Motivation: PR #7 (anti-memorization) showed (a) a longer daily history is reachable (CoinMetrics community
data, raw GitHub files; Drive `history_1d` 2019-09→2026-09), and (b) a relative-direction classifier with test
AUC 0.538 (linear) – 0.555 (GRU). Our main lesson is that AUC ≠ mean return, so (b) must be checked in P&L terms.

**H07-OOS (genuinely untouched period).** Data: CoinMetrics `PriceUSD` daily (00:00 UTC) for BTC and ETH.
Period 2018-01-01 → 2023-12-31 (none of it was used by this project). Strategy frozen exactly as H07:
long-only, sign of 30-day log return, 2% daily vol target per coin (30d realised vol, cap 3×), equal risk across
the two coins, 0.12% RT on turnover. Benchmark: buy & hold with the same vol scaling.
- PASS = Sharpe > 0 AND Sharpe ≥ benchmark Sharpe AND maxDD shallower than benchmark. Also reported by
  calendar year and per coin. Lookbacks N ∈ {7,14,21,45,60,90} reported as descriptive only (selection was N=30).

**H13 (PR #7 signal → money).** Universe: CoinMetrics assets with ≥ 700 daily prices since 2018 (stablecoins
excluded). Features = the PR #7 `build_features` set (last-day values, causal). Target = next-day return above the
cross-sectional median. Model = logistic regression (PR #7's linear reference; standardised features), fit once on
data ≤ 2023-06-28, frozen. Test 2024-07-01 → last date. Each day: long the top decile of predicted probability,
short the bottom decile, equal weight, 1-day hold, 0.12% RT on turnover (2× stress also reported).
- PASS = AUC replicates (> 0.52) AND net mean daily return > 0 with t > 2 AND both halves of the test period > 0.
- Caveat stated up front: the test window overlaps our VAL/HOLD calendar period (different data/hypothesis).

---
## Addendum D (2026-10-01, written BEFORE computing any result below): H15 on-chain & liquidity signals
Rationale ("outside the box"): everything so far used price/positioning, which markets price fast. Slow-moving
fundamental flows (on-chain, stablecoin liquidity) may be priced slowly. Data: CoinMetrics community daily.
Asset traded: BTC (ETH reported as a secondary check). Horizon: 30 days. All signals use data ≤ t−1 (one extra day
of publication lag), trade at t's close.

Split (new, for H15 only): DISC ≤ 2019-12-31 (BTC from 2013; stablecoins from 2018), VAL 2020-01-01→2022-12-31,
HOLD 2023-01-01→2026-05-24. (Overlap with H07's calendar is disclosed; these are different signals.)

Six signals, direction fixed by theory (sign = expected effect on forward BTC return):
- S1 MVRV: expanding-window percentile of log(CapMVRVCur) — high = overvalued → **negative**.
- S2 Exchange net flow: 30d Σ(FlowInExNtv − FlowOutExNtv)/SplyCur — coins moving to exchanges → **negative**.
- S3 Exchange supply trend: 90d change of SplyExNtv/SplyCur → **negative**.
- S4 Hash-ribbon: HashRate 30d MA / 60d MA − 1 (miner capitulation when < 0) → **positive**.
- S5 Stablecoin liquidity: 30d log growth of Σ SplyCur(usdt, usdc, dai, busd) → **positive**.
- S6 Network activity: log(AdrActCnt 30d MA / 365d MA) → **positive**.

Tests per signal: (a) Spearman corr(signal, fwd 30d log return) on non-overlapping month-end samples;
(b) timing strategy: long BTC when the signal is on its favourable side of its expanding median, else cash (0.12% RT),
vs buy & hold. ACCEPT = DISC corr sign as theorised with t>2, VAL same sign AND strategy Sharpe > buy&hold,
HOLD same sign AND Sharpe > buy&hold. 6 signals → Bonferroni noted (t>2.64 for 5% family-wise).

---
## Addendum E (2026-10-01, written BEFORE any simulation): H16 owner's "previous-candle range fade"
Rule (owner's spec): at the open of candle k place a SELL limit at high[k−1] and a BUY limit at low[k−1]
(valid for candle k only). A filled short takes profit with a BUY limit at low[k−1]; a filled long takes profit
with a SELL limit at high[k−1]. Positions can accumulate across candles. R = high[k−1] − low[k−1].
Data: Binance spot OHLC (github Speirsy11/crypto-dataset) BTC/ETH/SOL; signal candles 1h/4h/1d; fills resolved
on the 5-minute path. Conservative fill model: a limit fills only if price trades THROUGH it by 2 bp; no TP in
the same 5m bar as its entry; if SL and TP are both reachable inside one 5m bar, SL is assumed first.
Costs: limit (entry/TP) 0.02% per side (maker); market exits (SL/time/forced) 0.05% + 0.02% slippage.
Funding not modelled (both sides held; noted).

Risk variants (fixed now; nothing else will be tried):
- V0 raw: no stop, no cap (owner's rule as-is; open positions marked to market).
- V1 cap: at most 3 open positions per side; no stop.
- V2 owner's trend-hedge: cap 5 per side; when open shorts − open longs ≥ 2 the longs' TP is suspended (they ride
  the trend) and vice versa; TP re-armed when the imbalance falls below 2 (exit at market if already beyond TP).
- V3 stop+time: cap 3 per side; stop at 1R beyond entry; time-exit at market after 3 candles.
- V4 regime filter: V3, but orders only when Kaufman efficiency ratio ER(20) of closes < 0.3 (range-bound market).

Split: DISC 2017-08→2021-12, VAL 2022-01→2023-12, HOLD 2024-01→2026-09 (calendar overlaps earlier HOLDs; new
strategy, disclosed). Selection: best (variant, timeframe) by DISC net expectancy pooled over the 3 coins, among
those with daily-PnL t > 2; VAL must have net expectancy > 0 and t > 2; then HOLD once (expectancy > 0, PF > 1.1).

---
## Addendum F (2026-10-01, written BEFORE any computation): H17 chart-reading rules, low timeframes + HTF filters
Owner's request: derive conditional entry rules from the chart (not every move), focus on 15m / 1h, allow HTF
conditions, scaling-in and hedging. Translated into a FIXED grid of classic chart-reading setups.

Data: Binance spot OHLCV BTC/ETH/SOL 2017-08→2026-09 (5m path for exits). Signal TF ∈ {15m, 1h}.
Setups (long version; short = mirror), evaluated at the signal bar close:
- P1 engulfing: bar bullish, prior bar bearish, body engulfs prior body.
- P2 pin-bar rejection: lower wick ≥ 2×body and ≥ 60% of the bar range.
- P3 liquidity sweep & reclaim: low < min(low of prior 20 bars) and close > that min.
- P4 Donchian breakout (momentum): close > max(high of prior 20 bars).
- P5 compression breakout: ATR14/ATR100 in its lowest 20% (rolling 500 bars) and close > max(high prior 10).
- P6 inside-bar breakout: prior bar is inside its predecessor; close > predecessor high.
- P7 trend pullback: EMA20 > EMA50 and low ≤ EMA20 ≤ close.
- P8 capitulation reversal: volume > 3× 20-bar mean, range > 2×ATR14, bar opened above close of prior bar and
  closes in the upper half of its range after a drop (close < open of 3 bars ago).
- P9 3-bar exhaustion (mean reversion): 3 consecutive lower closes totalling > 2×ATR14.
- P10 London breakout: first close above the Asia range (00:00–07:00 UTC high) between 07:00–10:00 UTC.
HTF filters (completed higher-TF bars only): F0 none; F1 with 4h trend (4h close vs EMA50(4h)); F2 against the 4h
trend; F3 with daily trend (1d close vs EMA20(1d)); F4 range regime (daily ER(10) < 0.3).
Execution: entry at next bar open (taker 0.05% + 0.02% slip); stop = 1×ATR14(signal TF) from entry (stop-market,
taker + slip); target TP ∈ {1R, 2R, 3R} (limit, maker 0.02%); time exit 24h (15m) / 72h (1h) at market.
SL/TP ordering resolved on 5m bars; if both inside one 5m bar → stop first. Trades may overlap (scaling allowed).
Grid = 2 TF × 10 setups × 2 directions × 5 filters × 3 targets = 600 configs (pooled over 3 coins).
Split: DISC 2017-08→2021-12, VAL 2022–2023, HOLD 2024-01→2026-09 (HOLD touched once).
Acceptance: DISC net expectancy (R) > 0 with day-clustered t > 3.9 (Bonferroni 5% for 600) AND > 0 in each coin;
VAL: expectancy > 0, t > 2; HOLD: expectancy > 0. Survivors then: 2× cost stress, pyramiding (add on repeated
signal, cap 3) and hedge variant evaluated descriptively.

---
## Addendum G (2026-10-01, after H17 failed; written BEFORE fetching/computing H18 data)
H17 result: 0 of 600 configs passed (best DISC t = 1.79). Post-hoc diagnostic (23_chart_rules_gross.py): at 15m a
round trip costs 0.21–0.38 R (1-ATR stop), at 1h 0.10–0.18 R; the only setup with positive GROSS expectancy in
all three periods is P7 trend pullback (+0.03…+0.05 R). Because the BTC/ETH/SOL HOLD numbers of P7 have now been
seen, H18 is tested on UNTOUCHED coins only.

**H18 maker-entry trend pullback.** Coins: ADA, XRP, BNB, DOGE, LTC, LINK, AVAX, DOT (Binance spot, never used
in this project), full history to 2026-09. Rule: on signal TF bar close, if EMA20 > EMA50 (signal TF) place a BUY
limit at that bar's EMA20 for the next bar (mirror SELL limit when EMA20 < EMA50). Fill only if price trades through
by 2 bp. Max 3 concurrent positions per coin (scaling-in allowed). Stop = k×ATR14 below/above entry, target = m×k×ATR.
Costs: entry & TP maker 0.02%; stop/time exit taker 0.05% + 0.02% slip. Time exit 24h (15m) / 72h (1h).
Grid: TF {15m, 1h} × k {1, 2, 3} × m {2, 3} × HTF {none, 4h-aligned (4h close vs EMA50)} = 24 configs.
ACCEPT: pooled net expectancy > 0 with day-clustered t > 3.0 (Bonferroni 24), > 0 in ≥ 6 of 8 coins, and > 0 in each
of 2018–2021, 2022–2023, 2024–2026. BTC/ETH/SOL reported descriptively only.
