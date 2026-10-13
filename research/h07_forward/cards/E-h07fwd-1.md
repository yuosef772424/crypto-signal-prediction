---
id: E-h07fwd-1
mechanism: h07-trend-filter
timeframe: 1d
target: pnl
model_class: rule
reopens:
not_covered_by: F-0079=this card is the forward judging of the shadow variants that F-0079 leaves open (its R1 condition), not a re-run of the VAL/TEST comparison it closed
---
# Experiment card — E-h07fwd-1: forward paper trading of the frozen H07 rule and its two shadow variants

Card written after the ledger started (2026-09-30); the rule itself was frozen in research/edge_discovery
(`16_h07_oos.py`, REPORT.md Addendum C) and re-confirmed on Binance spot closes in `research/h07_volsizing`. This card
adds nothing to the rule: it records, in the standard format, the judging rules that `research/h07_forward/README.md`
and `E-h07comb-1` already state, so that they exist as a card. Written 2026-10-13 (review defect D5).

| Field | Value |
|---|---|
| Date / author | ledger started with the 2026-09-30 decision (P&L from 2026-10-01); card 2026-10-13 |
| Baseline | the frozen H07 rule, `tools/h07_forward.py` (`h07`), commit of this branch; no checkpoint, no dataset beyond the daily closes in the ledgers |
| The ONE change | none: no parameter is tuned. Forward evaluation of a frozen rule on data no study has touched |
| Failure being addressed | none (confirmation, not a fix). Shadow variants V3/V4 come from the `power` failure F-0079 (`research/h07_combos`) |
| Data / splits / holdout | Speirsy11/crypto-dataset daily closes (or a Binance CSV); decision days from 2026-09-30, bars from 2026-10-01 (time seal); nothing before that is used for judging |
| Seeds / budget | none (deterministic rule); one judgement, after 12 months of decision days (through 2027-09-30); no interim decisions |

## Frozen rule (never edited on the ledger)
- `h07`: per coin in {BTC, ETH}, long when the 30-day log return of the daily close is > 0; weight = 0.02 / std30
  (daily log returns), capped at 3, divided by the number of coins; cost 0.06% per side; weights decided at day d's
  close earn d -> d+1.
- Shadow variants, tracked in their own ledgers, never replacing `h07` before the judgement: `v3_10coins` (the same
  rule on BTC ETH SOL ADA BCH BNB DOGE TRX XRP ZEC, weight divided by the number of coins with >= 60 days of data) and
  `v4_ens_10coins` (signal = mean of 1{N-day log return > 0} for N in {10, 20, 30, 60}, on the same 10 coins).
- The ledger is append-only: `tools/h07_forward.py` refuses to rewrite a past row if the data changed.

## Judging rule for H07 (from `research/h07_forward/README.md`, pre-registered there)
After 12 months, H07 is supported if its Sharpe > the Sharpe of the 50/50 buy-and-hold of the same coins AND its max
drawdown is smaller. Otherwise it is not supported. Fewer than 12 months of decision days: no verdict.

## Judging test for the shadow variants (V3, V4)
Inputs: the forward ledgers only (`ledger.csv`, `ledger_v3_10coins.csv`, `ledger_v4_ens_10coins.csv`, bars from
2026-10-01). VAL and TEST are NOT pooled in: TEST was read for all variants and used when V3/V4 were chosen (see
`research/h07_combos/cards/E-h07comb-1_test_use_note.md`).
- Paired difference: ΔSharpe(variant - h07) with the stationary block bootstrap of E-h07comb-1 (block 20 days, 2000
  draws), interval at 1 - 0.05/4 = 98.75% (the same Bonferroni level as the card that produced the variants).
- A variant is adopted only if ΔSharpe > 0 with the interval excluding 0 AND its max drawdown, after scaling to
  `h07`'s realized volatility on the same days, is not worse than `h07`'s by more than 2 points.
- Otherwise it is not adopted and the shadow ledger is closed. The registry row F-0079 has `reopen_if` R1 reading
  "pooled VAL+forward"; that wording is superseded by this card (forward only). The registry is append-only, so a
  correction row is appended only if the owner asks for one.

## Mechanism and diagnostics
- Why H07 should hold: time-series momentum of BTC/ETH over 30 days with volatility-scaled exposure (REPORT.md
  Addendum C). Diagnostic reported with the verdict, not used to decide: exposure fraction of `h07`, and the
  give-back in the 10 days before an exit (as in E-h07comb-1).
- Expected power: with 12 months of daily data the Sharpe difference of two correlated trend strategies has an
  uncertainty of roughly ±0.5, so an inconclusive result for V3/V4 is the likely outcome and is a valid result.

## Accept / reject rule
- H07: as above (Sharpe and drawdown vs 50/50 buy-and-hold at 12 months).
- V3/V4: as above (paired bootstrap on the forward ledger). Inconclusive = not adopted.
- No change to any rule, weight, cost or universe while the ledger is running; a change starts a new ledger and a
  new card.

---
## Result (added after the run)
Not available before the 12-month judgement (earliest 2027-10-01).
