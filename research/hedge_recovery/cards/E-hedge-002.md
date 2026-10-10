---
id: E-hedge-002
mechanism: grid-hedge; trend-following
timeframe: 1h
target: pnl
model_class: rule
reopens:
not_covered_by: F-0065=this card is the experiment that produced F-0065 (its own result); F-0064=E-hedge-001's trigger is a realised loss of a base leg (≥1×NATR) and it closes hedges on a NATR-regime sign, this card has NO loss trigger: an opposite leg is opened only by an opposite base signal (reversal), its losers are left open with no action, it closes by a profit target only, and it adds a count-balancing rule (close longs while longs > shorts and longs are profitable), the same no-stop failure mode (F-0053) is re-tested here, so the drawdown rule is kept; F-0038=hedge at predicted range edges on resting limits (this card uses market fills at the next open, no range prediction); F-0053=previous-candle range fade with limit orders; F-0054=chart-setup study with fixed stop/target exits (this card's base arm B re-measures the same trend-pullback entry as a reference, reported, not claimed); F-0055=maker limit entry at EMA20 with scaling (this card reinforces only on a new base signal, with a 6-leg cap); F-0062=regime filters on wick-capture fills (no regime filter here)
---
# Experiment card — E-hedge-002: reversal hedge; ignore the losing opposite leg; close it on profit; balance counts

Written and committed BEFORE the run (rules: `docs/research/RESEARCH_RULES.md`, `docs/research/MODEL_DEV_PROTOCOL.md`).
Owner's rule (intent, 2026-10-10): when a reversal occurs and the open trade is a SHORT, open a BUY as a hedge request and do
not close the losing short. When the short has a good profit, close the short and keep the BUY until it wins or is reinforced
later if its conditions are met. If the number of BUY trades is greater than the number of SELL trades and the BUYs are in
profit, close part of the BUYs until the counts are equal. Each trade is 0.5% of capital.

| Field | Value |
|---|---|
| Date / author | 2026-10-10 / Claude (for the project owner) |
| Baseline (commit, checkpoint, dataset fingerprint) | same data as E-hedge-001: Speirsy11/crypto-dataset 1h spot OHLCV through 2026-09-30 23:00, BTC ETH SOL BNB XRP ADA DOGE |
| The ONE change | the reversal-hedge management (rules R1–R5 below) applied to the same base entries as arm B; sizing fixed at 0.5% of capital per leg |
| Failure being addressed (level, location, registry id) | none yet: a new hypothesis. Closest closed failure: F-0064 (E-hedge-001), differing in trigger, exit and balancing (see not_covered_by) |
| Data / splits / holdout | DEV 2019-01-01→2021-12-31 (rules fixed here, not tuned), VAL 2022-01-01→2023-12-31, TEST 2024-01-01→2026-09-30. TEST is NOT fresh (overlaps periods used by H06, H07, H18, F-0064); a pass must be re-checked on bars after 2026-09-30 |
| Seeds / budget | Random-entry control C: 5 seeds. CPU, minutes |

## Definitions (fixed before the run)
- **Capital and size:** capital = 100 units. Every leg (base or hedge) has size 0.5 units = 0.5% of capital. Per-leg return is in bp
  of its own notional. Account-level P&L and drawdown are in % of capital (leg size 0.5% × leg return).
- **Costs:** 0.12% round trip per leg, charged on every leg, including hedges.
- **Base entry (B):** the trend-pullback signal of E-hedge-001 (EMA20 > EMA50, low ≤ EMA20, close > EMA20 for long; mirror for short).
  Fill at the OPEN of the next bar. Decisions use closes only.
- **Conflict flag (R1):** when a new leg opens while opposite-direction legs are open, (a) each open opposite leg is flagged
  `conflict` (its loss is ignored: no stop, no time exit), and (b) the new leg is flagged `conflict` (it is the hedge request).
  A leg keeps its `conflict` flag until it closes.
- **Profit target (R2):** every `conflict` leg closes at the open of the next bar when its unrealised return at the close ≥ 1.0 × NATR_14
  (in %). This is my reading of "good profit" for the short and "until it wins" for the buy: both use the same threshold. If the owner
  means "any profit above zero" for the buy, that is a different rule and needs its own card.
- **Time exit (R3):** a non-`conflict` leg closes at the open 24 bars after its entry (same as arm B).
- **Reinforcement (R4):** a new base signal opens a new leg in its own direction, subject to the cap. If that direction already has
  `conflict` legs open, the new leg is also flagged `conflict` (it is part of the hedge set). Reinforcement is therefore the same
  entry rule as the base signal, with no separate condition.
- **Count balancing (R5):** at the close of each bar, while (open longs − queued long closes) > (open shorts − queued short closes)
  and at least one open long has unrealised return > 0 at this close, queue the close of the most profitable open long. Only this
  direction is in the rule; the symmetric case (more shorts, profitable shorts) is NOT applied here and is left for a later card.
- **Cap:** at most 6 open legs per coin (my choice, needed to keep the no-stop design finite; the same cap as E-hedge-001).
  Entries and hedges are refused at the cap.
- **Forced close:** any leg open at the end of its split closes at the last close and is reported as `eod`, not hidden.

## Arms (same base entries in every arm where possible)
- **B:** base entries + R3 time exit only (no R1, R2, R4, R5). Baseline with 0.5% sizing and no cap.
- **E2:** B + R1–R5 + cap (the card's candidate).
- **C (control):** random entries at the base entry rate (5 seeds) with R1–R5 + cap. Tests whether any gain comes from the rules
  rather than from the base signal.

## Predicted diagnostic movement
- Conflict legs are never stopped, so the realised P&L of the `conflict` legs should be negative in DEV/VAL/TEST (the F-0053 result
  with no stop) unless the profit target catches them; the share of `eod` legs and their loss must be reported.
- The count balancing (R5) only closes longs, so it should make longs and shorts closer in number but does not change the sign of
  the expected per-leg return. The predicted per-leg mean for E2 is therefore close to the mean of B's base legs, not above it.
- Random control C under the same rules should match E2 on the sign of the result if the rules, not the signal, drive it.
- Worst single-leg loss (bp) must be reported: the account-level drawdown can look small at 0.5% sizing while a single no-stop leg
  loses several times its notional.

## Accept / reject rule (written before the run)
- **Accept** (candidate for a fresh holdout, not for trading) only if ALL of:
  1. E2 mean net PnL per closed leg > 0 bp on VAL AND on TEST (pooled over 7 coins);
  2. E2 beats B on net PnL per closed leg on both VAL and TEST;
  3. E2 beats the mean of C (5 seeds) on net PnL per closed leg on both VAL and TEST;
  4. max per-coin mark-to-market drawdown of E2 ≤ 20% of capital on both VAL and TEST.
- **Reject** otherwise. The worst single-leg loss, the `eod` count and the share of P&L from `conflict` legs are reported even when
  E2 looks positive.

## Verified cause and mechanism
- The mechanism under test: a reversal hedge that is never stopped turns a loss into an open exposure whose P&L depends on whether
  the market returns; the profit target and count balancing decide when that exposure is realised. Whether this has positive
  expectancy is the question. The no-stop failure mode is already known from F-0053 and F-0064 and is re-tested here only through
  the sizing and drawdown rule.

---
## Result (added after the run)
Run: `research/hedge_recovery/sim2.py` → `E-hedge-002_results.csv` (7 coins × 3 splits × arms B, E2, C×5 seeds).
Pooled over 7 coins, mean net PnL per closed leg after 0.12% RT; account P&L in % of capital (leg size 0.5%).

| Split | B (base only) | E2 (rules R1–R5) | C (random + rules, 5 seeds) | E2 conflict legs | E2 clean legs | E2 max DD per coin |
|---|---|---|---|---|---|---|
| VAL 2022–23 | −5.6 bp (21,205 legs) | **−32.5 bp** (1,935 legs) | −32.0 bp (range −74.8 … +15.1) | −39.0 bp (1,760) | +32.5 bp (175) | 5.08% of capital |
| TEST 2024–26 | −7.8 bp (26,362) | **+4.3 bp** (2,249) | −17.3 bp (range −39.6 … +6.0) | +5.1 bp (2,093) | −5.5 bp (156) | 6.60% |
| DEV 2019–21 (not in the rule) | +10.5 bp | **−7,903 bp** (2,022) | −6,233 bp | −8,724 bp (1,831) | −26 bp (191) | **935.5%** of capital |

E2 win rate is 0.90–0.92 per leg (B: 0.45–0.47). Worst single leg: −100,939 bp on VAL, −26,107 bp on TEST, −1,367,230 bp on DEV.
Exit reasons on VAL: R2 profit target 1,151, R5 balance 694, time exit 48, end of data 42.

Accept-rule check:
1. E2 mean > 0 on VAL AND TEST: **fails** on VAL (−32.5).
2. E2 beats B on both: **fails** on VAL (−32.5 vs −5.6); passes on TEST.
3. E2 beats C on both: **fails** on VAL (−32.5 vs −32.0, a tie, not a win); passes on TEST.
4. max per-coin DD ≤ 20% of capital on VAL and TEST: passes (5.1%, 6.6%), but the rule is **not binding** at 0.5% sizing: the
   same rule fails badly on DEV (935.5% of capital, meaning the account would be wiped out). It is reported, not relied on.

- Failure level and location: economics, at the VAL comparison with B and with the random control under identical rules.
- Verified cause (tested by the control arm C and by the split of P&L by leg type): the 90% win rate is produced by the profit
  target R2 on the conflict legs, which is an accounting artifact of the no-stop design. The mean per leg is set by the
  unstopped tail of the losing legs that never reach the target, and the random control under the same rules has the same mean
  on VAL (−32.0 vs −32.5). So the rules add no edge beyond their exposure.
- The TEST result (+4.3 bp) is not repeated on VAL. TEST also overlaps periods used before, so it is not fresh.
- Verdict: **rejected.** Registry row added (F-0065). The card is not edited after this point.
