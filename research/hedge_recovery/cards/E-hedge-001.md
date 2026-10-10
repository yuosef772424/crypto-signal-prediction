---
id: E-hedge-001
mechanism: grid-hedge; trend-following; regime-filter
timeframe: 1h
target: pnl
model_class: rule
reopens:
not_covered_by: F-0064=this card is the experiment that produced F-0064 (its own result, not a different scope); F-0065=E-hedge-002, added after its run as a compliance declaration only (no loss trigger, no NATR-regime exit, opposite-signal hedge, profit target and count balancing, the shared no-stop failure mode is what both cards test); F-0038=two-order hedge placed at predicted range EDGES on resting limits; F-0053=previous-candle range fade with limit orders; F-0054=chart-setup study with fixed stop/target exits; F-0055=maker limit entry at EMA20 with scaling; F-0062=regime filters on wick-capture fills
---
# Experiment card — E-hedge-001: hedge losing positions, close hedges on the next signal or a NATR-regime exit

Written and committed BEFORE the run (rules: `docs/research/RESEARCH_RULES.md`, `docs/research/MODEL_DEV_PROTOCOL.md`).
Owner's rule (verbatim intent, 2026-10-10): when a trade loses, add an opposite trade as a hedge and wait for another
opportunity. If the next entry is a BUY, close the SELL counterpart (and symmetrically). A profitable trade that moves
against the NATR direction is closed even if it is a hedge.

| Field | Value |
|---|---|
| Date / author | 2026-10-10 / Claude (for the project owner) |
| Baseline (commit, checkpoint, dataset fingerprint) | commit of this card; data = Speirsy11/crypto-dataset 1h spot OHLCV, `tools/fetch_crypto_dataset.py --interval 1h --end "2026-09-30 23:00"`, BTC ETH SOL BNB XRP ADA DOGE |
| The ONE change | exit/position overlay (hedge on loss + hedge-close rules + NATR exit) applied to the SAME base entry signal, compared with the same entries without the overlay |
| Failure being addressed (level, location, registry id) | none yet: this is a new hypothesis. Closest closed failures are F-0038 and F-0053 (overlapping mechanism); see not_covered_by |
| Data / splits / holdout | DEV 2019-01-01→2021-12-31 (rules fixed here, not tuned), VAL 2022-01-01→2023-12-31, TEST 2024-01-01→2026-09-30. TEST overlaps the 2026 period already used for H06/H07 and is therefore NOT a fresh holdout; a verdict that passes must be re-checked on bars after 2026-09-30 |
| Seeds / budget | Random-entry control: 5 seeds. Deterministic otherwise. CPU, minutes |

## Definitions (fixed before the run)
- **Base entry (B):** trend = EMA20 > EMA50 on 1h closes. Long when trend is up, the bar's low ≤ EMA20 and its close > EMA20 (pullback
  touch and reclaim). Short mirror (trend down, high ≥ EMA20, close < EMA20). Entry at the OPEN of the next bar. One unit per leg.
- **Base time exit:** a base leg that has no hedge is closed at the open 24 bars after entry.
- **Cost:** 0.12% round trip per leg (top-50 assumption, RESEARCH_RULES cost table), charged on every leg including hedges.
- **NATR direction (my reading of the owner's phrase):** regime_up(t) = close_t > close_{t-14}, scaled by NATR_14(t) (ATR_14/close in %).
  The regime is used only as a sign: up or down. This is the one reading that needs owner confirmation.
- **Hedge trigger (H1):** at the close of bar t, a base leg whose unrealised loss is ≥ 1.0 × NATR_14(t) (in % of its entry price)
  and that has no hedge yet opens an opposite hedge leg at the open of bar t+1.
- **Hedge lifetime (H2):** a hedged base leg is NOT time-exited. Hedge legs are never hedged again.
- **New-signal close (H3):** when a base signal of direction d fires at bar t, every open hedge leg of direction −d closes at the open of bar t+1.
- **NATR exit (H4):** at the close of bar t, any open leg (base or hedge) whose unrealised PnL is > 0 and whose direction is against the
  NATR regime closes at the open of bar t+1, hedge or not.
- **Cap (H5):** at most 6 open legs per coin. When the cap is reached no new base or hedge leg opens. This is a risk limit added by
  the card (without it a losing leg can stay open forever, see F-0053), so the cap itself is part of the tested design.
- **Forced close:** any leg still open at the end of its split closes at the last bar's close (reported separately, not hidden).

## Arms (same base entries in every arm)
- **B:** base entries + base time exit only (no overlay). Baseline.
- **H:** B + H1–H5. The card's candidate.
- **C (control):** random entries at the same count and time as B's entries, with random direction (5 seeds), with H1–H5 applied.
  Tests whether any gain comes from the overlay mechanics rather than from the base signal.

## Predicted diagnostic movement (written before the run)
- Hedge legs open at regime-adverse points, so their mean net PnL should be ≤ 0 after costs (each pays 0.12%).
- H should reduce the realised loss per closed base leg versus B, but the mean net PnL per closed leg should NOT rise above B
  by more than the seed/split noise; the gain, if any, would come from closing losses at the hedge rather than from edge.
- The mark-to-market drawdown of H should be larger than its realised P&L suggests (open legs carry unrealised loss; F-0053).
  The report must show both realised and mark-to-market numbers.
- Arm C should show the same qualitative behaviour as H if the overlay mechanics, not the base signal, drive the result.

## Accept / reject rule
- **Accept** (candidate for a fresh holdout only, not for trading) if ALL of:
  1. H mean net PnL per closed leg > 0 bp on VAL AND on TEST (pooled over the 7 coins);
  2. H beats B on net PnL per closed leg on both VAL and TEST;
  3. H beats the mean of C (5 seeds) on net PnL per closed leg on both VAL and TEST;
  4. mark-to-market max drawdown of H ≤ 3 units per coin on both VAL and TEST. Unit = one leg's notional; capital per coin is
     defined as 6 units (the cap H5), so 3 units = 50% of that capital. Fixed here, before any run.
- **Reject** otherwise. The mark-to-market drawdown and the peak open legs are reported even when H looks positive on realised PnL.
- **Unexplained:** if H is positive on realised PnL but fails rule 4, it is not adopted; the failure is recorded.

## Verified cause and mechanism
- The mechanism to test: a hedge converts an open loss into a locked loss, and the cost of the hedge plus the cost of closing it
  may exceed the benefit; the NATR exit adds a second, regime-based close. Whether this produces positive expectancy is the question.
- Known risk, from the registry: any strategy that does not stop losses shows high realised win rate while its open legs hide a
  large loss (F-0053 V0). Rule 4 and the mark-to-market reporting are there to catch that.

---
## Result (added after the run)
Run: `research/hedge_recovery/sim.py` → `E-hedge-001_results.csv` (7 coins × 3 splits × arms B, H, C×5 seeds).
Pooled over the 7 coins, mean net PnL per closed leg after 0.12% RT:

| Split | B (base only) | H (overlay) | C (random + overlay, 5 seeds) | H hedge legs | max MTM DD per coin (H) |
|---|---|---|---|---|---|
| VAL 2022–23 | −5.6 bp (21,205 legs) | **−0.4 bp** (2,010 legs) | +1.5 bp (range −44.6 … +25.1) | +7.7 bp (205) | 4.67 units (ADA) |
| TEST 2024–26 | −7.8 bp (26,362 legs) | **+32.8 bp** (3,794 legs) | +9.5 bp (range −3.3 … +27.2) | −16.0 bp (444) | 18.35 units (BNB) |
| DEV 2019–21 (not in the rule) | +10.5 bp | **−5,722 bp** (3,170 legs) | −4,410 bp | −2.3 bp (404) | 2,318 units (DOGE) |

Accept-rule check:
1. H mean > 0 on VAL AND TEST: **fails** on VAL (−0.4).
2. H beats B on both: passes.
3. H beats C on both: **fails** on VAL (−0.4 vs +1.5); passes on TEST.
4. max MTM DD ≤ 3 units per coin on VAL and TEST: **fails** (4.67 on VAL, 18.35 on TEST).

- Failure level and location: economics, at the VAL comparison against the random-entry control with the same overlay (rule 3)
  and at the drawdown limit (rule 4). The positive TEST mean is not repeated on VAL.
- Verified cause (tested by the control arm C): the same overlay on random entries gives the same sign of result on VAL
  (+1.5 bp), so the overlay mechanics alone do not produce the difference. The DEV tail (−5,722 bp mean per leg, 2,318 units of
  drawdown) shows that losing legs with no stop grow without bound, which is the F-0053 mechanism. Its TEST gain is bought with an
  18-unit drawdown.
- TEST is not a fresh holdout (2024–26 overlaps periods used by H06, H07 and H18). This does not change the verdict, since the
  rule already fails on VAL.
- Verdict: **rejected.** Registry row added (F-0064). The card is not edited after this point.
