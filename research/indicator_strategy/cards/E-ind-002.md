---
id: E-ind-002
mechanism: breakout; trend-following; regime-filter
timeframe: 1d
target: pnl
model_class: rule
reopens: F-0054:R1
not_covered_by: F-0069=this card is the source of F-0069 (its own result, not a different scope); F-0055=maker limit entry at EMA20 with scaling on 1h, this card is a daily breakout with market fills and fixed ATR exit; F-0061=stop orders at PREDICTED candle extremes, this card uses indicator triggers on closes; F-0062=regime filters on wick-capture fills, the ADX cut here is an entry condition on breakout signals, not a fill-side filter; F-0064=E-hedge-001 hedge-and-hold, no hedging here; F-0065=E-hedge-002 reversal hedge, no hedging here
---
# Experiment card — E-ind-002: remove conditions from the daily breakouts (short side, blow-off ADX)

Written and committed BEFORE the run (rules: `docs/research/RESEARCH_RULES.md`, `docs/research/MODEL_DEV_PROTOCOL.md`).
Base configurations come from E-ind-001 (1d, no gate): `bb_break` and `donchian`. Exit unchanged (2×ATR stop, 3×ATR target,
30-bar time exit, cost 0.12% RT).

## Honesty about the split
The two edits below were chosen from a region analysis of DEV and VAL together, and then re-read on DEV alone. On DEV alone:
- removing shorts: longs +0.41R, shorts −0.07R (supported on DEV);
- removing ADX > 40 entries: +0.06R versus +0.30R (supported on DEV);
- adding an EMA200 trend condition to longs: +0.41R versus +0.37R (not supported on DEV, so it is dropped and not tested).

**VAL has already been looked at** (in the region analysis and in E-ind-001). Therefore the VAL numbers in this card are a
CONSISTENCY CHECK, not an independent confirmation. A configuration that passes here is an exploratory candidate only. The
verdict for any configuration that is taken further is made on TEST, once.

| Field | Value |
|---|---|
| Date / author | 2026-10-10 / Claude (for the project owner) |
| Baseline | E-ind-001 base rows: `bb_break` and `donchian`, 1d, gate none, and their adx25 variants |
| The ONE change (per arm) | arm A: remove the short side (`long_only`); arm B: remove entries when ADX > 40 (`not_adx40`); arm C: both A and B; the base is the screen configuration |
| Failure being addressed | none yet: a refinement of a screen candidate |
| Data / splits | DEV 2019–2021 (decision basis), VAL 2022–2023 (consistency check only, already viewed). TEST not touched |
| Seeds / budget | deterministic; 2 triggers × 4 arms, 7 coins, 2 splits; CPU, seconds |

## Accept rule (exploratory tier)
An arm is an **exploratory candidate** for a TEST card only if ALL of:
1. its DEV mean net R is higher than its base's DEV mean net R, AND DEV mean net R > 0;
2. its VAL mean net R is higher than its base's VAL mean net R (consistency check);
3. its VAL profit factor > 1.0 and VAL net R is positive on at least 5 of 7 coins.
A pass in this card does not establish the edge. It only allows one TEST evaluation of the chosen arm.

## Predicted diagnostic movement
- Arm A removes the short side. Predicted: DEV improves; VAL may not (shorts were positive in 2022).
- Arm B removes the ADX > 40 entries. Predicted: DEV improves; VAL may not (the VAL ADX > 40 subset was +0.20R).
- A "no change" outcome on VAL for either arm is a valid result and is reported as such.

---
## Result (added after the run)
Run: `research/indicator_strategy/variants.py` → `E-ind-002_summary.csv` (DEV and VAL only; TEST not loaded).
Pooled net R per trade. The base arm reproduces E-ind-001.

| Trigger | Arm | DEV net R | VAL net R (base) | VAL pf | VAL t | VAL coins+ | Passes exploratory rule |
|---|---|---|---|---|---|---|---|
| bb_break | long_only (no shorts) | +0.458 (+0.259) | +0.186 (+0.179) | 1.36 | 1.60 | 5/7 | yes |
| bb_break | not_adx40 | +0.308 | +0.192 | 1.40 | 2.11 | 6/7 | yes |
| bb_break | both | +0.543 | +0.124 | 1.23 | 0.99 | 5/7 | no (VAL below base) |
| donchian | long_only (no shorts) | +0.377 (+0.252) | **+0.216** (+0.196) | 1.45 | 2.00 | 6/7 | yes |
| donchian | not_adx40 | +0.300 | +0.183 | 1.40 | 2.27 | 7/7 | no (VAL below base) |
| donchian | both | +0.462 | +0.172 | 1.34 | 1.45 | 6/7 | no (VAL below base) |

- Removing shorts is the only edit that improves both DEV and VAL. Its VAL gain is small (+0.02R at most, and within noise: t 1.60 and 2.00).
- Removing ADX > 40 entries improves DEV but not VAL.
- Caveat: VAL was already viewed in E-ind-001. These are consistency checks, not confirmation.
- Verdict: **exploratory candidates.** The donchian long-only arm has the highest VAL mean and was selected for E-ind-003 by a rule stated before that card. Registry row F-0069 covers the failed arms.
