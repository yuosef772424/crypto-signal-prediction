---
id: E-disc-005
mechanism: trend-following; breakout; volatility-structure; volume-flow
timeframe: 4h;1d
target: pnl
model_class: rule
reopens: F-0054:R1
not_covered_by: F-0072=the campaign card that selected this family, its unseen-asset failure is recorded from E-disc-006; F-0053=previous-candle range fade with limit orders, not a band or channel trigger; F-0061=stop orders at predicted candle extremes, this card uses indicator triggers on closes; F-0062=regime filters on wick-capture fills, the gates here are entry conditions on trend or volatility triggers; F-0063=chart-image CNN, no image model here; F-0068=4h breakout and pullback families, re-run only as part of the multiple-testing family, with the same gates reported separately; F-0069=daily ADX>40 and short-side refinements, a different test; F-0070=cross-sectional momentum rotation, a portfolio with turnover costs, not a single-asset trade; F-0071=Bollinger breakout with ADX gate on unseen assets, a different trigger
---
# Experiment card — E-disc-005: three families not yet tested (Supertrend flip, Keltner squeeze, volume-confirmed Donchian)

Written and committed BEFORE the run (rules: `docs/research/RESEARCH_RULES.md`, `docs/research/MODEL_DEV_PROTOCOL.md`).
New families from the literature search (Supertrend, Keltner squeeze, volume confirmation). Not run in any earlier card.

| Field | Value |
|---|---|
| Date / author | 2026-10-10 / Claude, on branch `claude/strategy-discovery` |
| Baseline | same data and exit as E-disc-001: seven original coins, 4h and 1d, `/home/user/research/ohlc`, DEV 2019–2021 and VAL 2022–2023 only |
| The ONE change | the signal family; the gate set, the exit and the cost are the same as E-disc-001 |
| Failure being addressed | none yet: three new families. F-0068 is re-run only as part of the multiple-testing family, and its VAL gates are reported separately |
| Data / splits | DEV and VAL only. TEST, BCH/TRX/ZEC and the 2026-10 bars are not loaded |
| Seeds / budget | deterministic; 3 families × 3 gates × 2 timeframes = 18 new configurations; the BH correction covers all 60 configurations of the campaign (42 from E-disc-001 and 18 here) |

## Families (fixed before the run)
- `supertrend`: ATR(10) with multiplier 3. Long on a flip of the trend to up; short on a flip to down.
- `keltner_squeeze`: Keltner channel = EMA20 ± 1.5 × ATR14. A squeeze is when the Bollinger(20, 2) bands lie inside the Keltner
  channel on the previous bar. Long when close breaks above the upper Keltner band during or just after a squeeze; mirror for short.
- `volume_donchian`: close above the prior 20-bar high (long) or below the prior 20-bar low (short), AND the bar's volume is at least
  1.5 × the median volume of the prior 20 bars.

## Gates (same as E-disc-001)
`none`; `adx25`; `ema200_aligned`.

## Exit and cost (same as E-disc-001)
Stop 2 × ATR(14); target 3 × ATR(14); 30-bar time exit; cost 0.12% RT in R; entry at the next open.

## Accept / reject rule (written before the run)
A configuration is a **survivor** only if ALL of:
1. pooled mean net R > 0 on DEV AND on VAL;
2. Benjamini-Hochberg q-value of the one-sided VAL t-statistic, over the 60 configurations of the campaign, ≤ 0.10;
3. VAL profit factor ≥ 1.10;
4. VAL net R positive on at least 5 of the 7 coins.
A survivor goes to the robustness card and then to the unseen-asset test. It is not a finding.

## Predicted diagnostic movement
- The volume-confirmed Donchian should beat the plain Donchian on DEV if volume carries information; if it does not, the volume gate is not useful.
- Supertrend is a close relative of a moving-average trend rule, so it should resemble the pullback family and not beat the Donchian
  breakouts. A clear win would be unexpected and is reported.

---
## Result (added after the run)
(to be filled after the run; the card is not edited after the run)
