---
id: E-disc-002
mechanism: breakout; trend-following; regime-filter
timeframe: 1d
target: pnl
model_class: rule
reopens: F-0054:R1
not_covered_by: F-0072=volume-confirmed Donchian on unseen assets, a different trigger and filter from this card; F-0070=the momentum rotation of E-disc-004, this card varies exits and costs of breakout survivors and is not a rotation; F-0071=the unseen-asset test of S2 in E-disc-003, this card is the stability check that comes before it; F-0069=daily ADX>40 removal and short-side removal from E-ind-002, this card keeps the E-disc-001 survivors unchanged and only varies cost, entry delay and exit parameters; F-0061=stop orders at predicted candle extremes, no predicted level here; F-0062=regime filters on wick-capture fills, the ADX gate here is an entry condition on the breakout trigger
---
# Experiment card — E-disc-002: robustness of the three E-disc-001 survivors (DEV and VAL only)

Written and committed BEFORE the run (rules: `docs/research/RESEARCH_RULES.md`, `docs/research/MODEL_DEV_PROTOCOL.md`).

## Honesty about the split
The three survivors were selected on VAL (E-disc-001). VAL is therefore NOT an independent sample any more, and this card's
VAL numbers are a stability check, not a confirmation. The independent tests are E-disc-003 (unseen assets) and E-disc-004
(fresh bars after 2026-09-30). Nothing here may be called "validated".

## Survivors (from E-disc-001, fixed)
- S1: `1d|donchian20|adx25`
- S2: `1d|bb_break|adx25`
- S3: `1d|donchian20|none`

## Variants (fixed before the run; one change per row)
- `base`: the E-disc-001 settings (stop 2×ATR, target 3×ATR, 30 bars).
- `cost2`: cost 0.24% round trip instead of 0.12%.
- `delay1`: entry one extra bar later (the signal bar's next open is skipped; the position fills at the open after that).
- `grid`: 36 exit settings: stop ∈ {1.5, 2, 2.5, 3} ATR, target ∈ {2.5, 3, 3.5} ATR, time exit ∈ {20, 30, 40} bars (the base is one of them).

## Accept rule (stability, decided before the run)
A survivor is **robust** (candidate for E-disc-003) only if ALL of:
1. `cost2`: pooled mean net R > 0 on DEV AND on VAL;
2. `delay1`: pooled mean net R > 0 on VAL;
3. `grid`: at least 70% of the 36 exit settings have pooled mean net R > 0 on VAL;
4. year check on VAL: the mean net R is positive in 2022 AND in 2023 (the base settings).
A survivor that fails any condition is closed for this grid, and its failure is recorded in the registry.

## Predicted diagnostic movement
- `cost2` should reduce net R by about 0.01R per trade on 1d, so survivors with a large gross edge should stay positive.
- `delay1` should reduce net R modestly; a large drop would mean the edge sits in the first bar's move (a sign of look-ahead or
  fill sensitivity).
- The `grid` should be a broad plateau of positive values, not a single spike.

---
## Result (added after the run)
Run: `research/strategy_discovery/robustness.py` → `results/E-disc-002_grid.csv` and `results/E-disc-002_verdicts.csv`. DEV and VAL only.

| Survivor | DEV base | VAL base | DEV cost ×2 | VAL cost ×2 | VAL delay 1 bar | VAL grid positive share | VAL 2022 | VAL 2023 | Robust |
|---|---|---|---|---|---|---|---|---|---|
| S1 donchian20 + adx25 | +0.161 | +0.306 | +0.152 | +0.295 | +0.152 | 100% of 36 | +0.200 | +0.382 | **yes** |
| S2 bb_break + adx25 | +0.172 | +0.312 | +0.163 | +0.301 | +0.185 | 100% of 36 | +0.260 | +0.346 | **yes** |
| S3 donchian20, no gate | +0.252 | +0.196 | +0.243 | +0.184 | **+0.068** | 100% of 36 | +0.104 | +0.274 | **yes** |

All three pass the four conditions. Two cautions:
- **S3 is sensitive to entry timing:** a one-bar delay cuts its VAL mean from +0.196 to +0.068 (−65%). The rule allows this (it only needs to stay positive), but this is the weakest point of the three.
- The exit grid is a clean plateau: every one of the 36 stop/target/time settings is positive on VAL for all three survivors, so the result does not depend on a single exit setting.

VAL is not an independent sample for these survivors (see E-disc-001). This card is a stability check.
- Verdict: **all three robust under the stability rule.** They go to E-disc-003, which is the first independent test.
