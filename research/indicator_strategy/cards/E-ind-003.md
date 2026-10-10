---
id: E-ind-003
mechanism: breakout; trend-following
timeframe: 1d
target: pnl
model_class: rule
reopens: F-0054:R1
not_covered_by: F-0072=volume-confirmed Donchian on unseen assets, a different trigger and filter from this card; F-0070=cross-sectional momentum rotation of E-disc-004, a portfolio with turnover costs, not a single-asset breakout trade; F-0071=Bollinger band breakout with ADX gate on unseen assets, a different trigger from the Donchian channel of this card; F-0069=different scope, F-0069 removes ADX > 40 entries and tests the combined arm, E-ind-003 tests only the long-only daily Donchian with no ADX cut; F-0055=maker limit entry at EMA20 with scaling on 1h, this card is a daily Donchian breakout with market fills and a fixed ATR exit; F-0061=stop orders at PREDICTED candle extremes, this card uses an indicator trigger on the close; F-0062=regime filters on wick-capture fills, no regime filter here; F-0064=E-hedge-001 hedge-and-hold, no hedging here; F-0065=E-hedge-002 reversal hedge, no hedging here
---
# Experiment card — E-ind-003: final test of one configuration on TEST (run ONCE)

Written and committed BEFORE the TEST run (rules: `docs/research/RESEARCH_RULES.md`, `docs/research/MODEL_DEV_PROTOCOL.md`).

## The configuration (fixed before TEST is opened)
- Signal: Donchian breakout on 1d (close above the prior 20-bar high for longs), **long side only** (`long_only`).
- Exit: 2×ATR(14) stop, 3×ATR(14) target, 30-bar time exit, stop first inside a bar, cost 0.12% RT.
- Chosen in E-ind-002 as the highest VAL mean net R among the arms that passed the exploratory rule (DEV improved, VAL improved,
  VAL profit factor > 1, at least 5 of 7 coins positive on VAL). Its VAL gain over the base is +0.02R, which is small.

## Why this card is a test and not a discovery
- Every configuration in the screen (E-ind-001) and every arm in E-ind-002 was chosen with DEV and VAL. The base Donchian
  daily configuration is the screen survivor, and the long-only arm is a condition removed after reading DEV.
- TEST = 2024-01-01 → 2026-09-30. Some of these bars were used earlier by other hypotheses (H06, H07, H18), so TEST here is the
  best remaining sample, not a fresh one. A pass must be re-checked on bars after 2026-09-30 before any claim of an edge.

## Data / splits
- TEST only (2024-01-01 → 2026-09-30). The base arm (`donchian`, no gate) is reported alongside for context. It does NOT enter the rule.

## Accept / reject rule (TEST, decided before the run)
The configuration is **accepted as a candidate for out-of-time confirmation** only if ALL of:
1. pooled mean net R > 0 on TEST;
2. profit factor > 1.0 on TEST;
3. TEST net R is positive on at least 5 of 7 coins;
4. TEST t-statistic of the trade-level net R ≥ 2.0.
If any condition fails, the configuration is **rejected**, and a failure row is added to the registry. The base arm's TEST result
does not change the verdict.

## Predicted diagnostic movement
- Long-only should keep most of the Donchian daily edge in the uptrend years and should not depend on shorts. Its TEST mean is
  predicted to be positive but smaller than DEV's, since DEV's long-only gain was partly bull-market drift.

---
## Result (added after the run)
Run: `research/indicator_strategy/test_final.py`, executed ONCE on TEST (2024-01-01 → 2026-09-30). Output: `E-ind-003_test_summary.csv`, `E-ind-003_test_trades.csv`.

| Arm | n | Mean net R | Profit factor | Win rate | t | Coins positive |
|---|---|---|---|---|---|---|
| **chosen: donchian 1d, long only** | 205 | **+0.290** | **1.66** | 0.517 | **3.47** | 6/7 |
| context: donchian 1d, base (not in rule) | 344 | +0.194 | 1.42 | 0.497 | 3.07 | 7/7 |

Rule: mean net R > 0 (+0.290 ✓), pf > 1.0 (1.66 ✓), at least 5 of 7 coins positive (6 ✓), t ≥ 2.0 (3.47 ✓). **Verdict: ACCEPTED as a candidate for out-of-time confirmation.**

Where it succeeds and fails (TEST, chosen arm; figures in `figures/`):
- By year: 2024 +0.41R (n=96), 2025 +0.26R (n=57), 2026 +0.10R (n=52). The edge is shrinking, and the last year is weakest.
- By coin: DOGE +0.62R, XRP +0.45R, BNB +0.39R, BTC +0.26R, SOL +0.20R, ADA +0.13R, **ETH −0.05R**.
- By ADX: below 20 +0.56R (n=70); 20–25 +0.15R; 25–30 +0.41R; 30–40 +0.29R; **above 40 −0.16R (n=35)**.
- By trend: close above EMA200 +0.32R (n=155); below +0.19R (n=50).
- By volatility quartile: calmest +0.67R; the middle two +0.11R; most volatile +0.27R.
- By exit: target 99 trades (+1.49R each), stop 84 (−1.01R each), time 15, end of data 7. Max drawdown in trade sequence 17.5R.
- The equity curve (`figures/equity_curve_R.png`) is flat between 2022 and 2023, which the pooled VAL figure hides.

Caveats:
- TEST is not fresh: 2024–2026 overlaps periods used by earlier hypotheses. The verdict is a candidate, not an established edge.
- The 2026 slide and the ETH and ADX > 40 results are observations on the consumed TEST sample. They cannot be used to change this configuration now. Any change must be tested on bars after 2026-09-30.
- The ADX > 40 result on TEST (−0.16R) agrees with the DEV pattern and disagrees with VAL. That is a question for new data, not a rule.
- Verdict: **accepted as candidate for out-of-time confirmation**; no claim of a proven edge.
