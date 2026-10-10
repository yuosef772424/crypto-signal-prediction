---
id: E-disc-004
mechanism: momentum; trend-following; cross-sectional
timeframe: 1d
target: pnl
model_class: rule
reopens:
not_covered_by: F-0072=volume-confirmed Donchian on unseen assets, a different trigger and filter from this card; F-0070=this card is the source of F-0070 (its own result, not a different scope); F-0014=cross-sectional momentum rank MOM_RANK_6/24 as a feature for direction, a rolling IC screen of ranked features, not a portfolio rotation with turnover costs; F-0031=trained NIG-TimeNet rank portfolio, no model here; F-0041=cross-sectional rank of volatile coins, a mean-return test on losers, not a rotation; F-0007=Hurst and variance-ratio features, not momentum; F-0011=momentum proxy on absolute close, a feature screen, not a portfolio; F-0067=MACD histogram zero-cross as a trigger, not a cross-sectional rotation
---
# Experiment card — E-disc-004: cross-sectional momentum rotation across the seven original coins (DEV and VAL)

Written and committed BEFORE the run (rules: `docs/research/RESEARCH_RULES.md`, `docs/research/MODEL_DEV_PROTOCOL.md`).
This is a PORTFOLIO family, new to this project, so the trade-level R metric is not used: the outcome is the daily return of the
equal-weight portfolio after costs on turnover.

| Field | Value |
|---|---|
| Date / author | 2026-10-10 / Claude, on branch `claude/strategy-discovery` |
| Baseline | daily closes (UTC) of BTC ETH SOL BNB XRP ADA DOGE from `/home/user/research/ohlc`, through 2026-09-30. TEST not loaded, unseen assets and fresh bars not loaded |
| The ONE change | the rotation rule (lookback, number held, rebalance interval); the universe, the cost and the metric are fixed |
| Failure being addressed | none yet: a new family. The F-0054 reopen condition is about trade-level cost in R and does not apply here, so this card does not claim to reopen F-0054 |
| Data / splits | DEV 2019–2021, VAL 2022–2023. Nothing else |
| Seeds / budget | deterministic; 3 lookbacks × 3 numbers held × 2 rebalance intervals = 18 configurations; CPU, seconds |

## Rule (fixed before the run)
- At each rebalance day t (every `R` days), rank the seven coins by their return over the last `L` days (close_t / close_{t−L} − 1).
- Hold the top `K` coins with a positive return, with equal weight; the rest of the capital is in cash (zero return).
- The weights set at the close of t earn the returns of day t+1 (no look-ahead).
- Turnover cost: 0.12% per unit of notional traded (buying or selling), charged on the change of weights at each rebalance.
- Grid: L ∈ {30, 60, 120}, K ∈ {1, 2, 3}, R ∈ {7, 30}.

## Accept rule (discovery, written before the run)
A configuration is a **survivor** only if ALL of:
1. mean daily net return > 0 on DEV AND on VAL;
2. Benjamini-Hochberg q-value of the one-sided t-statistic of the VAL daily net returns, over the 18 configurations, ≤ 0.10;
3. VAL annualised Sharpe ratio ≥ 0.5 (mean / std of daily net returns × √365);
4. the VAL maximum drawdown is at most 35%.
A survivor is a candidate for an unseen-asset test; it is not a finding. VAL is not fresh for this family either: it has not
been used for any portfolio decision before, so this card is the first use of VAL here.

## Predicted diagnostic movement
- Cross-sectional momentum should beat an equal-weight buy-and-hold of the seven coins on drawdown more than on return.
- K = 1 should be the most volatile setting; K = 3 the closest to the basket.
- The turnover cost should bite most at R = 7 with K = 1.

---
## Result (added after the run)
Run: `research/strategy_discovery/rotation.py` → `results/E-disc-004_summary.csv` (18 configurations plus the buy-and-hold benchmark; DEV and VAL).

Survivors under the four conditions: **0 of 18.**
- Condition 2 (BH q ≤ 0.10 on VAL) fails for every configuration: the smallest VAL q is 0.47.
- Condition 4 (VAL max drawdown ≤ 35%) fails for every configuration: the least-bad VAL drawdown is −40% (L30|K3|R30).

Best VAL settings and the benchmark (equal weight of the seven coins, rebalanced daily):

| Setting | VAL mean daily net | VAL Sharpe | VAL max DD | VAL t | BH q |
|---|---|---|---|---|---|
| L30 K3 R30 | +0.0025 | 1.16 | −40% | 1.64 | 0.47 |
| L30 K2 R30 | +0.0025 | 1.10 | −46% | 1.55 | 0.47 |
| L120 K3 R30 | +0.0012 | 0.74 | −46% | 1.05 | 0.47 |
| benchmark (equal-weight buy and hold) | +0.0002 | 0.13 | −71% | 0.18 | — |

Reading: the rotation has a better VAL drawdown and Sharpe than the benchmark, but the VAL mean is small and not significant after the correction, and the drawdown is still above the limit. DEV looks stronger (Sharpe up to 1.9), but that period is the bull run of 2020–2021.
- Verdict: **rejected under the rule.** A registry row is recorded (F-0070), so the same rotation is not re-run without a reopen condition.
