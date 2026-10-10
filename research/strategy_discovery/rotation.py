"""
PURPOSE: Runs card E-disc-004: cross-sectional momentum rotation over the seven original coins on daily closes. Holds the top K
         coins by the return over L days (only those with a positive return), equal weight, rebalanced every R days; the rest in
         cash. Weights set at the close of day t earn day t+1 returns. Turnover is charged at 0.12% per unit traded. Writes the
         DEV and VAL metrics, the BH q-values on the VAL t-statistics and a buy-and-hold benchmark.
TAGS:    cross-sectional momentum, rotation portfolio, turnover cost, equal weight, Sharpe, max drawdown, E-disc-004, BH correction
PITFALLS: The universe is not constant: SOL, DOGE and BNB start later, so each day only coins with data and a lookback are ranked.
         The cost is charged on the change of weights (sum of |dw|) at each rebalance. No model, no parameter tuning beyond the grid.

Usage (repo root):
    python research/strategy_discovery/rotation.py --data /home/user/research/ohlc --out research/strategy_discovery/results
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trade_engine as E  # noqa: E402
from grid import COINS, SPLITS, bh_qvalues, one_sided_p  # noqa: E402

COST = 0.0012
LOOKBACKS = [30, 60, 120]
KS = [1, 2, 3]
REBALS = [7, 30]


def closes(data_dir):
    cols = {}
    for c in COINS:
        d = E.load(os.path.join(data_dir, f"ohlc_{c}_1h.parquet"), "1d")
        cols[c] = d.close
    return pd.DataFrame(cols).sort_index()


def rotation(P, L, K, R):
    """Daily net portfolio returns for one configuration."""
    rets = P.pct_change()
    mom = P / P.shift(L) - 1.0
    n = len(P)
    w = np.zeros(P.shape[1])
    out = np.full(n, np.nan)
    cost = np.zeros(n)
    for t in range(1, n):
        out[t] = float(np.nansum(w * rets.iloc[t].fillna(0).values))     # weights set before day t earn day t returns
        if t % R == 0:
            m = mom.iloc[t]
            valid = m.dropna()
            valid = valid[valid > 0].sort_values(ascending=False).head(K)
            new = np.zeros_like(w)
            if len(valid):
                new[[P.columns.get_loc(c) for c in valid.index]] = 1.0 / len(valid)
            cost[t] = COST * np.abs(new - w).sum()
            w = new
    net = pd.Series(out - cost, index=P.index).dropna()
    return net


def metrics(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return dict(n=len(x), mean=np.nan, sharpe=np.nan, mdd=np.nan, t=np.nan)
    sd = x.std(ddof=1)
    sharpe = x.mean() / sd * math.sqrt(365) if sd > 0 else np.nan
    eq = np.cumprod(1 + x)
    mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
    t = x.mean() / (sd / math.sqrt(len(x))) if sd > 0 else np.nan
    return dict(n=len(x), mean=float(x.mean()), sharpe=float(sharpe), mdd=mdd, t=float(t))


def run(data_dir, out_dir):
    P = closes(data_dir)
    rows = []
    for L in LOOKBACKS:
        for K in KS:
            for R in REBALS:
                net = rotation(P, L, K, R)
                for split, (s, e) in SPLITS.items():
                    sub = net.loc[s:e]
                    m = metrics(sub.values)
                    rows.append(dict(config=f"L{L}|K{K}|R{R}", L=L, K=K, R=R, split=split, **m))
    res = pd.DataFrame(rows)
    val = res[res.split == "VAL"].copy()
    val["p"] = [one_sided_p(t) for t in val.t]
    val["bh_q_val"] = bh_qvalues(val.p.values)
    res = res.merge(val[["config", "bh_q_val"]], on="config", how="left")
    # benchmark: equal weight of all available coins, rebalanced daily, no cost
    bench = P.pct_change().mean(axis=1)
    for split, (s, e) in SPLITS.items():
        m = metrics(bench.loc[s:e].dropna().values)
        rows.append(dict(config="benchmark_equal_weight_bh", L=np.nan, K=np.nan, R=np.nan, split=split, **m))
    res = pd.concat([res, pd.DataFrame(rows[-2:])], ignore_index=True)
    os.makedirs(out_dir, exist_ok=True)
    res.to_csv(os.path.join(out_dir, "E-disc-004_summary.csv"), index=False)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.data, a.out)
