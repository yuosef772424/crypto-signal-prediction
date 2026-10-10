"""
PURPOSE: Runs card E-srch-002: MACD histogram zero-cross on 15m and 1h, with gates none, adx25 and ema200_aligned, DEV and VAL on the
         seven original coins. Reports net and gross R next to the measured median cost in R, with Benjamini-Hochberg over the
         six configurations. Uses the shared trade engine with an explicit max_bars per timeframe (96 bars on 15m, 48 on 1h).
TAGS:    intraday, MACD, 15m, 1h, E-srch-002, cost in R, gross versus net, BH, DEV VAL
PITFALLS: The engine's default max_bars table has no 15m key, so max_bars is passed explicitly. Gates use the engine's gate names
         (adx25, ema200_up and ema200_dn for the aligned gate). Unseen coins and the fresh window are not loaded here.

Usage (repo root):
    python research/strategy_discovery/intraday_macd.py --out research/strategy_discovery/results
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
from grid import COINS, SPLITS, bh_qvalues, one_sided_p, tstat  # noqa: E402

DATA = {"15m": "/home/user/research/ohlc15", "1h": "/home/user/research/ohlc_full"}
MAX_BARS = {"15m": 96, "1h": 48}
GATE_NAMES = {"none": (), "adx25": ("adx25",), "ema200_aligned": ("ema200_up", "ema200_dn")}


def load_indicators(tf, coin):
    path = os.path.join(DATA[tf], f"ohlc_{coin}_{tf}.parquet" if tf == "15m" else f"ohlc_{coin}_1h.parquet")
    d = pd.read_parquet(path).sort_index()
    return E.indicators(d)


def run(out_dir):
    rows, trade_logs = [], []
    for tf in ("15m", "1h"):
        full = {c: load_indicators(tf, c) for c in COINS}
        for gname, gates in GATE_NAMES.items():
            for split, (s, e) in SPLITS.items():
                trades, costs = [], []
                for c in COINS:
                    dfull = full[c]
                    lg, sh = E.build_signal(dfull, "macd_cross", gates)
                    d = dfull.loc[s:e]
                    idx = dfull.index.get_indexer(d.index)
                    t = E.simulate(d, lg[idx], sh[idx], tf, max_bars=MAX_BARS[tf])
                    if len(t):
                        t.insert(0, "coin", c)
                        t.insert(0, "config", f"{tf}|macd_cross|{gname}")
                        t.insert(0, "split", split)
                        trades.append(t)
                        costs.append(float(np.median(E.COST / (t.risk_pct / 100.0))))
                allt = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame(columns=["net_r", "gross_r"])
                m = E.metrics(allt)
                rows.append(dict(config=f"{tf}|macd_cross|{gname}", tf=tf, gate=gname, split=split, n=m.get("n", 0),
                                 mean_net_r=m.get("mean_net_r", np.nan), mean_gross_r=m.get("mean_gross_r", np.nan),
                                 pf=m.get("pf", np.nan), t_stat=tstat(allt.net_r) if len(allt) else np.nan,
                                 median_cost_r=float(np.median(costs)) if costs else np.nan,
                                 coins_positive=int(sum(1 for c in COINS if len(allt) and
                                                        allt[allt.coin == c].net_r.mean() > 0)) if len(allt) else 0))
                if split in ("DEV", "VAL") and len(allt):
                    trade_logs.append(allt)
                print(f"{tf} {gname:14s} {split} n={rows[-1]['n']:6d} net={rows[-1]['mean_net_r']:+.3f} "
                      f"gross={rows[-1]['mean_gross_r']:+.3f} cost_R={rows[-1]['median_cost_r']:.3f} "
                      f"pf={rows[-1]['pf']:.2f} t={rows[-1]['t_stat']:+.2f}", flush=True)
    res = pd.DataFrame(rows)
    val = res[res.split == "VAL"].copy()
    val["p"] = [one_sided_p(t) for t in val.t_stat]
    val["bh_q"] = bh_qvalues(val.p.values)
    res = res.merge(val[["config", "bh_q"]], on="config", how="left")
    os.makedirs(out_dir, exist_ok=True)
    res.to_csv(os.path.join(out_dir, "E-srch-002_summary.csv"), index=False)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.out)
