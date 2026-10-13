"""
PURPOSE: Runs card E-srch-003: time-of-day seasonality on 15m bars. For each UTC start hour h, hold H bars (4, 8, 16) and direction
         (long or short), enters at the open of the bar starting at minute 0 of hour h and exits at the close H bars later, net of
         0.12% round trip. DEV and VAL on the seven original coins; BH over the 144 configurations on the VAL t-statistics.
TAGS:    seasonality, time of day, 15m, E-srch-003, hour of day, hold length, BH correction, DEV VAL
PITFALLS: Trades overlap by design (every hour opens one), so the t-statistic is optimistic; the BH count is over configurations only.
         Vectorised with numpy on the 15m open and close arrays; no stop or target.

Usage (repo root):
    python research/strategy_discovery/seasonality15m.py --out research/strategy_discovery/results
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid import COINS, SPLITS, bh_qvalues, one_sided_p  # noqa: E402

DATA = "/home/user/research/ohlc15"
COST = 0.0012
HOLDS = [4, 8, 16]
HOURS = list(range(24))


def coin_frame(c, s, e):
    d = pd.read_parquet(os.path.join(DATA, f"ohlc_{c}_15m.parquet")).sort_index()
    return d.loc[s:e]


def trade_returns(d, hour, hold, direction):
    """Net return per trade for one configuration on one coin slice."""
    idx = np.flatnonzero((d.index.hour == hour) & (d.index.minute == 0))
    O, C = d.open.values, d.close.values
    ex = idx + hold
    ok = ex < len(C)
    idx, ex = idx[ok], ex[ok]
    gross = direction * (C[ex] / O[idx] - 1.0)
    return gross - COST, gross


def run(out_dir):
    rows, per_coin_store = [], {}
    frames = {split: {c: coin_frame(c, s, e) for c in COINS} for split, (s, e) in SPLITS.items()}
    for hour in HOURS:
        for hold in HOLDS:
            for direction, dname in ((1, "long"), (-1, "short")):
                cfg = f"15m|h{hour:02d}|hold{hold}|{dname}"
                for split in SPLITS:
                    nets, grosses, coin_means = [], [], []
                    for c in COINS:
                        net, gross = trade_returns(frames[split][c], hour, hold, direction)
                        nets.append(net)
                        grosses.append(gross)
                        coin_means.append(net.mean() if len(net) else np.nan)
                    net = np.concatenate(nets) if nets else np.array([])
                    gross = np.concatenate(grosses) if grosses else np.array([])
                    n = len(net)
                    mean = float(net.mean()) if n else np.nan
                    sd = float(net.std(ddof=1)) if n > 1 else np.nan
                    t = mean / (sd / math.sqrt(n)) if n > 1 and sd > 0 else np.nan
                    rows.append(dict(config=cfg, hour=hour, hold=hold, direction=dname, split=split, n=n,
                                     mean_net=mean, mean_gross=float(gross.mean()) if n else np.nan,
                                     t_stat=t, coins_positive=int(sum(1 for v in coin_means if v == v and v > 0)),
                                     coins_total=int(sum(1 for v in coin_means if v == v))))
    res = pd.DataFrame(rows)
    val = res[res.split == "VAL"].copy()
    val["p"] = [one_sided_p(t) for t in val.t_stat]
    val["bh_q_val"] = bh_qvalues(val.p.values)
    res = res.merge(val[["config", "bh_q_val"]], on="config", how="left")
    res["survivor"] = False
    for cfg, g in res.groupby("config"):
        dev = g[g.split == "DEV"].iloc[0]
        val_r = g[g.split == "VAL"].iloc[0]
        ok = (dev.mean_net > 0) and (val_r.mean_net > 0) and (val_r.bh_q_val <= 0.10) and (val_r.coins_positive >= 5) and (val_r.n >= 500)
        res.loc[res.config == cfg, "survivor"] = bool(ok)
    os.makedirs(out_dir, exist_ok=True)
    res.to_csv(os.path.join(out_dir, "E-srch-003_summary.csv"), index=False)
    print("configurations:", res.config.nunique(), "survivors:", int(res[res.split == 'VAL'].survivor.sum()))
    best = res[res.split == "VAL"].sort_values("bh_q_val").head(8)
    print(best[["config", "n", "mean_net", "mean_gross", "t_stat", "bh_q_val", "coins_positive"]].round(5).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.out)
