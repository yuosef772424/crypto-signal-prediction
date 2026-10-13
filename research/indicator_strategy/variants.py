"""
PURPOSE: Runs card E-ind-002: the daily breakouts (bb_break, donchian) with arms base / long_only / not_adx40 / both, on the DEV
         and VAL splits only. Uses the same split and simulation code as screen.py so that the base arm reproduces E-ind-001.
TAGS:    variant arms, long_only, not_adx40, E-ind-002, exploratory tier, consistency check, daily breakout
PITFALLS: VAL has already been viewed in E-ind-001, so its numbers here are a consistency check, not confirmation.
         TEST is not loaded by this script.

Usage (repo root):
    python research/indicator_strategy/variants.py --data /home/user/research/ohlc --out research/indicator_strategy
"""
import argparse
import os

import numpy as np
import pandas as pd

import engine as E
from screen import COINS, SPLITS, tstat

TRIGGERS = ["bb_break", "donchian"]
ARMS = {"base": (), "long_only": ("long_only",), "not_adx40": ("not_adx40",), "both": ("long_only", "not_adx40")}


def run(data_dir, out_dir):
    full = {c: E.indicators(E.load(os.path.join(data_dir, f"ohlc_{c}_1h.parquet"), "1d")) for c in COINS}
    rows = []
    for trig in TRIGGERS:
        for arm, gates in ARMS.items():
            for split, (start, end) in SPLITS.items():
                trades = []
                for c in COINS:
                    d = full[c].loc[start:end]
                    lg, sh = E.build_signal(full[c], trig, gates)
                    idx = full[c].index.get_indexer(d.index)
                    t = E.simulate(d, lg[idx], sh[idx], "1d")
                    if len(t):
                        t.insert(0, "coin", c)
                        trades.append(t)
                allt = pd.concat(trades) if trades else pd.DataFrame()
                m = E.metrics(allt)
                coin_means = [t.net_r.mean() for t in trades]
                rows.append(dict(trigger=trig, arm=arm, split=split, n=m.get("n", 0), mean_net_r=m.get("mean_net_r", np.nan),
                                 pf=m.get("pf", np.nan), t_stat=tstat(allt.net_r) if len(allt) else np.nan,
                                 coins_positive=int(sum(v > 0 for v in coin_means)), coins_total=len(coin_means),
                                 short_share=float((allt.side == -1).mean()) if len(allt) else np.nan))
                print(f"{trig:9s} {arm:10s} {split} n={m.get('n',0):4d} net_R={m.get('mean_net_r',np.nan):+.3f} "
                      f"pf={m.get('pf',np.nan):.2f} t={rows[-1]['t_stat']:+.2f} coins+={rows[-1]['coins_positive']}/{len(coin_means)}",
                      flush=True)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "E-ind-002_summary.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.data, a.out)
