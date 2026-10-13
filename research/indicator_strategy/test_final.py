"""
PURPOSE: Runs card E-ind-003 ONCE on the TEST split (2024-01-01 → 2026-09-30): the daily Donchian breakout, long side only.
         The base arm (no gate) is written for context and does not enter the rule. Writes the verdict and the trade log.
TAGS:    TEST, final evaluation, E-ind-003, one-shot, donchian long_only, accept rule, trade log
PITFALLS: Running this script again is a new use of TEST and must be recorded as such. The verdict is the rule in the card,
         applied to the chosen arm only.

Usage (repo root):
    python research/indicator_strategy/test_final.py --data /home/user/research/ohlc --out research/indicator_strategy
"""
import argparse
import os

import numpy as np
import pandas as pd

import engine as E
from screen import COINS, tstat

START, END = "2024-01-01", "2026-09-30 23:00"


def run(data_dir, out_dir):
    full = {c: E.indicators(E.load(os.path.join(data_dir, f"ohlc_{c}_1h.parquet"), "1d")) for c in COINS}
    arms = {"chosen_donchian_long_only": ("long_only",), "context_donchian_base": ()}
    rows, all_trades = [], []
    for name, gates in arms.items():
        trades = []
        for c in COINS:
            d = full[c].loc[START:END]
            lg, sh = E.build_signal(full[c], "donchian", gates)
            idx = full[c].index.get_indexer(d.index)
            t = E.simulate(d, lg[idx], sh[idx], "1d")
            if len(t):
                t.insert(0, "coin", c)
                t.insert(0, "arm", name)
                trades.append(t)
        allt = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
        m = E.metrics(allt)
        coin_means = {c: t.net_r.mean() for t in trades for c in [t.coin.iloc[0]]}
        pos = sum(v > 0 for v in coin_means.values())
        row = dict(arm=name, n=m.get("n", 0), mean_net_r=m.get("mean_net_r", np.nan), pf=m.get("pf", np.nan),
                   win=m.get("win", np.nan), t_stat=tstat(allt.net_r) if len(allt) else np.nan,
                   coins_positive=pos, coins_total=len(coin_means))
        rows.append(row)
        all_trades.append(allt)
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    chosen = rows[0]
    ok = (chosen["mean_net_r"] > 0 and chosen["pf"] > 1.0 and chosen["coins_positive"] >= 5 and chosen["t_stat"] >= 2.0)
    print("VERDICT (E-ind-003 rule):", "ACCEPTED as candidate for out-of-time confirmation" if ok else "REJECTED")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).assign(verdict=("accepted" if ok else "rejected")).to_csv(
        os.path.join(out_dir, "E-ind-003_test_summary.csv"), index=False)
    pd.concat(all_trades, ignore_index=True).to_csv(os.path.join(out_dir, "E-ind-003_test_trades.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.data, a.out)
