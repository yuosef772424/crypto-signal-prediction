"""
PURPOSE: Runs card E-disc-002: robustness of the three E-disc-001 survivors on the DEV and VAL splits of the seven original coins.
         Variants: base, cost doubled, entry delayed one bar, and a 36-setting exit grid (stop, target, time exit). Writes a
         summary CSV with the pooled numbers and the year check, and the rule verdict per survivor.
TAGS:    robustness, E-disc-002, cost stress, entry delay, exit grid, plateau, year stability, survivor check
PITFALLS: VAL was used to select the survivors (E-disc-001), so VAL here is a stability check only. The delay variant shifts the
         signal arrays by one extra bar. TEST, BCH, TRX, ZEC and the 2026-10 bars are not loaded.

Usage (repo root):
    python research/strategy_discovery/robustness.py --data /home/user/research/ohlc --out research/strategy_discovery/results
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trade_engine as E  # noqa: E402
from grid import COINS, SPLITS, extra_columns, signal, apply_gate  # noqa: E402

SURVIVORS = {
    "S1": ("donchian20", "adx25"),
    "S2": ("bb_break", "adx25"),
    "S3": ("donchian20", "none"),
}
STOPS = [1.5, 2.0, 2.5, 3.0]
TARGETS = [2.5, 3.0, 3.5]
TIME_EXITS = [20, 30, 40]


def pooled(split_data, **kw):
    """Pooled trades over coins for one split and one set of exit settings. Returns (mean net R, trades DataFrame)."""
    trades = []
    for c, (d, lg, sh) in split_data.items():
        t = E.simulate(d, lg, sh, "1d", **kw)
        if len(t):
            t.insert(0, "coin", c)
            trades.append(t)
    allt = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame(columns=["net_r", "entry_time"])
    return (float(allt.net_r.mean()) if len(allt) else np.nan), allt


def run(data_dir, out_dir):
    full = {c: extra_columns(E.indicators(E.load(os.path.join(data_dir, f"ohlc_{c}_1h.parquet"), "1d"))) for c in COINS}
    rows, verdicts = [], []
    for name, (fam, gate) in SURVIVORS.items():
        prepared = {}
        for split, (start, end) in SPLITS.items():
            sd = {}
            for c in COINS:
                df = full[c]
                lg, sh = apply_gate(df, *signal(df, fam), gate)
                d = df.loc[start:end]
                idx = df.index.get_indexer(d.index)
                lg, sh = lg[idx], sh[idx]
                sd[c] = (d, lg, sh)
            prepared[split] = sd
        results = {}
        for split in SPLITS:
            sd = prepared[split]
            base_r, base_t = pooled(sd)
            cost2_r, _ = pooled(sd, cost=2 * E.COST)
            delayed = {c: (d, np.r_[False, lg[:-1]], np.r_[False, sh[:-1]]) for c, (d, lg, sh) in sd.items()}
            delay_r, _ = pooled(delayed)
            results[(split, "base")] = base_r
            results[(split, "cost2")] = cost2_r
            results[(split, "delay1")] = delay_r
            grid_vals = []
            for s_ in STOPS:
                for t_ in TARGETS:
                    for m_ in TIME_EXITS:
                        r, _ = pooled(sd, atr_stop=s_, atr_tgt=t_, max_bars=m_)
                        grid_vals.append(r)
                        rows.append(dict(survivor=name, config=f"1d|{fam}|{gate}", split=split, variant="grid",
                                         stop=s_, target=t_, time_exit=m_, mean_net_r=r))
            results[(split, "grid_pos_share")] = float(np.mean([v > 0 for v in grid_vals]))
            for variant in ("base", "cost2", "delay1"):
                rows.append(dict(survivor=name, config=f"1d|{fam}|{gate}", split=split, variant=variant,
                                 stop=2.0, target=3.0, time_exit=30, mean_net_r=results[(split, variant)]))
            # year check on the base settings
            if split == "VAL":
                years = {}
                for yr in (2022, 2023):
                    sub = {c: (d, lg, sh) for c, (d, lg, sh) in sd.items()}
                    _, t = pooled(sub)
                    years[yr] = float(t[t.entry_time.dt.year == yr].net_r.mean()) if len(t) else np.nan
                results["years"] = years
        ok_cost = results[("DEV", "cost2")] > 0 and results[("VAL", "cost2")] > 0
        ok_delay = results[("VAL", "delay1")] > 0
        ok_grid = results[("VAL", "grid_pos_share")] >= 0.70
        ok_year = all(v > 0 for v in results["years"].values())
        verdict = ok_cost and ok_delay and ok_grid and ok_year
        verdicts.append(dict(survivor=name, config=f"1d|{fam}|{gate}", dev_base=results[("DEV", "base")],
                             val_base=results[("VAL", "base")], dev_cost2=results[("DEV", "cost2")],
                             val_cost2=results[("VAL", "cost2")], val_delay1=results[("VAL", "delay1")],
                             val_grid_pos_share=results[("VAL", "grid_pos_share")], val_2022=results["years"][2022],
                             val_2023=results["years"][2023], ROBUST=bool(verdict)))
        print(name, fam, gate, "ROBUST" if verdict else "NOT ROBUST", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "E-disc-002_grid.csv"), index=False)
    pd.DataFrame(verdicts).to_csv(os.path.join(out_dir, "E-disc-002_verdicts.csv"), index=False)
    print(pd.DataFrame(verdicts).round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.data, a.out)
