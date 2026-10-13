"""
PURPOSE: Runs card E-disc-006 for one configuration: robustness on DEV and VAL (cost x2, one-bar delay, a 36-setting exit grid,
         year check), the unseen-asset test on BCH, TRX, ZEC over 2024 to 2026-09, and the fresh-bars window 2026-10. Dispatches
         signals to grid.py (E-disc-001 families) or grid2.py (E-disc-005 families).
TAGS:    survivor checks, robustness, unseen assets, fresh bars, E-disc-006, volume Donchian, sample size guard
PITFALLS: VAL was used for the selection of this configuration by the campaign BH rule; the VAL numbers here are stability checks.
         The fresh window is short (about eight daily bars); a verdict is given only with at least ten trades.

Usage (repo root):
    python research/strategy_discovery/survivor_checks.py --data /home/user/research/ohlc_full --family volume_donchian --gate none
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
import grid  # noqa: E402
import grid2  # noqa: E402
from grid import COINS, SPLITS, apply_gate  # noqa: E402

ALL10 = COINS + ["BCHUSDT", "TRXUSDT", "ZECUSDT"]
UNSEEN = ["BCHUSDT", "TRXUSDT", "ZECUSDT"]
STOPS = [1.5, 2.0, 2.5, 3.0]
TARGETS = [2.5, 3.0, 3.5]
TIME_EXITS = [20, 30, 40]
FRESH_START = pd.Timestamp("2026-10-01", tz="UTC")
UNSEEN_WIN = (pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2026-09-30 23:00", tz="UTC"))
MIN_TRADES = 10


def sig(d, fam):
    if fam in grid2.FAMILIES2:
        return grid2.signal2(d, fam)
    return grid.signal(d, fam)


def build(full, coin, fam, gate):
    df = full[coin]
    lg, sh = apply_gate(df, *sig(df, fam), gate)
    return df, lg, sh


def pooled(full, coins, fam, gate, start, end, **kw):
    trades = []
    for c in coins:
        df, lg, sh = build(full, c, fam, gate)
        d = df.loc[start:end]
        idx = df.index.get_indexer(d.index)
        t = E.simulate(d, lg[idx], sh[idx], "1d", **kw)
        if len(t):
            t.insert(0, "coin", c)
            trades.append(t)
    return pd.concat(trades, ignore_index=True) if trades else pd.DataFrame(columns=["coin", "net_r", "entry_time"])


def run(data_dir, out_dir, fam, gate):
    full = {c: grid2.extra(E.indicators(E.load(os.path.join(data_dir, f"ohlc_{c}_1h.parquet"), "1d"))) for c in ALL10}
    # signals need the extra columns of both families: grid.extra_columns adds Donchian 55, RSI(2), squeeze
    for c in ALL10:
        full[c] = grid.extra_columns(full[c])
    res = {}
    for split, (s, e) in SPLITS.items():
        base = pooled(full, COINS, fam, gate, s, e)
        res[(split, "base")] = base.net_r.mean()
        res[(split, "cost2")] = pooled(full, COINS, fam, gate, s, e, cost=2 * E.COST).net_r.mean()
        # one-bar delay: shift the signal arrays by one bar
        delayed_vals = []
        for c in COINS:
            df, lg, sh = build(full, c, fam, gate)
            lg2, sh2 = np.r_[False, lg[:-1]], np.r_[False, sh[:-1]]
            d = df.loc[s:e]
            idx = df.index.get_indexer(d.index)
            t = E.simulate(d, lg2[idx], sh2[idx], "1d")
            delayed_vals.extend(t.net_r.tolist())
        res[(split, "delay1")] = float(np.mean(delayed_vals)) if delayed_vals else np.nan
        if split == "VAL":
            grid_vals = []
            for st in STOPS:
                for tg in TARGETS:
                    for tx in TIME_EXITS:
                        grid_vals.append(pooled(full, COINS, fam, gate, s, e, atr_stop=st, atr_tgt=tg,
                                                max_bars=tx).net_r.mean())
            res["grid_pos_share"] = float(np.mean([v > 0 for v in grid_vals]))
            yrs = {}
            for yr in (2022, 2023):
                t = base[base.entry_time.dt.year == yr]
                yrs[yr] = float(t.net_r.mean()) if len(t) else np.nan
            res["years"] = yrs
    ok_R = (res[("DEV", "cost2")] > 0 and res[("VAL", "cost2")] > 0 and res[("VAL", "delay1")] > 0
            and res["grid_pos_share"] >= 0.70 and all(v > 0 for v in res["years"].values()))
    # unseen assets
    ua = pooled(full, UNSEEN, fam, gate, *UNSEEN_WIN)
    per = {c: ua[ua.coin == c].net_r.mean() for c in UNSEEN if (ua.coin == c).any()}
    pos = sum(v > 0 for v in per.values())
    pw = ua.net_r[ua.net_r > 0].sum()
    pl = -ua.net_r[ua.net_r <= 0].sum()
    pf = pw / pl if pl > 0 else np.inf
    ok_U = bool(len(ua) and ua.net_r.mean() > 0 and pf > 1.0 and pos >= 2)
    # fresh bars
    last = max(full[c].index.max() for c in ALL10)
    fr = pooled(full, ALL10, fam, gate, FRESH_START, last)
    fresh_verdict = "insufficient sample" if len(fr) < MIN_TRADES else ("positive" if fr.net_r.mean() > 0 else "negative")
    rows = [
        dict(test="R_robustness", metric="DEV cost2 / VAL cost2 / VAL delay1", value=f"{res[('DEV','cost2')]:+.3f} / {res[('VAL','cost2')]:+.3f} / {res[('VAL','delay1')]:+.3f}", verdict="PASS" if ok_R else "FAIL"),
        dict(test="R_robustness", metric="VAL grid positive share", value=f"{res['grid_pos_share']:.2f}", verdict=""),
        dict(test="R_robustness", metric="VAL 2022 / 2023", value=f"{res['years'][2022]:+.3f} / {res['years'][2023]:+.3f}", verdict=""),
        dict(test="U_unseen_assets", metric=f"n={len(ua)} mean={ua.net_r.mean():+.3f} pf={pf:.2f} coins+={pos}/3",
             value="; ".join(f"{c}={v:+.3f}" for c, v in per.items()), verdict="PASS" if ok_U else "FAIL"),
        dict(test="F_fresh_bars", metric=f"n={len(fr)} mean={(fr.net_r.mean() if len(fr) else float('nan')):+.3f} window_end={last}",
             value="", verdict=fresh_verdict),
    ]
    out = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    name = f"E-disc-006_{fam}_{gate}.csv"
    out.to_csv(os.path.join(out_dir, name), index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="research/strategy_discovery/results")
    ap.add_argument("--family", required=True)
    ap.add_argument("--gate", required=True)
    a = ap.parse_args()
    run(a.data, a.out, a.family, a.gate)
