"""
PURPOSE: Runs card E-disc-005: three new signal families (Supertrend flip, Keltner squeeze breakout, volume-confirmed Donchian)
         × 3 gates × 2 timeframes on the seven original coins, DEV and VAL. Benjamini-Hochberg is applied over all 60
         configurations of the campaign (the 42 from E-disc-001 and these 18), so that adding families does not hide the
         multiple-testing cost. Writes the summary and the survivor list.
TAGS:    supertrend, keltner squeeze, volume confirmation, E-disc-005, multiple testing, BH over the campaign, DEV VAL
PITFALLS: The 42 earlier rows are read from results/E-disc-001_summary.csv, not recomputed. Supertrend is implemented with the
         standard final-band recursion on ATR(10) × 3. Volume is taken from the 1h bars resampled to 4h or 1d (sum of volume).

Usage (repo root):
    python research/strategy_discovery/grid2.py --data /home/user/research/ohlc --out research/strategy_discovery/results
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
from grid import COINS, SPLITS, TIMEFRAMES, GATES, apply_gate, tstat, one_sided_p, bh_qvalues  # noqa: E402

FAMILIES2 = ["supertrend", "keltner_squeeze", "volume_donchian"]


def supertrend(d, n=10, mult=3.0):
    """Returns (trend, flip_up, flip_dn). trend = +1 up / -1 down; flips are the bars where the trend changes."""
    hl2 = (d.high + d.low) / 2
    atr = d.atr_st
    upper = (hl2 + mult * atr).values
    lower = (hl2 - mult * atr).values
    close = d.close.values
    fu = upper.copy()
    fl = lower.copy()
    trend = np.ones(len(d))
    for i in range(1, len(d)):
        fu[i] = upper[i] if (upper[i] < fu[i - 1] or close[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lower[i] if (lower[i] > fl[i - 1] or close[i - 1] < fl[i - 1]) else fl[i - 1]
        if trend[i - 1] == 1:
            trend[i] = -1 if close[i] < fl[i] else 1
        else:
            trend[i] = 1 if close[i] > fu[i] else -1
    t = pd.Series(trend, index=d.index)
    up = (t == 1) & (t.shift(1) == -1)
    dn = (t == -1) & (t.shift(1) == 1)
    return t, up.values, dn.values


def extra(d):
    d = d.copy()
    d["atr_st"] = (pd.concat([d.high - d.low, (d.high - d.close.shift()).abs(), (d.low - d.close.shift()).abs()],
                             axis=1).max(axis=1)).ewm(alpha=1 / 10, adjust=False).mean()
    kc_mid = d.close.ewm(span=20, adjust=False).mean()
    d["kc_up"] = kc_mid + 1.5 * d.atr
    d["kc_lo"] = kc_mid - 1.5 * d.atr
    d["squeeze_prev"] = ((d.bb_up < d.kc_up) & (d.bb_lo > d.kc_lo)).shift(1).fillna(False)
    d["don20_hi"] = d.high.rolling(20).max().shift(1)
    d["don20_lo"] = d.low.rolling(20).min().shift(1)
    d["vol_med20"] = d.volume.rolling(20).median().shift(1)
    return d


def signal2(d, fam):
    if fam == "supertrend":
        _, up, dn = supertrend(d)
        return up, dn
    if fam == "keltner_squeeze":
        recent_squeeze = d.squeeze_prev.values
        return ((d.close > d.kc_up).values & recent_squeeze), ((d.close < d.kc_lo).values & recent_squeeze)
    if fam == "volume_donchian":
        vol_ok = (d.volume > 1.5 * d.vol_med20).values
        return ((d.close > d.don20_hi).values & vol_ok), ((d.close < d.don20_lo).values & vol_ok)
    raise ValueError(fam)


def run(data_dir, out_dir):
    full = {}
    for c in COINS:
        for tf in TIMEFRAMES:
            full[(c, tf)] = extra(E.indicators(E.load(os.path.join(data_dir, f"ohlc_{c}_1h.parquet"), tf)))
    rows, logs = [], []
    for tf in TIMEFRAMES:
        for fam in FAMILIES2:
            for gate in GATES:
                for split, (start, end) in SPLITS.items():
                    trades, coin_means = [], []
                    for c in COINS:
                        dfull = full[(c, tf)]
                        lg, sh = signal2(dfull, fam)
                        lg, sh = apply_gate(dfull, lg, sh, gate)
                        d = dfull.loc[start:end]
                        idx = dfull.index.get_indexer(d.index)
                        t = E.simulate(d, lg[idx], sh[idx], tf)
                        if len(t):
                            t.insert(0, "coin", c)
                            t.insert(0, "config", f"{tf}|{fam}|{gate}")
                            t.insert(0, "split", split)
                            trades.append(t)
                            coin_means.append(t.net_r.mean())
                        else:
                            coin_means.append(np.nan)
                    allt = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame(columns=["net_r", "gross_r", "side"])
                    m = E.metrics(allt)
                    rows.append(dict(config=f"{tf}|{fam}|{gate}", tf=tf, family=fam, gate=gate, split=split, n=m.get("n", 0),
                                     mean_net_r=m.get("mean_net_r", np.nan), mean_gross_r=m.get("mean_gross_r", np.nan),
                                     pf=m.get("pf", np.nan), win=m.get("win", np.nan),
                                     t_stat=tstat(allt.net_r) if len(allt) else np.nan,
                                     coins_positive=int(sum(1 for v in coin_means if v == v and v > 0)),
                                     coins_total=int(sum(1 for v in coin_means if v == v))))
                    if split in ("DEV", "VAL") and trades:
                        logs.append(allt)
                    print(f"{tf} {fam:16s} {gate:14s} {split} n={rows[-1]['n']:5d} net_R={rows[-1]['mean_net_r']:+.3f} "
                          f"pf={rows[-1]['pf']:.2f} t={rows[-1]['t_stat']:+.2f}", flush=True)
    new = pd.DataFrame(rows)
    old = pd.read_csv(os.path.join(out_dir, "E-disc-001_summary.csv"))
    old = old[["config", "tf", "family", "gate", "split", "n", "mean_net_r", "mean_gross_r", "pf", "win", "t_stat",
               "coins_positive", "coins_total"]]
    allres = pd.concat([old, new], ignore_index=True)
    val = allres[allres.split == "VAL"].copy()
    val["p"] = [one_sided_p(t) for t in val.t_stat]
    val["bh_q_campaign"] = bh_qvalues(val.p.values)
    allres = allres.merge(val[["config", "bh_q_campaign"]].drop_duplicates("config"), on="config", how="left")
    os.makedirs(out_dir, exist_ok=True)
    allres.to_csv(os.path.join(out_dir, "E-disc-005_summary_campaign.csv"), index=False)
    print("written", len(allres), "rows; configurations in campaign:", allres.config.nunique())
    return allres


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.data, a.out)
