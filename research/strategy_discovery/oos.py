"""
PURPOSE: Runs card E-disc-003: the three robust E-disc-001 survivors on (A) unseen assets BCH, TRX, ZEC over 2024-01-01 to
         2026-09-30 and (B) fresh bars 2026-10-01 to the last available bar, for all ten coins. Indicators use the full history
         before each window (warm-up); signals and trades are restricted to the window.
TAGS:    out-of-sample, unseen assets, fresh bars, E-disc-003, survivor test, accept rule, sample size guard
PITFALLS: The fresh window holds only about eight daily bars per coin; the script reports the trade count and marks a verdict as
         "insufficient sample" below 10 trades. The unseen-asset period was used before for the other seven coins (consumed), so
         rule A is an asset holdout inside a consumed period.

Usage (repo root):
    python research/strategy_discovery/oos.py --data /home/user/research/ohlc_full --out research/strategy_discovery/results
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
from grid import extra_columns, signal, apply_gate  # noqa: E402

SURVIVORS = {"S1": ("donchian20", "adx25"), "S2": ("bb_break", "adx25"), "S3": ("donchian20", "none")}
UNSEEN = ["BCHUSDT", "TRXUSDT", "ZECUSDT"]
ALL10 = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "BCHUSDT", "TRXUSDT", "ZECUSDT"]
UNSEEN_WINDOW = ("2024-01-01", "2026-09-30 23:00")
FRESH_START = "2026-10-01"
MIN_TRADES = 10


def _utc(x):
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t


def window_trades(full, coins, fam, gate, start, end):
    start, end = _utc(start), _utc(end)
    out = []
    for c in coins:
        df = full[c]
        lg, sh = apply_gate(df, *signal(df, fam), gate)
        d = df.loc[start:end]
        idx = df.index.get_indexer(d.index)
        t = E.simulate(d, lg[idx], sh[idx], "1d")
        if len(t):
            t.insert(0, "coin", c)
            out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["coin", "net_r", "entry_time"])


def run(data_dir, out_dir):
    full = {c: extra_columns(E.indicators(E.load(os.path.join(data_dir, f"ohlc_{c}_1h.parquet"), "1d"))) for c in ALL10}
    last = max(full[c].index.max() for c in ALL10)
    rows = []
    for name, (fam, gate) in SURVIVORS.items():
        a = window_trades(full, UNSEEN, fam, gate, *UNSEEN_WINDOW)
        per_coin = {c: (a[a.coin == c].net_r.mean() if (a.coin == c).any() else np.nan) for c in UNSEEN}
        pos = sum(v > 0 for v in per_coin.values() if v == v)
        pf_w = a.net_r[a.net_r > 0].sum()
        pf_l = -a.net_r[a.net_r <= 0].sum()
        pf = pf_w / pf_l if pf_l > 0 else np.inf
        passA = bool(len(a) and a.net_r.mean() > 0 and pf > 1.0 and pos >= 2)
        rows.append(dict(survivor=name, config=f"1d|{fam}|{gate}", test="A_unseen_assets_2024_2026_09", n=len(a),
                         mean_net_r=float(a.net_r.mean()) if len(a) else np.nan, pf=float(pf), coins_positive=int(pos),
                         coins_total=len(UNSEEN), per_coin="; ".join(f"{c}={v:+.3f}" for c, v in per_coin.items() if v == v),
                         verdict="PASS" if passA else "FAIL"))
        b = window_trades(full, ALL10, fam, gate, FRESH_START, last)
        if len(b) < MIN_TRADES:
            verdictB = "insufficient sample"
        else:
            verdictB = "positive" if b.net_r.mean() > 0 else "negative"
        rows.append(dict(survivor=name, config=f"1d|{fam}|{gate}", test="B_fresh_2026_10", n=len(b),
                         mean_net_r=float(b.net_r.mean()) if len(b) else np.nan, pf=np.nan, coins_positive=np.nan,
                         coins_total=len(ALL10), per_coin=f"window end {last}", verdict=verdictB))
        print(name, fam, gate, "A:", "PASS" if passA else "FAIL", f"n={len(a)} R={a.net_r.mean():+.3f}" if len(a) else "A: n=0",
              "| B:", verdictB, f"n={len(b)}", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, "E-disc-003_results.csv"), index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.data, a.out)
