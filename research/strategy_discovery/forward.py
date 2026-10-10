"""
PURPOSE: Forward test of the frozen hypothesis in card E-fwd-001: daily Donchian 20 breakout, no gate, long and short, stop 2 x ATR14,
         target 3 x ATR14, 30-day time exit, cost 0.12% round trip, on the ten dataset coins. Only trades whose entry is on or after
         2026-10-09 (the first bar after the last bar of the historical data) are counted. Writes the closed forward trades and a
         summary with the decision metric of the card. The rules are frozen; this script must not be edited to change them.
TAGS:    forward test, E-fwd-001, frozen hypothesis, out of sample, decision metric, z-score, end condition, Donchian
PITFALLS: Open trades at the data end are reported as open and are not counted. The fresh 1h data must be downloaded into a NEW
         directory each run (the fetcher skips files that already exist). Indicators use the full history before the forward window.

Usage (repo root):
    python research/strategy_discovery/forward.py --data /home/user/research/forward_1h --out research/strategy_discovery/results
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
from grid import signal, apply_gate, extra_columns  # noqa: E402

UNIVERSE = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "BCHUSDT", "TRXUSDT", "ZECUSDT"]
FORWARD_START = pd.Timestamp("2026-10-09", tz="UTC")
END_TRADES = 300
END_DATE = pd.Timestamp("2028-08-10", tz="UTC")


def run(data_dir, out_dir):
    trades = []
    last_bar = None
    for c in UNIVERSE:
        path = os.path.join(data_dir, f"ohlc_{c}_1h.parquet")
        if not os.path.exists(path):
            print("missing", path)
            continue
        d = extra_columns(E.indicators(E.load(path, "1d")))
        last_bar = d.index.max() if last_bar is None else max(last_bar, d.index.max())
        lg, sh = apply_gate(d, *signal(d, "donchian20"), "none")
        t = E.simulate(d, lg, sh, "1d")
        if len(t):
            t.insert(0, "coin", c)
            trades.append(t)
    allt = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    fwd = allt[allt.entry_time >= FORWARD_START] if len(allt) else allt
    closed = fwd[fwd.reason != "eod"] if len(fwd) else fwd
    open_now = fwd[fwd.reason == "eod"] if len(fwd) else fwd
    n = len(closed)
    mean = float(closed.net_r.mean()) if n else float("nan")
    sd = float(closed.net_r.std(ddof=1)) if n > 1 else float("nan")
    # clustered standard error: trades are summed within entry month (overlap across coins and time)
    z_naive = mean / (sd / math.sqrt(n)) if n > 1 and sd > 0 else float("nan")
    if n > 1:
        dev = closed.net_r - mean
        g = dev.groupby(closed.entry_time.dt.to_period("M")).sum()
        se_cl = math.sqrt(float((g ** 2).sum())) / n
        z = mean / se_cl if se_cl > 0 else float("nan")
    else:
        se_cl, z = float("nan"), float("nan")
    ended = (n >= END_TRADES) or (last_bar is not None and last_bar >= END_DATE)
    if not ended:
        verdict = "running (end condition not reached)"
    elif n < END_TRADES:
        verdict = "insufficient sample at end date"
    elif mean > 0 and z >= 2:
        verdict = "consistent with an edge (pre-registered reading)"
    elif z <= -2:
        verdict = "rejected (pre-registered reading)"
    else:
        verdict = "inconclusive (pre-registered band)"
    summary = pd.DataFrame([dict(data_last_bar=str(last_bar), forward_start=str(FORWARD_START.date()), closed_trades=n,
                                 open_at_data_end=len(open_now), mean_net_r=mean, sd_net_r=sd, se_month_cluster=se_cl,
                                 z_month_cluster=z, z_naive_trade_level=z_naive,
                                 coins_with_trades=int(closed.coin.nunique()) if n else 0, verdict=verdict)])
    os.makedirs(out_dir, exist_ok=True)
    summary.to_csv(os.path.join(out_dir, "E-fwd-001_summary.csv"), index=False)
    if n:
        closed.to_csv(os.path.join(out_dir, "E-fwd-001_trades.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.data, a.out)
