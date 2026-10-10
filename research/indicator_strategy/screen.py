"""
PURPOSE: Runs card E-ind-001: every trigger × gate setting × timeframe (4h, 1d) × coin on the DEV and VAL splits only, with the
         card's fixed exit, and writes one row per configuration and split plus every trade (for region analysis). TEST is not
         loaded by this script by design.
TAGS:    indicator screen, E-ind-001, DEV VAL split, accept rule, trade log, t-statistic, profit factor, per-coin consistency
PITFALLS: Indicators are computed on the full history (warm-up before each split), signals and trades are restricted to the split
         window. The accept rule is applied to the printed summary, not to any single coin.

Usage (repo root):
    python research/indicator_strategy/screen.py --data /home/user/research/ohlc --out research/indicator_strategy
"""
import argparse
import os

import numpy as np
import pandas as pd

import engine as E

COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT"]
SPLITS = {"DEV": ("2019-01-01", "2021-12-31 23:00"), "VAL": ("2022-01-01", "2023-12-31 23:00")}  # no TEST here
TRIGGERS = ["rsi_mr", "macd_cross", "bb_break", "adx_trend", "donchian", "ema_pullback"]
GATES = [(), ("adx25",)]
TIMEFRAMES = ["4h", "1d"]


def tstat(x):
    x = np.asarray(x, float)
    if len(x) < 2 or x.std(ddof=1) == 0:
        return np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))


def run(data_dir, out_dir):
    trades_all, summary = [], []
    for tf in TIMEFRAMES:
        full = {c: E.indicators(E.load(os.path.join(data_dir, f"ohlc_{c}_1h.parquet"), tf)) for c in COINS}
        for trig in TRIGGERS:
            for gates in GATES:
                gname = "+".join(gates) if gates else "none"
                for split, (start, end) in SPLITS.items():
                    per_coin = []
                    for c in COINS:
                        d = full[c].loc[start:end]
                        lg, sh = E.build_signal(full[c], trig, gates)
                        lg, sh = lg[full[c].index.get_indexer(d.index)], sh[full[c].index.get_indexer(d.index)]
                        t = E.simulate(d, lg, sh, tf)
                        if len(t):
                            t.insert(0, "coin", c)
                            t.insert(0, "gate", gname)
                            t.insert(0, "trigger", trig)
                            t.insert(0, "tf", tf)
                            t.insert(0, "split", split)
                            trades_all.append(t)
                        per_coin.append((c, t))
                    allt = pd.concat([t for _, t in per_coin if len(t)]) if any(len(t) for _, t in per_coin) else pd.DataFrame()
                    m = E.metrics(allt)
                    coin_means = {c: (t.net_r.mean() if len(t) else np.nan) for c, t in per_coin}
                    summary.append(dict(tf=tf, trigger=trig, gate=gname, split=split, n=m.get("n", 0),
                                        mean_net_r=m.get("mean_net_r", np.nan), mean_gross_r=m.get("mean_gross_r", np.nan),
                                        win=m.get("win", np.nan), pf=m.get("pf", np.nan),
                                        t_stat=tstat(allt.net_r) if len(allt) else np.nan,
                                        coins_positive=int(sum(1 for v in coin_means.values() if v == v and v > 0)),
                                        coins_total=int(sum(1 for v in coin_means.values() if v == v))))
                    print(f"{tf} {trig:12s} gate={gname:8s} {split} n={m.get('n',0):5d} net_R={m.get('mean_net_r',np.nan):+.3f} "
                          f"pf={m.get('pf',np.nan):.2f} coins+={summary[-1]['coins_positive']}/{summary[-1]['coins_total']}",
                          flush=True)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(summary).to_csv(os.path.join(out_dir, "E-ind-001_summary.csv"), index=False)
    pd.concat(trades_all, ignore_index=True).to_csv(os.path.join(out_dir, "E-ind-001_trades.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.data, a.out)
