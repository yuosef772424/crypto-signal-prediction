"""
PURPOSE: Discovery screen for card E-disc-001: 7 signal families × 3 gate settings × 2 timeframes (4h, 1d) on the seven original
         coins, DEV and VAL splits only, with the card's fixed exit. Writes one row per configuration and split, the trade logs
         for DEV and VAL, and the Benjamini-Hochberg q-values on the VAL t-statistics. TEST and the unseen-asset coins are not loaded.
TAGS:    discovery grid, E-disc-001, signal families, gates, multiple testing, Benjamini-Hochberg, DEV VAL, trend, breakout,
         mean reversion, RSI2 pullback, squeeze breakout
PITFALLS: Indicators use the full history (warm-up before each split); signals and trades are restricted to the split window.
         The BH q-values are over the configurations in this grid only; a survivor still needs robustness checks and an
         unseen-asset test. Coins BCH, TRX, ZEC are reserved for E-disc-003 and are not read here.

Usage (repo root):
    python research/strategy_discovery/grid.py --data /home/user/research/ohlc --out research/strategy_discovery/results
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import trade_engine as E  # noqa: E402

COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT"]
SPLITS = {"DEV": ("2019-01-01", "2021-12-31 23:00"), "VAL": ("2022-01-01", "2023-12-31 23:00")}
TIMEFRAMES = ["4h", "1d"]
GATES = ["none", "adx25", "ema200_aligned"]
FAMILIES = ["ema_pullback", "donchian20", "donchian55", "bb_break", "adx_trend", "rsi2_pullback", "squeeze_break"]


def extra_columns(d):
    """Indicators not in the shared engine: Donchian channels of 55 bars, RSI(2), SMA(50), Bollinger bandwidth percentile."""
    d = d.copy()
    d["don55_hi"] = d.high.rolling(55).max().shift(1)
    d["don55_lo"] = d.low.rolling(55).min().shift(1)
    d["don20_hi"] = d.high.rolling(20).max().shift(1)
    d["don20_lo"] = d.low.rolling(20).min().shift(1)
    delta = d.close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 2, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 2, adjust=False).mean()
    d["rsi2"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    d["sma50"] = d.close.rolling(50).mean()
    width = (d.bb_up - d.bb_lo) / d.close
    d["bw_pct"] = width.rolling(120).rank(pct=True)
    return d


def signal(d, family):
    """Raw (long, short) arrays before gates. Crossings use bar t and t-1; all values are causal."""
    p = lambda s: s.shift(1)
    if family in ("ema_pullback", "bb_break", "adx_trend"):
        return E.build_signal(d, family, ())
    if family == "donchian20":
        return ((d.close > d.don20_hi).values, (d.close < d.don20_lo).values)
    if family == "donchian55":
        return ((d.close > d.don55_hi).values, (d.close < d.don55_lo).values)
    if family == "rsi2_pullback":
        return (((p(d.rsi2) < 10) & (d.rsi2 >= 10)).values, ((p(d.rsi2) > 90) & (d.rsi2 <= 90)).values)
    if family == "squeeze_break":
        squeezed = (p(d.bw_pct) < 0.2)
        return (((d.close > d.bb_up) & squeezed).values, ((d.close < d.bb_lo) & squeezed).values)
    raise ValueError(family)


def apply_gate(d, long, short, gate):
    long, short = np.asarray(long, bool).copy(), np.asarray(short, bool).copy()
    if gate == "none":
        return long, short
    if gate == "adx25":
        m = (d.adx >= 25).values
        return long & m, short & m
    if gate == "ema200_aligned":
        return long & (d.close > d.ema200).values, short & (d.close < d.ema200).values
    raise ValueError(gate)


def tstat(x):
    x = np.asarray(x, float)
    if len(x) < 2 or x.std(ddof=1) == 0:
        return np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))


def one_sided_p(t):
    """Upper-tail p-value from the normal approximation (trade-level, optimistic)."""
    if not np.isfinite(t):
        return 1.0
    return 0.5 * math.erfc(t / math.sqrt(2))


def bh_qvalues(p):
    """Benjamini-Hochberg q-values for a list of p-values."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(q, 0, 1)
    return out


def run(data_dir, out_dir, trade_dir):
    full = {}
    for c in COINS:
        for tf in TIMEFRAMES:
            d = E.indicators(E.load(os.path.join(data_dir, f"ohlc_{c}_1h.parquet"), tf))
            full[(c, tf)] = extra_columns(d)
    rows, trade_logs = [], []
    for tf in TIMEFRAMES:
        for fam in FAMILIES:
            for gate in GATES:
                for split, (start, end) in SPLITS.items():
                    trades, coin_means = [], []
                    for c in COINS:
                        d_full = full[(c, tf)]
                        lg, sh = signal(d_full, fam)
                        lg, sh = apply_gate(d_full, lg, sh, gate)
                        d = d_full.loc[start:end]
                        idx = d_full.index.get_indexer(d.index)
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
                    rows.append(dict(config=f"{tf}|{fam}|{gate}", tf=tf, family=fam, gate=gate, split=split,
                                     n=m.get("n", 0), mean_net_r=m.get("mean_net_r", np.nan),
                                     mean_gross_r=m.get("mean_gross_r", np.nan), pf=m.get("pf", np.nan),
                                     win=m.get("win", np.nan), t_stat=tstat(allt.net_r) if len(allt) else np.nan,
                                     coins_positive=int(sum(1 for v in coin_means if v == v and v > 0)),
                                     coins_total=int(sum(1 for v in coin_means if v == v))))
                    if split in ("DEV", "VAL") and trades:
                        trade_logs.append(allt)
                    print(f"{tf} {fam:14s} {gate:14s} {split} n={rows[-1]['n']:5d} net_R={rows[-1]['mean_net_r']:+.3f} "
                          f"pf={rows[-1]['pf']:.2f} t={rows[-1]['t_stat']:+.2f} coins+={rows[-1]['coins_positive']}/{rows[-1]['coins_total']}",
                          flush=True)
    res = pd.DataFrame(rows)
    val = res[res.split == "VAL"].copy()
    val["p_one_sided"] = [one_sided_p(t) for t in val.t_stat]
    val["bh_q_val"] = bh_qvalues(val.p_one_sided.values)
    res = res.merge(val[["config", "bh_q_val"]], on="config", how="left")
    os.makedirs(out_dir, exist_ok=True)
    res.to_csv(os.path.join(out_dir, "E-disc-001_summary.csv"), index=False)
    if trade_logs:
        os.makedirs(trade_dir, exist_ok=True)
        pd.concat(trade_logs, ignore_index=True).to_csv(os.path.join(trade_dir, "E-disc-001_trades_dev_val.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trades", default="/home/user/research/strategy_discovery",
                    help="trade logs are kept outside the repo (CI limit 1 MB per tracked file)")
    a = ap.parse_args()
    run(a.data, a.out, a.trades)
