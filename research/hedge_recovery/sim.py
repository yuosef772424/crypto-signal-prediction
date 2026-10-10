"""
PURPOSE: Runs experiment E-hedge-001 (cards/E-hedge-001.md): hedge losing positions with an opposite leg, close hedges on the
         next opposite-direction base signal, close profitable legs against the NATR regime, cap open legs. Arms B (no overlay),
         H (overlay) and C (random entries with the overlay), on DEV / VAL / TEST splits, 1h spot OHLCV, 7 coins.
TAGS:    grid-hedge, hedge recovery, E-hedge-001, trend pullback, NATR regime exit, mark-to-market drawdown, random control
PITFALLS: Decisions at bar j use closes up to j only; every action fills at the OPEN of bar j+1. Forced close at split end is
         counted separately (reason 'eod'). Rules are the card's, not tuned: changing any constant here is a new experiment.

Usage:
    python research/hedge_recovery/sim.py --data /home/user/research/ohlc --out research/hedge_recovery/E-hedge-001_results.csv
"""
import argparse
import csv
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT"]
SPLITS = {
    "DEV": ("2019-01-01", "2021-12-31 23:00"),
    "VAL": ("2022-01-01", "2023-12-31 23:00"),
    "TEST": ("2024-01-01", "2026-09-30 23:00"),
}
COST = 0.0012          # round trip per leg (card: 0.12% RT, top-50)
TIME_EXIT = 24         # bars (card: base time exit)
K_HEDGE = 1.0          # × NATR_14 loss trigger (card H1)
CAP = 6                # max open legs per coin (card H5)
REGIME_LAG = 14        # bars for the NATR-direction sign (card: close_t vs close_{t-14})
SEEDS = (0, 1, 2, 3, 4)


@dataclass
class Leg:
    d: int                      # +1 long, -1 short
    entry_px: float
    entry_bar: int
    kind: str                   # 'base' or 'hedge'
    hedged: bool = False
    open: bool = True
    exit_px: float = np.nan
    exit_bar: int = -1
    reason: str = ""
    parent: Optional["Leg"] = None

    def unreal(self, px):
        return self.d * (px / self.entry_px - 1.0)


def load(path, start, end):
    """Hourly OHLCV with the indicators used by the card, sliced to [start, end] after warm-up."""
    df = pd.read_parquet(path).sort_index()
    c, h, l = df.close, df.high, df.low
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    out = pd.DataFrame({
        "open": df.open, "high": h, "low": l, "close": c,
        "ema20": c.ewm(span=20, adjust=False).mean(),
        "ema50": c.ewm(span=50, adjust=False).mean(),
        "natr": atr / c * 100.0,                                  # percent
        "regime": np.sign(c - c.shift(REGIME_LAG)),               # NATR-direction sign (card)
    })
    out["sig_long"] = ((out.ema20 > out.ema50) & (out.low <= out.ema20) & (out.close > out.ema20)).astype(int)
    out["sig_short"] = ((out.ema20 < out.ema50) & (out.high >= out.ema20) & (out.close < out.ema20)).astype(int)
    # the card's warm-up is 200 bars; sliced after indicators so the first split bar has full history
    out = out.iloc[200:]
    return out.loc[start:end]


def simulate(O, C, natr, regime, sig_l, sig_s, mode):
    """mode 'B': base entries + base time exit. mode 'H': card's overlay (H1–H5) on the same base entries."""
    n = len(C)
    cap = CAP if mode == "H" else 10 ** 9
    legs: List[Leg] = []
    open_legs: List[Leg] = []
    pending = []                                  # ('close', leg, reason) | ('open', d, parent, kind)
    mtm = np.zeros(n)
    realized = 0.0
    max_open = 0

    def close(leg, px, bar, reason):
        nonlocal realized
        leg.open, leg.exit_px, leg.exit_bar, leg.reason = False, px, bar, reason
        realized += leg.d * (px / leg.entry_px - 1.0) - COST
        open_legs.remove(leg)

    for j in range(n):
        # 1) fills at this bar's OPEN for orders decided at the previous bar's close: closes first, then opens
        for act in sorted(pending, key=lambda a: 0 if a[0] == "close" else 1):
            if act[0] == "close":
                leg = act[1]
                if leg.open:
                    close(leg, O[j], j, act[2])
            else:
                _, d, parent, kind = act
                if len(open_legs) < cap:
                    leg = Leg(d=d, entry_px=O[j], entry_bar=j, kind=kind, parent=parent)
                    if parent is not None:
                        parent.hedged = True
                    legs.append(leg)
                    open_legs.append(leg)
        pending = []
        max_open = max(max_open, len(open_legs))

        if j == n - 1:                            # forced close at split end, reported separately
            for leg in list(open_legs):
                close(leg, C[j], j, "eod")
            mtm[j] = realized
            break

        # 2) decisions at this bar's CLOSE, executed at the next bar's open
        if mode == "H":
            queued_hedge = set()
            # H4: profitable legs running against the NATR regime are closed, hedge or not
            for leg in open_legs:
                if leg.unreal(C[j]) > 0 and regime[j] != 0 and leg.d != regime[j]:
                    pending.append(("close", leg, "H4_natr_exit"))
            # H1: unhedged base leg whose loss ≥ K × NATR_14 gets an opposite hedge leg
            for leg in open_legs:
                if leg.kind == "base" and not leg.hedged and leg.unreal(C[j]) * 100.0 <= -K_HEDGE * natr[j]:
                    pending.append(("open", -leg.d, leg, "hedge"))
                    queued_hedge.add(id(leg))
            # H3: a new base signal closes the open hedges of the opposite direction
            if sig_l[j]:
                for leg in open_legs:
                    if leg.kind == "hedge" and leg.d == -1:
                        pending.append(("close", leg, "H3_opposite_signal"))
            if sig_s[j]:
                for leg in open_legs:
                    if leg.kind == "hedge" and leg.d == 1:
                        pending.append(("close", leg, "H3_opposite_signal"))
        else:
            queued_hedge = set()

        # base time exit: unhedged base legs only (H2: hedged legs are not time-exited)
        for leg in open_legs:
            if leg.kind == "base" and not leg.hedged and id(leg) not in queued_hedge and j - leg.entry_bar >= TIME_EXIT:
                pending.append(("close", leg, "time_exit"))

        # base entry at this bar's signal (one leg per bar)
        if sig_l[j]:
            pending.append(("open", 1, None, "base"))
        elif sig_s[j]:
            pending.append(("open", -1, None, "base"))

        # mark-to-market equity at this bar's close (realised + unrealised, unit = one leg's notional)
        mtm[j] = realized + sum(leg.unreal(C[j]) for leg in open_legs)

    return legs, mtm, max_open


def summarise(legs, mtm, max_open):
    closed = [l for l in legs if not l.open]
    net = np.array([l.d * (l.exit_px / l.entry_px - 1.0) - COST for l in closed]) if closed else np.array([])
    base = np.array([l.d * (l.exit_px / l.entry_px - 1.0) - COST for l in closed if l.kind == "base"])
    hed = np.array([l.d * (l.exit_px / l.entry_px - 1.0) - COST for l in closed if l.kind == "hedge"])
    peak = np.maximum.accumulate(np.concatenate([[0.0], mtm]))[1:]
    dd = float(np.max(peak - mtm)) if len(mtm) else 0.0
    reasons = pd.Series([l.reason for l in closed]).value_counts().to_dict() if closed else {}
    return {
        "legs": len(closed),
        "base_legs": int(len(base)),
        "hedge_legs": int(len(hed)),
        "mean_net_bp": float(net.mean() * 1e4) if len(net) else np.nan,
        "mean_net_base_bp": float(base.mean() * 1e4) if len(base) else np.nan,
        "mean_net_hedge_bp": float(hed.mean() * 1e4) if len(hed) else np.nan,
        "win_rate": float((net > 0).mean()) if len(net) else np.nan,
        "realised_units": float(net.sum()) if len(net) else 0.0,
        "mtm_end_units": float(mtm[-1]) if len(mtm) else 0.0,
        "mtm_max_dd_units": dd,
        "max_open_legs": int(max_open),
        "reasons": ";".join(f"{k}={v}" for k, v in sorted(reasons.items())),
    }


def arm_inputs(df, arm, seed=0):
    if arm in ("B", "H"):
        return df.sig_long.values.astype(bool), df.sig_short.values.astype(bool)
    # arm C: random entries at the base arm's entry rate, random direction
    rng = np.random.default_rng(seed)
    rate = float((df.sig_long | df.sig_short).mean())
    enter = rng.random(len(df)) < rate
    up = rng.random(len(df)) < 0.5
    return enter & up, enter & ~up


def run(data_dir, out_csv):
    rows = []
    for split, (start, end) in SPLITS.items():
        for coin in COINS:
            path = os.path.join(data_dir, f"ohlc_{coin}_1h.parquet")
            df = load(path, start, end)
            O, C = df.open.values, df.close.values
            natr, regime = df.natr.values, df.regime.values
            for arm in ("B", "H", "C"):
                seeds = SEEDS if arm == "C" else (0,)
                for seed in seeds:
                    sl, ss = arm_inputs(df, arm, seed)
                    mode = "H" if arm in ("H", "C") else "B"
                    legs, mtm, max_open = simulate(O, C, natr, regime, sl, ss, mode)
                    s = summarise(legs, mtm, max_open)
                    s.update(split=split, coin=coin, arm=arm, seed=seed, bars=len(df))
                    rows.append(s)
                    print(f"{split} {coin:8s} {arm} s{seed} legs={s['legs']:5d} mean_bp={s['mean_net_bp']:8.2f} "
                          f"dd={s['mtm_max_dd_units']:6.2f} maxopen={s['max_open_legs']}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out_csv, index=False)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.data, a.out)
