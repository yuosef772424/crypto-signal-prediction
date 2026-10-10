"""
PURPOSE: Runs experiment E-hedge-002 (cards/E-hedge-002.md): reversal hedge. A new leg opened while opposite legs are open
         flags them as conflict (losers ignored, no stop, no time exit) and is itself a conflict leg; conflict legs close on a
         profit target of 1 × NATR_14; non-conflict legs time-exit after 24 bars; longs are closed while they outnumber shorts
         and are profitable; 6-leg cap per coin; each leg is 0.5% of capital. Arms B (base only), E2 (rules), C (random + rules).
TAGS:    grid-hedge, reversal hedge, E-hedge-002, conflict leg, count balancing, profit target, 0.5% sizing, random control
PITFALLS: Uses the base entry and loading code of sim.py (same study). Decisions use closes up to bar j only; fills at the OPEN of
         bar j+1. Drawdown and equity are in % of capital (leg size 0.5% × leg return). Count balancing is applied to longs only,
         as the owner specified; shorts are not balanced.

Usage:
    python research/hedge_recovery/sim2.py --data /home/user/research/ohlc --out research/hedge_recovery/E-hedge-002_results.csv
"""
import argparse
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from sim import COINS, SPLITS, COST, TIME_EXIT, CAP, SEEDS, load, arm_inputs

SIZE_PCT = 0.5                 # leg size, % of capital (owner: 0.5% of capital per trade)
TP_NATR = 1.0                  # profit target in × NATR_14 (%), for conflict legs (card R2)


@dataclass
class Leg:
    d: int
    entry_px: float
    entry_bar: int
    conflict: bool = False
    open: bool = True
    exit_px: float = np.nan
    exit_bar: int = -1
    reason: str = ""
    was_conflict: bool = False

    def ret(self, px):
        return self.d * (px / self.entry_px - 1.0)


def simulate(O, C, natr, sig_l, sig_s, mode):
    """mode 'B': base entries + time exit. mode 'E2': card rules R1–R5 + cap. Returns legs, equity (% of capital), max open."""
    n = len(C)
    cap = CAP if mode == "E2" else 10 ** 9
    legs: List[Leg] = []
    open_legs: List[Leg] = []
    pending = []
    realised = 0.0                       # sum of net fractional returns of closed legs
    equity = np.zeros(n)                 # % of capital, realised + unrealised
    max_open = 0

    def close(leg, px, bar, reason):
        nonlocal realised
        leg.open, leg.exit_px, leg.exit_bar, leg.reason = False, px, bar, reason
        realised += leg.ret(px) - COST
        open_legs.remove(leg)

    for j in range(n):
        # 1) fills at this bar's OPEN: closes first, then opens (cap checked at fill time)
        for act in sorted(pending, key=lambda a: 0 if a[0] == "close" else 1):
            if act[0] == "close":
                leg = act[1]
                if leg.open:
                    close(leg, O[j], j, act[2])
            else:
                _, d = act
                if len(open_legs) < cap:
                    leg = Leg(d=d, entry_px=O[j], entry_bar=j)
                    if mode == "E2":
                        opposite = [x for x in open_legs if x.d == -d]
                        if opposite:                               # R1: a reversal
                            leg.conflict = True
                            for x in opposite:
                                x.conflict = True
                                x.was_conflict = True
                        leg.was_conflict = leg.conflict
                    legs.append(leg)
                    open_legs.append(leg)
        pending = []
        max_open = max(max_open, len(open_legs))

        if j == n - 1:
            for leg in list(open_legs):
                close(leg, C[j], j, "eod")
            equity[j] = SIZE_PCT * realised
            break

        # 2) decisions at this bar's CLOSE, executed at the next bar's open
        queued = set()
        if mode == "E2":
            for leg in open_legs:                                  # R2: profit target for conflict legs
                if leg.conflict and leg.ret(C[j]) * 100.0 >= TP_NATR * natr[j]:
                    pending.append(("close", leg, "R2_profit_target"))
                    queued.add(id(leg))
        for leg in open_legs:                                      # R3: time exit for non-conflict legs
            if not leg.conflict and id(leg) not in queued and j - leg.entry_bar >= TIME_EXIT:
                pending.append(("close", leg, "time_exit"))
                queued.add(id(leg))
        if mode == "E2":                                           # R5: longs closed while they outnumber shorts
            longs_open = [x for x in open_legs if x.d == 1 and id(x) not in queued]
            shorts_open = [x for x in open_legs if x.d == -1 and id(x) not in queued]
            n_l, n_s = len(longs_open), len(shorts_open)
            while n_l > n_s:
                cands = [x for x in longs_open if id(x) not in queued and x.ret(C[j]) > 0]
                if not cands:
                    break
                best = max(cands, key=lambda x: x.ret(C[j]))
                pending.append(("close", best, "R5_balance"))
                queued.add(id(best))
                n_l -= 1
        if sig_l[j]:                                               # R4 / base entry: long
            pending.append(("open", 1))
        elif sig_s[j]:
            pending.append(("open", -1))

        equity[j] = SIZE_PCT * (realised + sum(x.ret(C[j]) for x in open_legs))
    return legs, equity, max_open


def summarise(legs, equity, max_open):
    closed = [x for x in legs if not x.open]
    if not closed:
        return {"legs": 0}
    net = np.array([x.ret(x.exit_px) - COST for x in closed])
    conflict = np.array([x.was_conflict for x in closed])
    reasons = pd.Series([x.reason for x in closed]).value_counts().to_dict()
    peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
    dd = float(np.max(peak - equity))
    return {
        "legs": len(closed),
        "long_legs": int(sum(x.d == 1 for x in closed)),
        "short_legs": int(sum(x.d == -1 for x in closed)),
        "conflict_legs": int(conflict.sum()),
        "mean_net_bp": float(net.mean() * 1e4),
        "mean_net_conflict_bp": float(net[conflict].mean() * 1e4) if conflict.any() else np.nan,
        "mean_net_clean_bp": float(net[~conflict].mean() * 1e4) if (~conflict).any() else np.nan,
        "win_rate": float((net > 0).mean()),
        "realised_pct_capital": float(SIZE_PCT * net.sum()),
        "equity_end_pct": float(equity[-1]),
        "dd_max_pct_capital": dd,
        "worst_leg_bp": float(net.min() * 1e4),
        "eod_legs": int(reasons.get("eod", 0)),
        "pnl_share_conflict": float(net[conflict].sum() / net.sum()) if net.sum() != 0 and conflict.any() else np.nan,
        "max_open_legs": int(max_open),
        "reasons": ";".join(f"{k}={v}" for k, v in sorted(reasons.items())),
    }


def run(data_dir, out_csv):
    rows = []
    for split, (start, end) in SPLITS.items():
        for coin in COINS:
            df = load(os.path.join(data_dir, f"ohlc_{coin}_1h.parquet"), start, end)
            O, C, natr = df.open.values, df.close.values, df.natr.values
            for arm in ("B", "E2", "C"):
                seeds = SEEDS if arm == "C" else (0,)
                for seed in seeds:
                    sl, ss = arm_inputs(df, "C" if arm == "C" else "B", seed)
                    mode = "B" if arm == "B" else "E2"
                    legs, eq, mo = simulate(O, C, natr, sl, ss, mode)
                    s = summarise(legs, eq, mo)
                    s.update(split=split, coin=coin, arm=arm, seed=seed, bars=len(df))
                    rows.append(s)
                    print(f"{split} {coin:8s} {arm:2s} s{seed} legs={s.get('legs',0):5d} mean_bp={s.get('mean_net_bp',np.nan):9.2f} "
                          f"dd%={s.get('dd_max_pct_capital',np.nan):6.2f} worst_bp={s.get('worst_leg_bp',np.nan):10.1f} "
                          f"conflict={s.get('conflict_legs',0)}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out_csv, index=False)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.data, a.out)
