"""R3-00b (clean check): no look-ahead through the 4h context, end to end, with the REAL shipped preset.

Expected behaviour (first principles, written before reading the rationale):
  * A sample whose last 1h bar OPENS at t is decided when that bar closes: information time I = t + 1h.
  * Therefore nothing that happens at or after I may change any input of that sample: not its 1h window, not its 4h
    window (a 4h bar closing at I is allowed, one closing after I is a leak), not any 4h indicator, cross-asset feature
    (MKT_*, breadth, MOM_ORTH_NATR) or phase-2 feature (15m, funding, OI, metrics) computed on a 4h bar.
Method: build the dataset twice from a synthetic 6-coin universe (real HOURLY_4H_OVERRIDES, 43 features). In the second
build EVERY raw stream of EVERY coin (1h/15m klines, funding, OI, metrics) is replaced from a boundary T on. Samples with
t + 1h <= T must be bit-identical in X_1h and X_4h. T is placed on and between 4h/8h boundaries and mid-bar.
Teeth: the same detector on a deliberately leaky configuration (legacy alignment, higher_tf_offset=0 = the 4h bar still
forming at t) must FAIL, otherwise the check proves nothing.

Run: python docs/research/audit/r3_00b_no_lookahead_perturbation.py     (~2 min)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _r3_synth import *  # noqa: E402,F401,F403

coins, k15, start, end = universe()
fails = []


def probe(ns, label, offsets, extra_T=()):
    ts_col = ns["TS_COL"]
    A = run(ns, coins, k15, start, end)
    tsA = np.array([k[1] for k in keyed(A, ts_col)])
    base = pd.Timestamp(int(np.percentile(tsA, 50)), tz="UTC").floor("8h") + pd.Timedelta(hours=24)
    Ts = [base + pd.Timedelta(hours=1 + o) for o in offsets] + [base + pd.Timedelta(minutes=m) for m in extra_T]
    res = []
    for T in Ts:
        B = run(ns, coins, k15, start, end, T=T)
        checked, changed, first_bad, n_after, n_after_changed = compare_before(A, B, ts_col, T)
        print(f"  {label:<22} T={T:%m-%d %H:%M} (T mod 4h = {(T.hour % 4)}h{T.minute:02d}) checked={checked} "
              f"X1h_changed={changed['1h']} X4h_changed={changed['4h']} | teeth: {n_after_changed}/{n_after} later X4h differ"
              + (f" | first={first_bad}" if first_bad else ""))
        res.append((T, changed, n_after_changed))
    return A, res


print("== shipped preset, stride 8 (grid 00/08/16 UTC): T = t+1h exactly, and mid-bar T ==")
_, r8 = probe(make_ns(8), "closed s8", [0], extra_T=(90, 195))
print("== shipped preset, stride 1 (every phase of t inside the 4h bar) ==")
_, r1 = probe(make_ns(1), "closed s1", [0, 1, 2, 3], extra_T=(90,))
for T, ch, ta in r8 + r1:
    if ch["1h"] or ch["4h"]:
        fails.append(("closed", T, ch))
    if ta == 0:
        fails.append(("no teeth", T, ta))

print("== coverage teeth: the replacement must reach every value feature of X_4h (otherwise 'unchanged' proves nothing) ==")
ns8 = make_ns(8)
A8 = run(ns8, coins, k15, start, end)
T8 = pd.Timestamp("2025-04-06 09:00", tz="UTC")
B8 = run(ns8, coins, k15, start, end, T=T8)
KA, KB = keyed(A8, ns8["TS_COL"]), keyed(B8, ns8["TS_COL"])
fo = A8["feature_order"]
reached = np.zeros(len(fo), bool)
for k, i in KA.items():
    if k[1] >= T8.value and k in KB:
        reached |= (A8["X_4h"][i].astype("float32") != B8["X_4h"][KB[k]].astype("float32")).any(axis=0)
unreached = [f for f, r in zip(fo, reached) if not r]
print(f"  features in X_4h never affected by the replacement: {unreached}  (availability flags are constant 1 in this synthetic universe)")
assert all(f.endswith("_available") for f in unreached), unreached
print(f"  {int(reached.sum())}/{len(fo)} features reached, incl. MKT_*, MOM_ORTH_NATR, MKT_BREADTH_24, FUND_*, OI_chg_1, ITD/EFF/VWAP, LSR_*, TAKER_LSR_1d")

print("== teeth: leaky config (legacy alignment, higher_tf_offset=0) must be caught ==")
_, rl = probe(make_ns(1, over={"higher_tf_mode": "legacy", "higher_tf_offset": 0}), "LEAKY legacy off=0", [0, 1, 2, 3])
caught = sum(1 for _, ch, _ in rl if ch["4h"] > 0)
print(f"  leaky config flagged at {caught}/{len(rl)} boundaries")
assert caught >= 1, "detector has no teeth: a known-leaky configuration was not flagged"
assert not fails, f"DEFECT: look-ahead through the 4h context: {fails}"
print("PASS: no sample changed when everything from T on was replaced, at every phase; detector catches a leaky config")
