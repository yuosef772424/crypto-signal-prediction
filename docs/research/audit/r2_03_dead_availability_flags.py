"""Repro R2-03: three of the 43 features are dead availability flags: EFF_RATIO_available, VWAP_DEVIATION_available,
VOL_CONC_HHI_available.

Their normalisation kind is UNIT_0_1 (FEATURE_KINDS), not BINARY_FLAG, so process_windows zeroes any window in which the
column is constant (hi == lo and kind not in NEVER_ZEROED_KINDS). A flag is constant (all 1 for a covered coin, all 0 for an
uncovered one) in almost every 32h window, so both states map to 0 and the mask carries no information; the other four
flags (FUND_/OI_/ITD_/MET_available) are BINARY_FLAG and do separate the states (+1 vs -1).

Expected (after fix): for each *_available column, a coin with the archive and a coin without it get different values
(+1 vs -1) in fully covered / fully uncovered windows.
Actual (b14b9bb): all zero for both.

Run:  python docs/research/audit/r2_03_dead_availability_flags.py   (~15 s)
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _r2_synth import *  # noqa: E402,F401,F403

coins = ["BTCUSDT", "C0USDT", "C1USDT", "C2USDT", "C3USDT", "C4USDT", "NOARCH1USDT"]
start, end = "2025-01-01", "2025-03-01"
rng = np.random.default_rng(1)
k15 = {c: make_coin_15m(rng, start, end, s0=100 + 10 * i) for i, c in enumerate(coins)}
tmp = tempfile.mkdtemp()
write_archives(tmp, coins, k15, start, end, rng, only=[c for c in coins if c != "NOARCH1USDT"])
ns = fresh_ns()
ds = build(ns, tmp, {c: to_1h(k) for c, k in k15.items()}, coins)
fo, X = ds["feature_order"], ds["X_1h"]
bounds = {b["name"]: b for b in ds["asset_bounds"]}
rows = lambda c: X[bounds[c]["start"] + 60:bounds[c]["end"]]        # skip the first windows (rolling warm-up)  # noqa: E731
flags = [c for c in fo if c.endswith("_available")]
bad = []
for f in flags:
    j = fo.index(f)
    covered, uncovered = float(rows("C0USDT")[:, :, j].mean()), float(rows("NOARCH1USDT")[:, :, j].mean())
    status = "ok" if covered != uncovered else "DEAD (same value for covered and uncovered coin)"
    print(f"{f:28s} kind={ns['classify_feature'](f):12s} covered coin {covered:+.2f} | uncovered coin {uncovered:+.2f}  {status}")
    if covered == uncovered:
        bad.append(f)
assert not bad, f"DEFECT: availability flags carry no information: {bad}"
print("PASS")
