"""Repro R2-04: a day-level feature classified CUMULATIVE (ITD_TRADES_LOG) is standardised inside each 32-row window by
IQR. A day-level series is a step function (2-3 distinct values per window), so the IQR is either the step itself
(minority side >= 25% of rows: output is exactly 1 for every step size) or ~0 (minority < 25%: divisor hits the 1e-8
floor and the row saturates at the +-5 clip). Either way the size of the step is erased: a 1e-6 change in log trade count
and a 2.7x change in trade count give the same value.

Expected (after fix): |z| grows with the step (tiny step -> |z| far below the big step) and <2% of ITD_TRADES_LOG values
in a built dataset sit at the clip.
Actual (b14b9bb): identical z for tiny and big steps (1.00 / 1.00 and 5.00 / 5.00); ~5% of values and ~29% of windows
hold clipped values on the synthetic Drive.

Run:  python docs/research/audit/r2_04_step_feature_saturation.py   (~15 s)
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _r2_synth import *  # noqa: E402,F401,F403

ns = fresh_ns()
name = "ITD_TRADES_LOG"
ctr, scl = np.zeros(1, dtype="float32"), np.ones(1, dtype="float32")


def zmax(step, n_rows):
    w = np.full((1, 32, 1), 7.0, dtype="float64")
    w[0, :n_rows, 0] = 7.0 + step
    return float(np.abs(ns["process_windows"](w.astype("float32"), [name], ctr, scl, "robust", ns["CONFIG"])).max())


ok = True
for n_rows in (9, 5):
    tiny, big = zmax(1e-3, n_rows), zmax(1.0, n_rows)
    print(f"step on {n_rows}/32 rows: max|z| tiny step (1e-3) = {tiny:.2f} | big step (1.0) = {big:.2f}   (kind={ns['classify_feature'](name)})")
    ok &= tiny < 0.5 * big

coins = ["BTCUSDT"] + [f"C{i}USDT" for i in range(8)]
rng = np.random.default_rng(1)
k15 = {c: make_coin_15m(rng, "2025-01-01", "2025-03-20", s0=100 * (i + 1)) for i, c in enumerate(coins)}
tmp = tempfile.mkdtemp()
write_archives(tmp, coins, k15, "2025-01-01", "2025-03-20", rng)
ns2 = fresh_ns()
ds = build(ns2, tmp, {c: to_1h(k) for c, k in k15.items()}, coins)
sat = 0.0
if name in ds["feature_order"]:
    v = ds["X_1h"][:, :, ds["feature_order"].index(name)]
    sat = float((np.abs(v) >= 4.99).mean())
    print(f"dataset: fraction of {name} values at the +-5 clip = {sat:.3f}; windows with any clipped value = {(np.abs(v) >= 4.99).any(axis=1).mean():.2f}")
assert ok and sat < 0.02, "DEFECT: step-function feature loses the size of the step (indicator / saturation)"
print("PASS")
