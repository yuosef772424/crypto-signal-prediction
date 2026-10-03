"""Repro R2-02: a 32-row window may straddle a hole in the data (rows deleted by add_features' dropna after missing
raw hours), so X contains a time jump while every other sample is 32 contiguous hours.

Round-1 R1-03 made the TARGET candle contiguous (ts + 1h) but the INPUT window is still cut by row position
(_align_single_tf_view: sliding_window_view over df rows). After a hole of h missing raw hours, indicator look-backs
delete ~50+ rows, so the first grid samples after the hole take their first rows from BEFORE the hole.

Expected (after fix): for every kept sample, ts - 31h is a row of the frame (window = 32 consecutive hours), i.e. such
samples are dropped like the label-gap ones.
Actual (b14b9bb): the samples right after the hole keep windows that span it (time jump of tens of hours).

Run:  python docs/research/audit/r2_02_window_spans_data_hole.py   (~5 s)
"""
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _r2_synth import *  # noqa: E402,F401,F403

rng = np.random.default_rng(3)
h = to_1h(make_coin_15m(rng, "2025-01-01", "2025-03-01"))
h = h.drop(pd.date_range("2025-02-01 09:00", "2025-02-01 11:00", freq="1h", tz="UTC"))     # 3 missing raw hours
ns = fresh_ns()
C = ns["CONFIG"]
C["phase2_data"]["use_intraday_15m"] = C["phase2_data"]["use_futures_metrics"] = False
C["market_context"]["enabled"] = C["market_breadth"]["enabled"] = C["momentum_orth_natr"]["enabled"] = False
C["funding_rate"]["enabled"] = C["open_interest"]["enabled"] = False
C["feature_order"] = None
ns["refresh_features"]()
with contextlib.redirect_stdout(io.StringIO()):
    dfs = ns["make_resample_fn"](C)(h.copy(), ["1h"])
    X, y, b, last = ns["prepare_single_asset"](dfs, config=C)
idx = dfs["1h"].index
win = C["window_sizes"]["1h"]
ts = pd.to_datetime(last[:, ns["TS_COL"]].astype("int64"), utc=True)
bad = []
for t in ts:
    p = idx.get_loc(t)
    if idx[p] - idx[p - win + 1] != pd.Timedelta(hours=win - 1):
        bad.append((t, idx[p - win + 1]))
print(f"raw hole: 3 h; frame hole: {idx.to_series().diff().max()}; samples {len(ts)}; windows spanning the hole: {len(bad)}")
for t, s in bad:
    print(f"  sample ts={t}  window starts {s}  (spans {(t - s) / pd.Timedelta('1h'):.0f} h instead of {win - 1})")
assert not bad, "DEFECT: input window straddles a data hole"
print("PASS")
