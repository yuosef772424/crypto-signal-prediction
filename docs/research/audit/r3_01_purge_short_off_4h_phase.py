"""Repro R3-01: the 129h purge assumes every sample time t is a multiple of 4h; for any other phase a val/test input window
reaches back further than the purge and contains candles that train targets were built from.

Expected: with the real split_data and the real closed-mode alignment, no val/test sample has an input bar (1h or 4h
window) at or after the open of a train sample's target candle (train t + 1h), for ANY stride/grid the pipeline accepts.
Actual: embargo_candles = 32*4h + horizon = 129, but the newest closed 4h bar closes at t - (t mod 4h), so the 4h window
starts at t - 128h - (t mod 4h) (up to 131h back). Samples at phase 1h/2h/3h right after the gap contain the target
candle of the last train sample.
The shipped setting (stride 8 on the 00/08/16 grid, t mod 4h = 0) is clean (control row). The defect is reachable by any
config with stride not a multiple of 4 (stride 1..3, 5..7...), align_windows_to_grid=False, or a start offset.

Uses the real align_multi_timeframes_time_based + split_data on frames whose value is the bar open time (so X_4h[:, 0, 0]
IS the real open hour of the first 4h bar of every window). ~5 s.
Run: python docs/research/audit/r3_01_purge_short_off_4h_phase.py
"""
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _r3_synth import *  # noqa: E402,F401,F403

ns = make_ns(8)
rows, total_bad = [], 0
for label, stride, grid in [("shipped: stride 8, grid", 8, True), ("stride 4, grid", 4, True), ("stride 1", 1, True),
                            ("stride 3", 3, True), ("stride 5", 5, True), ("stride 8, grid OFF (t=09:00+8k)", 8, False)]:
    cfg = closed_cfg(ns, stride, align_windows_to_grid=grid, split_dates={"train_end": "2025-02-01", "val_end": "2025-03-01"})
    ds = aligned_hour_dataset(ns, cfg, hour_frames("2025-01-01 02:00", 24 * 90))   # grid OFF: first window ends 02:00+31h = 09:00, so t mod 4h = 1
    with contextlib.redirect_stdout(io.StringIO()):
        train, val, test = ns["split_data"](ds, config=cfg)
    nv, worst_v = overlap_with_train_targets(ns, ds, val, train)
    nt, worst_t = overlap_with_train_targets(ns, ds, test, val)
    phases = sorted(set(((np.asarray(val["last_candles"])[:, ns["TS_COL"]] // H_NS) % 4).astype(int).tolist()))
    print(f"{label:<26} embargo={ns['embargo_candles'](ds, cfg)}h  val phases(t mod 4h)={phases}  "
          f"val windows containing a train target candle: {nv:>3}   test windows containing a val target candle: {nt:>3}"
          + (f"   e.g. sample t={worst_v[0]:%m-%d %H:%M} 4h window starts {worst_v[1]:%m-%d %H:%M}" if worst_v else ""))
    if not label.startswith("shipped"):
        total_bad += nv + nt
    else:
        assert nv == 0 and nt == 0, "control (shipped config) must be clean"
# information only: smallest explicit CONFIG['embargo_candles'] that leaves stride 1 clean (measured, not derived)
for e in (129, 130, 131, 132, 133):
    cfg = closed_cfg(ns, 1, embargo_candles=e, split_dates={"train_end": "2025-02-01", "val_end": "2025-03-01"})
    ds = aligned_hour_dataset(ns, cfg, hour_frames("2025-01-01 01:00", 24 * 90))
    with contextlib.redirect_stdout(io.StringIO()):
        train, val, test = ns["split_data"](ds, config=cfg)
    print(f"  stride 1 with embargo_candles={e}: val {overlap_with_train_targets(ns, ds, val, train)[0]}, test {overlap_with_train_targets(ns, ds, test, val)[0]}")
assert total_bad == 0, f"DEFECT: {total_bad} val/test samples contain candles used as train/val targets (purge 129h too short off the 4h phase)"
print("PASS")
