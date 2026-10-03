"""Repro R3-02: a 1h+4h dataset in the legacy alignment (the mode main.ipynb accepts with only a printed warning) is split with
a 33h purge although its 4h window spans 128h+, so val/test windows contain train targets.

Expected: split_data purges at least as much as the widest input window of any timeframe (the 129h rule the closed mode
uses), or main refuses a two-timeframe dataset whose purge is too short.
Actual: embargo_candles() widens only when higher_tf_mode == 'closed'; for legacy it returns base window + horizon = 33.
main.ipynb section 3 handles a dataset without higher_tf_mode='closed' by printing a warning about the alignment rule and
continuing (MODEL_TFS = ["1h", "4h"] is still accepted, the 4h branch is trained and evaluated).

Real align_multi_timeframes_time_based (legacy) + real split_data on frames whose value is the bar open time. ~3 s.
Run: python docs/research/audit/r3_02_legacy_two_tf_embargo_33h.py
"""
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _r3_synth import *  # noqa: E402,F401,F403

ns = make_ns(8)
cfg = closed_cfg(ns, 8, mode="legacy", split_dates={"train_end": "2025-02-01", "val_end": "2025-03-01"})
ds = aligned_hour_dataset(ns, cfg, hour_frames("2025-01-01 00:00", 24 * 90))
ds.pop("higher_tf_mode")                                    # what a legacy-built dataset looks like (key absent)
with contextlib.redirect_stdout(io.StringIO()):
    train, val, test = ns["split_data"](ds, config=cfg)
span_h = int(val["X_4h"].shape[1]) * 4
nv, worst = overlap_with_train_targets(ns, ds, val, train)
print(f"embargo_candles = {ns['embargo_candles'](ds, cfg)}h | 4h window = {val['X_4h'].shape[1]} bars = {span_h}h "
      f"| val samples {len(val['last_candles'])}: windows containing a train target candle = {nv}"
      + (f" (e.g. sample t={worst[0]:%m-%d %H:%M}, 4h window starts {worst[1]:%m-%d %H:%M})" if worst else ""))
assert nv == 0, "DEFECT: legacy two-timeframe split purges 33h but the 4h window spans 128h+"
print("PASS")
