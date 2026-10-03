"""Repro R2-05: the documented Colab configuration leaves no sealed holdout.

PROTOCOL.md: "val selects, test compares rounds, and a sealed holdout (the last 2-3 months) is seen by neither side and opened
once, at release". hourly_1h.md configures only split_dates={train_end: 2025-06-24, val_end: 2025-11-21}; split_data then makes
EVERYTHING after val_end + 33h the test split, up to the last candle of the data (registry: 2026-09), and the panel run exports
signals_test*.csv.gz for all of it to the evaluators. There is no config key or default that withholds the tail.

Expected (after fix): with the documented config, no test sample lies in the last 60 days of the data (a default
holdout_days >= 60, or an equivalent explicit split_dates key that the documented config sets).
Actual (b14b9bb): test reaches the last data timestamp; the last 60 days are inside test.

Run:  python docs/research/audit/r2_05_no_sealed_holdout.py   (~3 s)
"""
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _r2_synth import *  # noqa: E402,F401,F403

ns = fresh_ns()
C = ns["CONFIG"]
C["split_dates"] = {"train_end": "2025-06-24", "val_end": "2025-11-21"}        # exactly hourly_1h.md "الخطوات" step 2
grid = pd.date_range("2025-03-01", "2026-09-20", freq="8h", tz="UTC")           # stride-8 grid, data ends 2026-09-20
names = ["A", "B", "C"]
rows, bounds, off = [], [], 0
for nm in names:
    lc = np.zeros((len(grid), 7))
    lc[:, 3] = grid.as_unit("ns").asi8
    lc[:, :3] = 100.0
    lc[:, 4:] = 100.0
    rows.append(lc)
    bounds.append({"name": nm, "start": off, "end": off + len(grid)})
    off += len(grid)
lc = np.concatenate(rows)
n = len(lc)
ds = {"base_params": np.zeros((n, 2), "float32"), "last_candles": lc, "X_1h": np.zeros((n, 4, 2), "float32"),
      "y_high_class": np.zeros(n, "float32"), "y_high_reg": np.zeros(n, "float32"), "timeframes": ["1h"],
      "targets": ["high_class", "high_reg"], "asset_bounds": bounds, "base_tf": "1h", "window_sizes": {"1h": 32},
      "forecast_horizon": 1, "stride": 8}
with contextlib.redirect_stdout(io.StringIO()):
    train, val, test = ns["split_data"](ds, config=C)
data_end = grid.max()
ts_test = np.concatenate([sp["last_candles"][:, 3] for sp in test.values()]).astype("int64")
last_test = pd.Timestamp(int(ts_test.max()), tz="UTC")
in_tail = int((ts_test > (data_end - pd.Timedelta(days=60)).value).sum())
print(f"data ends {data_end}; last test sample {last_test}; test samples in the final 60 days: {in_tail} of {len(ts_test)}")
assert in_tail == 0, "DEFECT: no sealed holdout - the final 60 days are part of test (and exported for evaluation)"
print("PASS")
