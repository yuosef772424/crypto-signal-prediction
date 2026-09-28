"""Repro 03: after a hole in a coin's rows, the 1h label comes from a later candle.

prepare_single_asset takes the "future" candle as the next ROW (end_idx + 1), not the candle at
ts + 1h. Two ways a hole appears right after a grid window end:
  (a) hours missing from the raw file (AAAUSDT below: 5 h missing). resample('1h') re-inserts them as
      NaN-price rows (volume sum = 0, so dropna(how='all') keeps them); indicator look-backs then turn
      the following ~20 rows NaN and add_features' dropna deletes them too -> a 25 h hole;
  (b) rows deleted by add_features' final df.dropna() -- here a flat, zero-volume stretch makes some
      indicator NaN mid-series (BBBUSDT below), and the rows vanish silently.
Either way the label of the sample at grid time T is built from a candle hours later, so its horizon
differs from the other coins in the same 8h group (and its 32-row window spans more than 32h).

Expected (after fix): every kept sample's target candle opens exactly ts + forecast_horizon * 1h
(samples whose target candle is missing are dropped).
Actual (commit c62b4f0): the sample at the gap keeps a target from a later candle.

Run:  python docs/research/audit/repro_03_label_across_data_gap.py   (~5 s)
"""
import contextlib
import io
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _nbload import load_pipeline  # noqa: E402

ns = load_pipeline()
C = ns["CONFIG"]
ns["load_funding_open_interest"] = lambda *a, **k: None      # no Drive here: archives absent
ns["apply_hourly_preset"](ns["HOURLY_W32_S8_OVERRIDES"])

rng = np.random.default_rng(0)
coins = ["BTCUSDT", "AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT"]
idx_full = pd.date_range("2024-01-01", "2024-02-15", freq="1h", tz="UTC")
gap_after = pd.Timestamp("2024-02-01 08:00", tz="UTC")          # an 8h grid point
gap = pd.date_range(gap_after + pd.Timedelta("1h"), periods=5, freq="1h", tz="UTC")   # 09:00..13:00 missing
raw = {}
for c in coins:
    idx = idx_full.difference(gap) if c == "AAAUSDT" else idx_full
    close = 100 * np.exp(np.cumsum(rng.normal(0, .01, len(idx))))
    vol = rng.uniform(100, 200, len(idx))
    op = np.r_[close[0], close[:-1]]
    hi, lo = np.maximum(op, close) * 1.002, np.minimum(op, close) * 0.998
    if c == "BBBUSDT":                                   # 20 flat, zero-volume hours from 2024-01-13 12:00
        s = slice(300, 320)
        close[s] = op[s] = hi[s] = lo[s] = close[299]
        vol[s] = 0.0
        op[320] = close[299]
    raw[c] = pd.DataFrame({"open": op, "high": hi, "low": lo, "close": close, "volume": vol}, index=idx)

with contextlib.redirect_stdout(io.StringIO()):
    ds = ns["build_dataset_from_loader"]([{"name": c} for c in coins], lambda f, n: raw[n].copy(),
                                         ns["make_resample_fn"](C), max_workers=1)
bad = []
for coin in ("AAAUSDT", "BBBUSDT"):
    b = next(b for b in ds["asset_bounds"] if b["name"] == coin)
    lc = ds["last_candles"][b["start"]:b["end"]]
    ts = pd.to_datetime(lc[:, 3].astype("int64"), utc=True)
    src = raw[coin]
    for t, fut_close in zip(ts, lc[:, 4]):
        target_open = t + pd.Timedelta("1h")
        if target_open not in src.index or not np.isclose(src.loc[target_open, "close"], fut_close):
            got = src.index[np.isclose(src["close"].to_numpy(), fut_close) & (src.index > t)]
            bad.append((coin, t, target_open, got[0] if len(got) else "?"))
for coin, t, want, got in bad:
    print(f"{coin} sample ts={t} : target candle should open {want}, label taken from candle opening {got}")
print(f"{len(bad)} sample(s) with a target candle other than ts+1h")
assert not bad, "DEFECT: label built across a data gap"
print("PASS")
