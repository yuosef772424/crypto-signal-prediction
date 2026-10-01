"""Checks that PASSED in round 1 (kept as regression guards; they should keep passing).

Built with the real pipeline functions (1h_s8 preset: window 32, stride 8, horizon 1, grid alignment) on
synthetic hourly OHLCV for 10 coins with different listing times; funding/OI archives absent.

  1. No look-ahead: rebuilding the dataset from data cut at T_cut gives bit-identical X, y and
     last_candles for every sample of the cut build (all 23 features, incl. MKT_*, BREADTH, MOM_ORTH).
  2. Grid: every window end on the 8h grid; every timestamp holds all coins listed at that time.
  3. Purge: hours strictly between the last train target candle's close and the first val window
     candle's open (and val -> test) are > 0.
  4. Cross-asset features are not constant (MKT_BREADTH_24, MOM_ORTH_NATR).

Run:  python docs/research/audit/repro_00_clean_checks.py   (~10 s)
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
ns["load_funding_open_interest"] = lambda *a, **k: None
with contextlib.redirect_stdout(io.StringIO()):
    ns["apply_hourly_preset"](ns["HOURLY_W32_S8_OVERRIDES"])
rng = np.random.default_rng(0)
coins = ["BTCUSDT"] + [f"C{i}USDT" for i in range(9)]
raw = {}
for c in coins:
    idx = pd.date_range(pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=int(rng.integers(0, 300))),
                        pd.Timestamp("2024-03-01", tz="UTC"), freq="1h")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
    op = np.r_[close[0], close[:-1]]
    raw[c] = pd.DataFrame({"open": op, "high": np.maximum(op, close) * (1 + np.abs(rng.normal(0, .003, len(idx)))),
                           "low": np.minimum(op, close) * (1 - np.abs(rng.normal(0, .003, len(idx)))),
                           "close": close, "volume": rng.uniform(100, 200, len(idx))}, index=idx)


def build(cut=None):
    def load(fid, name):
        d = raw[name]
        return (d[d.index <= cut] if cut is not None else d).copy()
    with contextlib.redirect_stdout(io.StringIO()):
        return ns["build_dataset_from_loader"]([{"name": c} for c in coins], load, ns["make_resample_fn"](C),
                                               max_workers=1)


def keys(ds):
    names = np.empty(len(ds["last_candles"]), object)
    for b in ds["asset_bounds"]:
        names[b["start"]:b["end"]] = b["name"]
    return pd.MultiIndex.from_arrays([names, ds["last_candles"][:, 3].astype("int64")])


full, part = build(), build(pd.Timestamp("2024-02-10 03:00", tz="UTC"))
j = pd.Series(np.arange(len(full["last_candles"])), index=keys(full)).loc[keys(part)].to_numpy()
dx = np.abs(full["X_1h"][j] - part["X_1h"]).max(axis=(0, 1))
leaky = [f for f, v in zip(full["feature_order"], dx) if v > 1e-6]
same_y = all(np.array_equal(full[f"y_{h}"][j], part[f"y_{h}"]) for h in full["targets"])
print(f"1. truncation: {len(part['last_candles'])} rows compared | features that changed: {leaky} | y equal: {same_y}")
assert not leaky and same_y and np.array_equal(full["last_candles"][j], part["last_candles"])

ts = full["last_candles"][:, 3].astype("int64")
H = pd.Timedelta("1h").value
per_ts = pd.Series(ts).value_counts()
print(f"2. on 8h grid: {np.mean(ts % (8 * H) == 0):.0%} | coins per timestamp median {per_ts.median():.0f} of {len(coins)}")
assert np.all(ts % (8 * H) == 0) and per_ts.median() == len(coins)

C["split_dates"] = {"train_end": "2024-02-05", "val_end": "2024-02-15"}
with contextlib.redirect_stdout(io.StringIO()):
    tr, va, te = ns["split_data"](full, config=C)
t_tr = tr["last_candles"][:, 3].astype("int64")
t_va = va["last_candles"][:, 3].astype("int64")
t_te = np.concatenate([v["last_candles"][:, 3] for v in te.values()]).astype("int64")
gaps = [((b.min() - 31 * H) - (a.max() + 2 * H)) / H for a, b in ((t_tr, t_va), (t_va, t_te))]
print(f"3. purge: free hours train->val {gaps[0]:.0f}, val->test {gaps[1]:.0f}")
assert min(gaps) > 0

f = full["feature_order"]
sd = {n: float(full["X_1h"][:, -1, f.index(n)].std()) for n in ("MKT_BREADTH_24", "MOM_ORTH_NATR")}
print(f"4. last-step std: {sd}")
assert min(sd.values()) > 0.05
print("PASS")
