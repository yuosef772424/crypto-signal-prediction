"""Round-2 clean checks (expected: PASS at b14b9bb). Everything runs the real notebook functions on a synthetic Drive with
ALL phase-2 archives present (43 features), which round 1 never exercised.

1. No look-ahead: build the dataset twice; in variant B every source (1h candles, 15m candles, funding, OI, futures_metrics,
   all coins incl. the BTC reference and the breadth/rank universe) is replaced by fresh random data strictly after the
   knowable boundary T0 of a sample. X of every sample with ts <= T0 must be bit-identical, and samples after T0 must differ
   (so the perturbation really reached each feature family).
2. Availability masks: a coin with no archives at all gets flag 0 / value 0; a coin whose futures_metrics start later gets
   flag 0 before the start and 1 after (mask works, values neutral).
3. Scale round trip end to end: pipeline dataset (reg_target_scale=100) -> split_data -> main-style stamping ->
   PanelSplit -> oracle model output -> trainer.predict scaling -> export_signals -> report.summarize. The exported
   mu_*/pred_* must equal the realised returns/prices and AUC must be 1.0 (labels aligned with the trained label).

Run:  python docs/research/audit/r2_00_clean_checks.py   (~1 min)
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _r2_synth import *  # noqa: E402,F401,F403
from _nbload import ROOT  # noqa: E402

sys.path.insert(0, ROOT)

coins = ["BTCUSDT"] + [f"C{i}USDT" for i in range(8)]
start, end = "2025-01-01", "2025-03-20"
rng = np.random.default_rng(1)
k15 = {c: make_coin_15m(rng, start, end, s0=100 * (i + 1)) for i, c in enumerate(coins)}
T0 = pd.Timestamp("2025-02-20 08:00", tz="UTC")

# ───────── 1. look-ahead perturbation ─────────
rng2 = np.random.default_rng(999)
k15B = {}
for c, k in k15.items():
    keep = k[k.index <= T0 + pd.Timedelta("45min")]
    new = make_coin_15m(rng2, T0 + pd.Timedelta("1h"), end, s0=float(keep["close"].iloc[-1]))
    k15B[c] = pd.concat([keep, new])


def make(k15x, seed):
    h1 = {c: to_1h(k) for c, k in k15x.items()}
    tmp = tempfile.mkdtemp()
    write_archives(tmp, coins, k15x, start, end, np.random.default_rng(seed))
    return h1, tmp


hA, tA = make(k15, 5)
hB, tB = make(k15B, 6)


def splice(sub, tsfun, bound):
    for c in coins:
        pa, pb = (os.path.join(t, sub, f"{c}.csv.gz") for t in (tA, tB))
        a, b = pd.read_csv(pa), pd.read_csv(pb)
        pd.concat([a[tsfun(a) <= bound], b[tsfun(b) > bound]]).to_csv(pb, index=False, compression="gzip")


iso = lambda s: pd.to_datetime(s["timestamp"], utc=True)  # noqa: E731     # OI/metrics archives carry ISO text timestamps
splice("open_interest", iso, T0)                       # a row stamped h is knowable at h+1h -> rows <= T0 are knowable at T0+1h
splice("futures_metrics", iso, T0)
splice("funding_rate", iso, T0 + pd.Timedelta("1h"))
nsA, nsB = fresh_ns(), fresh_ns()
dsA, dsB = build(nsA, tA, hA, coins), build(nsB, tB, hB, coins)
fo = dsA["feature_order"]
assert len(fo) == 43, len(fo)
TS = nsA["TS_COL"]


def keyed(ds):
    return {(b["name"], int(ds["last_candles"][i, TS])): i for b in ds["asset_bounds"] for i in range(b["start"], b["end"])}


kA, kB = keyed(dsA), keyed(dsB)
before = [k for k in kA if k in kB and k[1] <= T0.value]
after = [k for k in kA if k in kB and k[1] > T0.value + 24 * 3600 * 10**9]
ia, ib = (np.array([m[k] for k in before]) for m in (kA, kB))
diff = np.abs(dsA["X_1h"][ia] - dsB["X_1h"][ib]).max(axis=(0, 1))
assert not (diff > 1e-6).any(), {fo[j]: float(diff[j]) for j in np.flatnonzero(diff > 1e-6)}
ja, jb = (np.array([m[k] for k in after]) for m in (kA, kB))
dafter = np.abs(dsA["X_1h"][ja] - dsB["X_1h"][jb]).max(axis=(0, 1))
families = {"funding": ["FUND_rate", "FUND_sum_1d"], "oi": ["OI_chg_1"], "metrics": ["LSR_GLOBAL", "TAKER_LSR_1d"],
            "15m daily": ["ITD_RVOL", "ITD_TRADES_Z", "VWAP_DEVIATION", "EFF_RATIO_24H"], "breadth": ["MKT_BREADTH_24"],
            "orth-mom": ["MOM_ORTH_NATR"], "btc ctx": ["MKT_CORR_20"], "price": ["close", "RSI_14"]}
for fam, cols in families.items():
    for c in cols:
        assert dafter[fo.index(c)] > 1e-3, f"perturbation did not reach {c} ({fam}) -> check is vacuous"
print(f"1. look-ahead: {len(before)} samples <= T0 identical on all 43 features "
      f"({sum(k[1] == T0.value for k in before)} at T0); {len(after)} later samples differ in every feature family")

# ───────── 2. availability masks ─────────
tmp = tempfile.mkdtemp()
rng3 = np.random.default_rng(7)
write_archives(tmp, coins, k15, start, end, rng3, only=[c for c in coins if c not in ("C6USDT", "C7USDT")])
write_archives(tmp, ["C7USDT"], k15, start, end, rng3, metrics_start="2025-02-15")     # metrics/OI start later
ns2 = fresh_ns()
ds2 = build(ns2, tmp, {c: to_1h(k) for c, k in k15.items()}, coins)
fo2, X2 = ds2["feature_order"], ds2["X_1h"]
bounds = {b["name"]: b for b in ds2["asset_bounds"]}
flag_cols = [c for c in fo2 if c.endswith("_available")]
b6 = bounds["C6USDT"]
x6 = X2[b6["start"]:b6["end"]]
for c in ["FUND_available", "OI_available", "MET_available", "ITD_available"]:
    j = fo2.index(c)
    assert (x6[:, :, j] == -1).all(), f"{c}: coin without archives must carry flag 0 (-1 after x*2-1)"
for c in ["FUND_rate", "FUND_sum_3d", "OI_chg_1", "LSR_GLOBAL", "TAKER_LSR_1d", "ITD_RVOL", "ITD_TRADES_Z", "VWAP_DEVIATION"]:
    assert (x6[:, :, fo2.index(c)] == 0).all(), f"{c}: value must be neutral 0 when no archive"
b7 = bounds["C7USDT"]
x7, ts7 = X2[b7["start"]:b7["end"]], pd.to_datetime(ds2["last_candles"][b7["start"]:b7["end"], TS].astype("int64"), utc=True)
j = fo2.index("MET_available")
pre, post = ts7 < pd.Timestamp("2025-02-15", tz="UTC"), ts7 >= pd.Timestamp("2025-02-15", tz="UTC") + pd.Timedelta("32h")
assert (x7[pre][:, :, j] == -1).all() and (x7[post][:, :, j] == 1).all(), "MET_available must switch 0 -> 1 at the archive start"
print("2. masks: no-archive coin -> flags 0/values 0; late-start coin flips 0 -> 1 at the start")

# ───────── 3. scale round trip through the panel path ─────────
from cross_asset.data import group_ns_for, panel_split_from  # noqa: E402
from cross_asset.report import summarize  # noqa: E402
from cross_asset.train import export_signals  # noqa: E402

C = nsA["CONFIG"]
C["split_dates"] = {"train_end": "2025-02-05", "val_end": "2025-03-01"}
import contextlib, io  # noqa: E401,E402
with contextlib.redirect_stdout(io.StringIO()):
    train, val, test = nsA["split_data"](dsA, config=C)
scale = float(dsA["reg_target_scale"])
assert scale == 100.0
for sp in (train, val, *test.values()):
    sp["reg_target_scale"] = scale                       # what main.ipynb cell 8 does
g = group_ns_for("1h", C["stride"])
tgt = ("high", "low")
va = panel_split_from(val, "1h", None, "val", targets=tgt, day_ns=g)
te = panel_split_from(test, "1h", None, "test", targets=tgt, day_ns=g)
assert va.target_scale == 100.0 and te.target_scale == 100.0
assert va.check()["ok"] and te.check()["ok"], (va.check(), te.check())
for ps in (va, te):
    y_scaled = ps.yreg
    logit = np.where(ps.ycls > 0, 6.0, -6.0).astype("float32")
    mu_pred = y_scaled.astype("float32")                # trainer.predict returns mu*reg_scale == y-units; oracle == y
    df = export_signals(ps, logit, mu_pred, ps.name, targets=ps.targets)
    assert np.allclose(df["fut_high"], df["pred_high"], rtol=1e-4), "pred_high != realised next high (scale/inverse)"
    assert np.allclose(df["fut_low"], df["pred_low"], rtol=1e-4), "pred_low != realised next low (scale/inverse)"
    assert np.allclose(df["mu_high"], df["fut_high"] / df["last_high"] - 1, atol=1e-4)
m, _ = summarize(df.assign(split="val"), df.assign(split="test"), group_ns=g)
assert m["auc_high"] > 0.999 and m["auc_low"] > 0.999, m
print("3. scale/inverse: oracle round trip exact (pred_* == realised prices), AUC(high/low) = 1.0")
print("PASS")
