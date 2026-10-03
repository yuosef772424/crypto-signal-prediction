"""R3-00 (clean checks): exact-timestamp 4h alignment, purge on the shipped grid, 4h normalisation window, float16 storage.

First-principles expectations (written before reading any rationale):
  E1  sample = last 1h bar opening at t, closing at t+1h, so information time I = t+1h. Every 4h bar in its window must have
      CLOSED <= I (open + 4h <= I); a bar closing after I is a leak; a bar closing exactly at I is allowed. The 4h window
      must be 32 consecutive bars. Tested on REAL timestamps (frames whose value is the bar's own open time), for sample
      phases t mod 4h = 0,1,2,3, for on-grid (stride 8) and off-grid data, and across holes in the 1h and 4h data.
  E2  purge: no val/test input bar (1h or 4h window) may be at or after the open of a train sample's target candle;
      tested with the REAL split_data on the shipped stride-8 grid.
  E3  4h normalisation: X_4h[close] == (close - median)/IQR of the 32 raw 4h closes of THAT window only (clip +-5).
  E4  float16: finite, |x| <= 5, error vs a float32 build <= 2.5e-3 (float16 spacing near 5 is 3.9e-3), also with raw volume
      x1e12, trades x1e9, open interest x1e9 (large-valued features).
Run: python docs/research/audit/r3_00_clean_checks.py     (~1 min)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _r3_synth import *  # noqa: E402,F401,F403

ns = make_ns(8)
H = H_NS

# ── E1: exact timestamps ────────────────────────────────────────────────────────────────────────────────────────────
print("== E1 exact-timestamp alignment (value = bar open hour) ==")
holes_1h = pd.date_range("2025-01-12 05:00", "2025-01-12 07:00", freq="1h", tz="UTC")
holes_4h = [pd.Timestamp("2025-01-20 08:00", tz="UTC")]
for stride, grid, start, drop1, drop4 in [(1, True, "2025-01-01 00:00", (), ()), (1, True, "2025-01-01 03:00", (), ()),
                                          (3, True, "2025-01-01 02:00", (), ()), (8, True, "2025-01-01 00:00", (), ()),
                                          (8, False, "2025-01-01 02:00", (), ()),
                                          (1, True, "2025-01-01 00:00", holes_1h, ()),
                                          (1, True, "2025-01-01 00:00", (), holes_4h)]:
    cfg = closed_cfg(ns, stride, align_windows_to_grid=grid)
    dfs = hour_frames(start, 24 * 40, drop_1h=drop1, drop_4h=drop4)
    out, ends = ns["align_multi_timeframes_time_based"](dfs, ["1h", "4h"], cfg["window_sizes"], stride, cfg)
    t = np.array([e.value // H for e in ends])
    last1, X4 = out["1h"][:, -1, 0], out["4h"][:, :, 0]
    assert (last1 == t).all(), "1h window does not end at the sample time"
    close4 = X4[:, -1] + 4                                             # close hour of the newest 4h bar in the window
    info = t + 1
    n_leak = int((close4 > info).sum())                                # closes after information time  = leak
    n_gt_t = int((close4 > t).sum())                                   # stricter documented rule: close <= t
    contiguous = bool((np.diff(X4, axis=1) == 4).all())
    contiguous1 = bool((np.diff(out["1h"][:, :, 0], axis=1) == 1).all())
    print(f"  stride={stride} grid={grid!s:5} start={start[-5:]} holes1h={len(drop1)} holes4h={len(drop4)}: N={len(t)} "
          f"leaks(close>t+1h)={n_leak} close>t={n_gt_t} slack(I-close)={int((info - close4).min())}..{int((info - close4).max())}h "
          f"4h contiguous={contiguous} 1h contiguous={contiguous1} phases(t mod 4)={sorted(set((t % 4).tolist()))}")
    # 1h-window contiguity is enforced later, in prepare_single_asset (R2-02), not by the aligner: only assert it without 1h holes.
    assert n_leak == 0 and n_gt_t == 0 and contiguous and (contiguous1 or len(drop1))
    assert len(t) > 50
# a bar closing exactly at I is dropped when t = 3 mod 4 (conservative, not a leak): show it.
cfg = closed_cfg(ns, 1)
out, ends = ns["align_multi_timeframes_time_based"](hour_frames("2025-01-01 00:00", 24 * 10), ["1h", "4h"], cfg["window_sizes"], 1, cfg)
t = np.array([e.value // H for e in ends]); slack = t + 1 - (out["4h"][:, -1, 0] + 4)
print(f"  t mod 4h = 3 -> slack {int(slack[t % 4 == 3].min())}h (bar closing exactly at t+1h is skipped: 1h staler than allowed, harmless)")

# ── E2: purge on the shipped grid (real split_data) ─────────────────────────────────────────────────────────────────
print("== E2 purge: real split_data, shipped stride 8, real embargo ==")
cfg = closed_cfg(ns, 8, split_dates={"train_end": "2025-02-01", "val_end": "2025-03-01"})
ds = aligned_hour_dataset(ns, cfg, hour_frames("2025-01-01 00:00", 24 * 90))
import contextlib, io  # noqa: E402
with contextlib.redirect_stdout(io.StringIO()):
    train, val, test = ns["split_data"](ds, config=cfg)
print(f"  embargo_candles={ns['embargo_candles'](ds, cfg)}  train/val/test = {len(train['last_candles'])}/{len(val['last_candles'])}/{len(test['last_candles'])}")
for name, sp in (("val", val), ("test", test)):
    n, worst = overlap_with_train_targets(ns, ds, sp, train)
    print(f"  {name}: samples whose window contains a train target candle = {n}")
    assert n == 0
n2, _ = overlap_with_train_targets(ns, ds, test, val)
print(f"  test vs val targets = {n2}")
assert n2 == 0

# ── E3/E4: normalisation window and float16 on the real preset (1 coin end to end) ─────────────────────────────────
print("== E3 4h normalisation window / E4 float16 ==")
coins, k15, start, end = universe(ncoin=6)
A = run(ns, coins, k15, start, end)
fo = A["feature_order"]
j = fo.index("close")
# recompute from the raw 4h windows of one coin through the real align + real feature frames
C = ns["CONFIG"]
c0 = coins[2]
import tempfile  # noqa: E402
from _r2_synth import write_archives  # noqa: E402
tmp = tempfile.mkdtemp()
write_archives(tmp, coins, k15, start, end, np.random.default_rng(11))
ns["_PHASE2_CACHE"].clear()
C["phase2_data"]["data_root"] = tmp
h1 = {c: to_1h(k15[c]) for c in coins}
with contextlib.redirect_stdout(io.StringIO()):
    dfs = ns["make_resample_fn"](C)(h1[c0].copy(), ["1h", "4h"])
    dfs = ns["add_funding_oi_features"](ns["add_intraday_15m_features"](dfs, c0, C), c0, C) if False else dfs
feat = ns["feature_order"](C)
avail = [f for f in feat if f in dfs["1h"].columns]
cfgw = C
raw, ends = ns["align_multi_timeframes_time_based"]({tf: dfs[tf][["close"]] for tf in ("1h", "4h")}, ["1h", "4h"], C["window_sizes"], 8, C)
c4 = raw["4h"][:, :, 0].astype("float64")
q75, q50, q25 = np.percentile(c4, [75, 50, 25], axis=1)
scale = np.maximum(np.maximum(q75 - q25, np.abs(q50) * ns["SCALE_REL_FLOOR"]), ns["SCALE_ABS_FLOOR"])
exp4 = np.clip((c4 - q50[:, None]) / scale[:, None], -5, 5)
# the same coin inside the full-universe dataset
K = keyed(A, ns["TS_COL"])
rows = [(i, e) for i, e in enumerate(ends) if (c0, e.value) in K]
assert len(rows) > 50, len(rows)
err = max(float(np.abs(A["X_4h"][K[(c0, e.value)], :, j].astype("float64") - exp4[i]).max()) for i, e in rows)
print(f"  4h close recomputed from its own 32-bar window: max |X_4h - expected| = {err:.2e} over {len(rows)} samples of {c0}")
assert err < 4e-3
c1 = raw["1h"][:, :, 0].astype("float64")                       # teeth: normalising 4h by the 1h window's stats would differ a lot
p75, p50, p25 = np.percentile(c1, [75, 50, 25], axis=1)
alt = np.clip((c4 - p50[:, None]) / np.maximum(np.maximum(p75 - p25, np.abs(p50) * ns["SCALE_REL_FLOOR"]), ns["SCALE_ABS_FLOOR"])[:, None], -5, 5)
print(f"  (teeth) 4h close normalised with the 1h window stats would differ from own-window by up to {float(np.abs(alt - exp4).max()):.2f}")
assert float(np.abs(alt - exp4).max()) > 0.5
X = A["X_4h"]
print(f"  float16 finite={bool(np.isfinite(X.astype('float32')).all())} max|x|={float(np.abs(X.astype('float32')).max()):.3f} dtype={X.dtype}")
assert np.isfinite(X.astype("float32")).all() and float(np.abs(X.astype("float32")).max()) <= 5.0

def worst_err(scale_k15=None, oi=None):
    n16 = make_ns(8)
    n32 = make_ns(8, over={"x_storage_dtype": None})
    a = run(n16, coins, k15, start, end, scale_k15=scale_k15, oi_factor=oi)
    b = run(n32, coins, k15, start, end, scale_k15=scale_k15, oi_factor=oi)
    assert a["X_1h"].dtype == np.float16 and b["X_1h"].dtype == np.float32
    e = max(float(np.abs(a[f"X_{tf}"].astype("float32") - b[f"X_{tf}"]).max()) for tf in ("1h", "4h"))
    fin = all(np.isfinite(a[f"X_{tf}"].astype("float32")).all() for tf in ("1h", "4h"))
    return e, fin

e0, f0 = worst_err()
e1, f1 = worst_err(scale_k15={"volume": 1e12, "quote_volume": 1e12, "taker_buy_volume": 1e12, "taker_buy_quote_volume": 1e12, "trades": 1e9}, oi=1e9)
print(f"  float16 vs float32 build: max abs error {e0:.2e} (normal), {e1:.2e} (volume x1e12, trades x1e9, OI x1e9); finite={f0 and f1}")
assert e0 <= 2.5e-3 and e1 <= 2.5e-3 and f0 and f1
print("PASS")
