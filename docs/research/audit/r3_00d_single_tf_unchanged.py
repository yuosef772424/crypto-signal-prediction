"""R3-00d (clean check): single-timeframe (and legacy 2-timeframe) pipeline output is byte-identical to the commit before the
two-timeframe change (4cb2d7b), built from the old notebook pulled out of git, NOT from tests/golden_pre_multi_tf.json.

Compares, for four configurations on the same synthetic universe: every key of the dataset (arrays byte-for-byte, dtype and
shape, scalars by repr), the split_data output (X, y, last_candles, base_params), and embargo_candles.
Run: python docs/research/audit/r3_00d_single_tf_unchanged.py     (~40 s)
"""
import contextlib
import hashlib
import io
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import _nbload  # noqa: E402
from _r2_synth import make_coin_15m, to_1h, write_archives  # noqa: E402

os.chdir(_nbload.ROOT)
OLD = "4cb2d7b"
old_nb = os.path.join(tempfile.mkdtemp(), "old_pipeline.ipynb")
with open(old_nb, "wb") as f:
    f.write(subprocess.check_output(["git", "show", f"{OLD}:crypto_data_pipeline_v6.ipynb"], cwd=_nbload.ROOT))
skip = ("اختبارات ذاتية لتعديلات هذا الدفتر",)
old = _nbload.load_notebook(old_nb, skip_contains=skip, ns={"RUN_HOURLY_4H_SELFTESTS": False})
new = _nbload.load_pipeline()

rng = np.random.default_rng(21)
coins = ["BTCUSDT", "ETHUSDT", "AUSDT", "BUSDT", "CUSDT"]
k15 = {c: make_coin_15m(rng, "2025-03-01", "2025-05-01", s0=50 * (i + 1)) for i, c in enumerate(coins)}
h1 = {c: to_1h(k) for c, k in k15.items()}
root = tempfile.mkdtemp()
write_archives(root, coins, k15, "2025-03-01", "2025-05-01", np.random.default_rng(3))


def digest(ds):
    h = hashlib.sha256()
    for k in sorted(ds):
        v = ds[k]
        h.update(k.encode())
        if isinstance(v, np.ndarray):
            h.update(np.ascontiguousarray(v).tobytes())
            h.update(str((v.dtype, v.shape)).encode())
        else:
            h.update(repr(v).encode())
    return h.hexdigest()[:16]


def build(ns, over, phase2):
    C = ns["CONFIG"]
    ns["apply_hourly_preset"](over)
    ns["_PHASE2_CACHE"].clear()
    C["phase2_data"]["data_root"] = root
    C["phase2_data"]["use_intraday_15m"] = C["phase2_data"]["use_futures_metrics"] = "auto" if phase2 else False
    C["feature_order"] = None
    if not phase2:
        C["funding_rate"]["enabled"] = C["open_interest"]["enabled"] = False
    ns["refresh_features"]()
    with contextlib.redirect_stdout(io.StringIO()):
        ds = ns["build_dataset_from_loader"]([{"name": c} for c in coins], lambda f, n: h1[n].copy(),
                                             ns["make_resample_fn"](C), max_workers=1, config=C)
    C["split_dates"] = {"train_end": "2025-04-01", "val_end": "2025-04-15"}
    C["holdout_start"] = None
    C["min_split_samples"] = 1
    return ds, C


def split_digest(ns, ds, C):
    with contextlib.redirect_stdout(io.StringIO()):
        parts = ns["split_data"](ds, config=C)
    h = hashlib.sha256()
    for p in parts:
        for m in ([p] if "y" in p else list(p.values())):
            for k in sorted(k for k in m if k.startswith("X_")):
                h.update(np.ascontiguousarray(m[k]).tobytes())
            for k in ("last_candles", "base_params"):
                h.update(np.ascontiguousarray(m[k]).tobytes())
            for yk in sorted(m["y"]):
                h.update(np.ascontiguousarray(m["y"][yk]).tobytes())
    return h.hexdigest()[:16]


S8 = dict(new["HOURLY_W32_S8_OVERRIDES"])
fails = []
for label, over, p2 in [("1h_s8 single, 43 features", S8, True), ("1h_s8 single, no phase 2", S8, False),
                        ("legacy 1h+4h (4h window 6)", {**S8, "tf_order": ["1h", "4h"], "base_tf": "1h", "window_sizes": {"1h": 32, "4h": 6}}, True),
                        ("single 1h stride 1", {**S8, "stride": 1}, False)]:
    o, Co = build(old, over, p2)
    n, Cn = build(new, over, p2)
    same_ds = digest(o) == digest(n)
    same_sp = split_digest(old, o, Co) == split_digest(new, n, Cn)
    same_em = old["embargo_candles"](o, Co) == new["embargo_candles"](n, Cn)
    print(f"{label:<28} N={len(o['base_params']):>5} dataset identical={same_ds} (keys equal={set(o) == set(n)}) split identical={same_sp} "
          f"embargo {old['embargo_candles'](o, Co)}={new['embargo_candles'](n, Cn)} X dtype {o['X_1h'].dtype}/{n['X_1h'].dtype}")
    if not (same_ds and same_sp and same_em):
        fails.append(label)
assert not fails, f"DEFECT: output changed vs {OLD}: {fails}"
print("PASS")
