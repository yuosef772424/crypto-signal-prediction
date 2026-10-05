"""R3-00c (clean check): the 1h and 4h arrays are gathered with the SAME sample indices everywhere, and float16 X is upcast
before the model does any arithmetic.

Method: every row carries a signature: X_1h[i] == i, X_4h[i] == i + 0.5 (float16, exact below 1024), y[i] == i, last_candles[i] == i.
The REAL functions are exercised: main.ipynb's make_shuffled_dataset (train batches), val dataset builder pattern via model_x,
_concat_splits, pool_test_dict (chicks pooled batch), and cross_asset.PanelSplit (batches, k-coin chunk_groups, group batches,
subset, take). Any batch where X_1h != X_4h - 0.5, or X differs from y/last_candles/idx, is a misalignment.
Model side: a real two-branch model called with float16 dict input must equal the float32-upcast call bit for bit, a tuple in
model-input order equals the dict; swapping the tuple order changes outputs (so order matters and callers must keep it).
Run: python docs/research/audit/r3_00c_gather_alignment.py     (~30 s, needs tensorflow)
"""
import ast
import contextlib
import io
import json
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402

# ── real functions from main.ipynb's code (now the workflow/ package; extracted by name, not copied) ──
src = "\n".join(open(os.path.join(ROOT, "workflow", f"{m}.py"), encoding="utf-8").read()
                for m in ("splits", "batches", "selective_eval", "pooling"))
want = {"_tfs_of", "model_x", "make_shuffled_dataset", "_to_float32_inputs", "_concat_splits", "_pool_by_asset",
        "pool_test_dict", "_split_members"}
fs = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name in want]
assert {f.name for f in fs} == want, want - {f.name for f in fs}
ns = {"np": np, "pd": pd, "tf": tf, "MODEL_TFS": ["1h", "4h"], "MODEL_TF": "1h"}
exec(compile(ast.Module(body=fs, type_ignores=[]), "main_extract", "exec"), ns)


def split(off, n):
    idx = np.arange(off, off + n)
    return {"X_1h": np.broadcast_to(idx.astype("float32")[:, None, None], (n, 32, 3)).astype("float16"),
            "X_4h": np.broadcast_to((idx + 0.5).astype("float32")[:, None, None], (n, 32, 3)).astype("float16"),
            "y": {"y_high_reg": idx.astype("float32")}, "last_candles": np.stack([idx] * 7, 1).astype("float64"),
            "base_params": np.stack([idx, idx], 1).astype("float32")}


bad, total = 0, 0
tr = split(0, 400)
ds = ns["make_shuffled_dataset"](ns["model_x"](tr), {"y_high_reg": tr["y"]["y_high_reg"]}, 64, seed=1)
for x, y in ds.take(6):
    a, b, c = x["1h"].numpy()[:, 0, 0], x["4h"].numpy()[:, 0, 0], y["y_high_reg"].numpy()
    total += 1
    bad += int(not (np.array_equal(a, b - 0.5) and np.array_equal(a, c) and x["1h"].dtype == tf.float32))
print(f"make_shuffled_dataset (train batches, float16 -> float32): {total} batches, misaligned {bad}")
test = {"A": split(400, 300), "B": split(700, 250)}
m, assets = ns["_concat_splits"](test, ["1h", "4h"])
ok = (np.array_equal(m["X_1h"][:, 0, 0], m["X_4h"][:, 0, 0] - 0.5) and np.array_equal(m["X_1h"][:, 0, 0], m["last_candles"][:, 0]))
print("_concat_splits (eval / collect_signals / signals export):", "aligned" if ok else "MISALIGNED")
bad += int(not ok)
td = {a: {**s, "y": {"high": s["y"]["y_high_reg"]}} for a, s in test.items()}
pooled, yt = ns["pool_test_dict"](td, ["1h", "4h"], n_per_asset=50)
ok = (np.array_equal(pooled["X_1h"][:, 0, 0], pooled["X_4h"][:, 0, 0] - 0.5)
      and np.array_equal(pooled["X_1h"][:, 0, 0], pooled["last_candles"][:, 0]) and np.array_equal(pooled["X_1h"][:, 0, 0], yt["y_high"]))
print("pool_test_dict (chicks pooled batch):", "aligned" if ok else "MISALIGNED")
bad += int(not ok)

from cross_asset.data import PanelSplit, _concat_dict  # noqa: E402
X, y, lc, assets = _concat_dict(test, ["1h", "4h"])
y = {"y_high_class": (y["y_high_reg"] > 0).astype("float32"), "y_high_reg": y["y_high_reg"], "y_low_class": np.ones(len(lc), "float32"),
     "y_low_reg": y["y_high_reg"], "y_close_class": np.ones(len(lc), "float32"), "y_close_reg": y["y_high_reg"]}
day_ns = 8 * 3600 * 10 ** 9
lc2 = lc.copy()
lc2[:, 3] = (np.arange(len(lc)) // 5) * day_ns
ps = PanelSplit(X, y, lc2, assets, "t", day_ns=day_ns)


def chk(b):
    a, c = b["x"]["1h"][:, 0, 0], b["x"]["4h"][:, 0, 0]
    return bool(np.array_equal(a, c - 0.5) and np.array_equal(a, b["idx"] + 400.0) and np.array_equal(b["yreg"][:, 0], a)
                and np.array_equal(b["ycls"][:, 0], (a > 0).astype("float32")) and b["x"]["1h"].dtype == np.float32)


n = nbad = 0
for b in ps.iter_batches(64, shuffle=True, seed=0, epoch=0, max_coins=3, min_coins=2):
    n += 1; nbad += not chk(b)
for b in ps.iter_batches(64):
    n += 1; nbad += not chk(b)
mem, sc = ps.chunk_groups(k=3, seed=0)
for b in ps.iter_group_batches(mem, 64, scored=sc):
    n += 1; nbad += not chk(b)
sub = ps.subset(last_days=10)
for b in sub.iter_batches(64):
    n += 1; nbad += not (np.array_equal(b["x"]["1h"][:, 0, 0], b["x"]["4h"][:, 0, 0] - 0.5) and np.array_equal(b["yreg"][:, 0], b["x"]["1h"][:, 0, 0]))
print(f"PanelSplit (train shuffled+k-coin sampling, val/test order, chunk_groups k=3, subset): {n} batches, misaligned {nbad}")
bad += nbad

# ── model side ──
import _nbload  # noqa: E402
mv = _nbload.load_model({"__name__": "mv"})        # package model/ (ex model_v2) without its import-time self test
model = mv["build_model_fn"]({"1h": 32, "4h": 32}, {"1h": 43, "4h": 43}, config=dict(mv["ANTI_MEMORIZATION_CONFIG"]))
rng = np.random.default_rng(0)
x1 = rng.normal(0, 1, (64, 32, 43)).clip(-5, 5).astype("float16")
x4 = rng.normal(0, 1, (64, 32, 43)).clip(-5, 5).astype("float16")
o16 = model.predict({"1h": x1, "4h": x4}, verbose=0)
o32 = model.predict({"1h": x1.astype("float32"), "4h": x4.astype("float32")}, verbose=0)
ot = model.predict((x1, x4), verbose=0)
osw = model.predict((x4, x1), verbose=0)
d_up = max(float(np.abs(o16[k] - o32[k]).max()) for k in o16)
d_tup = max(float(np.abs(ot[k] - o16[k]).max()) for k in o16)
d_sw = max(float(np.abs(osw[k] - o16[k]).max()) for k in o16)
print(f"model inputs {[i.name for i in model.inputs]}: float16 vs upcast maxdiff {d_up}, tuple(1h,4h) vs dict {d_tup}, "
      f"tuple(4h,1h) vs dict {d_sw:.3f} (order matters)")
assert d_up == 0.0 and d_tup == 0.0 and d_sw > 0
assert bad == 0, f"DEFECT: {bad} misaligned gathers"
print("PASS")
