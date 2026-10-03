"""Repro R3-03: audit_normalization (the pipeline's pre-training normalisation gate) returns std = inf on float16 X.

Expected: the audit reports the same per-feature statistics whether X is stored float32 or float16 (the storage type is
documented as an invisible memory optimisation; upcasting happens "before any arithmetic").
Actual: numpy reduces float16 in float16: v.std() overflows (sum of squares > 65504) and returns inf for every feature with
more than a few thousand values; the verdict column then reads "dispersion too high" for features that are fine. Real data
has ~14M values per feature, so on the real 1h+4h file the gate is unusable (all std = inf) unless the caller upcasts.

Uses the real shipped preset (HOURLY_4H_OVERRIDES, float16), synthetic 6-coin universe, stride 1 to reach a realistic size. ~15 s.
Run: python docs/research/audit/r3_03_audit_normalization_float16_std_inf.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _r3_synth import *  # noqa: E402,F401,F403

warnings.filterwarnings("ignore")
ns = make_ns(1)
coins, k15, start, end = universe()
A = run(ns, coins, k15, start, end)
X, fo = A["X_1h"], A["feature_order"]
assert X.dtype == np.float16
t16 = ns["audit_normalization"](X, fo, verbose=False).set_index("feature")
t32 = ns["audit_normalization"](X.astype("float32"), fo, verbose=False).set_index("feature")
n_inf = int(np.isinf(t16["std"]).sum())
flag = int((t16["verdict"] != t32["verdict"]).sum())
print(f"X_1h {X.shape} float16: std=inf for {n_inf}/{len(t16)} features; verdict differs from the float32 audit for {flag}")
print(pd.concat({"std_f16": t16["std"], "std_f32": t32["std"], "verdict_f16": t16["verdict"],
                 "verdict_f32": t32["verdict"]}, axis=1).head(6).to_string())
assert n_inf == 0 and flag == 0, "DEFECT: audit_normalization on float16 X gives std=inf / different verdicts than on float32"
print("PASS")
