"""Repro R2-06: a split without the reg_target_scale stamp is silently read as scale 1.0 by the panel path, so every
exported return/price is 100x wrong (no error, IC/AUC unaffected, so nothing looks broken).

split_data does not carry metadata; only main.ipynb cell 8 stamps 'reg_target_scale' on train/val/test. cross_asset
(panel_split_from) defaults to 1.0 when the key is absent. That is the state whenever main.ipynb is older than cross_asset
(REPO_SOURCE="auto" in main cell 2 prefers a stale copy under /content/drive/MyDrive/crypto over the branch, while
cross_asset is cloned from the branch when missing there), or anyone builds PanelSplit from split_data output directly.
The pipeline output itself carries enough to detect this: y_{t}_reg / (future/last - 1) is the scale.

Expected (after fix): either an error at construction, or the scale is recovered from last_candles, so that an oracle
model exports mu_high == realised return and pred_high == realised next high.
Actual (b14b9bb): silent; mu_high is 100x the realised return, pred_high is off by a factor of ~2.

Run:  python docs/research/audit/r2_06_panel_scale_unstamped.py   (~3 s)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
import numpy as np  # noqa: E402

from cross_asset.data import panel_split_from  # noqa: E402
from cross_asset.train import export_signals  # noqa: E402

rng = np.random.default_rng(0)
n, SCALE = 60, 100.0
ts = 1_750_003_200 * 10**9 + (np.arange(n) // 6) * 8 * 3600 * 10**9      # 6 coins per 8h group
last_close = 100 + rng.normal(0, 1, n)
last_high, last_low = last_close * 1.004, last_close * 0.996
fut_high = last_high * (1 + rng.normal(0, 0.01, n))
fut_low = last_low * (1 + rng.normal(0, 0.01, n))
lc = np.stack([last_high, last_low, last_close, ts.astype("float64"), last_close * 1.001, fut_low, fut_high], 1)
y = {"y_high_reg": ((fut_high / last_high - 1) * SCALE).astype("float32"), "y_low_reg": ((fut_low / last_low - 1) * SCALE).astype("float32"),
     "y_high_class": (fut_high > last_high).astype("float32"), "y_low_class": (fut_low > last_low).astype("float32")}
split = {"X_1h": np.zeros((n, 4, 2), "float32"), "y": y, "last_candles": lc, "base_params": np.zeros((n, 2), "float32")}
# no split["reg_target_scale"]: exactly what split_data returns
try:
    ps = panel_split_from(split, "1h", None, "val", targets=("high", "low"), day_ns=8 * 3600 * 10**9)
    df = export_signals(ps, np.zeros((n, 2), "float32"), ps.yreg.astype("float32"), "val", targets=ps.targets)   # oracle mu
except (ValueError, RuntimeError, KeyError) as exc:
    print(f"PASS: refused loudly ({exc})")
    raise SystemExit(0)
realised = fut_high / last_high - 1
print(f"target_scale used = {ps.target_scale} | mu_high[0] = {df['mu_high'][0]:+.4f} vs realised {realised[0]:+.4f} | "
      f"pred_high[0] = {df['pred_high'][0]:.2f} vs fut_high {fut_high[0]:.2f}")
assert np.allclose(df["mu_high"], realised, atol=1e-6) and np.allclose(df["pred_high"], fut_high, rtol=1e-6), \
    "DEFECT: unstamped split silently treated as scale 1.0"
print("PASS")
