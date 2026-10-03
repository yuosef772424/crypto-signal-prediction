"""Repro 01: panel_compare.csv "AUC high/low" is scored against a different label than the one trained.

Setup mirrors PANEL_PRESET="1h_s8": TARGET_MODE=None (pipeline 'return' labels: fut_high > last_high),
close suspended, groups = exact 8h timestamps. Predictions are a PERFECT score for the trained label.

Expected (after fix): cross_asset.report.summarize -> auc_high == 1.0, same as the AUC that
evaluate_k_coins computes from ps.ycls for the same rows.
Actual (commit c62b4f0): summarize re-derives cls_high as "above the group median" and reports < 1.0.

Run:  python docs/research/audit/repro_01_report_auc_label_mismatch.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from cross_asset.report import summarize  # noqa: E402

rng = np.random.default_rng(0)
H = 3600 * 10**9
n_ts, n_coins = 60, 40
ts = np.repeat(np.arange(n_ts) * 8 * H + 1_700_000_000 * 10**9 // (8 * H) * 8 * H, n_coins)
n = len(ts)
entry = 100 * np.exp(rng.normal(size=n))
last_high = entry * (1 + np.abs(rng.normal(0, .01, n)))
last_low = entry * (1 - np.abs(rng.normal(0, .01, n)))
mkt = np.repeat(rng.normal(0, .01, n_ts), n_coins)          # common market move per timestamp
fut_high = last_high * (1 + mkt + rng.normal(0, .01, n))
fut_low = last_low * (1 + mkt + rng.normal(0, .01, n))
fut_close = entry * (1 + mkt + rng.normal(0, .01, n))

trained_high = (fut_high > last_high).astype(int)             # pipeline y_high_class (return mode)
trained_low = (fut_low > last_low).astype(int)
df = pd.DataFrame({"asset": np.tile([f"C{i}" for i in range(n_coins)], n_ts), "timestamp": ts,
                   "entry": entry, "last_high": last_high, "last_low": last_low,
                   "fut_close": fut_close, "fut_high": fut_high, "fut_low": fut_low,
                   "p_up_high": trained_high.astype(float), "p_up_low": trained_low.astype(float),
                   "mu_high": 0.0, "mu_low": 0.0})

m, _ = summarize(df, df, group_ns=8 * H)
auc_trained = roc_auc_score(trained_high, df["p_up_high"])   # what evaluate_k_coins reports (ps.ycls)
print(f"AUC high vs trained label (evaluate_k_coins / panel_k_coins.csv): {auc_trained:.4f}")
print(f"AUC high reported by report.summarize (panel_compare.csv):         {m['auc_high']:.4f}")
print(f"AUC low  reported by report.summarize (panel_compare.csv):         {m['auc_low']:.4f}")
assert abs(m["auc_high"] - auc_trained) < 1e-9, "DEFECT: panel_compare AUC uses a different label than training"
print("PASS")
