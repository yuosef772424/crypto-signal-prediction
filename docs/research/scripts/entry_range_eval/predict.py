"""Step 1 of the entry_range evaluation: build the main.ipynb single-coin architecture, load the trained
weights and dump raw model outputs for VAL and TEST only.

Why a separate script: main.ipynb is being edited by another agent, so the model code is executed read-only
from model_v2 (1).ipynb (same cells main.ipynb %run's) and the MODEL_OVERRIDES logic is re-stated here
(TARGET_MODE=entry_range => close head enabled, enforce_order off, ANTI_MEMORIZATION_CONFIG on).

Holdout safety: rows with timestamp >= holdout_start - embargo are dropped right after loading the pickle,
BEFORE any model call, and never written to disk. The split rule mirrors crypto_data_pipeline_v6
`_split_global_time` (embargo = window + horizon candles).

usage: python predict.py --data pre.pkl --weights best.weights.h5 --out /path/outputs.npz
"""
import argparse
import json
import os
import pickle
import re

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def load_model_v2():
    """Executes the code cells of model_v2 (1).ipynb, skipping its top-level self-test call."""
    nb = json.load(open(os.path.join(REPO, "model_v2 (1).ipynb"), encoding="utf-8"))
    ns = {"__name__": "model_v2"}
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        src = "\n".join(l for l in src.splitlines() if not re.match(r"^\s*(%|!)", l))
        src = re.sub(r"^run_model_selftests\(.*\)\s*$", "pass", src, flags=re.M)
        exec(compile(src, "model_v2_cell", "exec"), ns)
    return ns


def split_masks(ts, train_end, val_end, hold_start, gap):
    tr = ts <= train_end
    va = (ts > train_end + gap) & (ts <= val_end)
    te = (ts > val_end + gap) & (ts < hold_start - gap)
    return tr, va, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    d = pickle.load(open(a.data, "rb"))
    lc = np.asarray(d["last_candles"], dtype="float64")
    ts = pd.to_datetime(lc[:, 3].astype("int64"), utc=True)
    gap = pd.Timedelta(hours=d["window_sizes"]["1h"] + d["forecast_horizon"])
    hold = pd.Timestamp(d["holdout_start"], tz="UTC")
    keep = np.asarray(ts < hold - gap)                       # drop holdout (+ its embargo) first
    print("dropped holdout/embargo rows:", int((~keep).sum()), "kept:", int(keep.sum()))
    lc, ts = lc[keep], ts[keep]
    X = d["X_1h"][keep]
    bp = np.asarray(d["base_params"])[keep]                 # [center, scale] of the window close (robust scaler)
    assets = np.empty(len(d["last_candles"]), dtype=object)
    for b in d["asset_bounds"]:
        assets[b["start"]:b["end"]] = b["name"]
    assets = assets[keep]
    feat = d["feature_order"]
    scale = float(d["reg_target_scale"])
    tr_end = pd.Timestamp(d["split_dates"]["train_end"], tz="UTC")
    va_end = pd.Timestamp(d["split_dates"]["val_end"], tz="UTC")
    _, mva, mte = split_masks(ts, tr_end, va_end, hold, gap)
    del d

    ns = load_model_v2()
    cfg = dict(ns["ANTI_MEMORIZATION_CONFIG"])
    cfg.update(enforce_order=False, price_targets=("high", "low", "close"),
               head_types={t: ["nig_regression", "binary_classification"] for t in ("high", "low", "close")})
    model = ns["build_model_fn"](X.shape[1], X.shape[2], config=cfg)
    n_params = model.count_params()
    model.load_weights(a.weights)
    print("weights loaded OK; params:", n_params)

    out = {}
    for name, m in (("val", mva), ("test", mte)):
        idx = np.where(m)[0]
        pred = model.predict(X[idx], batch_size=2048, verbose=0)
        for k, v in pred.items():
            out[f"{name}__{k}"] = np.asarray(v, dtype="float32").reshape(len(idx), -1)[:, 0]
        out[f"{name}__ts"] = lc[idx, 3].astype("int64")
        out[f"{name}__lc"] = lc[idx]
        out[f"{name}__bp"] = bp[idx]
        out[f"{name}__asset"] = assets[idx].astype(str)
        # a few raw features used by baselines (last bar of the window): RSI_14, NATR_14 for vol proxy
        out[f"{name}__feat_last"] = X[idx, -1, :].astype("float32")
        out[f"{name}__feat_win"] = X[idx][:, :, [feat.index("RSI_14"), feat.index("NATR_14"), feat.index("close")]]
        print(name, len(idx))
    out["feature_order"] = np.array(feat)
    out["scale"] = np.array(scale)
    np.savez_compressed(a.out, **out)


if __name__ == "__main__":
    main()
