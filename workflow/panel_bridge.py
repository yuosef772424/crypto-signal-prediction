"""
PURPOSE:  Section 7-h glue for the cross-asset panel model: asset names of a split and the current model's signals as the panel
          experiment's baseline (lifted out of the notebook cell, which keeps only the config and the run_panel_experiment call).
TAGS:     panel_names, panel_baseline, PANEL_BASELINE, panel model, cross_asset, run_panel_experiment, baseline signals
PITFALLS: panel_names needs split_asset_names (market_neutral module, section 7-f): run that cell first. The baseline must use the
          same split_dates and TARGET_MODE as the panel run. Executed into the notebook's shared namespace by workflow/_loader.py
          (never imported on its own). Lifted from the nested defs of main.ipynb section 7-h (cell 39).
"""
import os

import pandas as pd


def panel_names(dataset, which):
    """Asset name per sample of split ``which`` ("train"/"val"/"test"), or None when the data has no asset_bounds / another split mode."""
    try:
        return split_asset_names(dataset, which)
    except Exception as e:   # بيانات بلا asset_bounds أو وضع تقسيم آخر
        print(f"ℹ️ تعذّرت أسماء عملات {which}: {e}")
        return None


def panel_baseline(src, model, model_builder, dataset, val, test, tfs):
    """إشارات النموذج الحالي على نفس val/test (نفس أعمدة collect_signals + أسماء العملات).
    src: None | "model" (النموذج الحالي) | مسار best.weights.h5 | مجلد فيه signals_{val,test}.csv.gz."""
    if src is None:
        return None
    if isinstance(src, str) and os.path.isdir(src):
        return {"baseline": tuple(pd.read_csv(os.path.join(src, f"signals_{s}.csv.gz")) for s in ("val", "test"))}
    m = model
    if isinstance(src, str) and src.endswith(".h5"):
        m = model_builder()
        m.load_weights(src)
    v = collect_signals(m, val, tfs)
    names = panel_names(dataset, "val")
    if names is not None and len(names) == len(v):
        v = v.assign(asset=names)
    return {"baseline": (v.assign(split="val"), collect_signals(m, test, tfs).assign(split="test"))}
