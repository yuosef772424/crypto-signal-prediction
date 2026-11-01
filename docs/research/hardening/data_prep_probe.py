"""data_prep_probe.py - what does the pipeline's per-window normalization destroy? (audit of data preparation, 2026-11-01)

Runs the REAL pipeline (data/, via docs/research/audit/_nbload.load_pipeline) on real 1h candles (tools/fetch_crypto_dataset.py output),
with the level side-channel ON so the same samples carry (a) the pipeline-normalized window the model sees and (b) the train-fitted level
columns LVL_*. Then:

  1. RETAINED: for each level quantity Q (known at entry), how much of Q can a linear model recover from the NORMALIZED window alone
     (test R^2)? Low = the normalization erased it.
  2. PREDICT: how well do the feature views predict the next-horizon magnitude (|return|, log range) and direction (test Spearman IC /
     AUC, pooled over coins, train < val < test in time)?
  3. COINS: how much of the cross-coin variation of the volatility level does each view carry?

    python docs/research/hardening/data_prep_probe.py --data-dir /tmp/realdata --out docs/research/hardening/data_prep_probe_results.json

Everything is fitted on train, scored on test; ridge penalties are chosen on val only. Not a model: a linear probe of information content
(the same instrument as failure F-0029's probe), so the numbers say what is IN each view, not what a network could extract.
"""
import argparse
import contextlib
import glob
import io
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))
sys.path.insert(0, ROOT)
import _nbload  # noqa: E402
from stats.metrics import auc, spearman  # noqa: E402

LEVEL = {"window": 32, "long_window": 256}
SPLITS = {"train_end": "2023-12-31", "val_end": "2024-12-31"}


def load_frames(data_dir, start, interval="1h"):
    frames = {}
    for p in sorted(glob.glob(os.path.join(data_dir, f"ohlc_*_{interval}.parquet"))):
        sym = os.path.basename(p)[len("ohlc_"):-len(f"_{interval}.parquet")]
        d = pd.read_parquet(p).sort_index()
        d = d[d.index >= pd.Timestamp(start, tz="UTC")]
        frames[sym] = d[["open", "high", "low", "close", "volume"]].astype("float64")
    if not frames:
        raise FileNotFoundError(f"no ohlc_*_{interval}.parquet in {data_dir}")
    return frames


def build(frames, horizon):
    ns = _nbload.load_pipeline()
    q = lambda f, *a, **k: _quiet(f, *a, **k)  # noqa: E731
    ns["reset_config"]()
    q(ns["apply_hourly_preset"], ns["HOURLY_W32_S8_OVERRIDES"])
    ns["update_config"]({"tf_order": ["1h"], "window_sizes": {"1h": 32}, "forecast_horizon": horizon, "level_channel": dict(LEVEL),
                         "holdout_start": None, "split_dates": dict(SPLITS), "disk_backed": False,
                         "phase2_data": {"use_intraday_15m": False, "use_futures_metrics": False},
                         "funding_rate": {"enabled": False}, "open_interest": {"enabled": False}})
    q(ns["refresh_features"])
    cfg = ns["CONFIG"]
    ds = q(ns["build_dataset"], [{"name": s} for s in frames], load_asset_fn=lambda fid, name: frames[name],
           resample_fn=ns["make_resample_fn"](cfg), max_workers=1, config=cfg)
    cfg = dict(cfg)
    split = q(ns["split_data"], ds, config=cfg)
    return ns, ds, split, cfg


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def ridge_fit(Xtr, ytr, Xva, yva, alphas=(1.0, 10.0, 100.0, 1e3, 1e4, 1e5)):
    """Standardize on train, ridge by closed form, alpha picked on val R^2 / correlation. Returns (predict_fn, alpha)."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A = (Xtr - mu) / sd
    ym = ytr.mean()
    G, b = A.T @ A, A.T @ (ytr - ym)
    best = None
    for a in alphas:
        w = np.linalg.solve(G + a * np.eye(G.shape[0]), b)
        pv = ((Xva - mu) / sd) @ w + ym
        score = -np.mean((yva - pv) ** 2)
        if best is None or score > best[0]:
            best = (score, a, w)
    _, a, w = best
    return (lambda X: ((X - mu) / sd) @ w + ym), a


def r2(y, p):
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def views(split, level_idx, other_idx):
    X = np.asarray(split["X_1h"], dtype="float64")
    non = X[:, :, other_idx]
    mu, sd = non.mean(1, keepdims=True), non.std(1, keepdims=True) + 1e-4
    z = (non - mu) / sd                                       # what the model's InstanceNorm hands the encoder
    stats = np.concatenate([non.mean(1), np.log(non.std(1) + 1e-4)], axis=1)       # the model's stats path
    return {
        "norm_last": non[:, -1, :],
        "norm_window": non.reshape(len(non), -1),
        "inst_z_window": z.reshape(len(z), -1),
        "inst_stats": stats,
        "level_last": X[:, -1, :][:, level_idx],
        "lvl_cols_window": X[:, :, level_idx].reshape(len(X), -1),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    frames = load_frames(a.data_dir, a.start)
    ns, ds, (train, val, test), cfg = build(frames, a.horizon)
    fo = ds["feature_order"]
    level_cols = ds["level_channel"]["columns"]
    level_idx = [fo.index(c) for c in level_cols]
    other_idx = [j for j in range(len(fo)) if fo[j] not in level_cols]
    test_assets = None
    if "y" not in test:                                               # per-asset dict -> one pooled test (+ the asset of every row)
        keys = sorted(test)
        test_assets = np.concatenate([np.full(len(test[s]["last_candles"]), s) for s in keys])
        test = {k: np.concatenate([np.asarray(test[s][k]) for s in keys]) for k in ("X_1h", "last_candles")}
    L = {"train": train, "val": val, "test": test}
    V = {k: views(v, level_idx, other_idx) for k, v in L.items()}

    def targets(sp):
        lc = np.asarray(sp["last_candles"], dtype="float64")
        last_c, fut_c, fut_lo, fut_hi = lc[:, 2], lc[:, 4], lc[:, 5], lc[:, 6]
        ret = np.log(fut_c / last_c)
        return {"abs_ret": np.log(np.abs(ret) + 1e-5), "log_range": np.log(np.log(fut_hi / fut_lo) + 1e-5), "up": (ret > 0).astype(float)}, lc

    T = {k: targets(v)[0] for k, v in L.items()}
    ts_test = pd.to_datetime(np.asarray(L["test"]["last_candles"])[:, 3].astype("int64"), utc=True)
    out = {"n": {k: int(len(v["norm_last"])) for k, v in V.items()}, "coins": list(frames), "features": len(other_idx), "level_cols": level_cols,
           "horizon": a.horizon, "splits": SPLITS, "n_norm_features": len(other_idx)}

    # 1) RETAINED: recover each level quantity (train-standardized, last step) from the normalized window only
    ret = {}
    for j, c in enumerate(level_cols):
        y = {k: V[k]["level_last"][:, j] for k in V}
        for name in ("norm_last", "norm_window", "inst_z_window", "inst_stats"):
            f, alpha = ridge_fit(V["train"][name], y["train"], V["val"][name], y["val"])
            ret.setdefault(c, {})[name] = r2(y["test"], f(V["test"][name]))
    out["retained_r2_test"] = ret

    # 2) PREDICT: magnitude and direction from each view (+ combinations)
    combos = {
        "norm_window": ["norm_window"], "inst_z_window": ["inst_z_window"], "inst_z+stats": ["inst_z_window", "inst_stats"],
        "level_last": ["level_last"], "norm_window+level": ["norm_window", "level_last"],
        "inst_z+stats+level": ["inst_z_window", "inst_stats", "level_last"],
    }
    pred = {}
    months = ts_test.to_period("M")
    for tname in ("abs_ret", "log_range", "up"):
        for cname, parts in combos.items():
            g = lambda s: np.concatenate([V[s][p] for p in parts], axis=1)  # noqa: E731
            f, alpha = ridge_fit(g("train"), T["train"][tname], g("val"), T["val"][tname])
            p = f(g("test"))
            ic = spearman(p, T["test"][tname])
            m = pd.DataFrame({"p": p, "y": T["test"][tname], "m": months}).groupby("m").apply(lambda d: spearman(d.p.to_numpy(), d.y.to_numpy()))
            row = {"ic": ic, "alpha": alpha, "months_pos": f"{int((m > 0).sum())}/{len(m)}", "n_test": int(len(p))}
            if tname == "up":
                row["auc"] = auc(T["test"][tname], p)
            else:
                row["r2"] = r2(T["test"][tname], p)
            pred.setdefault(tname, {})[cname] = row
    out["predict_test"] = pred

    # 3) PER FEATURE: kind, share of values at the +-clip bound, share of constant (zeroed) windows, and how much of the volatility level
    #    (LVL_logvol) the feature's own normalized window still carries (test R^2 of a ridge on that one feature's 32 steps)
    Xtr, Xva, Xte = (np.asarray(L[k]["X_1h"], dtype="float64") for k in ("train", "val", "test"))
    jv = level_cols.index("LVL_logvol")
    lv = {k: np.asarray(L[k]["X_1h"], dtype="float64")[:, -1, level_idx[jv]] for k in L}
    clip = float(ns["CONFIG"].get("clip_abs", ns["CLIP_ABS"]))
    per = {}
    for j in other_idx:
        name = fo[j]
        col = Xtr[:, :, j]
        f, _ = ridge_fit(col, lv["train"], Xva[:, :, j], lv["val"])
        per[name] = {"kind": ns["classify_feature"](name), "clipped_share": float((np.abs(col) >= clip - 1e-6).mean()),
                     "windows_with_clip": float((np.abs(col) >= clip - 1e-6).any(axis=1).mean()),
                     "constant_window_share": float((col.max(1) == col.min(1)).mean()),
                     "logvol_r2": r2(lv["test"], f(Xte[:, :, j]))}
    out["per_feature"] = per
    # exact-zero next-horizon return (close[t+h] == close[t]): the stale-bar problem, with CONFIG['stale_bars'] at its default (off)
    out["zero_return_share"] = {k: float((np.asarray(L[k]["last_candles"])[:, 4] == np.asarray(L[k]["last_candles"])[:, 2]).mean()) for k in L}
    # between-coin share of the volatility level in the test period (eta^2 of LVL_logvol on the asset) and per-coin mean vol (in train sd)
    if test_assets is not None:
        g = pd.DataFrame({"v": lv["test"], "a": test_assets})
        out["coin_logvol_eta2_test"] = float(g.groupby("a").v.transform("mean").var() / g.v.var())
        out["coin_logvol_mean_z_test"] = {k: float(v) for k, v in g.groupby("a").v.mean().items()}
    out["note"] = "pooled test; ridge penalties chosen on val; all statistics fitted on train"
    txt = json.dumps(out, indent=1, default=float)
    print(txt)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
