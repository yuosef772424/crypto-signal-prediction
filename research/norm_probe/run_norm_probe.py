"""
PURPOSE:  E-normprobe-1. CPU-only static probe of the INPUT normalization of 32-bar H/L/C windows: V0 current
          (window_scale, robust), V1 log-returns + global Min-Max, V2 + Standardization, V3 / trailing sigma clip 5,
          V4 = V2 clipped at 5, V5 train-fitted quantile transform. Per variant: effective rank, near-duplicate share,
          NN-distance distribution, top-1/top-5 PCA variance, ridge/logistic linear probes with month-block bootstrap.
TAGS:     input normalization, log returns, min-max, standardization, effective rank, near duplicates, linear probe,
          representation collapse, bootstrap month blocks, E-normprobe-1
PITFALLS: All statistics (Min-Max range, mean/std, quantiles) are fitted on bars with time <= TRAIN_END only. A window is
          dropped for ALL variants if bars t-47..t+1 touch a gap/NaN or a flat zero-volume run >= 168 h. Code package
          modules are NOT imported (they run in one shared namespace): V0 re-implements data/normalize.py
          process_windows for H/L/C and tests/test_norm_probe.py pins it to the real function.

Usage:
    python research/norm_probe/run_norm_probe.py --data-dir /tmp/claude-0/data/h1 --out research/norm_probe/results
    python research/norm_probe/run_norm_probe.py --synthetic --out /tmp/x      # GARCH-like heavy tails, smoke run
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

T = 32                      # window length: data/defaults.py window_sizes
STRIDE = 4
FLAT_RUN = 168              # hours
NEED_BACK = 47              # bars before the window end that must be clean (window 32 + prev close + 14 sigma bars)
SIGMA_N = 14
SIGMA_FLOOR = 1e-4
CLIP = 5.0
TRAIN_END = pd.Timestamp("2024-12-31 23:00", tz="UTC")     # last train target bar
VAL_START = pd.Timestamp("2025-01-02 00:00", tz="UTC")     # first val window end (48 h embargo)
END = pd.Timestamp("2026-09-30 23:00", tz="UTC")
GRID_START = pd.Timestamp("2022-01-01 00:00", tz="UTC")
VARIANTS = ["V0", "V1", "V2", "V3", "V4", "V5"]
TARGETS = ["high", "low", "close"]
SCALE_REL_FLOOR, SCALE_ABS_FLOOR = 1e-3, 1e-8              # data/normalize.py:231 and :241


# ───────────────────────────── data ─────────────────────────────
def load_panel(data_dir, coins=None):
    """{coin: DataFrame(open high low close volume) on the common hourly grid (NaN where the bar is missing)}."""
    grid = pd.date_range(GRID_START, END, freq="1h")
    out = {}
    for f in sorted(glob.glob(os.path.join(data_dir, "ohlc_*_1h.parquet"))):
        coin = os.path.basename(f).split("_")[1].replace("USDT", "")
        if coins and coin not in coins:
            continue
        d = pd.read_parquet(f)
        d.index = pd.DatetimeIndex(d.index).tz_convert("UTC") if d.index.tz else pd.DatetimeIndex(d.index).tz_localize("UTC")
        out[coin] = d[~d.index.duplicated()].reindex(grid)
    return out


def synthetic_panel(n_coins=4, seed=0):
    """GARCH(1,1)-like heavy-tailed hourly OHLCV on the common grid (for tests and smoke runs; NOT used for results)."""
    rng = np.random.default_rng(seed)
    grid = pd.date_range(GRID_START, END, freq="1h")
    n = len(grid)
    out = {}
    for k in range(n_coins):
        z = rng.standard_t(4, n) / np.sqrt(2.0)
        r = np.zeros(n)
        h = 1e-4
        for i in range(1, n):
            h = 2e-6 + 0.08 * r[i - 1] ** 2 + 0.9 * h
            r[i] = np.sqrt(h) * z[i]
        c = 100.0 * np.exp(np.cumsum(r))
        o = np.r_[c[0], c[:-1]]
        sp = np.sqrt(h if False else 1e-4) * np.abs(rng.standard_normal(n))
        hi = np.maximum(o, c) * np.exp(sp * np.abs(rng.standard_t(4, n)) * 0.5)
        lo = np.minimum(o, c) * np.exp(-sp * np.abs(rng.standard_t(4, n)) * 0.5)
        out[f"S{k}"] = pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c,
                                     "volume": rng.gamma(2.0, 100.0, n)}, index=grid)
    return out


def flat_zero_volume_mask(df, min_run=FLAT_RUN):
    """True for bars inside a run (>= min_run consecutive hours) of flat (high == low) zero-volume bars."""
    flat = ((df["volume"].to_numpy() == 0) & (df["high"].to_numpy() == df["low"].to_numpy())).astype(np.int8)
    m = np.zeros(len(flat), dtype=bool)
    i, n = 0, len(flat)
    while i < n:
        if flat[i]:
            j = i
            while j < n and flat[j]:
                j += 1
            if j - i >= min_run:
                m[i:j] = True
            i = j
        else:
            i += 1
    return m


def bar_returns(H, L, C):
    """Per-bar log returns relative to the PREVIOUS close: columns [high, low, close]; row 0 is NaN."""
    pc = np.r_[np.nan, C[:-1]]
    return np.stack([np.log(H / pc), np.log(L / pc), np.log(C / pc)], axis=1)


def trailing_sigma(rc, n=SIGMA_N, floor=SIGMA_FLOOR):
    """sigma[t] = std of the n close log-returns strictly BEFORE bar t (past-only); NaN until n values exist."""
    s = pd.Series(rc).shift(1).rolling(n).std().to_numpy()
    return np.where(np.isfinite(s), np.maximum(s, floor), np.nan)


# ───────────────────────────── transforms ─────────────────────────────
def v0_current(H, L, C):
    """The current method for the 3 price columns (data/normalize.py process_windows, kind price_level, robust):
    (x - median(close window)) / max(IQR(close window), |median| * 1e-3, 1e-8), constant columns -> 0, clip +-5.
    Inputs (N, T) each; output (N, T, 3) in the order [high, low, close]."""
    X = np.stack([H, L, C], axis=2)
    q75, q50, q25 = np.percentile(C, [75, 50, 25], axis=1)
    s = np.maximum(np.maximum(q75 - q25, np.abs(q50) * SCALE_REL_FLOOR), SCALE_ABS_FLOOR)
    out = (X - q50[:, None, None]) / s[:, None, None]
    const = X.max(axis=1) == X.min(axis=1)                       # (N, 3)
    out = np.where(const[:, None, :], 0.0, out)
    return np.clip(out, -CLIP, CLIP)


class MinMax:
    name = "V1"

    def fit(self, bars):
        self.lo, self.hi = bars.min(axis=0), bars.max(axis=0)
        return self

    def transform(self, R, S=None):
        return (R - self.lo) / (self.hi - self.lo)


class Standard:
    name = "V2"

    def __init__(self, clip=None):
        self.clip = clip

    def fit(self, bars):
        self.mu, self.sd = bars.mean(axis=0), bars.std(axis=0)
        return self

    def transform(self, R, S=None):
        z = (R - self.mu) / self.sd
        return np.clip(z, -self.clip, self.clip) if self.clip else z


class SigmaScaled:
    name = "V3"

    def fit(self, bars):
        return self

    def transform(self, R, S):
        return np.clip(R / S[:, :, None], -CLIP, CLIP)


class Quantile:
    name = "V5"
    LEVELS = np.linspace(0.0, 1.0, 1001)

    def fit(self, bars):
        self.q = np.quantile(bars, self.LEVELS, axis=0)
        return self

    def transform(self, R, S=None):
        out = np.empty_like(R)
        for j in range(R.shape[2]):
            xp, fp = self.q[:, j], self.LEVELS
            keep = np.r_[True, np.diff(xp) > 0]               # strictly increasing knots for interp
            out[:, :, j] = np.interp(R[:, :, j], xp[keep], fp[keep])
        return out


# ───────────────────────────── windows ─────────────────────────────
def build_samples(panel):
    """Window ends on the shared stride grid; one common valid-sample set for ALL variants.
    Returns dict of arrays: coin id, end index e, split label, per-window raw OHLC (N,T), returns (N,T,3), sigma (N,T),
    targets, month label."""
    recs = []
    coins = list(panel)
    for ci, coin in enumerate(coins):
        d = panel[coin]
        H, L, C = (d[k].to_numpy(float) for k in ("high", "low", "close"))
        times = d.index
        bad = ~(np.isfinite(H) & np.isfinite(L) & np.isfinite(C) & (C > 0) & (H > 0) & (L > 0))
        bad |= flat_zero_volume_mask(d.fillna({"volume": 1.0}))
        cum = np.r_[0, np.cumsum(bad)]
        n = len(d)
        e = np.arange(NEED_BACK, n - 1)
        e = e[(cum[e + 2] - cum[e - NEED_BACK]) == 0]          # bars e-47 .. e+1 clean
        hour = ((times[e] - GRID_START) / pd.Timedelta("1h")).astype(int)
        e = e[hour % STRIDE == 0]
        t_end, t_tgt = times[e], times[e + 1]
        split = np.where(t_tgt <= TRAIN_END, "train", np.where((t_end >= VAL_START) & (t_tgt <= END), "val", "skip"))
        R = bar_returns(np.where(bad, np.nan, H), np.where(bad, np.nan, L), np.where(bad, np.nan, C))
        S = trailing_sigma(R[:, 2])
        idx = e[:, None] + np.arange(-(T - 1), 1)[None, :]
        recs.append(dict(
            coin=np.full(len(e), ci), e=e, split=split, month=np.asarray(t_end.strftime("%Y-%m")),
            Hw=H[idx], Lw=L[idx], Cw=C[idx], Rw=R[idx], Sw=S[idx],
            y_reg=np.stack([H[e + 1] / H[e] - 1, L[e + 1] / L[e] - 1, C[e + 1] / C[e] - 1], axis=1),
            y_cls=np.stack([H[e + 1] > H[e], L[e + 1] > L[e], C[e + 1] > C[e]], axis=1).astype(float),
            bars_ok=None))
    cat = {k: np.concatenate([r[k] for r in recs]) for k in recs[0] if k != "bars_ok"}
    cat["coins"] = coins
    cat["bars"] = {}
    for ci, coin in enumerate(coins):                              # train bars for fitting (all clean bars <= TRAIN_END)
        d = panel[coin]
        H, L, C = (d[k].to_numpy(float) for k in ("high", "low", "close"))
        bad = ~(np.isfinite(H) & np.isfinite(L) & np.isfinite(C)) | flat_zero_volume_mask(d.fillna({"volume": 1.0}))
        R = bar_returns(np.where(bad, np.nan, H), np.where(bad, np.nan, L), np.where(bad, np.nan, C))
        m = (d.index <= TRAIN_END) & np.isfinite(R).all(axis=1)
        cat["bars"][coin] = R[m]
    cat["train_bars"] = np.concatenate(list(cat["bars"].values()))
    keep = cat["split"] != "skip"
    for k in list(cat):
        if isinstance(cat[k], np.ndarray) and len(cat[k]) == len(keep) and k != "train_bars":
            cat[k] = cat[k][keep]
    ok = np.isfinite(cat["Rw"]).all(axis=(1, 2)) & np.isfinite(cat["Sw"]).all(axis=1)
    for k in list(cat):
        if isinstance(cat[k], np.ndarray) and len(cat[k]) == len(ok) and k != "train_bars":
            cat[k] = cat[k][ok]
    return cat


def make_inputs(smp, train_bars=None):
    """{variant: (N, T*3) float64}; fitted on train bars only. Returns (inputs, fitted transforms)."""
    train_bars = smp["train_bars"] if train_bars is None else train_bars
    tf = {"V1": MinMax().fit(train_bars), "V2": Standard().fit(train_bars),
          "V4": Standard(clip=CLIP).fit(train_bars), "V5": Quantile().fit(train_bars), "V3": SigmaScaled()}
    X = {"V0": v0_current(smp["Hw"], smp["Lw"], smp["Cw"])}
    for v in ("V1", "V2", "V3", "V4", "V5"):
        X[v] = tf[v].transform(smp["Rw"], smp["Sw"])
    return {v: x.reshape(len(x), -1) for v, x in X.items()}, tf


# ───────────────────────────── geometry metrics ─────────────────────────────
def eff_rank_ratio(X):
    """participation ratio (sum ev)^2 / sum ev^2 of the covariance, / dim: the definition of model/diagnostics.py:201."""
    c = np.cov(X, rowvar=False)
    ev = np.maximum(np.linalg.eigvalsh(c), 0.0)
    return float((ev.sum() ** 2 / max((ev ** 2).sum(), 1e-30)) / len(ev)), ev


def geometry(X):
    pr, ev = eff_rank_ratio(X)
    sd = X.std(axis=0)
    cor = np.cov((X - X.mean(0))[:, sd > 1e-12] / sd[sd > 1e-12], rowvar=False)
    evc = np.maximum(np.linalg.eigvalsh(cor), 0.0)
    pc = evc.sum() ** 2 / max((evc ** 2).sum(), 1e-30) / len(evc)
    p = ev / ev.sum()
    ent = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()) / len(ev))
    evs = np.sort(ev)[::-1]
    return dict(eff_rank_cov=pr, eff_rank_corr=float(pc), eff_rank_entropy=ent,
                top1_var=float(evs[0] / ev.sum()), top5_var=float(evs[:5].sum() / ev.sum()))


def nn_distances(X, coin, e, n_sub=15000, seed=0, excl=48, chunk=1500):
    """Nearest-neighbour Euclidean distance within a random subsample; same-coin windows closer than `excl` bars are
    not neighbours (overlapping windows would be trivially close)."""
    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(len(X), min(n_sub, len(X)), replace=False))
    A, c, t = X[sel], coin[sel], e[sel]
    sq = (A ** 2).sum(1)
    out = np.empty(len(A))
    for i in range(0, len(A), chunk):
        d2 = sq[i:i + chunk, None] + sq[None, :] - 2.0 * A[i:i + chunk] @ A.T
        bad = (c[i:i + chunk, None] == c[None, :]) & (np.abs(t[i:i + chunk, None] - t[None, :]) < excl)
        d2[bad] = np.inf
        out[i:i + chunk] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
    return out


def nn_summary(nn):
    med = float(np.median(nn))
    return dict(nn_median=med, nn_p10=float(np.percentile(nn, 10)), nn_p90=float(np.percentile(nn, 90)),
                nn_p10_over_med=float(np.percentile(nn, 10) / med), nn_p90_over_med=float(np.percentile(nn, 90) / med),
                near_dup_share=float((nn < 0.01 * med).mean()))


# ───────────────────────────── probes ─────────────────────────────
def spearman(a, b_rank):
    ra = rankdata(a)
    ra = ra - ra.mean()
    return float((ra * b_rank).sum() / np.sqrt((ra ** 2).sum() * (b_rank ** 2).sum()))


def auc(score, y):
    r = rankdata(score)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y > 0].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def r2(pred, y, base_mean):
    return float(1.0 - ((y - pred) ** 2).sum() / ((y - base_mean) ** 2).sum())


LAM_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def _ridge_fit(X, Y, lam_rel):
    mx, my = X.mean(0), Y.mean(0)
    Xc = X - mx
    G = Xc.T @ Xc
    W = np.linalg.solve(G + lam_rel * np.trace(G) / len(G) * np.eye(len(G)), Xc.T @ (Y - my))
    return mx, my, W


def _ridge_pred(m, X):
    return (X - m[0]) @ m[2] + m[1]


def probe_variant(Xtr, Xva, ytr_reg, ytr_cls, e_tr):
    """Ridge (3 reg targets) and logistic (3 class targets) fitted on train; regularization chosen on the last 20% of
    train (time order, 48 h gap). Returns val predictions {'high_reg': ..., 'high_cls': ...}, chosen settings."""
    cut = np.quantile(e_tr, 0.8)
    fit_m, tail_m = e_tr < cut - 48, e_tr >= cut
    preds, chosen = {}, {}
    scale = float(np.sqrt(((Xtr - Xtr.mean(0)) ** 2).mean()))
    for j, t in enumerate(TARGETS):
        y = ytr_reg[:, j]
        best = max(LAM_GRID, key=lambda lam: r2(_ridge_pred(_ridge_fit(Xtr[fit_m], y[fit_m, None], lam), Xtr[tail_m])[:, 0],
                                                y[tail_m], y[fit_m].mean()))
        m = _ridge_fit(Xtr, y[:, None], best)
        preds[f"{t}_reg"] = _ridge_pred(m, Xva)[:, 0]
        chosen[f"{t}_reg"] = best
        yc = ytr_cls[:, j]

        def tail_auc(C):
            lr = LogisticRegression(C=C, max_iter=300).fit(Xtr[fit_m] / scale, yc[fit_m])
            return auc(lr.decision_function(Xtr[tail_m] / scale), yc[tail_m])
        bestC = max(C_GRID, key=tail_auc)
        lr = LogisticRegression(C=bestC, max_iter=300).fit(Xtr / scale, yc)
        preds[f"{t}_cls"] = lr.decision_function(Xva / scale)
        chosen[f"{t}_cls"] = bestC
    return preds, chosen


METRIC_KEYS = [f"{t}_{k}" for t in TARGETS for k in ("reg_ic", "reg_r2", "cls_auc")]


def skill_all(preds, yva_reg, yva_cls, ranks_y, base_mean, idx=None):
    """All probe metrics on val rows `idx` (None = all). ranks_y: pre-ranked, centered y per target (full val) is used
    only when idx is None; with idx the ranks are recomputed."""
    out = {}
    for j, t in enumerate(TARGETS):
        p, y, yc = preds[f"{t}_reg"], yva_reg[:, j], yva_cls[:, j]
        pc = preds[f"{t}_cls"]
        if idx is not None:
            p, y, yc, pc = p[idx], y[idx], yc[idx], pc[idx]
        ry = rankdata(y)
        ry = ry - ry.mean()
        out[f"{t}_reg_ic"] = spearman(p, ry)
        out[f"{t}_reg_r2"] = r2(p, y, base_mean[j])
        out[f"{t}_cls_auc"] = auc(pc, yc)
    return out


def month_bootstrap(preds_by_v, yva_reg, yva_cls, months, base_mean, reps=300, seed=0):
    """Paired month-block bootstrap: same resampled months for every variant. Returns {variant: [metric dict per rep]}."""
    rng = np.random.default_rng(seed)
    um = np.unique(months)
    rows_of = {m: np.flatnonzero(months == m) for m in um}
    res = {v: [] for v in preds_by_v}
    for _ in range(reps):
        pick = rng.choice(um, len(um), replace=True)
        idx = np.concatenate([rows_of[m] for m in pick])
        for v, p in preds_by_v.items():
            res[v].append(skill_all(p, yva_reg, yva_cls, None, base_mean, idx))
    return res


# ───────────────────────────── driver ─────────────────────────────
def decide(point, boot):
    """Pre-registered rule (research/norm_probe/cards/E-normprobe-1.md)."""
    rows = {}
    for v in VARIANTS[1:]:
        d = {k: np.array([b[k] - b0[k] for b, b0 in zip(boot[v], boot["V0"])]) for k in ("high_reg_ic", "low_reg_ic")}
        disq = any(np.percentile(d[k], 97.5) < 0 for k in d)
        rows[v] = dict(disqualified=bool(disq), rank_val=point[v]["val"]["eff_rank_cov"],
                       dup_val=point[v]["val"]["near_dup_share"],
                       dup_below_v0=bool(point[v]["val"]["near_dup_share"] < point["V0"]["val"]["near_dup_share"]
                                         or point[v]["val"]["near_dup_share"] == point["V0"]["val"]["near_dup_share"] == 0.0))
    ok = {v: r for v, r in rows.items() if not r["disqualified"] and r["dup_below_v0"]
          and r["rank_val"] > point["V0"]["val"]["eff_rank_cov"]}
    pick = max(ok, key=lambda v: ok[v]["rank_val"]) if ok else "V0 (reject: no variant qualifies)"
    return dict(pick=pick, per_variant=rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n-coins", type=int, default=4)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--nn-sub", type=int, default=15000)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    panel = synthetic_panel(a.n_coins) if a.synthetic else load_panel(a.data_dir)
    smp = build_samples(panel)
    split = smp["split"]
    tr, va = split == "train", split == "val"
    print("coins", smp["coins"], "train", int(tr.sum()), "val", int(va.sum()), "train bars", len(smp["train_bars"]), flush=True)
    X, tf = make_inputs(smp)
    point, rows, preds_by_v, chosen = {}, [], {}, {}
    base_mean = smp["y_reg"][tr].mean(0)
    for v in VARIANTS:
        point[v] = {}
        for sp_name, m in (("train", tr), ("val", va)):
            g = geometry(X[v][m])
            g.update(nn_summary(nn_distances(X[v][m], smp["coin"][m], smp["e"][m], a.nn_sub)))
            point[v][sp_name] = g
        if v == "V1":
            point[v]["val"]["share_outside_01"] = float(((X[v][va] < 0) | (X[v][va] > 1)).mean())
            point[v]["train"]["share_outside_01"] = float(((X[v][tr] < 0) | (X[v][tr] > 1)).mean())
        if v in ("V0", "V3", "V4"):
            for sp_name, m in (("train", tr), ("val", va)):
                point[v][sp_name]["share_at_clip"] = float((np.abs(X[v][m]) >= CLIP - 1e-9).mean())
        preds_by_v[v], chosen[v] = probe_variant(X[v][tr], X[v][va], smp["y_reg"][tr], smp["y_cls"][tr], smp["e"][tr])
        point[v]["val"].update(skill_all(preds_by_v[v], smp["y_reg"][va], smp["y_cls"][va], None, base_mean))
        print(v, {k: round(x, 4) for k, x in point[v]["val"].items()}, f"{time.time() - t0:.0f}s", flush=True)
    boot = month_bootstrap(preds_by_v, smp["y_reg"][va], smp["y_cls"][va], smp["month"][va], base_mean, a.reps)
    # CSVs
    for v in VARIANTS:
        for sp_name in ("train", "val"):
            rows.append(dict(variant=v, split=sp_name, **point[v][sp_name]))
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "norm_probe_results.csv"), index=False)
    brow = []
    for v in VARIANTS:
        for k in METRIC_KEYS:
            x = np.array([b[k] for b in boot[v]])
            d = x - np.array([b[k] for b in boot["V0"]])
            brow.append(dict(variant=v, metric=k, point=point[v]["val"][k], lo=np.percentile(x, 2.5), hi=np.percentile(x, 97.5),
                             delta_vs_v0=point[v]["val"][k] - point["V0"]["val"][k], delta_lo=np.percentile(d, 2.5),
                             delta_hi=np.percentile(d, 97.5), delta_sd=d.std()))
    pd.DataFrame(brow).to_csv(os.path.join(a.out, "norm_probe_bootstrap.csv"), index=False)
    dec = decide(point, boot)
    meta = dict(coins=smp["coins"], n_train=int(tr.sum()), n_val=int(va.sum()), T=T, stride=STRIDE, reps=a.reps,
                synthetic=a.synthetic, chosen_regularization=chosen, decision=dec, seconds=time.time() - t0)
    json.dump(meta, open(os.path.join(a.out, "norm_probe_meta.json"), "w"), indent=1, default=float)
    print(json.dumps(dec, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
