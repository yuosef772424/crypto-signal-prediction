"""
PURPOSE:  E-gate-001 run (phase G2): do gate-style derived multi-timeframe states (tools/gate_features.py, gate-style-v1,
          unverified) carry out-of-sample information on the 4h / 24h direction and on the range in NATR units of 10 coins?
          Logistic + HistGradientBoosting, k-sweep, learning curve, 20 label-shift nulls, old (raw indicators) vs new (states),
          annual out-of-sample refits, costs. Accept rule: research/gate_features/cards/E-gate-001.md (fixed before the run).
TAGS:     gate features, E-gate-001, G2, k-sweep, learning curve, label-shift null, old vs new feature set, run contract
PITFALLS: Reads data ONLY through tools/holdout_guard.load_unsealed (window end 2026-10-01). Unknown config keys raise. Stops at
          --max-minutes and then marks the manifest `truncated: true` (a truncated run is not evidence). Run from the repo root.

Usage:
    python research/gate_features/run_gate001.py --config research/gate_features/config_e_gate_001.json --out <dir> [--max-minutes 50]
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import gate_features as gf          # noqa: E402
import holdout_guard as hg          # noqa: E402

DEFAULTS = {
    "data_dir": None, "coins": ["ADAUSDT", "BCHUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "ETHUSDT", "SOLUSDT", "TRXUSDT", "XRPUSDT", "ZECUSDT"],
    "pattern": "ohlc_{symbol}_5m.parquet", "load_start": "2017-01-01", "end": "2026-10-01",
    "train_start": "2018-01-01", "train_end": "2023-01-01", "val_end": "2025-01-01", "compare_end": "2026-10-01",
    "horizons": [4, 24], "ks": [1, 3, 5, 10, 20, "all"], "n_null": 20, "train_stride": 4, "seed": 0, "cost_rt": 0.0014,
    "years": [2021, 2022, 2023, 2024, 2025], "taus": [0.25, 0.5, 0.75], "max_rows": 0,
}
QUANTILE_SHIFT_DAYS = (30, 300)


class Deadline(Exception):
    pass


class Clock:
    def __init__(self, minutes):
        self.t0, self.limit = time.time(), minutes * 60
        self.last = None

    def stage(self, name):
        print(f"[{self.minutes():6.2f} min] {name}", file=sys.stderr, flush=True)

    def check(self):
        if time.time() - self.t0 > self.limit:
            raise Deadline

    def minutes(self):
        return (time.time() - self.t0) / 60


def load_config(path):
    cfg = json.load(open(path))
    bad = set(cfg) - set(DEFAULTS)
    if bad:
        raise KeyError(f"unknown config keys: {sorted(bad)}")
    out = {**DEFAULTS, **cfg}
    if not out["data_dir"]:
        raise KeyError("data_dir is required")
    return out


# ------------------------------------------------------------------ dataset
def coin_frame(df5, horizons):
    """Decision table + targets for one coin (index = 1h bar open t; entry = open at t+1h)."""
    dt = gf.decision_table(df5)
    grid = pd.date_range(df5.index[0].floor("5min"), df5.index[-1], freq="5min")
    g = df5.reindex(grid)
    op, hi, lo = g["open"], g["high"], g["low"]
    entry_t = dt.index + pd.Timedelta(hours=1)
    entry = op.reindex(entry_t).to_numpy()
    dt["entry"] = entry
    for H in horizons:
        n = H * 12
        exit_ = op.reindex(entry_t + pd.Timedelta(hours=H)).to_numpy()
        dt[f"ret{H}"] = exit_ / entry - 1
        hmax = hi.rolling(n).max().shift(-(n - 1)).reindex(entry_t).to_numpy()
        lmin = lo.rolling(n).min().shift(-(n - 1)).reindex(entry_t).to_numpy()
        natr = dt["60__natr14"].to_numpy() / 100
        dt[f"mag{H}"] = (hmax - lmin) / entry / natr
    return dt


def build_dataset(cfg, clock):
    frames, files = [], []
    for sym in cfg["coins"]:
        df5 = hg.load_unsealed(cfg["data_dir"], sym, cfg["load_start"], cfg["end"], pattern=cfg["pattern"])
        files.append((f"{sym}", int(len(df5)), str(df5.index[0]), str(df5.index[-1])))
        f = coin_frame(df5, cfg["horizons"])
        f.insert(0, "coin", sym)
        frames.append(f.loc[f.index >= pd.Timestamp(cfg["train_start"], tz="UTC")])
        clock.check()
    d = pd.concat(frames)
    d = d.loc[d.index < pd.Timestamp(cfg["compare_end"], tz="UTC")]
    if cfg["max_rows"]:
        d = d.iloc[:cfg["max_rows"]]
    return d, files


def column_sets(d):
    st = [f"{tf}__{c}" for tf in gf.TIMEFRAMES for c in gf.STATE_COLUMNS] + ["overall_agg", "signal"]
    raw = [f"{tf}__{c}" for tf in gf.TIMEFRAMES for c in gf.RAW_COLUMNS]
    natr = [f"{tf}__{gf.NATR_COLUMN}" for tf in gf.TIMEFRAMES]
    return st, raw, natr


# ------------------------------------------------------------------ design matrices / ranking
def rank_columns(X, y):
    """Between-level R2 of y on each column (levels = values if <= 25 unique else 10 train quantile bins; NaN = own level)."""
    ym = y.mean()
    tot = ((y - ym) ** 2).sum()
    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        x = X[:, j]
        nan = np.isnan(x)
        u = np.unique(x[~nan])
        if len(u) <= 25:
            codes = np.searchsorted(u, np.where(nan, u[0] if len(u) else 0, x))
        else:
            edges = np.unique(np.quantile(x[~nan], np.linspace(0, 1, 11)[1:-1]))
            codes = np.searchsorted(edges, np.where(nan, edges[0], x))
        codes = np.where(nan, codes.max() + 1 if len(codes) else 0, codes)
        n = np.bincount(codes)
        s = np.bincount(codes, weights=y)
        ok = n > 0
        out[j] = (n[ok] * (s[ok] / n[ok] - ym) ** 2).sum() / tot if tot > 0 else 0
    return out


class Design:
    """Turns selected columns into a model matrix. 'state': one-hot with an explicit missing level (NaN is a level, not 0).
    'raw': train-median imputation + missing flags + standardization."""

    def __init__(self, kind, Xtr):
        self.kind = kind
        if kind == "state":
            self.enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
            self.enc.fit(np.nan_to_num(Xtr, nan=-99.0))
        else:
            self.med = np.nanmedian(Xtr, axis=0)
            filled = np.where(np.isnan(Xtr), self.med, Xtr)
            self.mu, self.sd = filled.mean(0), filled.std(0) + 1e-12
            self.flag = np.isnan(Xtr).any(0)

    def __call__(self, X):
        if self.kind == "state":
            return self.enc.transform(np.nan_to_num(X, nan=-99.0))
        nan = np.isnan(X)
        Z = (np.where(nan, self.med, X) - self.mu) / self.sd
        return np.hstack([Z, nan[:, self.flag].astype(float)]) if self.flag.any() else Z


def gbm_clf():
    return HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_depth=3, min_samples_leaf=500,
                                          l2_regularization=1.0, early_stopping=False, random_state=0)


def gbm_reg(tau):
    return HistGradientBoostingRegressor(loss="quantile", quantile=tau, max_iter=100, learning_rate=0.05, max_depth=3,
                                         min_samples_leaf=500, l2_regularization=1.0, early_stopping=False, random_state=0)


def fit_direction(model, kind, Xtr, ytr):
    """Returns predict(X) -> P(up)."""
    if model == "gbm":
        m = gbm_clf().fit(Xtr, ytr)
        return lambda X: m.predict_proba(X)[:, 1]
    des = Design(kind, Xtr)
    m = LogisticRegression(C=0.1, max_iter=300).fit(des(Xtr), ytr)
    return lambda X: m.predict_proba(des(X))[:, 1]


def pinball(y, q, tau):
    d = y - q
    return np.maximum(tau * d, (tau - 1) * d).mean()


def fit_magnitude(model, kind, Xtr, zy, taus, nat_tr=None):
    """Returns predict(X) -> (n, len(taus)) quantiles of y (NATR units). zy = log y on train."""
    if model == "gbm":
        ms = [gbm_reg(t).fit(Xtr, zy) for t in taus]
        return lambda X: np.exp(np.column_stack([m.predict(X) for m in ms]))
    des = Design(kind, Xtr)
    m = Ridge(alpha=10.0).fit(des(Xtr), zy)
    rq = np.quantile(zy - m.predict(des(Xtr)), taus)
    return lambda X: np.exp(m.predict(des(X))[:, None] + rq[None, :])


def mag_loss(y, Q, taus):
    return float(np.mean([pinball(y, Q[:, i], t) for i, t in enumerate(taus)]))


# ------------------------------------------------------------------ evaluation helpers
def trade_stats(p, ret, coin, day, cost, top=0.1):
    """Top decile by |p-0.5|: side = sign(p-0.5). Gross/net percent per trade, 2x cost, best-5% removal, per-coin, day-clustered t."""
    conf = np.abs(p - 0.5)
    sel = conf >= np.quantile(conf, 1 - top)
    side = np.where(p[sel] > 0.5, 1.0, -1.0)
    g = side * ret[sel] * 100
    c, dd = coin[sel], day[sel]
    cost_pct = cost * 100
    out = {"n_trades": int(sel.sum()), "gross_pct": float(g.mean()), "net1x_pct": float((g - cost_pct).mean()),
           "net2x_pct": float((g - 2 * cost_pct).mean())}
    for lab, k in (("1x", 1), ("2x", 2)):
        net = np.sort(g - k * cost_pct)
        out[f"net{lab}_trim5_pct"] = float(net[:int(np.floor(len(net) * 0.95))].mean())
    daily = pd.Series(g - cost_pct).groupby(dd).mean()
    out["net1x_day_t"] = float(daily.mean() / (daily.std() / np.sqrt(len(daily)))) if len(daily) > 2 else float("nan")
    per = pd.DataFrame({"coin": c, "gross": g}).groupby("coin")["gross"].agg(["count", "mean"])
    big = per[per["count"] >= 20]
    out["coins_ge20"] = int(len(big))
    out["coins_net1x_positive"] = int((big["mean"] - cost_pct > 0).sum())
    out["coins_gross_gt_cost"] = int((big["mean"] > cost_pct).sum())
    return out, per.assign(net1x=per["mean"] - cost_pct)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def eff_samples(d, tr_mask, n_feat, col):
    daily = d.loc[tr_mask].groupby([d.loc[tr_mask].index.floor("1D"), "coin"])[col].first().unstack("coin")
    corr = daily.corr().to_numpy()
    n = corr.shape[0]
    rho = float((corr.sum() - n) / (n * (n - 1)))
    n_ind = n / (1 + (n - 1) * rho)
    days = int(daily.shape[0])
    return {"train_days": days, "mean_cross_coin_corr_daily_first_ret": rho, "independent_coins": float(n_ind),
            "effective_samples": float(days * n_ind), "features_all_state_set": n_feat,
            "ratio_eff_per_feature": float(days * n_ind / n_feat)}


# ------------------------------------------------------------------ main experiment
def run(cfg, out_dir, max_minutes):
    clock = Clock(max_minutes)
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])
    res = {"truncated": False, "stage": "load"}
    tabs = {}
    n_conf = {"direction": 0, "magnitude": 0, "null_refits": 0}
    try:
        d, files = build_dataset(cfg, clock)
        st_cols, raw_cols, natr_cols = column_sets(d)
        Xs, Xr = d[st_cols].to_numpy(np.float32), d[raw_cols].to_numpy(np.float32)
        Xn = np.log(np.clip(d[natr_cols].to_numpy(np.float64), 1e-6, None)).astype(np.float32)
        t = d.index
        coin, day = d["coin"].to_numpy(), t.floor("1D").to_numpy()
        T = lambda s: pd.Timestamp(s, tz="UTC")
        hour = t.hour.to_numpy()
        res.update({"rows": int(len(d)), "rows_per_coin": d["coin"].value_counts().sort_index().to_dict(),
                    "n_state_cols": len(st_cols), "n_raw_cols": len(raw_cols)})

        def masks(H):
            h = pd.Timedelta(hours=1 + H)
            ok = d[f"ret{H}"].notna().to_numpy() & d[f"mag{H}"].notna().to_numpy() & np.isfinite(d[f"mag{H}"].to_numpy()) & (d[f"mag{H}"].to_numpy() > 0)
            tr = ok & (t >= T(cfg["train_start"])) & (t + h <= T(cfg["train_end"]))
            va = ok & (t >= T(cfg["train_end"])) & (t + h <= T(cfg["val_end"]))
            cp = ok & (t >= T(cfg["val_end"])) & (t + h <= T(cfg["compare_end"]))
            return tr, va, cp

        ks_of = lambda n: [n if k == "all" else min(k, n) for k in cfg["ks"]]
        tables = {k: [] for k in ("dir_sweep", "null", "years", "econ", "percoin", "mag_sweep", "mag_years", "lc", "mag_null")}
        res["capacity"] = {}
        for H in cfg["horizons"]:
            res["stage"] = f"direction H={H}"; clock.stage(res["stage"])
            tr, va, cp = masks(H)
            strd = tr & (hour % cfg["train_stride"] == 0)
            y = (d[f"ret{H}"].to_numpy() > 0).astype(int)
            ret = d[f"ret{H}"].to_numpy()
            res["capacity"][f"H{H}"] = eff_samples(d, tr, len(st_cols), f"ret{H}")
            res["capacity"][f"H{H}"]["rows_train_fit"] = int(strd.sum())
            res["capacity"][f"H{H}"]["rows_val"] = int(va.sum())
            res["capacity"][f"H{H}"]["rows_compare"] = int(cp.sum())
            res["capacity"][f"H{H}"]["train_base_rate_up"] = float(y[tr].mean())
            sets = {"state": (Xs, st_cols), "raw": (Xr, raw_cols)}
            fitted = {}
            for sname, (X, cols) in sets.items():
                order = np.argsort(-rank_columns(X[strd], y[strd].astype(float)))
                for model in ("logistic", "gbm"):
                    for k in ks_of(len(cols)):
                        clock.check()
                        sel = order[:k]
                        pr = fit_direction(model, "state" if sname == "state" else "raw", X[strd][:, sel], y[strd])
                        row = {"H": H, "set": sname, "model": model, "k": int(k)}
                        for nm, mk in (("train", strd), ("val", va), ("compare", cp)):
                            p = pr(X[mk][:, sel])
                            row[f"auc_{nm}"] = float(roc_auc_score(y[mk], p))
                        row["gap"] = row["auc_train"] - row["auc_val"]
                        tables["dir_sweep"].append(row)
                        n_conf["direction"] += 1
                        fitted[(sname, model, k)] = (pr, sel)
            sw = pd.DataFrame([r for r in tables["dir_sweep"] if r["H"] == H])
            best = sw[sw["set"] == "state"].sort_values("auc_val", ascending=False).iloc[0]
            model, kb = best["model"], int(best["k"])
            pr, sel = fitted[("state", model, kb)]
            info = {"H": H, "model": model, "k": kb, "auc_val": float(best["auc_val"]), "auc_compare": float(best["auc_compare"])}
            info["best_raw_auc_val"] = float(sw[sw["set"] == "raw"]["auc_val"].max())
            # --- null: label shifts through ranking + fit + max over configs (state set)
            res["stage"] = f"null H={H}"; clock.stage(res["stage"])
            nulls = []
            coins = d["coin"].to_numpy()
            idx_by_coin = {c: np.flatnonzero(coins == c) for c in cfg["coins"]}
            for s in range(cfg["n_null"]):
                ys = y.copy()
                for c, ix in idx_by_coin.items():
                    off = int(rng.integers(QUANTILE_SHIFT_DAYS[0] * 24, QUANTILE_SHIFT_DAYS[1] * 24))
                    ys[ix] = np.roll(y[ix], off)
                order_n = np.argsort(-rank_columns(Xs[strd], ys[strd].astype(float)))
                for mdl in ("logistic", "gbm"):
                    for k in ks_of(len(st_cols)):
                        clock.check()
                        prn = fit_direction(mdl, "state", Xs[strd][:, order_n[:k]], ys[strd])
                        a = float(roc_auc_score(y[va], prn(Xs[va][:, order_n[:k]])))
                        tables["null"].append({"H": H, "shift": s, "model": mdl, "k": int(k), "auc_val": a})
                        n_conf["null_refits"] += 1
            nl = pd.DataFrame([r for r in tables["null"] if r["H"] == H])
            peaks = nl.groupby("shift")["auc_val"].max()
            sel_mdl_peaks = nl[nl["model"] == model].groupby("shift")["auc_val"].max()
            real_model_peak = float(sw[(sw["set"] == "state") & (sw["model"] == model)]["auc_val"].max())
            info.update({"null_peak_max": float(peaks.max()), "null_peak_mean": float(peaks.mean()), "null_peak_sd": float(peaks.std()),
                         "null_p99_normal": float(peaks.mean() + 2.33 * peaks.std()), "null_mean_all": float(nl["auc_val"].mean()),
                         "real_model_peak_val": real_model_peak, "null_model_peak_max": float(sel_mdl_peaks.max()),
                         "n_null": int(peaks.shape[0])})
            # --- years: annual out-of-sample refits of the selected config
            res["stage"] = f"years H={H}"; clock.stage(res["stage"])
            yrs = []
            for Y in cfg["years"]:
                clock.check()
                hh = pd.Timedelta(hours=1 + H)
                trY = masks(H)[0] | masks(H)[1] | masks(H)[2]
                trY = trY & (t >= T(cfg["train_start"])) & (t + hh <= T(f"{Y}-01-01")) & (hour % cfg["train_stride"] == 0)
                teY = d[f"ret{H}"].notna().to_numpy() & (t >= T(f"{Y}-01-01")) & (t + hh <= T(f"{Y + 1}-01-01"))
                oY = np.argsort(-rank_columns(Xs[trY], y[trY].astype(float)))[:kb]
                pY = fit_direction(model, "state", Xs[trY][:, oY], y[trY])
                p = np.clip(pY(Xs[teY][:, oY]), 1e-6, 1 - 1e-6)
                base = np.clip(y[trY].mean(), 1e-6, 1 - 1e-6)
                ll, ll0 = log_loss(y[teY], p), log_loss(y[teY], np.full(teY.sum(), base))
                yrs.append({"H": H, "year": Y, "n": int(teY.sum()), "auc": float(roc_auc_score(y[teY], p)), "logloss": float(ll),
                            "logloss_intercept": float(ll0), "gain": float(ll0 - ll)})
            tables["years"] += yrs
            info["years_gain_positive"] = int(sum(r["gain"] > 0 for r in yrs))
            # --- learning curve: most recent 25/50/100 % of the training rows
            res["stage"] = f"learning curve H={H}"; clock.stage(res["stage"])
            trs = np.flatnonzero(strd)
            for frac in (0.25, 0.5, 1.0):
                clock.check()
                tcut = np.quantile(t[trs].asi8, 1 - frac) if frac < 1 else t[trs].asi8.min()
                use = trs[t[trs].asi8 >= tcut]
                oL = np.argsort(-rank_columns(Xs[use], y[use].astype(float)))[:kb]
                pL = fit_direction(model, "state", Xs[use][:, oL], y[use])
                tables["lc"].append({"H": H, "frac": frac, "rows": int(len(use)), "auc_val": float(roc_auc_score(y[va], pL(Xs[va][:, oL])))})
            # --- economics of the selected (train-only) model, top decile by confidence
            for nm, mk in (("val", va), ("compare", cp)):
                p = pr(Xs[mk][:, sel])
                s_, per = trade_stats(p, ret[mk], coin[mk], day[mk], cfg["cost_rt"])
                tables["econ"].append({"H": H, "period": nm, **s_})
                for c_, r_ in per.iterrows():
                    tables["percoin"].append({"H": H, "period": nm, "coin": c_, "trades": int(r_["count"]),
                                              "gross_pct": float(r_["mean"]), "net1x_pct": float(r_["net1x"])})
            res.setdefault("direction", {})[f"H{H}"] = info

            # ================= magnitude =================
            res["stage"] = f"magnitude H={H}"; clock.stage(res["stage"])
            zy = np.log(d[f"mag{H}"].to_numpy())
            yy = d[f"mag{H}"].to_numpy()
            taus = cfg["taus"]
            sets_m = {"state": (Xs, st_cols), "raw": (Xr, raw_cols)}
            base_losses = {}
            q0 = np.quantile(yy[strd], taus)
            base_losses["const"] = mag_loss(yy[va], np.tile(q0, (va.sum(), 1)), taus)
            for mdl in ("logistic", "gbm"):
                nm_ = "ridge" if mdl == "logistic" else "gbm"
                pq = fit_magnitude(mdl, "raw", Xn[strd], zy[strd], taus)
                base_losses[f"natr_{nm_}"] = mag_loss(yy[va], pq(Xn[va]), taus)
            base_name = min(base_losses, key=base_losses.get)
            res.setdefault("magnitude_baselines", {})[f"H{H}"] = {**base_losses, "best": base_name}
            for sname, (X, cols) in sets_m.items():
                order = np.argsort(-rank_columns(X[strd], zy[strd]))
                for mdl in ("logistic", "gbm"):
                    for k in ks_of(len(cols)):
                        clock.check()
                        sel_m = order[:k]
                        Xtr_m = np.hstack([X[strd][:, sel_m], Xn[strd]])
                        pq = fit_magnitude(mdl, "state" if sname == "state" else "raw", Xtr_m, zy[strd], taus)
                        lv = mag_loss(yy[va], pq(np.hstack([X[va][:, sel_m], Xn[va]])), taus)
                        tables["mag_sweep"].append({"H": H, "set": sname, "model": "ridge" if mdl == "logistic" else "gbm", "k": int(k),
                                                    "loss_val": lv, "rel_improve_vs_best_natr": 1 - lv / base_losses[base_name]})
                        n_conf["magnitude"] += 1
            mw = pd.DataFrame([r for r in tables["mag_sweep"] if r["H"] == H])
            mb = mw[mw["set"] == "state"].sort_values("loss_val").iloc[0]
            mmodel, mk_ = ("logistic" if mb["model"] == "ridge" else "gbm"), int(mb["k"])
            minfo = {"H": H, "model": mb["model"], "k": mk_, "rel_improve_val": float(mb["rel_improve_vs_best_natr"]),
                     "best_raw_rel_improve_val": float(mw[mw["set"] == "raw"]["rel_improve_vs_best_natr"].max()), "baseline": base_name}
            # annual refits (state + natr vs best of the three NATR-only baselines refit)
            for Y in cfg["years"]:
                clock.check()
                hh = pd.Timedelta(hours=1 + H)
                okm = np.isfinite(zy) & d[f"ret{H}"].notna().to_numpy()
                trY = okm & (t >= T(cfg["train_start"])) & (t + hh <= T(f"{Y}-01-01")) & (hour % cfg["train_stride"] == 0)
                teY = okm & (t >= T(f"{Y}-01-01")) & (t + hh <= T(f"{Y + 1}-01-01"))
                oY = np.argsort(-rank_columns(Xs[trY], zy[trY]))[:mk_]
                pq = fit_magnitude(mmodel, "state", np.hstack([Xs[trY][:, oY], Xn[trY]]), zy[trY], taus)
                lm = mag_loss(yy[teY], pq(np.hstack([Xs[teY][:, oY], Xn[teY]])), taus)
                bl = {"const": mag_loss(yy[teY], np.tile(np.quantile(yy[trY], taus), (teY.sum(), 1)), taus)}
                for mdl in ("logistic", "gbm"):
                    pb = fit_magnitude(mdl, "raw", Xn[trY], zy[trY], taus)
                    bl["natr_" + ("ridge" if mdl == "logistic" else "gbm")] = mag_loss(yy[teY], pb(Xn[teY]), taus)
                tables["mag_years"].append({"H": H, "year": Y, "n": int(teY.sum()), "loss_model": lm, **{f"loss_{k}": v for k, v in bl.items()},
                                            "rel_improve": float(1 - lm / min(bl.values()))})
            minfo["years_improve_positive"] = int(sum(r["rel_improve"] > 0 for r in tables["mag_years"] if r["H"] == H))
            # magnitude null: label shifts of the selected configuration (ranking + fit)
            res["stage"] = f"magnitude null H={H}"; clock.stage(res["stage"])
            for s in range(cfg["n_null"]):
                clock.check()
                zs = zy.copy()
                for c, ix in idx_by_coin.items():
                    off = int(rng.integers(QUANTILE_SHIFT_DAYS[0] * 24, QUANTILE_SHIFT_DAYS[1] * 24))
                    zs[ix] = np.roll(zy[ix], off)
                okz = np.isfinite(zs)
                use = strd & okz
                on = np.argsort(-rank_columns(Xs[use], zs[use]))[:mk_]
                pq = fit_magnitude(mmodel, "state", np.hstack([Xs[use][:, on], Xn[use]]), zs[use], taus)
                lv = mag_loss(yy[va], pq(np.hstack([Xs[va][:, on], Xn[va]])), taus)
                tables["mag_null"].append({"H": H, "shift": s, "rel_improve_val": float(1 - lv / base_losses[base_name])})
                n_conf["null_refits"] += 1
            mn = np.array([r["rel_improve_val"] for r in tables["mag_null"] if r["H"] == H])
            minfo.update({"null_rel_improve_max": float(mn.max()), "null_rel_improve_mean": float(mn.mean())})
            res.setdefault("magnitude", {})[f"H{H}"] = minfo
        res["stage"] = "done"
    except Deadline:
        res["truncated"] = True
    res["configurations"] = n_conf
    res["configurations_total_evaluated"] = int(sum(n_conf.values()))
    res["files"] = files
    res["runtime_min"] = round(clock.minutes(), 2)
    # ---- outputs
    outs = {}
    for k, rows in tables.items():
        if rows:
            p = os.path.join(out_dir, f"{k}.csv")
            pd.DataFrame(rows).round(6).to_csv(p, index=False)
            outs[os.path.basename(p)] = sha256_file(p)
    rp = os.path.join(out_dir, "results.json")
    json.dump({k: v for k, v in res.items() if k != "files"}, open(rp, "w"), indent=1, default=str, sort_keys=True)
    outs["results.json"] = sha256_file(rp)
    try:
        sha = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    except Exception:                                            # noqa: BLE001
        sha = "unknown"
    fp = hashlib.sha256(json.dumps(sorted(files)).encode()).hexdigest()
    manifest = {"run_id": os.path.basename(os.path.abspath(out_dir)), "card": "research/gate_features/cards/E-gate-001.md", "git_sha": sha,
                "gate_features_version": gf.VERSION, "config": cfg, "input_fingerprint_sha256": fp, "inputs": files,
                "rows": res.get("rows"), "configurations_evaluated": res["configurations_total_evaluated"],
                "configurations_detail": n_conf, "runtime_min": res["runtime_min"], "truncated": res["truncated"], "outputs_sha256": outs}
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1, default=str)
    return manifest, res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-minutes", type=float, default=50)
    a = ap.parse_args(argv)
    manifest, res = run(load_config(a.config), a.out, a.max_minutes)
    print(json.dumps(manifest["outputs_sha256"], indent=1))
    print("truncated:", manifest["truncated"], "configs:", manifest["configurations_evaluated"], "minutes:", manifest["runtime_min"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
