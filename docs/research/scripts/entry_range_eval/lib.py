"""Shared helpers for the entry_range evaluation (predict.py -> discover_val.py -> eval_test.py).

Everything here is deliberately small and numpy/pandas only. The two guards that matter:
  * VAL is the only split used to pick rules/thresholds (discover_val.py writes frozen.json);
  * eval_test.py reads frozen.json and touches TEST exactly once.
Definition in force (empirically verified on val, see docs/research/entry_range_eval.md):
  class heads = same-type direction (future_high > last_high, future_low > last_low, future_close > last_close)
  reg heads   = non-negative magnitudes from P = last_close, x100 (up=H/P-1, dn=1-L/P, |C/P-1|).
"""
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_auc_score

SCALE = 100.0
FEE_MAKER, FEE_TAKER = 0.0002, 0.0005


def _rsi14(c):
    """Wilder RSI(14) on the 32-close window (approximation: seeded on the window, not on full history)."""
    d = np.diff(c, axis=1)
    g, l = np.clip(d, 0, None), np.clip(-d, 0, None)
    ag, al = g[:, :14].mean(1), l[:, :14].mean(1)
    for i in range(14, d.shape[1]):
        ag = (ag * 13 + g[:, i]) / 14
        al = (al * 13 + l[:, i]) / 14
    return 100 - 100 / (1 + ag / np.maximum(al, 1e-12))


def load_split(npz, split, coin_rv=None):
    """DataFrame for one split from predict.py's npz. coin_rv: {asset: long-run vol} fitted on VAL only."""
    z = np.load(npz, allow_pickle=True)
    g = lambda k: z[f"{split}__{k}"]
    lc = g("lc")
    lh, ll, P, ts, fc, fl, fh = lc.T
    bp = g("bp")
    c = g("feat_win")[:, :, 2].astype("float64") * bp[:, 1:2] + bp[:, 0:1]     # exact price reconstruction
    lr = np.diff(np.log(c), axis=1)
    prev = c[:, -2]
    top, bot = np.maximum(prev, P), np.minimum(prev, P)                        # open ~ previous close (24/7 market)
    rng = lh - ll
    df = pd.DataFrame({
        "asset": g("asset"), "ts": pd.to_datetime(ts.astype("int64"), utc=True),
        "P": P, "lh": lh, "ll": ll, "fc": fc, "fh": fh, "fl": fl,
        "mu_up": g("y_high") / SCALE, "mu_dn": g("y_low") / SCALE, "mu_c": g("y_close") / SCALE,
        "p_h": g("y_high_class_logits"), "p_l": g("y_low_class_logits"), "p_c": g("y_close_class_logits"),
        "epi_h": g("y_high_epistemic"), "epi_l": g("y_low_epistemic"), "epi_c": g("y_close_epistemic"),
        "ale_h": g("y_high_aleatoric"), "ale_l": g("y_low_aleatoric"), "ale_c": g("y_close_aleatoric"),
        "conf_c": g("y_close_confidence"),
        "ret1": P / prev - 1, "rsi14": _rsi14(c),
        "rv8": lr[:, -8:].std(1), "rv32": lr.std(1),
        "wick": np.where(rng > 0, ((bot - ll) - (lh - top)) / np.where(rng > 0, rng, 1), 0.0),
    })
    df["ret"] = df.fc / df.P - 1
    df["dir"] = (df.fc > df.P).astype(int)
    df["up_exc"] = np.maximum(df.fh / df.P - 1, 0)
    df["dn_exc"] = np.maximum(1 - df.fl / df.P, 0)
    df["abs_ret"] = df.ret.abs()
    df["asym_real"] = df.up_exc - df.dn_exc
    df["tsi"] = pd.factorize(df.ts)[0]
    df["month"] = df.ts.dt.strftime("%Y-%m")
    if coin_rv is None:
        coin_rv = df.groupby("asset").rv32.mean().to_dict()
    df["coin_rv"] = df.asset.map(coin_rv).fillna(np.nanmedian(list(coin_rv.values())))
    return df, coin_rv


# ----------------------------------------------------------------------------- statistics
def cluster_t(x, cl):
    """Mean of x and its t-stat with standard errors clustered by `cl` (integer timestamp ids)."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return np.nan, np.nan, n
    m = x.mean()
    u, inv = np.unique(cl, return_inverse=True)
    if len(u) < 3:
        return m, np.nan, n
    s = np.bincount(inv, weights=x - m)
    se = np.sqrt((s ** 2).sum()) / n
    return m, (m / se if se > 0 else np.nan), n


def pval(t):
    return float(2 * (1 - norm.cdf(abs(t)))) if np.isfinite(t) else np.nan


def rank_ic(df, s, mask=None, target="ret", min_n=10):
    """Per-timestamp Spearman(s, target); returns mean IC, t over timestamps, per-month mean IC, n timestamps."""
    d = pd.DataFrame({"tsi": df.tsi.values, "s": np.asarray(s, float), "y": df[target].values, "m": df.month.values})
    if mask is not None:
        d = d[np.asarray(mask)]
    d = d.dropna()
    d = d[d.groupby("tsi").s.transform("size") >= min_n]
    d["rs"] = d.groupby("tsi").s.rank()
    d["ry"] = d.groupby("tsi").y.rank()
    ic = d.groupby("tsi").apply(lambda g: np.corrcoef(g.rs, g.ry)[0, 1] if g.rs.std() > 0 and g.ry.std() > 0 else np.nan,
                                include_groups=False).dropna()
    if len(ic) < 3:
        return np.nan, np.nan, {}, len(ic)
    mon = d.groupby("tsi").m.first().reindex(ic.index)
    return float(ic.mean()), float(ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)))), ic.groupby(mon).mean().to_dict(), len(ic)


def partial_ic(df, s, y, ctrl, min_n=10):
    """Mean per-timestamp partial Spearman(s, y | ctrl)."""
    d = pd.DataFrame({"tsi": df.tsi.values, "s": np.asarray(s, float), "y": np.asarray(y, float), "c": np.asarray(ctrl, float)}).dropna()
    for k in "syc":
        d["r" + k] = d.groupby("tsi")[k].rank()
    out = []
    for _, g in d.groupby("tsi"):
        if len(g) < min_n:
            continue
        r = np.corrcoef(g[["rs", "ry", "rc"]].values.T)
        den = np.sqrt((1 - r[0, 2] ** 2) * (1 - r[1, 2] ** 2))
        if den > 0:
            out.append((r[0, 1] - r[0, 2] * r[1, 2]) / den)
    out = np.array(out)
    return float(out.mean()), float(out.mean() / (out.std(ddof=1) / np.sqrt(len(out)))), len(out)


def auc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if 0 < y.mean() < 1 else np.nan


# ----------------------------------------------------------------------------- rules
def base_scores(df, zc=None):
    """All direction scores computed from model outputs only (s > 0 => predicts UP). zc: {name: std} frozen on val."""
    up, dn, pc, ph, pl = df.mu_up.values, df.mu_dn.values, df.p_c.values, df.p_h.values, df.p_l.values
    tot = np.maximum(up + dn, 1e-12)
    S = {
        "pc": pc - 0.5,
        "asym": up - dn,
        "nasym": (up - dn) / tot,
        "pos_impl": dn / tot - 0.5,                       # where P sits inside the predicted [L,H] range
        "p_mean": (ph + pl + pc) / 3 - 0.5,
        "p_hl": (ph + pl) / 2 - 0.5,
    }
    sg = np.sign
    agree = (sg(ph - .5) == sg(pl - .5)) & (sg(pl - .5) == sg(pc - .5))
    S["agree3"] = np.where(agree, sg(pc - .5) * np.minimum.reduce([abs(ph - .5), abs(pl - .5), abs(pc - .5)]), 0.0)
    S["pc_x_muc"] = (pc - .5) * df.mu_c.values * SCALE
    S["sign_pc_muc"] = sg(pc - .5) * df.mu_c.values
    S["asym_agree_pmean"] = np.where(sg(up - dn) == sg(S["p_mean"]), up - dn, 0.0)
    S["asym_agree_pc"] = np.where(sg(up - dn) == sg(pc - .5), up - dn, 0.0)
    zc = zc or {k: S[k].std() for k in ("nasym", "pc", "p_mean")}
    S["z_nasym_pc"] = S["nasym"] / zc["nasym"] + S["pc"] / zc["pc"]
    S["z_nasym_pmean"] = S["nasym"] / zc["nasym"] + S["p_mean"] / zc["p_mean"]
    return S, zc


def baseline_scores(df):
    """(b) reversal: -RSI14 and -last-bar return; (c) last-candle wick imbalance (lower-upper wick)/range."""
    return {"neg_rsi14": -(df.rsi14.values - 50), "neg_ret1": -df.ret1.values, "wick": df.wick.values}


GATES = ["none", "epi_c_lo", "epi_c_hi", "ale_c_lo", "ale_c_hi", "epi_hl_lo", "epi_hl_hi", "no_collapse"]


def gate_values(df):
    return {"epi_c": df.epi_c.values, "ale_c": df.ale_c.values, "epi_hl": (df.epi_h.values + df.epi_l.values) / 2}


def gate_consts(dfv):
    """Frozen gate thresholds from VAL: medians, and 10%-of-median 'near-zero' cut-offs for the collapse gate."""
    gv = gate_values(dfv)
    c = {k: float(np.median(v)) for k, v in gv.items()}
    c["nz_epi"] = 0.1 * float(np.median(np.concatenate([dfv.epi_h, dfv.epi_l, dfv.epi_c])))
    c["nz_ale"] = 0.1 * float(np.median(np.concatenate([dfv.ale_h, dfv.ale_l, dfv.ale_c])))
    return c


def gate_mask(df, gate, c):
    gv = gate_values(df)
    if gate == "none":
        return np.ones(len(df), bool)
    if gate == "no_collapse":
        eps = np.stack([df.epi_h, df.epi_l, df.epi_c]).min(0)
        als = np.stack([df.ale_h, df.ale_l, df.ale_c]).min(0)
        return (eps >= c["nz_epi"]) & (als >= c["nz_ale"])
    k, side = gate.rsplit("_", 1)
    return gv[k] <= c[k] if side == "lo" else gv[k] >= c[k]


COVERAGES = [1.0, 0.5, 0.2, 0.1, 0.05]


def select(s, gmask, thr):
    return gmask & (np.abs(s) > 0) & (np.abs(s) >= thr)


def coverage_thr(s, gmask, cov):
    a = np.abs(s[gmask & (np.abs(s) > 0)])
    return 0.0 if cov >= 1.0 else float(np.quantile(a, 1 - cov))


def hit_stats(df, s, sel, invert=False):
    """Accuracy of sign(s) (flipped if invert) vs the close-return direction over selected rows, clustered t vs 0.5."""
    pred = (s > 0).astype(int)
    if invert:
        pred = 1 - pred
    h = (pred[sel] == df.dir.values[sel]).astype(float)
    m, t, n = cluster_t(h - 0.5, df.tsi.values[sel])
    return {"n": int(n), "acc": float(h.mean()) if n else np.nan, "t": t, "base": float(df.dir.values[sel].mean()) if n else np.nan}


def balanced_acc(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    if y.min() == y.max():
        return np.nan
    return 0.5 * ((pred[y == 1] == 1).mean() + (pred[y == 0] == 0).mean())


def month_table(df, s, sel, invert=False):
    pred = (s > 0).astype(int)
    if invert:
        pred = 1 - pred
    d = pd.DataFrame({"m": df.month.values[sel], "h": (pred[sel] == df.dir.values[sel]).astype(float)})
    g = d.groupby("m").h.agg(["mean", "size"])
    return g


# ----------------------------------------------------------------------------- tradability
def simulate(df, s, sel, invert=False, taker_entry=False):
    """Maker-limit entry at the model's predicted level, taker exit at the bar close (1h OHLC, worst-case ordering).
    long : buy limit at P(1-mu_dn), filled if future_low <= level;  short: sell limit at P(1+mu_up), filled if future_high >= level.
    Exit at future_close. Any TP inside the same bar is ambiguous under 1h OHLC: worst case = never counted
    (an optimistic bound where a TP at the opposite predicted level counts when touched is returned as net_opt)."""
    d = df[sel].copy()
    long = (s[sel] > 0) ^ invert
    P = d.P.values
    lvl = np.where(long, P * (1 - d.mu_dn.values), P * (1 + d.mu_up.values))
    filled = np.where(long, d.fl.values <= lvl, d.fh.values >= lvl)
    if taker_entry:
        lvl, filled = P, np.ones(len(d), bool)
    fee_in = FEE_TAKER if taker_entry else FEE_MAKER
    gross = np.where(long, d.fc.values / lvl - 1, 1 - d.fc.values / lvl)
    net = gross - fee_in - FEE_TAKER
    # optimistic: TP at the opposite predicted level counts if touched (maker exit)
    tp = np.where(long, P * (1 + d.mu_up.values), P * (1 - d.mu_dn.values))
    tp_hit = np.where(long, d.fh.values >= tp, d.fl.values <= tp)
    gross_opt = np.where(tp_hit, np.where(long, tp / lvl - 1, 1 - tp / lvl), gross)
    net_opt = gross_opt - fee_in - np.where(tp_hit, FEE_MAKER, FEE_TAKER)
    out = d[["ts", "tsi", "month", "asset"]].copy()
    out["long"], out["filled"], out["gross"], out["net"], out["net_opt"] = long, filled, gross, net, net_opt
    return out


def sim_summary(out):
    f = out[out.filled]
    if len(f) < 3:
        return {"signals": len(out), "fills": len(f)}
    m, t, n = cluster_t(f.net.values, f.tsi.values)
    mo, to, _ = cluster_t(f.net_opt.values, f.tsi.values)
    mg, tg, _ = cluster_t(f.gross.values, f.tsi.values)
    mm = f.groupby("month").net.mean()
    per_signal = np.where(out.filled, out.net, 0.0).mean()
    return {"signals": len(out), "fills": len(f), "fill_rate": len(f) / len(out), "gross_bp": mg * 1e4, "gross_t": tg,
            "net_bp": m * 1e4, "net_t": t, "net_opt_bp": mo * 1e4, "net_opt_t": to,
            "per_signal_net_bp": per_signal * 1e4, "months_pos": int((mm > 0).sum()), "months": int(len(mm)),
            "long_share": float(out.long.mean())}


# ----------------------------------------------------------------------------- reports shared by val/test
def vol_fit(dfv):
    """Vol-only predictors fitted on VAL (log-linear on past realised vol + coin long-run vol); returns coef dict."""
    X = np.column_stack([np.ones(len(dfv)), np.log(dfv.rv8 + 1e-6), np.log(dfv.rv32 + 1e-6), np.log(dfv.coin_rv + 1e-6)])
    out = {}
    for name, y in (("up", dfv.up_exc), ("dn", dfv.dn_exc), ("close", dfv.abs_ret)):
        out[name] = np.linalg.lstsq(X, np.log(y.values + 1e-4), rcond=None)[0]
    return out


def vol_pred(df, coef):
    X = np.column_stack([np.ones(len(df)), np.log(df.rv8 + 1e-6), np.log(df.rv32 + 1e-6), np.log(df.coin_rv + 1e-6)])
    return X @ coef


def magnitude_report(df, coefs):
    """Magnitude heads vs vol-only: rank IC, partial IC (controlling the vol-only prediction), AUC of
    'above-median magnitude' inside rv32 quintiles for the head and for the vol-only predictor."""
    rows = {}
    q = pd.qcut(df.rv32, 5, labels=False, duplicates="drop").values
    for name, mu, y in (("up", df.mu_up, df.up_exc), ("dn", df.mu_dn, df.dn_exc), ("close", df.mu_c, df.abs_ret)):
        vp = vol_pred(df, coefs[name])
        ic_m, t_m, _, _ = rank_ic(df, mu.values, target=y.name)
        ic_v, t_v, _, _ = rank_ic(df, vp, target=y.name)
        pic, pt, _ = partial_ic(df, mu.values, y.values, vp)
        a_m, a_v = [], []
        for k in np.unique(q):
            m = q == k
            lab = (y.values[m] > np.median(y.values[m])).astype(int)
            a_m.append(auc(lab, mu.values[m]))
            a_v.append(auc(lab, vp[m]))
        rows[name] = {"ic_head": ic_m, "t_head": t_m, "ic_volonly": ic_v, "t_volonly": t_v, "partial_ic": pic, "partial_t": pt,
                      "auc_q_head": a_m, "auc_q_vol": a_v,
                      "pooled_rho_head": float(pd.Series(mu.values).corr(pd.Series(y.values), method="spearman")),
                      "pooled_rho_vol": float(pd.Series(vp).corr(pd.Series(y.values), method="spearman"))}
    return rows


def collapse_report(df):
    """Share of near-zero epistemic/aleatoric (absolute thresholds in target units x100, and 10% of median) + accuracy there."""
    rows = {}
    for h, mu in (("h", None), ("l", None), ("c", None)):
        for kind in ("epi", "ale"):
            v = df[f"{kind}_{h}"].values
            med = np.median(v)
            rows[f"{kind}_{h}"] = {"min": float(v.min()), "p1": float(np.quantile(v, .01)), "median": float(med),
                                   "share_lt_0.05": float((v < 0.05).mean()), "share_lt_10pct_median": float((v < 0.1 * med).mean()),
                                   "cv": float(v.std() / v.mean())}
    p = {h: df[f"p_{h}"].values for h in "hlc"}
    rows["p_spread"] = {h: {"std": float(v.std()), "p5": float(np.quantile(v, .05)), "p95": float(np.quantile(v, .95))} for h, v in p.items()}
    near = (df[["epi_h", "epi_l", "epi_c"]].min(1) < 0.05) | (df[["ale_h", "ale_l", "ale_c"]].min(1) < 0.05)
    rows["near_zero_rows"] = int(near.sum())
    rows["acc_pc_near_zero"] = float(((df.p_c > .5).astype(int) == df.dir)[near].mean()) if near.sum() else None
    return rows


def rule_report(df, s, invert=False, sel=None):
    """Full metric block for one score on a split: AUC, accuracy vs balance, BA, IC (mean,t), coverage table, months."""
    sel_all = (np.abs(s) > 0) if sel is None else sel
    pred = (s > 0).astype(int) ^ int(invert)
    ss = -s if invert else s
    ic, t, mon_ic, nts = rank_ic(df, ss, mask=sel_all)
    hs = hit_stats(df, s, sel_all, invert)
    rep = {"n": hs["n"], "acc": hs["acc"], "acc_t": hs["t"], "base": hs["base"],
           "auc": auc(df.dir.values[sel_all], ss[sel_all]), "ba": balanced_acc(df.dir.values[sel_all], pred[sel_all]),
           "ic": ic, "ic_t": t, "ic_by_month": mon_ic, "cov": {}}
    for cov in COVERAGES[1:]:
        thr = coverage_thr(ss, sel_all, cov)
        m = select(ss, sel_all, thr)
        h = hit_stats(df, s, m, invert)
        rep["cov"][cov] = {"n": h["n"], "acc": h["acc"], "t": h["t"], "base": h["base"]}
    mt = month_table(df, s, sel_all, invert)
    rep["months"] = {k: (float(v["mean"]), int(v["size"])) for k, v in mt.iterrows()}
    rep["months_above_50"] = int((mt["mean"] > 0.5).sum())
    return rep
