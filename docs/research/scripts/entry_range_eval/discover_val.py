"""Rule discovery on VAL only. Writes frozen.json (rules, signs, thresholds, vol-only fit) and val_configs.csv.

Protocol: every (rule x uncertainty-gate x coverage) configuration is one trial (M trials, two-sided: an inverted rule
is the same trial with the opposite sign). Selection statistic = clustered-by-timestamp t of (accuracy - 0.5).
Multiple testing: Bonferroni over M, plus a permutation null of the MAX |t| over all M trials, built by circularly
shifting the label cross-sections in time (keeps cross-sectional/market structure, breaks score-label alignment).
TEST is never loaded here.
"""
import argparse
import json

import numpy as np
import pandas as pd

import lib

ap = argparse.ArgumentParser()
ap.add_argument("--npz", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--perms", type=int, default=200)
a = ap.parse_args()

dfv, coin_rv = lib.load_split(a.npz, "val")
S, zc = lib.base_scores(dfv)
gc = lib.gate_consts(dfv)
gm = {g: lib.gate_mask(dfv, g, gc) for g in lib.GATES}

# ---- definition check (which entry_range definition produced these weights?) ----
up, dn = dfv.fh / dfv.P - 1, 1 - dfv.fl / dfv.P
rng = dfv.fh - dfv.fl
pos = np.where(rng > 0, (dfv.fc - dfv.fl) / rng, .5)
cur = {"high": (dfv.fh > dfv.lh).astype(int), "low": (dfv.fl > dfv.ll).astype(int), "close": dfv.dir}
old = {"high": (up.clip(lower=0) > .002).astype(int), "low": (dn.clip(lower=0) > .002).astype(int), "close": (pos > .5).astype(int)}
defcheck = {"auc_current_def": {t: lib.auc(cur[t], dfv[f"p_{h}"]) for t, h in (("high", "h"), ("low", "l"), ("close", "c"))},
            "auc_first_def": {t: lib.auc(old[t], dfv[f"p_{h}"]) for t, h in (("high", "h"), ("low", "l"), ("close", "c"))},
            "close_head_range": [float(dfv.mu_c.min() * 100), float(dfv.mu_c.max() * 100)],
            "spearman_muc_abs_ret": float(dfv.mu_c.corr(dfv.abs_ret, method="spearman")),
            "spearman_muc_range_pos": float(pd.Series(dfv.mu_c.values).corr(pd.Series(pos), method="spearman")),
            "spearman_muup_up": float(dfv.mu_up.corr(pd.Series(up.clip(lower=0)), method="spearman")),
            "spearman_mudn_dn": float(dfv.mu_dn.corr(pd.Series(dn.clip(lower=0)), method="spearman")),
            "class_balance_val": {k: float(v.mean()) for k, v in cur.items()}}

# ---- all trials ----
labels_mat = np.full((dfv.tsi.max() + 1, dfv.asset.nunique()), np.nan)
ai = pd.factorize(dfv.asset)[0]
labels_mat[dfv.tsi.values, ai] = dfv.dir.values
T = labels_mat.shape[0]
rng_np = np.random.default_rng(0)
shifts = rng_np.integers(5, T - 5, size=a.perms)
tsi, y = dfv.tsi.values, dfv.dir.values


def tstat(h_minus, cl):
    m = h_minus.mean()
    u, inv = np.unique(cl, return_inverse=True)
    s = np.bincount(inv, weights=h_minus - m)
    se = np.sqrt((s ** 2).sum()) / len(h_minus)
    return m / se if se > 0 else 0.0


trials, null_max = [], np.zeros(a.perms)
for r, s in S.items():
    pred = (s > 0).astype(int)
    for g in lib.GATES:
        for cov in lib.COVERAGES:
            thr = lib.coverage_thr(s, gm[g], cov)
            sel = lib.select(s, gm[g], thr)
            n, ncl = int(sel.sum()), len(np.unique(tsi[sel])) if sel.any() else 0
            row = {"rule": r, "gate": g, "cov": cov, "thr": thr, "n": n, "ncl": ncl, "acc": np.nan, "t": np.nan, "base": np.nan}
            if n >= 300 and ncl >= 30:
                h = (pred[sel] == y[sel]).astype(float)
                row.update(acc=float(h.mean()), t=float(tstat(h - .5, tsi[sel])), base=float(y[sel].mean()))
                row["_idx"] = (r, g, cov)
                for k, sh in enumerate(shifts):          # permutation null for this trial
                    lab = labels_mat[(tsi[sel] + sh) % T, ai[sel]]
                    ok = ~np.isnan(lab)
                    if ok.sum() > 100:
                        hh = (pred[sel][ok] == lab[ok]).astype(float)
                        null_max[k] = max(null_max[k], abs(tstat(hh - .5, tsi[sel][ok])))
            trials.append(row)

df_tr = pd.DataFrame(trials)
M = len(df_tr)                      # ALL trials counted, including ones skipped for tiny n (conservative)
ok = df_tr.t.notna()
df_tr["p_raw"] = df_tr.t.map(lambda t: lib.pval(t) if np.isfinite(t) else np.nan)
df_tr["p_bonf"] = (df_tr.p_raw * M).clip(upper=1)
best = df_tr[ok].reindex(df_tr[ok].t.abs().sort_values(ascending=False).index).iloc[0]
perm_p = float((null_max >= abs(best.t)).mean())
perm_p = max(perm_p, 1 / (a.perms + 1))
# also: best plain (no gate, full coverage) rule as a context frozen candidate
plain = df_tr[ok & (df_tr.gate == "none") & (df_tr["cov"] == 1.0)]
best_plain = plain.reindex(plain.t.abs().sort_values(ascending=False).index).iloc[0]

# ---- baselines on val (sign of wick chosen on val: 2 extra trials) ----
B = lib.baseline_scores(dfv)
base_rows = {}
for k, s in B.items():
    sel = np.abs(s) > 0
    hs = lib.hit_stats(dfv, s, sel)
    ic, t, _, _ = lib.rank_ic(dfv, s)
    base_rows[k] = {"acc": hs["acc"], "acc_t": hs["t"], "auc": lib.auc(dfv.dir, s), "ic": ic, "ic_t": t}
wick_invert = bool(base_rows["wick"]["acc"] < 0.5)

# second pre-declared criterion (fixed before touching TEST): among the 13 full-coverage, ungated rules pick the largest
# |per-timestamp rank-IC t| on val, sign = sign of the IC. Its own family size is 13 (Bonferroni x13).
ic_tab = {r: lib.rank_ic(dfv, s)[:2] for r, s in S.items()}
r_ic = max(ic_tab, key=lambda r: abs(ic_tab[r][1]))
best_ic = {"rule": r_ic, "ic": ic_tab[r_ic][0], "ic_t": ic_tab[r_ic][1], "invert": bool(ic_tab[r_ic][0] < 0),
           "p_bonf_13": min(1.0, lib.pval(ic_tab[r_ic][1]) * len(ic_tab))}

coefs = lib.vol_fit(dfv)
frozen = {
    "n_trials_M": M, "n_valid_trials": int(ok.sum()), "perm_p_max_t": perm_p, "perms": a.perms,
    "best": {k: (v.item() if hasattr(v, "item") else v) for k, v in best.drop("_idx", errors="ignore").items()},
    "best_plain": {k: (v.item() if hasattr(v, "item") else v) for k, v in best_plain.drop("_idx", errors="ignore").items()},
    "zc": zc, "gate_consts": gc, "coin_rv": coin_rv, "vol_coefs": {k: v.tolist() for k, v in coefs.items()},
    "best_ic": best_ic, "wick_invert": wick_invert, "baselines_val": base_rows, "definition_check": defcheck,
    "baseline_thr": {k: {str(c): lib.coverage_thr(s, np.ones(len(s), bool), c) for c in lib.COVERAGES} for k, s in B.items()},
}
# invert flag: the rule is applied with the sign that val prefers
for k in ("best", "best_plain"):
    frozen[k]["invert"] = bool(frozen[k]["acc"] < 0.5)
json.dump(frozen, open(a.out, "w"), indent=1, default=float)
df_tr.drop(columns=["_idx"], errors="ignore").to_csv(a.out.replace(".json", "_configs.csv"), index=False)

# ---- val-side descriptive reports for the write-up (val only) ----
full_rules = {r: lib.rule_report(dfv, S[r]) for r in S}
tab = pd.DataFrame({r: {"auc": v["auc"], "acc": v["acc"], "acc_t": v["acc_t"], "ic": v["ic"], "ic_t": v["ic_t"]} for r, v in full_rules.items()}).T
pd.set_option("display.width", 200)
print("== definition check ==\n", json.dumps(defcheck, indent=1))
print("== val full-coverage rules ==\n", tab.round(4))
print("== baselines val ==\n", pd.DataFrame(base_rows).T.round(4), "wick_invert", wick_invert)
print(f"== M={M} trials ({int(ok.sum())} valid); best:", best[["rule", "gate", "cov", "n", "acc", "t"]].to_dict(), "p_bonf", min(1, lib.pval(best.t) * M), "perm_p", perm_p)
print("best_ic:", best_ic)
print("best_plain:", best_plain[["rule", "gate", "cov", "n", "acc", "t"]].to_dict())
print("top 12 trials by |t|:\n", df_tr[ok].reindex(df_tr[ok].t.abs().sort_values(ascending=False).index).head(12)[["rule", "gate", "cov", "n", "acc", "base", "t", "p_bonf"]].round(4))
print("== magnitude vs vol-only (val) ==\n", json.dumps(lib.magnitude_report(dfv, coefs), indent=1, default=float))
print("== uncertainty collapse (val) ==\n", json.dumps(lib.collapse_report(dfv), indent=1, default=float))
