"""Single evaluation of the FROZEN rules (frozen.json from discover_val.py) on TEST.

`--split val` exists only as a dry run of this very code path (bug hunting on val); the reported result is one run
with `--split test`. Nothing is selected or tuned here: rules, signs, gate thresholds, vol-only coefficients and coin
long-run vols all come from frozen.json (fitted on VAL).
"""
import argparse
import json

import numpy as np
import pandas as pd

import lib

ap = argparse.ArgumentParser()
ap.add_argument("--npz", required=True)
ap.add_argument("--frozen", required=True)
ap.add_argument("--split", default="test", choices=["val", "test"])
ap.add_argument("--out", required=True)
a = ap.parse_args()

fz = json.load(open(a.frozen))
df, _ = lib.load_split(a.npz, a.split, coin_rv=fz["coin_rv"])
S, _ = lib.base_scores(df, zc=fz["zc"])
B = lib.baseline_scores(df)
gc = fz["gate_consts"]
res = {"split": a.split, "n": len(df), "timestamps": int(df.tsi.nunique()), "class_balance": {
    "close": float(df.dir.mean()), "high": float((df.fh > df.lh).mean()), "low": float((df.fl > df.ll).mean())}}
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)


def block(name, s, invert, gate="none"):
    sel = lib.gate_mask(df, gate, gc) & (np.abs(s) > 0)
    rep = lib.rule_report(df, s, invert, sel)
    rep["ic_by_month"] = {k: float(v) for k, v in rep["ic_by_month"].items()}
    res[name] = rep
    return sel


cands = {
    "primary_acc_best": (S[fz["best"]["rule"]], fz["best"]["invert"], fz["best"]["gate"]),
    "best_ic": (S[fz["best_ic"]["rule"]], fz["best_ic"]["invert"], "none"),
    "ref_pc": (S["pc"], False, "none"),
    "base_neg_rsi14": (B["neg_rsi14"], False, "none"),
    "base_neg_ret1": (B["neg_ret1"], False, "none"),
    "base_wick": (B["wick"], fz["wick_invert"], "none"),
}
sels = {k: block(k, s, inv, g) for k, (s, inv, g) in cands.items()}
res["frozen_rules"] = {"primary_acc_best": fz["best"], "best_ic": fz["best_ic"], "wick_invert": fz["wick_invert"],
                       "M": fz["n_trials_M"], "perm_p_val": fz["perm_p_max_t"]}

# ---- all 13 rules on this split (descriptive only; nothing is re-selected from this table) ----
res["all_rules_desc"] = {r: {k: v for k, v in lib.rule_report(df, s).items() if k in ("auc", "acc", "acc_t", "ic", "ic_t")} for r, s in S.items()}

# ---- p-values for the frozen claims ----
pr = res["primary_acc_best"]
res["p_values"] = {"primary_acc_t": pr["acc_t"], "primary_p_single": lib.pval(pr["acc_t"]),
                   "primary_p_bonf_M": min(1.0, lib.pval(pr["acc_t"]) * fz["n_trials_M"]),
                   "best_ic_t": res["best_ic"]["ic_t"], "best_ic_p_single": lib.pval(res["best_ic"]["ic_t"]),
                   "best_ic_p_bonf_13": min(1.0, lib.pval(res["best_ic"]["ic_t"]) * 13)}

# ---- realized asymmetry up_exc - dn_exc ----
asym = S["asym"]
ic_a, t_a, _, _ = lib.rank_ic(df, asym, target="asym_real")
res["asym_real"] = {"ic_asym_vs_asym_real": ic_a, "t": t_a,
                    "auc_asym_vs_sign_asym_real": lib.auc((df.asym_real > 0).astype(int), asym),
                    "auc_pc_vs_sign_asym_real": lib.auc((df.asym_real > 0).astype(int), df.p_c),
                    "auc_asym_vs_dir": lib.auc(df.dir, asym),
                    "base_asym_real_pos": float((df.asym_real > 0).mean())}

# ---- the three questions ----
res["magnitude_vs_vol"] = lib.magnitude_report(df, {k: np.array(v) for k, v in fz["vol_coefs"].items()})
res["close_class"] = {"auc_pooled": lib.auc(df.dir, df.p_c), "ic_pc_vs_ret": lib.rank_ic(df, df.p_c.values)[:2],
                      "auc_high": lib.auc((df.fh > df.lh).astype(int), df.p_h), "auc_low": lib.auc((df.fl > df.ll).astype(int), df.p_l)}
res["collapse"] = lib.collapse_report(df)

# ---- tradability: maker-limit entry at the predicted level, taker exit at close, worst-case ordering ----
res["trade"] = {}
for name in ("primary_acc_best", "best_ic", "ref_pc"):
    s, inv, g = cands[name]
    sel0 = sels[name]
    ss = -s if inv else s
    for cov in (1.0, 0.1, 0.05):
        thr = lib.coverage_thr(ss, sel0, cov)
        m = lib.select(ss, sel0, thr)
        sim = lib.simulate(df, s, m, inv)
        res["trade"][f"{name}@{int(cov * 100)}%"] = lib.sim_summary(sim)
    # reference: market entry at P (taker in and out) for the same rule at full coverage
    res["trade"][f"{name}@100%_market_entry"] = lib.sim_summary(lib.simulate(df, s, sel0, inv, taker_entry=True))
json.dump(res, open(a.out, "w"), indent=1, default=float)


def show(name):
    r = res[name]
    print(f"{name:18s} n={r['n']:6d} acc={r['acc']:.4f} (t={r['acc_t']:.2f}) base={r['base']:.3f} BA={r['ba']:.4f} AUC={r['auc']:.4f} IC={r['ic']:.4f} (t={r['ic_t']:.2f})"
          f" | cov50/20/10/5 acc: " + "/".join(f"{r['cov'][c]['acc']:.3f}" for c in lib.COVERAGES[1:])
          + f" | months>50%: {r['months_above_50']}/{len(r['months'])}")


print(f"== {a.split}: n={res['n']} timestamps={res['timestamps']} balance={res['class_balance']}")
for k in cands:
    show(k)
print("p-values:", json.dumps(res["p_values"], default=float))
print("all rules (descriptive):\n", pd.DataFrame(res["all_rules_desc"]).T.round(4))
print("asym_real:", json.dumps(res["asym_real"], default=float))
print("close_class:", json.dumps(res["close_class"], default=float))
print("magnitude:\n", pd.DataFrame({h: {k: v for k, v in d.items() if not k.startswith("auc_q")} | {"auc_q_head_mean": np.mean(d["auc_q_head"]), "auc_q_vol_mean": np.mean(d["auc_q_vol"])} for h, d in res["magnitude_vs_vol"].items()}).T.round(4))
print("collapse:", json.dumps({k: v for k, v in res["collapse"].items() if k in ("p_spread", "near_zero_rows", "acc_pc_near_zero")}, default=float))
print("collapse epi/ale share<0.05:", {k: v["share_lt_0.05"] for k, v in res["collapse"].items() if isinstance(v, dict) and "share_lt_0.05" in v})
print("trade:\n", pd.DataFrame(res["trade"]).T.round(3).to_string())
