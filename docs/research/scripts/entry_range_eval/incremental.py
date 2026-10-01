"""Post-hoc (after the single frozen TEST run; NOT used for any selection): are the model's cross-sectional rank ICs
just the short-term reversal in disguise? Reports pooled Spearman with -RSI14 and the partial IC controlling -RSI14
(per-timestamp partial Spearman) on VAL and TEST for the main candidate scores."""
import argparse
import json

import numpy as np
import pandas as pd

import lib

ap = argparse.ArgumentParser()
ap.add_argument("--npz", required=True)
ap.add_argument("--frozen", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
fz = json.load(open(a.frozen))
out = {}
for split in ("val", "test"):
    df, _ = lib.load_split(a.npz, split, coin_rv=fz["coin_rv"])
    S, _ = lib.base_scores(df, zc=fz["zc"])
    B = lib.baseline_scores(df)
    rows = {}
    for name, s in {"p_mean_inverted": -S["p_mean"], "asym": S["asym"], "nasym": S["nasym"], "pc": S["pc"], "asym_agree_pc": S["asym_agree_pc"]}.items():
        ic, t, _, _ = lib.rank_ic(df, s)
        rho = float(pd.Series(s).corr(pd.Series(B["neg_rsi14"]), method="spearman"))
        p_rsi = lib.partial_ic(df, s, df.ret.values, B["neg_rsi14"])
        p_ret1 = lib.partial_ic(df, s, df.ret.values, B["neg_ret1"])
        rows[name] = {"ic": ic, "ic_t": t, "rho_with_neg_rsi14": rho, "partial_ic_ctrl_rsi": p_rsi[0], "partial_t_rsi": p_rsi[1],
                      "partial_ic_ctrl_ret1": p_ret1[0], "partial_t_ret1": p_ret1[1]}
    out[split] = rows
    print(split, "\n", pd.DataFrame(rows).T.round(4).to_string())
json.dump(out, open(a.out, "w"), indent=1, default=float)
