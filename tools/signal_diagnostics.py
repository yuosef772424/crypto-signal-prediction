#!/usr/bin/env python3
"""
PURPOSE:  Are the model's confidence (conf_<t>) and uncertainty (wst_<t>, the Student-t width in return units) informative about its error, or noise? On the signals table (workflow.selective_eval.collect_signals / the `signals_{val,test}.csv.gz` exported by run_reports) it computes, per head: the pooled Spearman between the signal and the absolute error, the WITHIN-ASSET mean Spearman (assets differ in volatility, which makes any scale-like quantity correlate with the error pooled), a permutation null (the signal shuffled inside each asset: z score and p), and decile tables of the mean absolute error by signal decile.
TAGS:     confidence, uncertainty, calibration, informative signal, permutation test, within-asset correlation, Spearman, decile table, signals csv, S03
PITFALLS: The error is |actual - predicted| in return units: actual = fut_<t>/entry - 1 and predicted = mu_<t> (return modes), or |fut_<t> - pred_<t>|/entry in entry_range tables (columns pred_high/pred_low); close in entry_range is skipped. Higher uncertainty should go with a LARGER error (positive rho) and higher confidence with a SMALLER one (negative rho); a clean monotone relation of the opposite sign is information too (an inverted signal), not randomness: read the sign. With ~10^5 rows any non-zero rho is "significant": judge the effect size (rho, decile spread), not only z. Assets with fewer than min_n rows are left out of the within-asset statistic. The statistic of the null is the same mean-of-asset-Spearmans, so pooled and within-asset numbers answer different questions. The model that produced the table decides what this says: a 2-epoch profile model says little about a fully trained one.

Usage:  python tools/signal_diagnostics.py signals_test.csv.gz [signals_val.csv.gz ...] [--targets high,low,close] [--perm 200]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from core.schema import TARGET_COLUMNS as HEAD_TARGETS  # noqa: E402  (the one definition of the head names)


def _error_and_signals(df, t):
    """(abs error in return units, confidence, uncertainty) of head ``t`` or None when the table lacks it."""
    need = {"entry", f"fut_{t}"}
    if not need <= set(df.columns):
        return None
    entry = df["entry"].to_numpy("float64")
    if f"pred_{t}" in df.columns:                                        # entry_range tables: prices
        err = np.abs(df[f"fut_{t}"].to_numpy("float64") - df[f"pred_{t}"].to_numpy("float64")) / entry
    elif f"mu_{t}" in df.columns and "entry_close_reg" not in df.columns:
        err = np.abs(df[f"fut_{t}"].to_numpy("float64") / entry - 1.0 - df[f"mu_{t}"].to_numpy("float64"))
    else:
        return None
    conf = df[f"conf_{t}"].to_numpy("float64") if f"conf_{t}" in df.columns else None
    unc = df[f"wst_{t}"].to_numpy("float64") if f"wst_{t}" in df.columns else None
    return err, conf, unc


def _rank(a):
    return pd.Series(a).rank(method="average").to_numpy()


def _within_asset(assets, x, y, n_perm, seed, min_n):
    """Mean over assets of the Spearman(x, y) and its permutation null (x shuffled inside each asset)."""
    groups = []
    for _, idx in pd.Series(np.arange(len(assets))).groupby(np.asarray(assets)):
        idx = idx.to_numpy()
        if len(idx) < min_n:
            continue
        rx, ry = _rank(x[idx]), _rank(y[idx])
        rx, ry = rx - rx.mean(), ry - ry.mean()
        d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
        if d > 0:
            groups.append((rx, ry, d))
    if not groups:
        return {"n_assets": 0, "mean_rho": float("nan"), "null_mean": float("nan"), "null_std": float("nan"), "z": float("nan"), "p_two_sided": float("nan")}
    obs = float(np.mean([(rx * ry).sum() / d for rx, ry, d in groups]))
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for b in range(n_perm):
        null[b] = np.mean([(rx[rng.permutation(len(rx))] * ry).sum() / d for rx, ry, d in groups])
    sd = float(null.std(ddof=1)) if n_perm > 1 else float("nan")
    p = float((1 + np.sum(np.abs(null - null.mean()) >= abs(obs - null.mean()))) / (n_perm + 1))
    return {"n_assets": len(groups), "mean_rho": obs, "null_mean": float(null.mean()), "null_std": sd,
            "z": (obs - float(null.mean())) / sd if sd and sd > 0 else float("nan"), "p_two_sided": p}


def _deciles(signal, err, bins=10):
    q = pd.qcut(pd.Series(signal).rank(method="first"), bins, labels=False)
    g = pd.DataFrame({"q": q, "err": err}).groupby("q")["err"].agg(["mean", "size"])
    return {"mean_abs_error": [float(v) for v in g["mean"]], "n": [int(v) for v in g["size"]],
            "decile_trend_rho": float(pd.Series(g["mean"].to_numpy()).corr(pd.Series(np.arange(len(g))), method="spearman"))}


def check_head(df, t, n_perm=200, seed=0, min_n=50, bins=10):
    """The diagnostics of head ``t`` (dict) or None when the table cannot give its error."""
    got = _error_and_signals(df, t)
    if got is None:
        return None
    err, conf, unc = got
    ok = np.isfinite(err)
    out = {"n": int(ok.sum()), "mean_abs_error": float(err[ok].mean())}
    assets = df["asset"].to_numpy() if "asset" in df.columns else np.zeros(len(df), dtype=int)
    for name, sig in (("confidence", conf), ("uncertainty", unc)):
        if sig is None:
            continue
        m = ok & np.isfinite(sig)
        if m.sum() < 3 * min_n:
            continue
        out[name] = {"pooled_rho": float(pd.Series(sig[m]).corr(pd.Series(err[m]), method="spearman")),
                     "within_asset": _within_asset(assets[m], sig[m], err[m], n_perm, seed, min_n),
                     "deciles": _deciles(sig[m], err[m], bins),
                     "expected_sign": "negative" if name == "confidence" else "positive"}
    return out


def check_all(df, targets=HEAD_TARGETS, **kw):
    return {t: r for t in targets if (r := check_head(df, t, **kw)) is not None}


def verdict(r):
    """One line per signal: informative (right sign) / inverted (opposite sign) / noise, by the within-asset effect and its null."""
    lines = []
    for t, h in r.items():
        for name in ("confidence", "uncertainty"):
            if name not in h:
                continue
            w = h[name]["within_asset"]
            rho, z = w["mean_rho"], w["z"]
            want = -1 if name == "confidence" else 1
            if not np.isfinite(rho) or abs(z) < 3 or abs(rho) < 0.02:
                kind = "noise-like (effect < 0.02 or |z| < 3)"
            else:
                kind = "informative (expected sign)" if np.sign(rho) == want else "INVERTED (opposite sign)"
            lines.append(f"{t:>5} {name:<11} within-asset rho {rho:+.3f} (z {z:+.1f}), pooled {h[name]['pooled_rho']:+.3f}: {kind}")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tables", nargs="+")
    ap.add_argument("--targets", default=",".join(HEAD_TARGETS))
    ap.add_argument("--perm", type=int, default=200)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    res = {}
    for path in a.tables:
        res[os.path.basename(path)] = check_all(pd.read_csv(path), targets=tuple(a.targets.split(",")), n_perm=a.perm)
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        for name, r in res.items():
            print(f"== {name}")
            print("\n".join(verdict(r)) or "(no head with an error definition in this table)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
