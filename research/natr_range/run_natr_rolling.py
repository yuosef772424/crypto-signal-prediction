"""
PURPOSE:  E-natr-004 Stage A (phase R3, reopening F-0084): rolling recalibration of the per-NATR-decile p10/p90 correction of the 1h
          range forecast. Base regression fitted once on 2020-2023, decile edges fixed (pooled NATR 2020-2024), factors refitted at the
          start of every calendar month on the trailing 12 months of REALIZED rows strictly before that month (target bar ended before
          the month starts; deciles with < 200 window rows use the pooled factor). Three arms on the exploration val 2025-01..2026-09 of
          the 10 open coins: control (no correction), static (E-natr-003 2024 factors), rolling (arm under test, ruled).
TAGS:     NATR range forecast, E-natr-004, R3, rolling recalibration, adaptive conformal, decile correction, F-0084, drift, Colab job,
          exploration
PITFALLS: Reads data ONLY through holdout_guard.load_unsealed (window end <= 2026-10-01; a sealed coin raises); no sealed bucket is opened and
          the forward data (>= 2026-10-01, Stage B) is NOT read: Stage B needs a reviewed forward-read guard path that does not exist yet.
          The 12-month window, monthly refit, 200-row minimum and the rule 8/9 constants are constants of the card (natr_range.ROLL_*,
          STAB_*, QUARTER_*), not config keys: unknown config keys raise. The static and control arms are reported, only rolling is
          ruled. The refit raises on any window row whose target ends after the refit month starts. E-natr-003 code is reused, not changed.

Usage (via the Colab runner job research/natr_range/e_natr_004_job.json):
    python research/natr_range/run_natr_rolling.py --config <json> --out <dir> [--max-minutes 30]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import colab_runner as cr           # noqa: E402
import holdout_guard as hg          # noqa: E402
import natr_range as nr             # noqa: E402
import run_natr_calibrate as rcal   # noqa: E402  (E-natr-003 pieces: windows, sparse rule, top-decile rules)
import run_natr_confirm as rc       # noqa: E402

CARD = "research/natr_range/cards/E-natr-004.md"
_NO_OOF = ("oof_first_year", "oof_last_year", "oof_start_year")
DEFAULTS = {k: v for k, v in rcal.DEFAULTS.items() if k not in _NO_OOF}
SPARSE, VAL_YEARS, MIN_TOP_N, STEP, Deadline = rcal.SPARSE, rcal.VAL_YEARS, rcal.MIN_TOP_N, rcal.STEP, rcal.Deadline


def complete_config(cfg):
    bad = set(cfg) - set(DEFAULTS)
    if bad:
        raise KeyError(f"unknown config keys: {sorted(bad)}")
    out = {**DEFAULTS, **cfg}
    if not out["data_dir"]:
        raise KeyError("data_dir is required")
    return out


def load_config(path):
    return complete_config(json.load(open(path)))


def rolling_verdict(sv, top, ctrl_top, quarters, stab, n_coins, min_coins):
    """Card rules on top of the E-natr-002 sparse-rule verdict `sv` (rules 1-4): 5 top decile 74..86 % in each val year with >= 200 rows,
    6 closer to 80 % than the control in each year, 7 enough coins, 8 every quarter with >= 200 top-decile rows within 74..86 % and >= 5 such
    quarters, 9 factor stability (max |ln change| <= 0.05, mean <= 0.02). `quarters` = quarter_top_coverage table of the rolling arm."""
    eps = 1e-12
    year_ok, closer_ok = True, True
    for yr in VAL_YEARS:
        t, c = top.get(yr), ctrl_top.get(yr)
        if t is None or t["n"] < MIN_TOP_N or not np.isfinite(t["coverage"]) or abs(t["coverage"] - nr.COVER_TARGET) > nr.DECILE_TOL + eps:
            year_ok = False
        if t is None or c is None or not np.isfinite(t["coverage"]) or not np.isfinite(c) or not abs(t["coverage"] - nr.COVER_TARGET) < abs(c - nr.COVER_TARGET):
            closer_ok = False
    ev = quarters[quarters["n"] >= nr.QUARTER_MIN_N]
    quarters_ok = bool(len(ev) >= nr.QUARTER_MIN_EVALUABLE and (np.abs(ev["coverage"] - nr.COVER_TARGET) <= nr.DECILE_TOL + eps).all())
    stab_ok = bool(stab["n_changes"] > 0 and stab["max_abs_ln"] <= nr.STAB_MAX_LN + eps and stab["mean_abs_ln"] <= nr.STAB_MEAN_LN + eps)
    out = {k: sv[k] for k in sv if k != "accepted"}
    out.update({"top_decile_year_ok": year_ok, "top_decile_closer_than_control_ok": closer_ok, "coins_ok": bool(n_coins >= min_coins),
                "quarters_ok": quarters_ok, "n_evaluable_quarters": int(len(ev)), "stability_ok": stab_ok, "rules_1_to_4_ok": bool(sv["accepted"])})
    out["accepted"] = bool(sv["accepted"] and year_ok and closer_ok and out["coins_ok"] and quarters_ok and stab_ok)
    return out


def _arm_tables(ev, yd, arm):
    d = ev["deciles"].copy()
    d.insert(0, "arm", arm)
    return d, yd.assign(arm=arm)


def run(cfg, out_dir, max_minutes, lock=None):
    t0 = time.time()
    cfg = complete_config(cfg)
    fs, be, ce, ve = rcal.check_windows({**cfg, "oof_last_year": 2024})          # cheap failures first, before any data is touched
    os.makedirs(out_dir, exist_ok=True)
    parts, inputs, counts, skipped, stale_dropped = [], [], {}, {}, {}
    truncated = False
    try:
        for sym in cfg["coins"]:
            if time.time() - t0 > max_minutes * 60:
                raise Deadline
            try:
                d = hg.load_unsealed(cfg["data_dir"], sym, cfg["load_start"], cfg["end"], pattern=cfg["pattern"], lock=lock)
            except FileNotFoundError:
                skipped[sym] = "file not found"
                continue
            if len(d) == 0:
                skipped[sym] = "no rows in window"
                continue
            inputs.append(cfg["pattern"].format(symbol=sym))
            n0 = len(d)
            d = nr.drop_stale_runs(d, cfg["stale_min_run_hours"], cfg["source_minutes"])
            stale_dropped[sym] = int(n0 - len(d))
            bars = nr.resample_bars(d, STEP, cfg["source_minutes"])
            if len(bars) < nr.BASELINE_WINDOW + nr.NATR_PERIOD + 2:
                skipped[sym] = "too few 1h bars"
                continue
            r = nr.build_rows(bars, STEP)
            r["coin"] = sym
            parts.append(r)
            counts[f"{sym}_source_bars"], counts[f"{sym}_rows"] = int(n0), int(len(r))
            del d, bars
        allr = pd.concat(parts) if parts else pd.DataFrame()
        res = {"card": CARD, "truncated": False, "skipped_coins": skipped, "stale_source_rows_dropped": stale_dropped, "windows": {
            "fit_start": str(fs), "base_end": str(be), "calib_end": str(ce), "val_end": str(ve)},
            "hyper_parameters": {"window_months": nr.ROLL_WINDOW_MONTHS, "refit": "monthly", "min_rows_per_decile": nr.ROLL_MIN_N,
                                 "fallback": "pooled factor", "stability_max_abs_ln": nr.STAB_MAX_LN, "stability_mean_abs_ln": nr.STAB_MEAN_LN}}
        if len(allr) == 0:
            raise ValueError("no rows at all")
        t = allr["tgt_open"]
        base, calib = allr[(t >= fs) & (t < be)], allr[(t >= be) & (t < ce)]
        val_all = allr[(t >= ce) & (t < ve)]
        need = [f"b_{tg}_{nr.qname(q)}" for tg in nr.TARGETS for q in nr.TAUS]
        val = val_all.dropna(subset=need)
        coins_all = set(base["coin"]) & set(calib["coin"]) & set(val["coin"])
        # --- fixed pieces (as E-natr-003): base fit on `base`, edges from train rows 2020-2024
        train_rows = allr[(t >= fs) & (t < ce)]
        models = nr.fit_models_until(base, be)
        edges = nr.decile_edges(train_rows["natr"])
        static = nr.fit_decile_corrections(models, edges, calib, fit_end=ce)
        # --- the change: monthly rolling factors from realized rows strictly before each month (guarded in code); val rows of the
        # month itself are never used for that month. `allr` holds every realized row, val rows included, which is allowed because the
        # window ends before the month starts.
        row_f, ftab = nr.rolling_row_factors(models, edges, allr, val)
        rolling = {"range": {"row_factors": row_f}}
        first = ftab[ftab["month"] == ftab["month"].min()].sort_values("decile")
        jan_vs_static = float(np.max(np.abs(np.column_stack([first["c10"], first["c90"]]) - static["range"]["factors"][:, [0, 2]])))
        arms = {"control": None, "static": static, "rolling": rolling}
        ev = {a: nr.evaluate_frozen(models, edges, val, c) for a, c in arms.items()}
        yd = {a: nr.decile_year_coverage(models, edges, val, c) for a, c in arms.items()}
        qt = {a: nr.quarter_top_coverage(models, edges, val, c) for a, c in arms.items()}
        tops = {a: {int(r.year): {"coverage": float(r.coverage), "n": int(r.n)} for r in yd[a][yd[a]["decile"] == nr.TOP_DECILE].itertuples()} for a in arms}
        ctrl_top = {y: v["coverage"] for y, v in tops["control"].items()}
        stab = nr.factor_stability(ftab)
        sv = {a: rc.sparse_verdict(ev[a], SPARSE) for a in arms}
        verdict = rolling_verdict(sv["rolling"], tops["rolling"], ctrl_top, qt["rolling"], stab, len(coins_all), cfg["min_coins"])
        res.update({"verdict": verdict, "static_verdict_reported_not_ruled": {k: v for k, v in sv["static"].items() if k != "reason"},
                    "control_verdict_reported_not_ruled": {k: v for k, v in sv["control"].items() if k != "reason"},
                    "top_decile": tops, "stability": stab, "n_fallback_decile_months": int(ftab["fallback"].sum()),
                    "n_refits": int(ftab["month"].nunique()), "rolling_first_month_vs_static_max_abs_diff": jan_vs_static,
                    "n_coins_all_windows": len(coins_all), "n_base_rows": int(len(base)), "n_calib_rows": int(len(calib)), "n_val_rows": int(len(val)),
                    "n_val_rows_dropped_no_baseline": int(len(val_all) - len(val)),
                    "n_crossed_val": {a: ev[a]["n_crossed"] for a in arms},
                    "extras": {a: {k: v for k, v in ev[a]["extras"].items() if not isinstance(v, dict)} for a in arms}})
        dts, yds = zip(*[_arm_tables(ev[a], yd[a], a) for a in arms])
        pd.concat(dts).round(6).to_csv(os.path.join(out_dir, "deciles.csv"), index=False)
        pd.concat(yds).round(6).to_csv(os.path.join(out_dir, "year_deciles.csv"), index=False)
        pd.concat([qt[a].assign(arm=a) for a in arms]).round(6).to_csv(os.path.join(out_dir, "quarters.csv"), index=False)
        ftab.round(6).to_csv(os.path.join(out_dir, "monthly_factors.csv"), index=False)
        pd.DataFrame(ev["rolling"]["months"]).round(6).to_csv(os.path.join(out_dir, "months.csv"), index=False)
        pd.DataFrame([{**{k: v for k, v in verdict.items() if not isinstance(v, (list, dict))}, "stability_max_abs_ln": stab["max_abs_ln"],
                       "stability_mean_abs_ln": stab["mean_abs_ln"],
                       **{f"{a}_{k}": v for a in arms for k, v in res["extras"][a].items() if k in ("pinball_model", "pinball_baseline")}}]).round(6).to_csv(
            os.path.join(out_dir, "summary.csv"), index=False)
        frozen = {"version": 1, "card": CARD, "timeframe": "1h", "step_minutes": STEP, "source_minutes": cfg["source_minutes"], "natr_period": nr.NATR_PERIOD,
                  "baseline_window": nr.BASELINE_WINDOW, "windows": res["windows"], "coins": sorted(coins_all),
                  "models": {f"{tg}_{nr.qname(q)}": {"a": ab[0], "b": ab[1]} for (tg, q), ab in models.items()},
                  "edges": [float(e) for e in edges], "hyper_parameters": res["hyper_parameters"],
                  "static_correction_factors": static["range"]["factors"].tolist(),
                  "rule": {"cover_target": nr.COVER_TARGET, "overall_tol": nr.OVERALL_TOL, "decile_tol": nr.DECILE_TOL,
                           "min_pinball_deciles": nr.MIN_PINBALL_DECILES, "ic_month_share_pct": nr.MIN_IC_MONTH_SHARE_PCT,
                           "quarter_min_n": nr.QUARTER_MIN_N, "quarter_min_evaluable": nr.QUARTER_MIN_EVALUABLE}, **SPARSE}
        with open(os.path.join(out_dir, "frozen_stage_b.json"), "w") as fh:
            fh.write(json.dumps(frozen, indent=1, sort_keys=True) + "\n")
    except Deadline:
        truncated = True
        res = {"card": CARD, "truncated": True, "verdict": {"accepted": False, "evaluable": False}}
    res["truncated"] = truncated
    res["runtime_min"] = round((time.time() - t0) / 60, 2)
    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=float)
    n_fits = len(nr.TARGETS) * len(nr.TAUS)
    manifest = cr.write_manifest(out_dir, cfg, inputs=inputs, input_root=cfg["data_dir"], row_counts=counts, runtime_seconds=time.time() - t0,
                                 n_configurations=n_fits, repo_dir=ROOT,
                                 extra={"truncated": truncated, "card": CARD, "accept_rule_evaluations": 1, "baseline_computations": 3})
    return manifest, res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-minutes", type=float, default=30)
    a = ap.parse_args(argv)
    manifest, res = run(load_config(a.config), a.out, a.max_minutes)
    print(json.dumps(manifest["outputs"], indent=1))
    print("truncated:", manifest["truncated"], "minutes:", round(manifest["runtime_seconds"] / 60, 2))
    print(json.dumps(res["verdict"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
