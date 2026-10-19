"""
PURPOSE:  E-natr-003 Stage A (phase R3, reopening F-0083): localize the top-NATR-decile under-coverage on train data (forward-chain
          out-of-fold, pre-registered rule), fit the base 1h log-log quantile regression on 2020-2023 and a per-NATR-decile
          multiplicative p10/p90 correction on the 2024 calibration slice (train only), and judge the corrected forecast on the
          exploration val 2025-01..2026-09 of the 10 open coins with the E-natr-001/002 rule set plus the per-year top-decile rule.
TAGS:     NATR range forecast, E-natr-003, R3, high-NATR calibration, split conformal, CQR, decile correction, localization, F-0083,
          stale bars, Colab job, exploration
PITFALLS: Reads data ONLY through holdout_guard.load_unsealed (window end <= 2026-10-01; a sealed coin raises); no sealed bucket is opened.
          Windows are checked BEFORE any data is loaded: no fit window may reach 2025-01-01 (natr_range.TRAIN_END). Unknown config keys
          raise. Padded history is masked by natr_range.drop_stale_runs (the loader does not require volume > 0). The control arm
          (same base fit, no correction) is reported so the effect of the correction is separable from the 2020-2023 window.
          Accepted needs the localization label "static" (card Step 1). Stage B (forward data >= 2026-10-01) is a different card.

Usage (via the Colab runner job research/natr_range/e_natr_003_job.json):
    python research/natr_range/run_natr_calibrate.py --config <json> --out <dir> [--max-minutes 30]
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
import run_natr_confirm as rc       # noqa: E402  (sparse_verdict: the E-natr-002 sparse-decile rule)

CARD = "research/natr_range/cards/E-natr-003.md"
DEFAULTS = {
    "data_dir": None, "coins": ["ADAUSDT", "BCHUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "ETHUSDT", "SOLUSDT", "TRXUSDT", "XRPUSDT", "ZECUSDT"],
    "pattern": "{symbol}.csv.gz", "source_minutes": 15, "load_start": "2019-10-01", "end": "2026-10-01",
    "fit_start": "2020-01-01", "base_end": "2024-01-01", "calib_end": "2025-01-01", "val_end": "2026-10-01",
    "stale_min_run_hours": 168, "oof_first_year": 2021, "oof_last_year": 2024, "oof_start_year": 2020, "min_coins": 8,
}
SPARSE = {"sparse_rule": {"min_decile_n": 200, "min_evaluable_deciles": 6, "pinball_win_frac": 0.8}}   # E-natr-002 amendment, unchanged
VAL_YEARS = (2025, 2026)
MIN_TOP_N = 200                       # top-decile rows needed per val year (card rule 5)
STEP = 60


class Deadline(Exception):
    pass


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


def _ts(s):
    return pd.Timestamp(s, tz="UTC")


def check_windows(cfg):
    """Raises before any data is loaded: fit_start < base_end <= calib_end <= TRAIN_END (2025-01-01) < val_end <= the time seal."""
    fs, be, ce, ve = (_ts(cfg[k]) for k in ("fit_start", "base_end", "calib_end", "val_end"))
    if not (fs < be <= ce):
        raise ValueError(f"need fit_start < base_end <= calib_end, got {fs}, {be}, {ce}")
    if ce > nr.TRAIN_END:
        raise ValueError(f"calib_end {ce} is after the train end {nr.TRAIN_END}: no 2025+ data may be used for fitting")
    if not ce < ve:
        raise ValueError("val_end must be after calib_end")
    if cfg["oof_last_year"] >= nr.TRAIN_END.year:
        raise ValueError("oof_last_year must be a train year (< 2025)")
    return fs, be, ce, ve


def stage_a_verdict(sv, top, ctrl_top, loc, n_coins, min_coins):
    """Card rules on top of the E-natr-002 sparse-rule verdict `sv` (rules 1-4): 5 top decile 74..86 % in each val year with >= 200 rows,
    6 the corrected top-decile coverage closer to 80 % than the control's in each val year, 7 enough coins and localization == static."""
    eps = 1e-12
    year_ok, closer_ok = True, True
    for yr in VAL_YEARS:
        t = top.get(yr)
        if t is None or t["n"] < MIN_TOP_N or not np.isfinite(t["coverage"]) or abs(t["coverage"] - nr.COVER_TARGET) > nr.DECILE_TOL + eps:
            year_ok = False
        c = ctrl_top.get(yr)
        if t is None or c is None or not np.isfinite(t["coverage"]) or not np.isfinite(c) or not abs(t["coverage"] - nr.COVER_TARGET) < abs(c - nr.COVER_TARGET):
            closer_ok = False
    out = {k: sv[k] for k in sv if k not in ("accepted",)}
    out.update({"top_decile_year_ok": year_ok, "top_decile_closer_than_control_ok": closer_ok, "coins_ok": bool(n_coins >= min_coins),
                "localization_static_ok": loc["label"] == "static", "rules_1_to_4_ok": bool(sv["accepted"])})
    out["accepted"] = bool(sv["accepted"] and year_ok and closer_ok and out["coins_ok"] and out["localization_static_ok"])
    return out


def _tables(ev, arm):
    d = ev["deciles"].copy()
    d.insert(0, "arm", arm)
    return d


def run(cfg, out_dir, max_minutes, lock=None):
    t0 = time.time()
    cfg = complete_config(cfg)
    fs, be, ce, ve = check_windows(cfg)                              # cheap failures first, before any data is touched
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
            "fit_start": str(fs), "base_end": str(be), "calib_end": str(ce), "val_end": str(ve)}}
        if len(allr) == 0:
            raise ValueError("no rows at all")
        t = allr["tgt_open"]
        base, calib = allr[(t >= fs) & (t < be)], allr[(t >= be) & (t < ce)]
        val_all = allr[(t >= ce) & (t < ve)]
        need = [f"b_{tg}_{nr.qname(q)}" for tg in nr.TARGETS for q in nr.TAUS]
        val = val_all.dropna(subset=need)
        coins_all = set(base["coin"]) & set(calib["coin"]) & set(val["coin"])
        # --- step 1: localization on train data only
        train_rows = allr[(t >= fs) & (t < ce)]
        oof = nr.oof_decile_diagnostic(train_rows, cfg["oof_first_year"], cfg["oof_last_year"], cfg["oof_start_year"])
        # --- the change: base fit on `base`, correction fitted on `calib` (train only; both guarded in code)
        models = nr.fit_models_until(base, be)
        edges = nr.decile_edges(train_rows["natr"])
        corr = nr.fit_decile_corrections(models, edges, calib, fit_end=ce)
        ev_ctrl = nr.evaluate_frozen(models, edges, val)
        ev_cor = nr.evaluate_frozen(models, edges, val, corr)
        yd_ctrl = nr.decile_year_coverage(models, edges, val)
        yd_cor = nr.decile_year_coverage(models, edges, val, corr)
        top = {int(r.year): {"coverage": float(r.coverage), "n": int(r.n)} for r in yd_cor[yd_cor["decile"] == nr.TOP_DECILE].itertuples()}
        ctrl_top = {int(r.year): float(r.coverage) for r in yd_ctrl[yd_ctrl["decile"] == nr.TOP_DECILE].itertuples()}
        loc = nr.classify_localization(oof, ctrl_top)
        sv = rc.sparse_verdict(ev_cor, SPARSE)
        verdict = stage_a_verdict(sv, top, ctrl_top, loc, len(coins_all), cfg["min_coins"])
        verdict_ctrl = rc.sparse_verdict(ev_ctrl, SPARSE)
        fit_rows = pd.concat([base, calib])
        factors = corr["range"]["factors"]
        res.update({"localization": loc, "verdict": verdict, "control_verdict_reported_not_ruled": {k: v for k, v in verdict_ctrl.items() if k != "reason"},
                    "top_decile_corrected": top, "top_decile_control": ctrl_top, "n_coins_all_windows": len(coins_all),
                    "n_fit_rows_at_or_after_2025": int((pd.DatetimeIndex(fit_rows["tgt_open"]) >= nr.TRAIN_END).sum()),
                    "n_base_rows": int(len(base)), "n_calib_rows": int(len(calib)), "n_val_rows": int(len(val)),
                    "n_val_rows_dropped_no_baseline": int(len(val_all) - len(val)), "n_crossed_val": {"corrected": ev_cor["n_crossed"], "control": ev_ctrl["n_crossed"]},
                    "extras_corrected": {k: v for k, v in ev_cor["extras"].items() if not isinstance(v, dict)},
                    "extras_control": {k: v for k, v in ev_ctrl["extras"].items() if not isinstance(v, dict)}})
        pd.concat([_tables(ev_ctrl, "control"), _tables(ev_cor, "corrected")]).round(6).to_csv(os.path.join(out_dir, "deciles.csv"), index=False)
        pd.concat([yd_ctrl.assign(arm="control"), yd_cor.assign(arm="corrected")]).round(6).to_csv(os.path.join(out_dir, "year_deciles.csv"), index=False)
        oof.round(6).to_csv(os.path.join(out_dir, "oof_diagnostic.csv"), index=False)
        pd.DataFrame({"decile": range(10), "n_calib": corr["range"]["n"], "c10": factors[:, 0], "c50": factors[:, 1], "c90": factors[:, 2]}).round(6).to_csv(
            os.path.join(out_dir, "corrections.csv"), index=False)
        pd.DataFrame([{**{k: v for k, v in verdict.items() if not isinstance(v, (list, dict))}, "localization": loc["label"],
                       "pooled_oof_top_coverage": loc["pooled_oof_top_coverage"], **{f"corrected_{k}": v for k, v in res["extras_corrected"].items()},
                       **{f"control_{k}": v for k, v in res["extras_control"].items()}}]).round(6).to_csv(os.path.join(out_dir, "summary.csv"), index=False)
        pd.DataFrame(ev_cor["months"]).round(6).to_csv(os.path.join(out_dir, "months.csv"), index=False)
        frozen = {"version": 1, "card": CARD, "timeframe": "1h", "step_minutes": STEP, "source_minutes": cfg["source_minutes"], "natr_period": nr.NATR_PERIOD,
                  "baseline_window": nr.BASELINE_WINDOW, "windows": res["windows"], "coins": sorted(coins_all),
                  "models": {f"{tg}_{nr.qname(q)}": {"a": ab[0], "b": ab[1]} for (tg, q), ab in models.items()},
                  "edges": [float(e) for e in edges], "correction_factors": factors.tolist(), "correction_n_calib": [int(n) for n in corr["range"]["n"]],
                  "rule": {"cover_target": nr.COVER_TARGET, "overall_tol": nr.OVERALL_TOL, "decile_tol": nr.DECILE_TOL,
                           "min_pinball_deciles": nr.MIN_PINBALL_DECILES, "ic_month_share_pct": nr.MIN_IC_MONTH_SHARE_PCT}, **SPARSE}
        with open(os.path.join(out_dir, "frozen_stage_b.json"), "w") as fh:
            fh.write(json.dumps(frozen, indent=1, sort_keys=True) + "\n")
    except Deadline:
        truncated = True
        res = {"card": CARD, "truncated": True, "verdict": {"accepted": False, "evaluable": False}}
    res["truncated"] = truncated
    res["runtime_min"] = round((time.time() - t0) / 60, 2)
    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=float)
    n_fits = len(nr.TARGETS) * len(nr.TAUS) + (cfg["oof_last_year"] - cfg["oof_first_year"] + 1) * len(nr.TAUS)
    manifest = cr.write_manifest(out_dir, cfg, inputs=inputs, input_root=cfg["data_dir"], row_counts=counts, runtime_seconds=time.time() - t0,
                                 n_configurations=n_fits, repo_dir=ROOT,
                                 extra={"truncated": truncated, "card": CARD, "accept_rule_evaluations": 1, "baseline_computations": 2})
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
    print("localization:", json.dumps(res.get("localization", {}).get("label")))
    print(json.dumps(res["verdict"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
