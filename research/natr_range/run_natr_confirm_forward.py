"""
PURPOSE:  E-natr-004-confirm (phase R3, Stage B; PREPARED, NOT QUEUED): the forward confirmation of the rolling 1h NATR range calibration.
          Loads the FROZEN Stage A config (base coefficients, decile edges, hyper-parameters), reads the 10 open coins' forward bars
          (open time >= 2026-10-01) through holdout_guard.load_forward, refits the p10/p90 decile factors every month on realized rows
          strictly before it, and applies the card's accept rules once to the pooled forward window (control and static reported).
TAGS:     NATR range forecast, E-natr-004-confirm, R3, Stage B, forward confirmation, time seal, load_forward, rolling recalibration,
          frozen config, Colab job
PITFALLS: Not allowed before the evaluation end (2027-01-01, or 2027-04-01 after the one pre-registered extension): the run refuses
          when `forward_end` is later than today. Everything that can fail cheaply (config keys, card vs frozen-config / code hashes,
          rule constants) is checked BEFORE any bar is read. Forward bars come ONLY from load_forward (one log row, copied to
          access_log_row.csv for the reviewer to commit); bars before 2026-10-01 come from load_unsealed and are history only (NATR,
          90-bar baseline, rolling windows): rows with target open < 2026-10-01 are never evaluated. Sealed coins raise. No sealed
          bucket is opened. Nothing is refitted except the monthly factors.

Usage (via the Colab runner job research/natr_range/e_natr_004_confirm_job.json, never by hand):
    python research/natr_range/run_natr_confirm_forward.py --config <json> --out <dir> [--max-minutes 30]
"""
import argparse
import hashlib
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
import run_natr_confirm as rc       # noqa: E402  (sparse_verdict)

CARD = "research/natr_range/cards/E-natr-004-confirm.md"
ALLOWED_ENDS = ("2027-01-01", "2027-04-01")          # first evaluation, and the ONE pre-registered 3-month extension
DEFAULTS = {"data_dir": None, "coins": None, "pattern": "{symbol}.csv.gz", "source_minutes": 15, "history_start": "2025-06-01",
            "forward_end": "2027-01-01", "stale_min_run_hours": 168, "min_coins": 8}
MIN_TOP_N, MIN_EVAL_DECILES, STEP = 200, 6, 60
STAB_ANCHOR = pd.Timestamp("2026-09-01", tz="UTC")   # first refit month of the stability rule (computed from pre-seal rows only)


class Deadline(Exception):
    pass


def complete_config(cfg):
    bad = set(cfg) - set(DEFAULTS)
    if bad:
        raise KeyError(f"unknown config keys: {sorted(bad)}")
    out = {**DEFAULTS, **cfg}
    if not out["data_dir"]:
        raise KeyError("data_dir is required")
    if out["forward_end"] not in ALLOWED_ENDS:
        raise ValueError(f"forward_end must be one of {ALLOWED_ENDS} (card: first evaluation or the one extension)")
    return out


def load_config(path):
    return complete_config(json.load(open(path)))


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def check_frozen():
    """Card vs frozen config / code hashes and rule constants. Raises before any bar is read."""
    meta, _ = hg._parse_card(_abs(CARD))
    fp = _abs(meta["frozen_config_file"])
    if _sha(fp) != meta["frozen_config_sha256"]:
        raise hg.HoldoutOpenError(f"{CARD}: frozen_config_sha256 does not match {meta['frozen_config_file']}")
    if _sha(_abs(meta["frozen_code_file"])) != meta["frozen_code_sha256"]:
        raise hg.HoldoutOpenError(f"{meta['frozen_code_file']} differs from the code frozen by the card (frozen_code_sha256)")
    fr = json.load(open(fp))
    want = {"cover_target": nr.COVER_TARGET, "overall_tol": nr.OVERALL_TOL, "decile_tol": nr.DECILE_TOL,
            "min_pinball_deciles": nr.MIN_PINBALL_DECILES, "ic_month_share_pct": nr.MIN_IC_MONTH_SHARE_PCT}
    if {k: fr["rule"].get(k) for k in want} != want:
        raise ValueError("frozen rule constants differ from tools/natr_range.py")
    hp = fr["hyper_parameters"]
    if hp["window_months"] != nr.ROLL_WINDOW_MONTHS or hp["min_rows_per_decile"] != nr.ROLL_MIN_N or hp["refit"] != "monthly":
        raise ValueError("frozen hyper-parameters differ from tools/natr_range.py")
    if fr["timeframe"] != "1h" or fr["step_minutes"] != STEP or fr["natr_period"] != nr.NATR_PERIOD or fr["baseline_window"] != nr.BASELINE_WINDOW:
        raise ValueError("frozen file does not describe the 1h NATR_14 / 90-bar configuration of this code")
    models = {(tg, t): (float(fr["models"][f"{tg}_{nr.qname(t)}"]["a"]), float(fr["models"][f"{tg}_{nr.qname(t)}"]["b"]))
              for tg in nr.TARGETS for t in nr.TAUS}
    edges = np.asarray(fr["edges"], dtype=float)
    if len(edges) != 9:
        raise ValueError("frozen file needs 9 decile edges")
    return fr, models, edges, meta["frozen_config_sha256"]


def confirm_verdict(sv, top, ctrl_top_cov, quarters, stab, n_coins, min_coins, forward_end):
    """Card rules. `sv` = sparse_verdict of the rolling arm (rules 1-4), `top` = {"n", "coverage"} of the pooled forward top decile,
    `ctrl_top_cov` = control top-decile coverage, `quarters` = quarter_top_coverage table, `stab` = factor_stability.
    status: insufficient_extend_3_months (top decile < 200 rows or < 6 evaluable deciles, window ends 2027-01-01), insufficient_final
    (same after the extension), confirmed (all rules), rejected (sufficient window, any rule fails)."""
    eps = 1e-12
    sufficient = bool(top["n"] >= MIN_TOP_N and sv["n_evaluable_deciles"] >= MIN_EVAL_DECILES)
    if not sufficient:
        status = "insufficient_extend_3_months" if forward_end == ALLOWED_ENDS[0] else "insufficient_final"
        return {**{k: sv[k] for k in sv if k != "accepted"}, "sufficient": False, "status": status, "accepted": False}
    c = top["coverage"]
    r5 = bool(np.isfinite(c) and abs(c - nr.COVER_TARGET) <= nr.DECILE_TOL + eps)
    r6 = bool(np.isfinite(c) and np.isfinite(ctrl_top_cov) and abs(c - nr.COVER_TARGET) < abs(ctrl_top_cov - nr.COVER_TARGET))
    ev = quarters[quarters["n"] >= nr.QUARTER_MIN_N]
    r8 = bool(len(ev) >= 1 and (np.abs(ev["coverage"] - nr.COVER_TARGET) <= nr.DECILE_TOL + eps).all())
    r9 = bool(stab["n_changes"] > 0 and stab["max_abs_ln"] <= nr.STAB_MAX_LN + eps and stab["mean_abs_ln"] <= nr.STAB_MEAN_LN + eps)
    r7 = bool(n_coins >= min_coins)
    rules = {"r1_overall": bool(sv["overall_ok"]), "r2_deciles": bool(sv["deciles_ok"]), "r3_pinball": bool(sv["pinball_ok"]), "r4_ic": bool(sv["ic_ok"]),
             "r5_top_decile": r5, "r6_closer_than_control": r6, "r7_coins": r7, "r8_quarters": r8, "r9_stability": r9}
    ok = all(rules.values())
    return {**{k: sv[k] for k in sv if k != "accepted"}, "sufficient": True, "rules": rules, "n_evaluable_quarters": int(len(ev)),
            "status": "confirmed" if ok else "rejected", "accepted": bool(ok)}


def _arm_deciles(ev, arm):
    d = ev["deciles"].copy()
    d.insert(0, "arm", arm)
    return d


def run(cfg, out_dir, max_minutes, lock=None, now=None):
    t0 = time.time()
    cfg = complete_config(cfg)
    fend = pd.Timestamp(cfg["forward_end"], tz="UTC")
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now < fend:
        raise ValueError(f"forward_end {fend.date()} is in the future (today {now.date()}): the card forbids evaluating a partial window")
    fr, models, edges, fsha = check_frozen()                           # cheap failures first, before any bar is read
    coins = cfg["coins"] or fr["coins"]
    lock = lock or hg.load_lock()
    os.makedirs(out_dir, exist_ok=True)
    hist_end = hg.FORWARD_START
    hist, inputs = {}, []
    for sym in coins:                                                   # history = pre-seal bars (load_unsealed: raises on a sealed coin)
        hist[sym] = hg.load_unsealed(cfg["data_dir"], sym, cfg["history_start"], hist_end, pattern=cfg["pattern"], lock=lock)
    fwd = hg.load_forward(cfg["data_dir"], coins, hist_end, fend, _abs(CARD), pattern=cfg["pattern"], lock=lock)   # ONE log row
    log_lines = open(os.path.join(lock.lock_dir, "holdout_access_log.csv"), encoding="utf-8").read().splitlines()
    with open(os.path.join(out_dir, "access_log_row.csv"), "w", encoding="utf-8", newline="") as fh:
        fh.write(log_lines[0] + "\n" + log_lines[-1] + "\n")
    parts, skipped, stale, counts = [], {}, {}, {}
    truncated = False
    try:
        for sym in coins:
            if time.time() - t0 > max_minutes * 60:
                raise Deadline
            d = pd.concat([hist[sym], fwd[hg._norm(sym)]]).sort_index()
            if len(fwd[hg._norm(sym)]) == 0 or len(d) == 0:
                skipped[sym] = "no forward rows"
                continue
            inputs.append(cfg["pattern"].format(symbol=sym))
            n0 = len(d)
            d = nr.drop_stale_runs(d, cfg["stale_min_run_hours"], cfg["source_minutes"])
            stale[sym] = int(n0 - len(d))
            bars = nr.resample_bars(d, STEP, cfg["source_minutes"])
            if len(bars) < nr.BASELINE_WINDOW + nr.NATR_PERIOD + 2:
                skipped[sym] = "too few 1h bars"
                continue
            r = nr.build_rows(bars, STEP)
            r["coin"] = sym
            parts.append(r)
            counts[f"{sym}_source_bars"], counts[f"{sym}_rows"] = int(n0), int(len(r))
        allr = pd.concat(parts)
        t = allr["tgt_open"]
        need = [f"b_{tg}_{nr.qname(q)}" for tg in nr.TARGETS for q in nr.TAUS]
        span = allr[(t >= STAB_ANCHOR) & (t < fend)].dropna(subset=need)       # September rows only anchor the stability rule
        is_fwd = np.asarray(pd.DatetimeIndex(span["tgt_open"]) >= hg.FORWARD_START)
        row_f, ftab = nr.rolling_row_factors(models, edges, allr, span)
        val = span[is_fwd]
        if len(val) == 0:
            raise ValueError("no forward rows with a baseline")
        rolling = {"range": {"row_factors": row_f[is_fwd]}}
        static_f = np.asarray(fr["static_correction_factors"], dtype=float)
        static = {"range": {"edges": edges, "factors": static_f}}
        arms = {"control": None, "static": static, "rolling": rolling}
        ev = {a: nr.evaluate_frozen(models, edges, val, c) for a, c in arms.items()}
        yd = {a: nr.decile_year_coverage(models, edges, val, c) for a, c in arms.items()}
        qt = {a: nr.quarter_top_coverage(models, edges, val, c) for a, c in arms.items()}
        dec = nr.assign_decile(val["natr"], edges)
        tops = {}
        for a in arms:
            row = ev[a]["deciles"].iloc[nr.TOP_DECILE]
            tops[a] = {"n": int(row["n"]), "coverage": float(row["coverage"])}
        sv = {a: rc.sparse_verdict(ev[a], fr) for a in arms}
        stab = nr.factor_stability(ftab)
        coins_used = sorted(set(val["coin"]))
        verdict = confirm_verdict(sv["rolling"], tops["rolling"], tops["control"]["coverage"], qt["rolling"], stab, len(coins_used),
                                  cfg["min_coins"], cfg["forward_end"])
        res = {"card": CARD, "frozen_config_sha256": fsha, "truncated": False, "forward_end": cfg["forward_end"], "verdict": verdict,
               "first_eval_open": str(val["tgt_open"].min()), "last_eval_open": str(val["tgt_open"].max()), "n_forward_rows": int(len(val)),
               "coins_used": coins_used, "skipped_coins": skipped, "stale_source_rows_dropped": stale, "top_decile": tops,
               "rolling_minus_static_top_decile_pp": 100 * (tops["rolling"]["coverage"] - tops["static"]["coverage"]),
               "stability": stab, "n_fallback_decile_months": int(ftab["fallback"].sum()), "n_refits": int(ftab["month"].nunique()),
               "n_crossed": {a: ev[a]["n_crossed"] for a in arms}, "extras": {a: {k: v for k, v in ev[a]["extras"].items() if not isinstance(v, dict)} for a in arms},
               "static_verdict_reported_not_ruled": {k: v for k, v in sv["static"].items() if k != "reason"},
               "control_verdict_reported_not_ruled": {k: v for k, v in sv["control"].items() if k != "reason"},
               "top_decile_rows_by_decile_check": int((dec == nr.TOP_DECILE).sum())}
        pd.concat([_arm_deciles(ev[a], a) for a in arms]).round(6).to_csv(os.path.join(out_dir, "deciles.csv"), index=False)
        pd.concat([qt[a].assign(arm=a) for a in arms]).round(6).to_csv(os.path.join(out_dir, "quarters.csv"), index=False)
        ftab.round(6).to_csv(os.path.join(out_dir, "monthly_factors.csv"), index=False)
        pd.DataFrame(ev["rolling"]["months"]).round(6).to_csv(os.path.join(out_dir, "months.csv"), index=False)
        pd.DataFrame([{**{k: v for k, v in verdict.items() if not isinstance(v, (list, dict))}, "stability_max_abs_ln": stab["max_abs_ln"],
                       "stability_mean_abs_ln": stab["mean_abs_ln"], "top_decile_n": tops["rolling"]["n"], "top_decile_coverage": tops["rolling"]["coverage"],
                       **{f"{a}_pinball_model": res["extras"][a]["pinball_model"] for a in arms}}]).round(6).to_csv(
            os.path.join(out_dir, "summary.csv"), index=False)
    except Deadline:
        truncated = True
        res = {"card": CARD, "truncated": True, "verdict": {"accepted": False, "status": "not_evaluable"}}
    res["truncated"] = truncated
    res["runtime_min"] = round((time.time() - t0) / 60, 2)
    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=float)
    manifest = cr.write_manifest(out_dir, cfg, inputs=inputs, input_root=cfg["data_dir"], row_counts=counts, runtime_seconds=time.time() - t0,
                                 n_configurations=len(nr.TARGETS) * len(nr.TAUS), repo_dir=ROOT,
                                 extra={"truncated": truncated, "card": CARD, "frozen_config_sha256": fsha, "accept_rule_evaluations": 1})
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
