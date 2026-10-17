"""
PURPOSE:  E-natr-002 (phase R3, confirmation step; PREPARED, NOT QUEUED): the single frozen H2 run of the 1h NATR range forecast.
          Checks the frozen rules against the card, opens bucket H2 ONCE through holdout_guard.open_holdout, resamples Drive
          history_15m to 1h, evaluates the frozen coefficients and decile edges (no refit) with the fixed accept rule and writes
          the run-contract bundle plus access_log_row.csv (the log row appended in the clone, for the reviewer to commit).
TAGS:     H2 confirmation, E-natr-002, R3, frozen rules, holdout, Colab job, NATR range forecast, run contract
PITFALLS: Everything that can fail cheaply (config keys, frozen-file hash vs card, rule constants vs tools/natr_range.py) is checked
          BEFORE the bucket is opened, because a successful open burns H2 even if the run crashes later. Never run it twice
          (the guard refuses). Run from the repo root on Colab with Drive mounted; CSP_DRIVE_ROOT/history_15m is data_dir.

Usage (via the Colab runner job research/natr_range/h2_job.json, never by hand):
    python research/natr_range/run_natr_confirm.py --config <json> --out <dir> --max-minutes 20
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
import colab_runner as cr           # noqa: E402
import holdout_guard as hg          # noqa: E402
import natr_range as nr             # noqa: E402

DEFAULTS = {"data_dir": None, "card": "research/natr_range/cards/E-natr-002.md", "frozen_rules": "research/natr_range/frozen_rules.json",
            "pattern": "{symbol}.csv.gz", "load_start": "2024-10-01"}
BUCKET = "H2"


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


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


def _check_frozen(cfg):
    """Validate the frozen file against the card and the code constants. Raises before anything is opened."""
    raw = open(_abs(cfg["frozen_rules"]), "rb").read()
    sha = __import__("hashlib").sha256(raw).hexdigest()
    meta, _ = hg._parse_card(_abs(cfg["card"]))
    if meta.get("frozen_rules_sha256") != sha:
        raise hg.HoldoutOpenError(f"{cfg['card']}: frozen_rules_sha256 does not match {cfg['frozen_rules']} ({sha})")
    fr = json.loads(raw)
    want = {"cover_target": nr.COVER_TARGET, "overall_tol": nr.OVERALL_TOL, "decile_tol": nr.DECILE_TOL,
            "min_pinball_deciles": nr.MIN_PINBALL_DECILES, "ic_month_share_pct": nr.MIN_IC_MONTH_SHARE_PCT}
    if fr["rule"] != want:
        raise ValueError(f"frozen rule constants {fr['rule']} differ from tools/natr_range.py {want}")
    if fr["timeframe"] != "1h" or fr["step_minutes"] != 60 or fr["natr_period"] != nr.NATR_PERIOD or fr["baseline_window"] != nr.BASELINE_WINDOW:
        raise ValueError("frozen file does not describe the 1h NATR_14 / 90-bar configuration of this code")
    models = {}
    for tg in nr.TARGETS:
        for t in nr.TAUS:
            m = fr["models"][f"{tg}_{nr.qname(t)}"]
            models[(tg, t)] = (float(m["a"]), float(m["b"]))
    edges = np.asarray(fr["edges"], dtype=float)
    if len(edges) != 9:
        raise ValueError("frozen file needs 9 decile edges")
    return fr, models, edges, sha


def run(cfg, out_dir, max_minutes, lock=None):
    t0 = time.time()
    cfg = complete_config(cfg)
    fr, models, edges, sha = _check_frozen(cfg)                      # all cheap failures happen before the bucket is opened
    lock = lock or hg.load_lock()
    os.makedirs(out_dir, exist_ok=True)
    h = hg.open_holdout(_abs(cfg["card"]), BUCKET, lock=lock)       # the ONE open: logs a row in holdout_access_log.csv
    log_path = os.path.join(lock.lock_dir, "holdout_access_log.csv")
    lines = open(log_path, encoding="utf-8").read().splitlines()
    with open(os.path.join(out_dir, "access_log_row.csv"), "w", encoding="utf-8", newline="") as fh:
        fh.write(lines[0] + "\n" + lines[-1] + "\n")
    start, end = pd.Timestamp(fr["eval_start"], tz="UTC"), pd.Timestamp(fr["eval_end"], tz="UTC")
    parts, skipped, used, inputs, counts = [], {}, [], [], {}
    truncated = False
    for sym in h.symbols:
        if time.time() - t0 > max_minutes * 60:
            truncated = True
            break
        try:
            d = h.load(cfg["data_dir"], sym, cfg["load_start"], fr["eval_end"], pattern=cfg["pattern"])
        except FileNotFoundError:
            skipped[sym] = "file not found"
            continue
        if len(d) == 0:
            skipped[sym] = "no rows in window"
            continue
        inputs.append(cfg["pattern"].format(symbol=sym))
        bars = nr.resample_bars(d, fr["step_minutes"], fr["source_minutes"])
        if len(bars) < nr.BASELINE_WINDOW + nr.NATR_PERIOD + 2:
            skipped[sym] = "too few 1h bars"
            continue
        r = nr.build_rows(bars, fr["step_minutes"])
        r["coin"] = sym
        r = r[(r["tgt_open"] >= start) & (r["tgt_open"] < end)]
        r = r.dropna(subset=[f"b_{tg}_{nr.qname(t)}" for tg in nr.TARGETS for t in nr.TAUS])
        if len(r) == 0:
            skipped[sym] = "no rows with a baseline in the evaluation window"
            continue
        parts.append(r)
        used.append(sym)
        counts[f"{sym}_rows"] = int(len(r))
    res = {"bucket": BUCKET, "card": cfg["card"], "frozen_rules_sha256": sha, "truncated": truncated, "coins_used": used,
           "skipped_coins": skipped, "n_bucket_symbols": len(h.symbols)}
    if parts and not truncated:
        val = pd.concat(parts)
        ev = nr.evaluate_frozen(models, edges, val)
        verdict = dict(ev["verdict"])
        verdict["coins_ok"] = bool(len(used) >= fr["min_coins"])
        verdict["accepted"] = bool(verdict["accepted"] and verdict["coins_ok"])
        res.update({"verdict": verdict, "extras": ev["extras"], "n_crossed": ev["n_crossed"], "n_coins_used": len(used)})
        for name in ("deciles", "months", "slices"):
            ev[name].round(6).to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
        pd.DataFrame([{**verdict, **{k: v for k, v in ev["extras"].items() if not isinstance(v, dict)}, "n_coins_used": len(used),
                       "n_crossed": ev["n_crossed"]}]).round(6).to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    else:
        res["verdict"] = {"accepted": False, "evaluable": False}
    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=float)
    manifest = cr.write_manifest(out_dir, cfg, inputs=inputs, input_root=cfg["data_dir"], row_counts=counts, runtime_seconds=time.time() - t0,
                                 n_configurations=1, repo_dir=ROOT,
                                 extra={"truncated": truncated, "card": cfg["card"], "bucket": BUCKET, "frozen_rules_sha256": sha})
    return manifest, res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-minutes", type=float, default=20)
    a = ap.parse_args(argv)
    manifest, res = run(load_config(a.config), a.out, a.max_minutes)
    print(json.dumps(manifest["outputs"], indent=1))
    print("truncated:", manifest["truncated"], "coins used:", len(res["coins_used"]), "skipped:", len(res["skipped_coins"]))
    print(json.dumps(res["verdict"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
