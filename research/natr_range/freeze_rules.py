"""
PURPOSE:  Writes research/natr_range/frozen_rules.json for the H2 confirmation: the 1h quantile-regression coefficients and the
          NATR decile edges of E-natr-001 run1 (fitted on train only), the data window, the minimum coin count and the accept
          rule constants, byte-deterministic so that its sha256 can sit in the confirmation card (frozen_rules_sha256).
TAGS:     frozen rules, sha256, H2 confirmation, E-natr-002, R3
PITFALLS: Reads only the committed run1 bundle (results.json), never data. Re-running it must give the same bytes; if the
          output changes, the card's frozen_rules_sha256 no longer matches and the guard refuses to open H2.

Usage:
    python research/natr_range/freeze_rules.py
"""
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import natr_range as nr  # noqa: E402

STUDY = os.path.join(ROOT, "research", "natr_range")


def build():
    run1 = os.path.join(STUDY, "results", "e_natr_001_run1")
    res = json.load(open(os.path.join(run1, "results.json")))
    manifest = json.load(open(os.path.join(run1, "manifest.json")))
    tf = res["timeframes"]["1h"]
    return {"version": 1, "parent_card": "research/natr_range/cards/E-natr-001.md", "source_run": "research/natr_range/results/e_natr_001_run1",
            "source_run_git_sha": manifest["git_sha"], "timeframe": "1h", "step_minutes": 60, "source_minutes": 15,
            "natr_period": nr.NATR_PERIOD, "baseline_window": nr.BASELINE_WINDOW, "models": tf["models"], "edges": tf["edges"],
            "eval_start": "2025-01-01", "eval_end": "2026-10-01", "min_coins": 20,
            "rule": {"cover_target": nr.COVER_TARGET, "overall_tol": nr.OVERALL_TOL, "decile_tol": nr.DECILE_TOL,
                     "min_pinball_deciles": nr.MIN_PINBALL_DECILES, "ic_month_share_pct": nr.MIN_IC_MONTH_SHARE_PCT}}


def main():
    path = os.path.join(STUDY, "frozen_rules.json")
    body = (json.dumps(build(), indent=1, sort_keys=True) + "\n").encode("utf-8")
    with open(path, "wb") as f:
        f.write(body)
    print(path, hashlib.sha256(body).hexdigest())


if __name__ == "__main__":
    main()
