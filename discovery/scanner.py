"""
PURPOSE:  scan_candidates: evaluate every candidate x every target in one call and return the leaderboard.
TAGS:     scan_candidates, leaderboard, scanner, candidates x targets
PITFALLS: Uses evaluate_candidate (evaluation module) and the axis's registry-free evaluation; windows come from rolling_splits. Executed into the notebook's shared namespace by discovery/_loader.py (never imported on its own): names from the axis/pipeline notebooks resolve at call time. Extracted verbatim from signal_discovery_lab.ipynb cell 19 (section 6).
"""
import numpy as np
import pandas as pd


def scan_candidates(candidates, windows, targets=("close", "high", "low"),
                    feature_order=None, **eval_kwargs):
    """يُقيِّم كل مرشّح × كل هدف عبر evaluate_candidate (مع حارس
    clean_reg_target تلقائياً)، ويُرجع لوحة قيادة (leaderboard) مُرتَّبة —
    اتساق الإشارة أوّلاً، ثم قوة IC المطلقة. لا يتوقّف عند أوّل خطأ (يُسجَّل
    ويُكمل بقية المرشّحين) — مفيد خاصة لمرشّحين قد لا يحملهما كل dataset
    (مثل FUND_rate_extreme_position)."""
    rows = []
    for cand in candidates:
        predict_fn = make_candidate_predict_fn(cand, feature_order=feature_order)
        for target in targets:
            try:
                report = evaluate_candidate(predict_fn, target, windows, **eval_kwargs)
                rows.append({"name": cand["name"], "track": cand["track"], "target": target,
                            "hypothesis": cand.get("hypothesis", ""),
                            "mean_ic": report["mean_ic"], "std_ic": report["std_ic"],
                            "frac_significant": report["frac_significant"],
                            "consistent_sign": report["consistent_sign"],
                            "n_ok": report["n_ok"], "status": "ok"})
            except Exception as e:
                rows.append({"name": cand["name"], "track": cand["track"], "target": target,
                            "status": f"error: {type(e).__name__}: {e}"})
    df = pd.DataFrame(rows)
    ok = df[df["status"] == "ok"].copy()
    if len(ok):
        ok["abs_mean_ic"] = ok["mean_ic"].abs()
        ok = ok.sort_values(["consistent_sign", "abs_mean_ic"], ascending=[False, False])
        df = pd.concat([ok.drop(columns="abs_mean_ic"), df[df["status"] != "ok"]], ignore_index=True)
    return df
