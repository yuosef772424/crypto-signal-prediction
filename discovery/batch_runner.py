"""
PURPOSE:  Batch runner: parallel evaluation of the whole candidate registry with automatic registration (classify_result, run_batch_and_register).
TAGS:     run_batch_and_register, classify_result, batch runner, registry, register_hypothesis, imap_ordered, parallel
PITFALLS: Writes to the experiment registry (registry_path): self-tests must pass a temporary path. Uses default_workers/imap_ordered from the pipeline. Executed into the notebook's shared namespace by discovery/_loader.py (never imported on its own): names from the axis/pipeline notebooks resolve at call time. Extracted verbatim from signal_discovery_lab.ipynb cell 25 (section 8).
"""
import numpy as np
import pandas as pd


def classify_result(report, min_frac_significant=0.34, min_n_ok=5):
    """معيار قبول موحّد — نفس المنطق يُطبَّق بصرف النظر عن مصدر المرشّح
    (راجع "كيف تُقيَّم نتائج كل هذه الأدوات" في خطة المشروع). `n_ok` أقلّ من
    الحدّ الأدنى يعني عدد نوافذ ناجحة غير كافٍ للحكم أصلاً — لا "مرفوضة"
    مُتسرِّعة على دليل ضعيف."""
    if report.get("n_ok", 0) < min_n_ok:
        return "قيد الاختبار"
    if report.get("consistent_sign") and report.get("frac_significant", 0) >= min_frac_significant:
        return "مقبولة"
    return "مرفوضة"


def _batch_eval_job(job):
    cand, target, windows_, feature_order, eval_kwargs = job
    predict_fn = make_candidate_predict_fn(cand, feature_order=feature_order)
    try:
        report = evaluate_candidate(predict_fn, target, windows_, **eval_kwargs)
        return {"cand": cand, "target": target, "report": report, "status": "ok"}
    except Exception as e:
        return {"cand": cand, "target": target, "status": "error", "error": f"{type(e).__name__}: {e}"}


def run_batch_and_register(candidates, windows, targets=("close", "high", "low"), feature_order=None,
                           id_prefix="SCAN", max_workers=None, registry_path=None, **eval_kwargs):
    """المُشغّل الدفعي الكامل: يقيّم كل (مرشّح × هدف) بالتوازي عبر
    imap_ordered/default_workers (من دفتر التحضير)، يصنّف كل نتيجة عبر
    classify_result، ويسجّلها تلقائياً في experiment_registry. يُرجع
    (leaderboard, registered_ids) — الأولى للعرض السريع، والثانية لتتبّع ما
    كُتب فعلاً.

    ``registry_path``: مرّره (مثلاً tempfile) لتوجيه التسجيل بعيداً عن السجلّ
    الحقيقي — مفيد للاختبار الذاتي؛ اتركه ``None`` للمسار الافتراضي الحقيقي.
    """
    jobs = [(cand, target, windows, feature_order, eval_kwargs) for cand in candidates for target in targets]
    n_workers = max_workers or default_workers(len(jobs))

    rows, registered = [], []
    for res in imap_ordered(_batch_eval_job, jobs, max_workers=n_workers):
        cand, target = res["cand"], res["target"]
        if res["status"] == "error":
            rows.append({"name": cand["name"], "track": cand["track"], "target": target,
                        "status": "error", "error": res["error"]})
            continue
        report = res["report"]
        status = classify_result(report)
        hyp_id = f"{id_prefix}_{cand['name']}_{target}"
        register_hypothesis(
            hyp_id=hyp_id,
            hypothesis=f"{cand.get('hypothesis', cand['name'])} (هدف: {target})",
            source=cand["track"],
            status=status,
            report={"per_window": report["per_window"], "mean_ic": report["mean_ic"],
                    "std_ic": report["std_ic"], "frac_significant": report["frac_significant"],
                    "consistent_sign": report["consistent_sign"], "n_ok": report["n_ok"]},
            notes=(f"مُسجَّلة آلياً عبر run_batch_and_register. mean_ic={report['mean_ic']:.4f}, "
                  f"consistent_sign={report['consistent_sign']}, "
                  f"frac_significant={report['frac_significant']:.2f}."),
            registry_path=registry_path,
        )
        registered.append(hyp_id)
        rows.append({"name": cand["name"], "track": cand["track"], "target": target, "status": status,
                    "mean_ic": report["mean_ic"], "std_ic": report["std_ic"],
                    "frac_significant": report["frac_significant"],
                    "consistent_sign": report["consistent_sign"], "n_ok": report["n_ok"]})

    df = pd.DataFrame(rows)
    ok = df[df["status"] != "error"].copy()
    if len(ok):
        ok["abs_mean_ic"] = ok["mean_ic"].abs()
        ok = ok.sort_values(["status", "abs_mean_ic"], ascending=[True, False])
        df = pd.concat([ok.drop(columns="abs_mean_ic"), df[df["status"] == "error"]], ignore_index=True)
    return df, registered


# مثال استخدام حقيقي (يكتب في experiment_registry الحقيقي — شغّله عمداً، لا تلقائياً):
# batch_leaderboard, registered_ids = run_batch_and_register(
#     CANDIDATE_SIGNALS + EXPLORATORY_CANDIDATES + GENERATIVE_CANDIDATES + LEGACY_BACKTEST_CANDIDATES,
#     windows, feature_order=FEATURE_ORDER)
# print(batch_leaderboard.to_string(index=False))
# print(f"سُجِّل {len(registered_ids)} مدخلاً: {registered_ids}")
