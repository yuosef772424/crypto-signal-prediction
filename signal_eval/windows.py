"""
PURPOSE:  evaluate_windows: aggregates IC + decile + permutation over several time windows and judges consistency (mean_ic, frac_significant, consistent_sign), not one number.
TAGS:     evaluate_windows, windows aggregation, consistent_sign, frac_significant, mean_ic, std_ic, per_window
PITFALLS: A failed window (too few samples) is skipped with a warning, not fatal. Executed into the one shared signal_eval namespace by signal_eval/_loader.py (never imported on its own): names from other modules resolve at call time.

تجميع التقييم عبر عدّة نوافذ زمنية — يستهلك مخرجات core.py لكل نافذة على
حدة، ثم يُلخّص الاتساق عبرها. الاتساق عبر عدّة نوافذ دليل أقوى بكثير من IC
قوي على تقسيم واحد قد يكون حظّاً (راجع منهجية التقييم في خطة المشروع).

## ٥) تجميع التقييم عبر عدّة نوافذ زمنية

الاتساق عبر نوافذ متعدّدة (عبر `rolling_splits`) دليل أقوى بكثير من IC قويّ على تقسيم واحد قد يكون حظّاً.
"""
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd



def evaluate_windows(window_results: List[Tuple[str, np.ndarray, np.ndarray]],
                     ic_method: str = "spearman", n_shuffles: int = 1000,
                     min_samples: int = 10, seed: Optional[int] = None,
                     verbose: bool = True) -> Dict[str, Any]:
    """يقيس فرضية واحدة عبر عدّة نوافذ (كل عنصر: اسم النافذة، تنبؤات، عوائد
    فعليّة) — لكل نافذة IC + عُشر + خطّ أساس عشوائي على حدة، ثم تلخيص شامل.

    نوافذ فشلت (بيانات غير كافية) تُستبعَد من التلخيص مع تحذير صريح، لا
    تُسقِط العملية كلها — فرضية قد تنجح على أغلب النوافذ حتى لو فشلت واحدة
    بسبب نقص بيانات محلي (عملة جديدة الإدراج في تلك الفترة مثلاً).

    Returns:
        قاموس يحوي ``per_window`` (تفصيل كل نافذة)، و``mean_ic``/``std_ic``
        (عبر النوافذ الناجحة)، و``frac_significant`` (نسبة النوافذ التي
        تجاوزت الصدفة بـp<0.05 وبنفس اتجاه IC الكلي — لا يكفي p صغيراً وحده،
        الاتجاه المتّسق هو ما يهمّ)، و``consistent_sign`` (هل كل النوافذ
        الناجحة اتّفقت على إشارة IC، موجبة كانت أم سالبة؟).
    """
    per_window = []
    for name, preds, actuals in window_results:
        try:
            ic = compute_ic(preds, actuals, method=ic_method, min_samples=min_samples)
            dec = decile_spread(preds, actuals, min_per_decile=max(1, min_samples // 10) or 1)
            perm = permutation_baseline(preds, actuals, n_shuffles=n_shuffles,
                                        method=ic_method, seed=seed)
            per_window.append({"window": name, "status": "ok", "ic": ic,
                               "spread": dec["spread"], "monotonic": dec["monotonic"],
                               "p_value": perm["p_value"], "percentile": perm["percentile"]})
        except ValueError as exc:
            per_window.append({"window": name, "status": "skipped", "note": str(exc)[:120]})
            if verbose:
                print(f"   ⚠️ [{name}] تُخطّيت: {str(exc)[:120]}")

    ok = [w for w in per_window if w["status"] == "ok"]
    table = pd.DataFrame(per_window)

    if not ok:
        if verbose:
            print("❌ لا نافذة واحدة ناجحة — لا يمكن تلخيص شيء.")
        return {"per_window": table, "mean_ic": None, "std_ic": None,
               "frac_significant": None, "consistent_sign": None, "n_ok": 0}

    ics = np.array([w["ic"] for w in ok])
    mean_ic, std_ic = float(ics.mean()), float(ics.std())
    overall_sign = np.sign(mean_ic)
    consistent_sign = bool(np.all(np.sign(ics) == overall_sign)) if overall_sign != 0 else False
    frac_significant = float(np.mean([
        w["p_value"] < 0.05 and np.sign(w["ic"]) == overall_sign for w in ok]))

    if verbose:
        print(f"📊 {len(ok)}/{len(window_results)} نافذة ناجحة | "
              f"IC: متوسط={mean_ic:.4f} انحراف={std_ic:.4f} | "
              f"اتّساق الإشارة عبر كل النوافذ: {'نعم' if consistent_sign else 'لا'} | "
              f"نوافذ معنوية (p<0.05) بنفس الاتجاه: {frac_significant:.0%}")

    return {"per_window": table, "mean_ic": mean_ic, "std_ic": std_ic,
           "frac_significant": frac_significant, "consistent_sign": consistent_sign,
           "n_ok": len(ok)}
