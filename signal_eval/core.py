"""
PURPOSE:  The three independent measurement tools of phase 0: compute_ic (Spearman/Pearson IC), decile_spread and permutation_baseline - pure functions on arrays, no dependence on the dataset shape.
TAGS:     compute_ic, decile_spread, permutation_baseline, information coefficient, IC, spearman, decile, permutation test, null distribution, p_value, seed
PITFALLS: Independent of rolling_splits/dataset shapes by design (tested on synthetic data only). permutation_baseline needs a seed for a reproducible registry entry. Executed into the one shared signal_eval namespace by signal_eval/_loader.py (never imported on its own): names from other modules resolve at call time.

## ٢) معامل الارتباط (Information Coefficient)

الأداة الأساسية لكل شيء لاحق: رقم واحد يلخّص "هل ترتيب التنبؤات يطابق ترتيب العوائد الفعلية؟" — لا أكثر ولا أقل.

## ٣) فحص العُشر (Decile Spread)

يكشف إشارة قد لا يظهر أثرها في IC واحد يلخّص كل العيّنات — التركّز في الأطراف.

## ٤) خطّ أساس عشوائي (Permutation Baseline)

الإجابة المباشرة على "أفضل من الصدفة؟" — لا الاكتفاء برقم IC مجرَّد.
"""
def compute_ic(predictions: np.ndarray, actuals: np.ndarray,
               method: str = "spearman", min_samples: int = 10) -> float:
    """معامل الارتباط بين تنبؤات فرضية والعوائد الفعلية المحقَّقة.

    ``method='spearman'`` (افتراضي): ارتباط الرتب — لا يفترض علاقة خطية بين
    التنبؤ والعائد، فقط أن الترتيب صحيح (تنبؤ أعلى ↔ عائد أعلى غالباً). هذا
    أنسب لتقييم فرضية قد تُنتج "درجة" غير مُعايَرة (لا سعراً حقيقياً)،
    ولمقاومته الأفضل للقيم الشاذة القليلة الشائعة في عوائد الأصول المالية.
    ``method='pearson'``: ارتباط خطي كلاسيكي — أضيق افتراضاً (يفترض علاقة
    خطية فعلاً)، لكن أسرع حساباً وأكثر حساسية لعلاقة خطية قوية فعلية.

    يتجاهل صمتاً أي عيّنة NaN في أي من المصفوفتين (لا يرفضها العملية كلها،
    ولا يُصفّرها بصمت أيضاً — ببساطة تُستبعَد من حساب الارتباط). يرفع
    ``ValueError`` إن قلّ عدد العيّنات الصالحة عن ``min_samples`` — IC على
    عيّنات قليلة جداً رقم عشوائي بلا معنى إحصائي، لا نتيجة تستحق الثقة.

    Returns:
        رقم واحد بين -1 و1. صفر بالضبط (لا NaN) إن كانت إحدى المصفوفتين
        ثابتة تماماً (تباين صفري) بعد إزالة NaN — لا ارتباط مُعرَّف رياضياً
        مع قيمة ثابتة، وصفر هنا هو الإعلان الصريح "لا إشارة" لا خطأ حسابي.
    """
    predictions = np.asarray(predictions, dtype="float64")
    actuals = np.asarray(actuals, dtype="float64")
    if predictions.shape != actuals.shape:
        raise ValueError(f"الشكلان يجب أن يتطابقا: {predictions.shape} != {actuals.shape}")
    if method not in ("spearman", "pearson"):
        raise ValueError(f"method غير معروفة: {method!r} — المتاح: spearman, pearson")

    mask = np.isfinite(predictions) & np.isfinite(actuals)
    n = int(mask.sum())
    if n < min_samples:
        raise ValueError(
            f"❌ {n} عيّنة صالحة فقط (بعد استبعاد NaN) — أقل من الحدّ الأدنى "
            f"{min_samples}. IC على عدد بهذا الصغر رقم عشوائي لا يُعتمَد عليه.")

    p, a = predictions[mask], actuals[mask]
    if np.std(p) < 1e-12 or np.std(a) < 1e-12:
        return 0.0

    if method == "spearman":
        p = pd.Series(p).rank().to_numpy()
        a = pd.Series(a).rank().to_numpy()
    corr = np.corrcoef(p, a)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0


# ══════════════════════════════════════════════════════════════════════════
# ٢) فحص العُشر (decile spread)
# ══════════════════════════════════════════════════════════════════════════


def decile_spread(predictions: np.ndarray, actuals: np.ndarray,
                  n_deciles: int = 10, min_per_decile: int = 5) -> Dict[str, Any]:
    """يقسّم العيّنات إلى ``n_deciles`` حسب رتبة التنبؤ، ويحسب متوسط العائد
    الفعلي لكل عُشر — يكشف إشارة حقيقية حتى لو كان IC الكلي صغيراً (تركيز
    الأثر في الأطراف لا يظهر بالضرورة في معامل ارتباط واحد يلخّص كل العيّنات).

    ``monotonic``: هل متوسط العائد يزداد (أو يتناقص) بانتظام من أدنى عُشر
    لأعلاه؟ **هذا أهم من ``spread`` وحده**: فرق كبير بين الطرفين مع منتصف
    عشوائي غير مرتّب مرشّح أقوى لأن يكون صدفة من فرق نظيف رتيب عبر كل الأعشار.

    Args:
        min_per_decile: أقل عدد عيّنات مقبول في كل عُشر — دون ذلك يُرفَع
            ``ValueError`` بدل نتيجة على عيّنات قليلة جداً لا تمثّل شيئاً.

    Returns:
        قاموس ``{'deciles': DataFrame(index, mean_actual, n), 'spread':
        float, 'monotonic': bool, 'top_minus_bottom_t_stat': float}``.
        ``top_minus_bottom_t_stat`` تقريب سريع (فرق المتوسطات ÷ الخطأ
        المعياري المجمَّع) — ليس اختباراً إحصائياً صارماً، فقط مؤشر أوّلي
        لحجم الفرق نسبة لتشتّت كل عُشر؛ الحسم الفعلي عبر ``permutation_baseline``.
    """
    predictions = np.asarray(predictions, dtype="float64")
    actuals = np.asarray(actuals, dtype="float64")
    mask = np.isfinite(predictions) & np.isfinite(actuals)
    p, a = predictions[mask], actuals[mask]
    n = len(p)
    if n < n_deciles * min_per_decile:
        raise ValueError(
            f"❌ {n} عيّنة صالحة لا تكفي لـ{n_deciles} أعشار بحدّ أدنى "
            f"{min_per_decile} لكل عُشر (تحتاج {n_deciles * min_per_decile} على الأقل).")

    order = np.argsort(p, kind="stable")
    bucket = np.empty(n, dtype="int64")
    bucket[order] = (np.arange(n) * n_deciles) // n

    rows = []
    for d in range(n_deciles):
        vals = a[bucket == d]
        rows.append({"decile": d, "mean_actual": float(vals.mean()),
                    "std_actual": float(vals.std()), "n": int(len(vals))})
    table = pd.DataFrame(rows)

    means = table["mean_actual"].to_numpy()
    diffs = np.diff(means)
    monotonic = bool(np.all(diffs >= 0) or np.all(diffs <= 0))
    spread = float(means[-1] - means[0])

    top, bottom = a[bucket == n_deciles - 1], a[bucket == 0]
    pooled_se = np.sqrt(top.var(ddof=1) / len(top) + bottom.var(ddof=1) / len(bottom))
    t_stat = float(spread / pooled_se) if pooled_se > 1e-12 else 0.0

    return {"deciles": table, "spread": spread, "monotonic": monotonic,
           "top_minus_bottom_t_stat": t_stat}


# ══════════════════════════════════════════════════════════════════════════
# ٣) خطّ أساس عشوائي (permutation baseline)
# ══════════════════════════════════════════════════════════════════════════


def permutation_baseline(predictions: np.ndarray, actuals: np.ndarray,
                         n_shuffles: int = 1000, method: str = "spearman",
                         seed: Optional[int] = None) -> Dict[str, Any]:
    """يبني توزيعاً خالياً (null distribution) لـIC عبر بعثرة ``actuals``
    عشوائياً ``n_shuffles`` مرة (بلا تغيير ``predictions``)، ويقارن IC
    الحقيقي بهذا التوزيع — الإجابة المباشرة على "هل هذا أفضل من الصدفة؟"
    بدل الاكتفاء برقم IC مجرَّد.

    ``percentile``: أين يقع IC الحقيقي ضمن توزيع الصدفة (0-100). قريب من
    100 (أو 0 لإشارة سالبة قوية) = IC الحقيقي أقوى من الغالبية الساحقة من
    عمليات البعثرة العشوائية. ``p_value``: تقريب أحادي الجانب — نسبة عمليات
    البعثرة التي أعطت |IC| ≥ |IC الحقيقي| (اختبار ثنائي الطرف ضمنياً، يلائم
    عدم معرفة اتجاه الإشارة مسبقاً في هذا السياق الاستكشافي).

    ``seed``: لتكرار نفس نتيجة البعثرة بالضبط — مهم لسجلّ التجارب (نتيجة
    فرضية يجب أن تكون قابلة لإعادة الإنتاج، لا تتغيّر بين تشغيلين).
    """
    predictions = np.asarray(predictions, dtype="float64")
    actuals = np.asarray(actuals, dtype="float64")
    real_ic = compute_ic(predictions, actuals, method=method)

    mask = np.isfinite(predictions) & np.isfinite(actuals)
    p, a = predictions[mask], actuals[mask]
    rng = np.random.default_rng(seed)

    null_ics = np.empty(n_shuffles, dtype="float64")
    for i in range(n_shuffles):
        shuffled = rng.permutation(a)
        null_ics[i] = compute_ic(p, shuffled, method=method, min_samples=1)

    percentile = float((null_ics < real_ic).mean() * 100.0)
    p_value = float((np.abs(null_ics) >= abs(real_ic)).mean())

    return {"real_ic": real_ic, "null_mean": float(null_ics.mean()),
           "null_std": float(null_ics.std()), "percentile": percentile,
           "p_value": p_value, "n_shuffles": n_shuffles}
