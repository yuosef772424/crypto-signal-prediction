"""
PURPOSE:  The only layer that assumes the shape of rolling_splits/build_dataset output: concat_splits, extract_actuals, extract_feature_last_diff, momentum_predict_fn, evaluate_hypothesis_over_rolling_windows.
TAGS:     concat_splits, extract_actuals, extract_feature_last_diff, momentum_predict_fn, evaluate_hypothesis_over_rolling_windows, rolling_splits, split['y'], feature_order, keep_asset_test_separate
PITFALLS: Targets are nested under split['y']; feature_order is NOT stored in the windows (pass dataset['feature_order'] via partial). A flat split is recognised by the 'base_params' key. Executed into the one shared signal_eval namespace by signal_eval/_loader.py (never imported on its own): names from other modules resolve at call time.

طبقة التكامل مع دفتر التحضير (crypto_data_pipeline) — الجزء الوحيد الذي
يفترض شكل بيانات معيّناً (مخرجات rolling_splits/_take). بمعزل عمداً عن
core.py وwindows.py (المُختبَرين بلا أي اعتماد على هذا الشكل) — أي خلل في
افتراض هنا محصور في هذا الملف، لا يمسّ صحّة القياس نفسه.

**العقد المفترض — تحقّقتُه من مصدر ``_take``/``add_y_prefix`` الفعلي مباشرة،
لا تخميناً:** كل قسم (من ``_take`` داخل ``rolling_splits``/``split_data``) قاموس:

* ``base_params``, ``last_candles``: مصفوفات (N, ...) — موجودتان دائماً، تُستخدَمان
  هنا كعلامة "هذا قسم بيانات مسطّح" (خلاف قاموس أصول ``{اسم: قسم}``).
* ``X_{tf}``: مصفوفة (N,T,F) لكل فريم — مسطّحة كما هو متوقَّع.
* ``y``: **قاموس فرعي واحد** ``{اسم_الهدف: مصفوفة (N,)}`` — لا مفاتيح
  ``y_head`` مباشرة على مستوى القسم نفسه؛ كلها متداخلة تحت هذا المفتاح
  الواحد (تصميم متعمَّد في الدفتر الأصلي، توافقاً مع طريقة Keras في مطابقة
  مخرجات متعدّدة بالاسم). ``add_y_prefix`` (إن ``prefix_y=True``، الافتراضي)
  يُضيف بادئة ``y_`` لمفاتيح *داخل* هذا القاموس الفرعي فقط.
* لا ``feature_order`` على مستوى القسم — موجودة فقط في ``dataset`` الأصلي
  قبل التقطيع (مؤكَّد سابقاً من تشغيل حقيقي).

``test`` قد يكون قسماً مسطّحاً كما فوق، أو ``{اسم_الأصل: قسم}``
(``keep_asset_test_separate=True``) — كل قيمة فيه قسم مسطّح بنفس الشكل.


## ٦) طبقة التكامل مع دفتر التحضير

**العقد المفترض — تحقّقتُه من مصدر `_take`/`add_y_prefix` الفعلي في دفتر التحضير مباشرة، بعد أن أخطأتُ فيه مرّتين بالتخمين:** كل قسم قاموس فيه `X_{tf}` (مصفوفة `(N,T,F)`)، و`base_params`/`last_candles` (علامة "قسم مسطّح")، و**الأهداف متداخلة تحت مفتاح فرعي واحد `y`** — `split['y']['y_close_reg']`، لا `split['y_close_reg']` مباشرة. لا `feature_order` على مستوى القسم إطلاقاً (فقط في `dataset` الأصلي قبل التقطيع — مرّرها صراحةً). `test` قد يكون قاموساً مسطّحاً كما فوق، أو `{اسم_الأصل: قسم}` (`keep_asset_test_separate`).

⚠️ هذا القسم وحده يفترض شكل بيانات دفتر التحضير — لو تغيّر شكل مخرجات `rolling_splits` مستقبلاً، هنا فقط يحتاج تعديلاً، لا الأقسام (٢)-(٥).
"""
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional, List, Tuple, Dict, Any, Callable
import numpy as np



def concat_splits(split: Any) -> Dict[str, Any]:
    """يُسطّح ``test`` إن كان ``{اسم_الأصل: قسم}`` (keep_asset_test_separate)
    بدمج كل الأصول في قاموس واحد — X مُلحَقة بمحور العيّنات، وy (القاموس
    الفرعي) تُدمَج مفتاحاً مفتاحاً. قسم مسطّح أصلاً يُعاد كما هو بلا نسخ.

    ✅ **التمييز بمفتاح ``'base_params'`` تحديداً** — موجود في كل قسم مسطّح
    (يضعه ``_take`` دائماً)، وغير موجود أبداً بين أسماء الأصول في قاموس
    ``{اسم: قسم}``. أدقّ من التخمين بشكل القيمة (``isinstance(..., dict)``
    وحده لا يكفي: قيمة ``split['y']`` في القسم المسطّح نفسه قاموس أيضاً،
    فقد يُلتبَس بقاموس أصول لو اعتُمد نوع القيمة فقط — راجع سجلّ التعديلات).
    """
    if not isinstance(split, dict) or not split:
        raise ValueError(f"split فارغ أو ليس قاموساً (النوع: {type(split).__name__}).")

    if "base_params" in split:
        return split          # مسطّح أصلاً

    parts = list(split.items())
    bad = next((name for name, p in parts
               if not (isinstance(p, dict) and "base_params" in p)), None)
    if bad is not None:
        raise ValueError(
            f"العنصر '{bad}' لا يبدو قسم بيانات صالحاً (بلا 'base_params') — "
            f"هل split فعلاً {{اسم_الأصل: قسم}} كما هو مفترض؟")

    split_parts = [p for _, p in parts]
    out: Dict[str, Any] = {
        "base_params": np.concatenate([p["base_params"] for p in split_parts], axis=0),
        "last_candles": np.concatenate([p["last_candles"] for p in split_parts], axis=0),
    }
    for k in [k for k in split_parts[0] if k.startswith("X_")]:
        out[k] = np.concatenate([p[k] for p in split_parts if k in p], axis=0)

    y_keys = set()
    for p in split_parts:
        y_keys.update((p.get("y") or {}).keys())
    out["y"] = {yk: np.concatenate([p["y"][yk] for p in split_parts if yk in (p.get("y") or {})],
                                  axis=0)
               for yk in y_keys}
    return out


def extract_actuals(split: Dict[str, Any], target_key: str = "y_close_reg") -> np.ndarray:
    """يقرأ مصفوفة الهدف الفعلي (العائد المحقَّق) من ``split['y'][...]`` —
    ✅ **الأهداف متداخلة تحت مفتاح فرعي واحد ``'y'``**، لا مفاتيح ``y_head``
    مباشرة على القسم نفسه (تأكّدتُ من هذا من مصدر ``_take``/``add_y_prefix``
    الفعلي — راجع توثيق الوحدة أعلى الملف). يقبل ``target_key`` مع البادئة
    ``'y_'`` أو بدونها؛ يجرّب كليهما (``add_y_prefix`` قد يكون فعل أو لم يفعل
    حسب ``prefix_y`` عند بناء النافذة).
    """
    split = concat_splits(split)
    y = split.get("y")
    if not isinstance(y, dict):
        raise ValueError(
            "لا مفتاح 'y' (قاموس أهداف فرعي) في هذا القسم — تحقّق أنك مرّرت "
            "قسماً من مخرجات rolling_splits/split_data مباشرة، لا شيئاً آخر.")
    alt_key = target_key[len("y_"):] if target_key.startswith("y_") else f"y_{target_key}"
    for k in (target_key, alt_key):
        if k in y:
            return np.asarray(y[k], dtype="float64")
    raise KeyError(f"لا '{target_key}' ولا '{alt_key}' في split['y'] — المتاح: {list(y.keys())}")


def extract_feature_last_diff(split: Dict[str, Any], feature: str = "close",
                              tf: Optional[str] = None,
                              feature_order: Optional[List[str]] = None) -> np.ndarray:
    """فرق آخر خطوتين زمنيتين لميزة واحدة داخل نافذة كل عيّنة — "أحدث تغيّر"
    لتلك الميزة، مقياس خام لكن كافٍ تماماً لأي فرضية تُقيَّم بـIC سبيرمان
    (ارتباط رتبي لا يتأثّر بالمقياس المطلق، فلا حاجة لعكس التطبيع إلى سعر
    حقيقي — أي تحويل رتيب لنفس الترتيب يعطي نفس IC).

    ✅ **``feature_order`` غير محفوظة داخل نوافذ ``rolling_splits``** (قِيس
    فعلياً: موجودة في ``dataset['feature_order']`` الأصلي، لا في كل قسم
    ``train``/``val``/``test`` الناتج عن التقطيع) — مرّرها صراحةً من
    ``dataset`` الأصلي. القسم نفسه يبقى يُحاوَل أولاً (توافقاً مع أي مصدر
    بيانات يحفظها فعلاً)، لكن التمرير الصريح هو المسار الموثوق.

    ``tf``: الفريم المطلوب (افتراضياً أول عمود X_* موجود في القسم).
    ``feature``: اسمها كما في ``feature_order`` بالضبط.
    """
    split = concat_splits(split)
    feature_order = feature_order if feature_order is not None else split.get("feature_order")
    if not feature_order:
        raise ValueError(
            "لا 'feature_order' — لا في القسم ولا مُمرَّرة صراحةً. مرّرها من "
            "dataset الأصلي: extract_feature_last_diff(split, feature_order=dataset['feature_order']).")
    if feature not in feature_order:
        raise KeyError(f"'{feature}' غير موجودة في feature_order ({len(feature_order)} ميزة).")
    idx = feature_order.index(feature)

    if tf is None:
        x_keys = [k for k in split if k.startswith("X_")]
        if not x_keys:
            raise ValueError("لا عمود X_* في قسم البيانات.")
        tf = x_keys[0][2:]
    X = np.asarray(split[f"X_{tf}"])
    if X.shape[-1] <= idx:
        raise ValueError(f"عمود X_{tf} له {X.shape[-1]} ميزة فقط، لا يكفي للفهرس {idx}.")
    return X[:, -1, idx] - X[:, -2, idx]


def momentum_predict_fn(train: Dict, val: Dict, test: Dict,
                        feature: str = "close", tf: Optional[str] = None,
                        feature_order: Optional[List[str]] = None) -> np.ndarray:
    """فرضية الأساس (المرحلة ٠، اختبار سلامة المحور): التنبؤ = آخر تغيّر في
    السعر داخل نافذة كل عيّنة (استمرار الاتجاه الأخير) — لا تدريب، لا تحتاج
    ``train``/``val`` إطلاقاً، تُنتَج مباشرة من ``test``.

    مرّر ``feature_order`` (من ``dataset['feature_order']`` الأصلي) عبر
    ``functools.partial`` قبل تمريرها لـ``evaluate_hypothesis_over_rolling_windows``
    — راجع مثال الاستخدام في نهاية الدفتر.
    """
    return extract_feature_last_diff(test, feature=feature, tf=tf, feature_order=feature_order)


def evaluate_hypothesis_over_rolling_windows(
    windows: List[Tuple[Dict, Dict, Any]],
    predict_fn: Callable[[Dict, Dict, Dict], np.ndarray],
    target_key: str = "y_close_reg",
    window_names: Optional[List[str]] = None,
    ic_method: str = "spearman", n_shuffles: int = 1000,
    min_samples: int = 10, seed: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """يقيس فرضية واحدة عبر مخرجات ``rolling_splits`` مباشرة — لكل نافذة
    ``(train, val, test)``: ``predict_fn(train, val, test)`` تُنتج تنبؤات
    بطول ``test`` (بعد تسطيحه إن لزم)، تُقارَن بـ``test[target_key]``.

    ``predict_fn`` قد تتجاهل ``train``/``val`` كلياً (فرضية يدوية بلا تدريب،
    كـ``momentum_predict_fn``)، أو تُدرِّب نموذجاً على ``train`` — العقد نفسه
    في الحالتين.
    """
    window_names = window_names or [f"نافذة {i+1}" for i in range(len(windows))]
    if len(window_names) != len(windows):
        raise ValueError("عدد الأسماء يجب أن يطابق عدد النوافذ.")

    results = []
    for name, (train, val, test) in zip(window_names, windows):
        test_flat = concat_splits(test)
        preds = predict_fn(train, val, test)
        actuals = extract_actuals(test_flat, target_key=target_key)
        if len(preds) != len(actuals):
            raise ValueError(
                f"[{name}] طول التنبؤات ({len(preds)}) لا يطابق طول الأهداف "
                f"({len(actuals)}) — راجع predict_fn.")
        results.append((name, np.asarray(preds, dtype="float64"), actuals))

    return evaluate_windows(results, ic_method=ic_method, n_shuffles=n_shuffles,
                            min_samples=min_samples, seed=seed, verbose=verbose)
