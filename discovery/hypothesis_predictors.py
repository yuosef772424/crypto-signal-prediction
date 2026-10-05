"""
PURPOSE:  Candidate predictors of the lab's hypothesis rounds: systematic hypothesis generation (RSI vol-adjusted, asymmetric momentum, isolation forest), legacy backtest retests (CMO, TSI, DPO), ridge composite, fractal reversal.
TAGS:     make_isolation_forest_predict_fn, make_dpo_reversion, make_ridge_composite_predict_fn, make_fractal_reversal_predict_fn, _rsi_vol_adjusted, _asymmetric_momentum, _cmo_reversal, _tsi_momentum
PITFALLS: Only definitions: the candidate lists (GENERATIVE_CANDIDATES, LEGACY_BACKTEST_CANDIDATES, FRACTAL_REVERSAL_CANDIDATE) and their runs stay in the notebook cells. Fractal needs a dataset built with exclude_from_features=['open', ...]. Executed into the notebook's shared namespace by discovery/_loader.py (never imported on its own): names from the axis/pipeline notebooks resolve at call time. Extracted verbatim from signal_discovery_lab.ipynb cells 14, 16, 22 and 45 (sections 5, 7, 12).
"""
import numpy as np
import pandas as pd


def _rsi_vol_adjusted(X_last, feature_order):
    """(RSI-50) مطبَّعة بالتقلب الحالي (NATR_14) — عتبة تطرف نسبية للنظام
    السعري، لا 30/70 الثابتة. سالبة (نفترض ارتداداً، لا استمراراً)."""
    rsi = X_last[:, feature_order.index("RSI_14")]
    natr = X_last[:, feature_order.index("NATR_14")]
    return -(rsi - 50.0) / (natr + 1e-6)


def _asymmetric_momentum(X_last, feature_order):
    """في الهبوط (RET_1<0) نعتمد الزخم القصير (RET_1) — استجابة سريعة؛ في
    الصعود نعتمد زخماً أبطأ (RET_6) — تأكيداً أبطأ (الذعر أسرع من الجشع).
    ⚠️ تنبيه للحذر عند تصميم مرشّحين مماثلين: تحجيم بسيط (`ret1 * وزن`، لو
    كان الوزن ثابتاً في كل شقّ) هو تحويل رتيب لـRET_1 عالمياً، وسبيرمان
    (المقياس المُستخدَم في IC هنا) لا يتأثر بتحويل رتيب — أي IC سيتطابق مع
    IC خام RET_1 تماماً رغم اختلاف الشكل. هنا نستخدم متغيّرين مختلفين حسب
    الحالة (تحويل غير رتيب)، فترتيبه لا يطابق ترتيب RET_1 ولا RET_6 وحدهما
    (تحقّق تجريبي على بيانات حقيقية: سبيرمان مع RET_1 ≈0.73، مع RET_6 ≈0.66
    — لا ±1.0 لأيّهما)."""
    ret1 = X_last[:, feature_order.index("RET_1")]
    ret6 = X_last[:, feature_order.index("RET_6")]
    return np.where(ret1 < 0, ret1, ret6)


def make_isolation_forest_predict_fn(features, feature_order=None, contamination=0.1, random_state=42):
    """`kind="trained"`: يُدرِّب Isolation Forest على train (مُطبَّع بمتوسط/
    انحراف train نفسه)، ثم يُرجع درجة الشذوذ (`decision_function`، أعلى =
    أقلّ شذوذاً) على test — نفس نمط make_ridge_composite_predict_fn تماماً،
    لكن بلا هدف (كشف شذوذ غير مُشرَف، لا انحدار)."""
    def predict_fn(train, val, test):
        from sklearn.ensemble import IsolationForest
        train_flat = concat_splits(train)
        Xtr = extract_feature_matrix(train_flat, features, feature_order=feature_order)
        mu, sigma = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-9
        model = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=100)
        model.fit((Xtr - mu) / sigma)
        Xte = extract_feature_matrix(test, features, feature_order=feature_order)
        return model.decision_function((Xte - mu) / sigma)
    return predict_fn
def _cmo_reversal(series_2d, length=14):
    """CMO (Chande Momentum Oscillator) عبر pandas_ta_classic على كل عيّنة
    على حدة — عكسه لصياغة فرضية ارتداد (CMO متطرف يعكس)، بنفس منطق
    RSI_14_reversion. من أقوى نتائج الاستكشاف القديم (profit_factor≈1.31
    في الباكتيست الخام، بلا تصحيح للاختبارات المتعددة هناك)."""
    import pandas_ta_classic as ta
    out = np.full(series_2d.shape[0], np.nan)
    for i in range(series_2d.shape[0]):
        cmo = ta.cmo(pd.Series(series_2d[i]), length=length)
        if cmo is not None and len(cmo):
            out[i] = cmo.iloc[-1]
    return -out


def _tsi_momentum(series_2d, fast=5, slow=13, signal=5):
    """TSI (True Strength Index) عبر pandas_ta_classic — زخم مزدوج التنعيم،
    بلا عكس (فرضية استمرار، لا ارتداد). بارامترات مُقصَّرة (5/13/5 بدل
    13/25/13 الأصلية في الباكتيست القديم) — راجع "ملحق ٣" أعلاه لسبب هذا
    التكيّف (`window_size=32` لا يكفي لتقارب TSI(13,25))."""
    import pandas_ta_classic as ta
    out = np.full(series_2d.shape[0], np.nan)
    for i in range(series_2d.shape[0]):
        tsi = ta.tsi(pd.Series(series_2d[i]), fast=fast, slow=slow, signal=signal)
        if tsi is not None and len(tsi):
            out[i] = tsi.iloc[-1, 0]
    return out


def make_dpo_reversion(length):
    """DPO (Detrended Price Oscillator) بصيغته `centered=False` — لا إزاحة
    للخلف، بلا تسرّب معلومات مستقبلية، قابل للتنفيذ حيّاً (بخلاف
    `centered=True` المُوثَّق أعلى الدفتر كمثال "تقطير الرؤية المتأخّرة"،
    ذاك يُستخدَم كهدف لا كمدخل). عكسه لصياغة فرضية ارتداد — بنفس منطق
    استراتيجية صاحب المشروع الأصلية (شراء عند DPO سالب جداً). تجربة مباشرة
    من صاحب المشروع على BTC/1H: طول=2 → profit_factor=1.14، طول=14 →
    profit_factor=0.89 (خاسر) — كلا الطولين هنا للمقارنة عبر IC الصارم."""
    import pandas_ta_classic as ta

    def _fn(series_2d):
        out = np.full(series_2d.shape[0], np.nan)
        for i in range(series_2d.shape[0]):
            dpo = ta.dpo(pd.Series(series_2d[i]), length=length, centered=False)
            if dpo is not None and len(dpo):
                out[i] = dpo.iloc[-1]
        return -out
    return _fn
from sklearn.linear_model import RidgeCV
# ✅ extract_feature_matrix مُعرَّفة مرّة واحدة في قسم ٤ (إطار المرشّح) —
# يُعاد استخدامها هنا كما هي، ومن make_isolation_forest_predict_fn لاحقاً.


def make_ridge_composite_predict_fn(features, target, feature_order=None, alphas=(0.1, 1.0, 10.0, 100.0)):
    """يُدرِّب RidgeCV على train (مغلق، بلا حِقَب) متنبّئاً بـclean_reg_target
    لنفس target، ثم يُنبئ على test — نفس عقد predict_fn المُستخدَم مع
    evaluate_candidate."""
    def predict_fn(train, val, test):
        train_flat = concat_splits(train)
        Xtr = extract_feature_matrix(train_flat, features, feature_order=feature_order)
        ytr = clean_reg_target(train_flat, target)
        mu, sigma = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-9
        model = RidgeCV(alphas=alphas)
        model.fit((Xtr - mu) / sigma, ytr)
        Xte = extract_feature_matrix(test, features, feature_order=feature_order)
        return model.predict((Xte - mu) / sigma)
    return predict_fn
def _fractal_reversal_signal(high_2d, low_2d, window=5):
    """فركتال ويليامز على آخر خطوة "مؤكَّدة" فقط (T-1-window//2)، لا آخر
    خطوة في النافذة — تفادياً لتسرّب معلومة مستقبلية عبر center=True (راجع
    الخلية النصية أعلاه). -1 = آخر تطرّف مؤكَّد كان قمّة (نتوقّع ارتداداً
    هبوطياً)، +1 = كان قاعاً (نتوقّع ارتداداً صعودياً)، 0 = لا تطرّف عندها."""
    if window % 2 == 0:
        raise ValueError("window يجب أن يكون فردياً (مركز واضح لكل جهة).")
    half = window // 2
    confirmed_idx = high_2d.shape[1] - 1 - half
    if confirmed_idx < half:
        raise ValueError(
            f"طول النافذة الزمنية ({high_2d.shape[1]}) أقصر من اللازم "
            f"لتأكيد فركتال بعرض {window} — لا نقطة زمنية صالحة."
        )
    out = np.zeros(high_2d.shape[0])
    for i in range(high_2d.shape[0]):
        h, l = high_2d[i], low_2d[i]
        lo, hi = confirmed_idx - half, confirmed_idx + half + 1
        is_high = h[confirmed_idx] == h[lo:hi].max()
        is_low = l[confirmed_idx] == l[lo:hi].min()
        if is_high and not is_low:
            out[i] = -1.0
        elif is_low and not is_high:
            out[i] = 1.0
    return out


def make_fractal_reversal_predict_fn(feature_order=None, window=5):
    """`kind="series"` مزدوج (high وlow معاً) — يفشل بخطأ صريح إن لم تحمل
    dataset الحالية high/low فعلياً (راجع العائق ١ أعلاه)، لا صمتاً بصفر."""
    def predict_fn(train, val, test):
        test_flat = concat_splits(test)
        fo = feature_order if feature_order is not None else test_flat.get("feature_order")
        if not fo or "high" not in fo or "low" not in fo:
            raise ValueError(
                "fractal_reversal يحتاج 'high' و'low' في feature_order — "
                "أعد بناء dataset بـ exclude_from_features=['open', 'volume'] "
                "(بدل القائمة الافتراضية التي تستبعد high/low أيضاً)."
            )
        high_2d = extract_feature_series(test_flat, "high", feature_order=fo)
        low_2d = extract_feature_series(test_flat, "low", feature_order=fo)
        return _fractal_reversal_signal(high_2d, low_2d, window=window)
    return predict_fn
