"""
PURPOSE:  Single-candidate framework: extract feature values from windows and build predict_fn closures for feature, interaction, custom, series and generic candidate dicts.
TAGS:     extract_feature_last_value, extract_feature_matrix, extract_feature_series, make_candidate_predict_fn, make_feature_predict_fn, candidate dict, predict_fn
PITFALLS: A candidate dict is {name, track, feature | kind, transform, hypothesis}; any feature in feature_order is a candidate immediately. Executed into the notebook's shared namespace by discovery/_loader.py (never imported on its own): names from the axis/pipeline notebooks resolve at call time. Extracted verbatim from signal_discovery_lab.ipynb cell 8 (section 4).
"""
import numpy as np
import pandas as pd


def extract_feature_last_value(split, feature, tf=None, feature_order=None):
    """آخر قيمة (خطوة زمنية أخيرة) لميزة واحدة داخل نافذة كل عيّنة — حالة
    المؤشر عند لحظة القرار، لا فرقها كـextract_feature_last_diff."""
    split = concat_splits(split)
    feature_order = feature_order if feature_order is not None else split.get("feature_order")
    if not feature_order:
        raise ValueError("مرّر feature_order صراحةً (dataset['feature_order']).")
    idx = feature_order.index(feature)
    if tf is None:
        tf = [k for k in split if k.startswith("X_")][0][2:]
    X = np.asarray(split[f"X_{tf}"])
    return X[:, -1, idx]


def extract_feature_matrix(split, features=None, tf=None, feature_order=None):
    """مصفوفة كل الميزات (أو مجموعة فرعية) في آخر خطوة زمنية — (N, len(features)).
    أعمّ من extract_feature_last_value: تُستخدَم لأي مرشّح يحتاج أكثر من ميزة
    معاً (تفاعل، مركَّب Ridge، تدريب Isolation Forest، أو دالة مخصَّصة كاملة
    على متجه الميزات — kind='custom'/'trained' أدناه)."""
    split = concat_splits(split)
    feature_order = feature_order if feature_order is not None else split.get("feature_order")
    if not feature_order:
        raise ValueError("مرّر feature_order صراحةً (dataset['feature_order']).")
    if tf is None:
        tf = [k for k in split if k.startswith("X_")][0][2:]
    X = np.asarray(split[f"X_{tf}"])[:, -1, :]
    if features is not None:
        idx = [feature_order.index(f) for f in features]
        X = X[:, idx]
    return X


def extract_feature_series(split, feature, tf=None, feature_order=None):
    """كل الخطوات الزمنية لميزة واحدة داخل نافذة كل عيّنة — (N, T)، لا آخر
    خطوة فقط كـextract_feature_last_value. لازمة لأي مؤشر مشتقّ يحتاج
    تاريخاً كاملاً ضمن النافذة (CMO/TSI عبر pandas_ta مثلاً، لا يُحسَبان من
    قيمة أخيرة وحدها) — kind='series' أدناه."""
    split = concat_splits(split)
    feature_order = feature_order if feature_order is not None else split.get("feature_order")
    if not feature_order:
        raise ValueError("مرّر feature_order صراحةً (dataset['feature_order']).")
    idx = feature_order.index(feature)
    if tf is None:
        tf = [k for k in split if k.startswith("X_")][0][2:]
    X = np.asarray(split[f"X_{tf}"])
    return X[:, :, idx]


def make_feature_predict_fn(feature, transform=None, tf=None, feature_order=None):
    """predict_fn جاهزة لـevaluate_candidate من أي ميزة في feature_order.
    مرّر transform (مثلاً lambda v: -(v-50.0) لعكس RSI حول نقطة المنتصف)
    لصياغة فرضية اتجاه محدّدة بدل القيمة الخام."""
    def predict_fn(train, val, test):
        v = extract_feature_last_value(test, feature=feature, tf=tf, feature_order=feature_order)
        return transform(v) if transform else v
    return predict_fn


def make_interaction_predict_fn(feat_a, feat_b, op="mul", transform=None, tf=None, feature_order=None):
    """predict_fn من تفاعل بين ميزتين موجودتين (ضرب أو فرق) — يغطي فرضيات
    "تفاعلات" (قسم ٣ في خطة المشروع، مثال: حجم×جسم الشمعة) بلا الحاجة لعمود
    ميزة جديد محسوب مسبقاً في خط الأنابيب."""
    def predict_fn(train, val, test):
        a = extract_feature_last_value(test, feature=feat_a, tf=tf, feature_order=feature_order)
        b = extract_feature_last_value(test, feature=feat_b, tf=tf, feature_order=feature_order)
        v = a * b if op == "mul" else (a - b)
        return transform(v) if transform else v
    return predict_fn


def make_custom_predict_fn(fn, tf=None, feature_order=None):
    """predict_fn من دالة مخصَّصة على متجه الميزات الكامل عند آخر خطوة زمنية
    (`fn(X_last, feature_order) -> np.ndarray`) — بلا تدريب (test فقط، بلا
    استخدام train/val). يغطي فرضيات "توليد ممنهج" بلا حاجة لعمود ميزة/تفاعل
    ثنائي جاهز: مقلوب الافتراض الضمني (مثلاً RSI مطبَّع بالتقلب)، كسر
    التناظر (معادلتان مختلفتان للصعود/الهبوط عمداً)، إلخ — راجع قسم "توليد
    فرضيات ممنهج" أدناه."""
    def predict_fn(train, val, test):
        X_last = extract_feature_matrix(test, tf=tf, feature_order=feature_order)
        return fn(X_last, feature_order)
    return predict_fn


def make_series_predict_fn(fn, feature="close", tf=None, feature_order=None):
    """predict_fn من دالة مخصَّصة تُطبَّق على كامل نافذة ميزة واحدة عبر
    الزمن (`fn(series_2d) -> np.ndarray`، حيث `series_2d` شكلها (N, T)) —
    بخلاف kind='custom' (متجه كل الميزات معاً، لكن آخر خطوة فقط)، هذا لميزة
    واحدة تحتاج تاريخاً كاملاً ضمن النافذة (مؤشر مُشتقّ عبر pandas_ta مثل
    CMO/TSI لا يُحسَب من قيمة أخيرة وحدها)."""
    def predict_fn(train, val, test):
        series = extract_feature_series(test, feature=feature, tf=tf, feature_order=feature_order)
        return fn(series)
    return predict_fn


def make_candidate_predict_fn(cand, tf=None, feature_order=None):
    """يبني predict_fn من قاموس مرشّح واحد بصرف النظر عن نوعه (`kind`) —
    نقطة التفرّع الوحيدة، فلا يحتاج scan_candidates/run_batch_and_register
    معرفة الفرق بين الأنواع:

    * الافتراضي (بلا `kind`): ميزة واحدة عبر make_feature_predict_fn.
    * `kind="interaction"`: تفاعل بين ميزتين (`feat_a`/`feat_b`/`op`).
    * `kind="custom"`: دالة مخصَّصة بلا تدريب على متجه الميزات الكامل (`fn`).
    * `kind="trained"`: مرشّح يحتاج تدريباً على train (مثل Isolation Forest) —
      `builder(feature_order=...)` يُرجع predict_fn جاهزة (نفس نمط
      make_ridge_composite_predict_fn في قسم البحث التركيبي).
    * `kind="series"`: دالة مخصَّصة على كامل نافذة ميزة واحدة عبر الزمن
      (`fn`/`feature`) — لمؤشر مشتقّ يحتاج تاريخاً كاملاً (CMO/TSI)."""
    kind = cand.get("kind", "feature")
    if kind == "interaction":
        return make_interaction_predict_fn(cand["feat_a"], cand["feat_b"], op=cand.get("op", "mul"),
                                           transform=cand.get("transform"), tf=tf, feature_order=feature_order)
    if kind == "custom":
        return make_custom_predict_fn(cand["fn"], tf=tf, feature_order=feature_order)
    if kind == "trained":
        return cand["builder"](feature_order=feature_order)
    if kind == "series":
        return make_series_predict_fn(cand["fn"], feature=cand.get("feature", "close"), tf=tf, feature_order=feature_order)
    return make_feature_predict_fn(cand["feature"], transform=cand.get("transform"), tf=tf, feature_order=feature_order)
