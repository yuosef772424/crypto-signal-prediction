"""
PURPOSE:  Phase-3 discovery tools without prior rules: Matrix Profile, SHAP interaction pairs, ensemble feature ranking, K-means/HDBSCAN regimes, cheap cross-asset coupling test.
TAGS:     make_matrix_profile_predict_fn, discover_shap_interaction_pairs, discover_ensemble_feature_ranking, make_cluster_regime_predict_fn, make_cross_asset_predict_fn, shap, hdbscan, stumpy
PITFALLS: Optional heavy imports (shap, xgboost, hdbscan, stumpy) happen inside the functions. Only definitions here: the runs stay in the notebook cells. Executed into the notebook's shared namespace by discovery/_loader.py (never imported on its own): names from the axis/pipeline notebooks resolve at call time. Extracted verbatim from signal_discovery_lab.ipynb cells 28, 31, 34, 38 and 41 (sections 9, 10).
"""
import numpy as np
import pandas as pd


def make_matrix_profile_predict_fn(target, feature="close", feature_order=None, k=3):
    """أساس Matrix Profile: بحث أقرب جار (k-NN) بمسافة إقليدية بعد تطبيع-Z
    — لكل نافذة اختبار، تُطبَّع نافذة `feature` (افتراضياً `close`)، تُقارَن
    بمكتبة نوافذ train كلّها، ويُتنبَّأ بمتوسط العائد الفعلي (`target`) الذي
    تلا أقرب k نافذة تاريخياً مشابهة شكلياً. راجع الشرح أعلاه لسبب الحساب
    المباشر بدل `stumpy.mass`."""
    def predict_fn(train, val, test):
        train_flat = concat_splits(train)
        train_series = extract_feature_series(train_flat, feature, feature_order=feature_order)
        train_targets = clean_reg_target(train_flat, target)
        test_series = extract_feature_series(test, feature, feature_order=feature_order)

        def znorm(a):
            mu = a.mean(axis=1, keepdims=True)
            sd = a.std(axis=1, keepdims=True) + 1e-9
            return (a - mu) / sd

        Ztr, Zte = znorm(train_series), znorm(test_series)
        preds = np.empty(len(Zte))
        for i in range(len(Zte)):
            d = np.linalg.norm(Ztr - Zte[i], axis=1)
            k_eff = min(k, len(d))
            nn_idx = np.argpartition(d, k_eff - 1)[:k_eff]
            preds[i] = train_targets[nn_idx].mean()
        return preds
    return predict_fn


def discover_shap_interaction_pairs(train, target, feature_order, top_k=5, max_depth=4,
                                    n_estimators=200, random_state=42):
    """يُدرِّب RandomForestRegressor ضحلاً على متجه الميزات الكامل (آخر خطوة
    زمنية، عبر extract_feature_matrix)، يحسب SHAP interaction values
    (`shap.TreeExplainer`)، ويُرجع أقوى top_k أزواج ميزات (بمعزل عن القطر —
    الأثر الرئيسي المنفرد لكل ميزة، لا تفاعلاً) مرتّبة تنازلياً بمقدار
    التفاعل المطلق المتوسط."""
    from sklearn.ensemble import RandomForestRegressor
    import shap
    train_flat = concat_splits(train)
    X = extract_feature_matrix(train_flat, feature_order=feature_order)
    y = clean_reg_target(train_flat, target)
    model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators,
                                  random_state=random_state, n_jobs=-1).fit(X, y)
    explainer = shap.TreeExplainer(model)
    interaction_values = explainer.shap_interaction_values(X)
    mean_abs = np.abs(interaction_values).mean(axis=0)
    np.fill_diagonal(mean_abs, 0.0)
    F = len(feature_order)
    pairs = [(i, j, mean_abs[i, j]) for i in range(F) for j in range(i + 1, F)]
    pairs.sort(key=lambda t: t[2], reverse=True)
    return [(feature_order[i], feature_order[j], float(mag)) for i, j, mag in pairs[:top_k]]


def discover_ensemble_feature_ranking(train, target, feature_order, top_k=10, random_state=42):
    """إجماع أهمية ميزات من ثلاثة نماذج مختلفة الطبيعة (RandomForestRegressor
    + GradientBoostingRegressor + XGBRegressor)، بمتوسط `feature_importances_`
    الثلاثة بالتساوي. انحدار على `clean_reg_target` (لا تصنيف ثنائي كالأصل
    في `xgboost.ipynb`) ليتوافق مع صرامة هذا الدفتر. تُشغَّل مرّة واحدة فقط
    (على `train` النافذة الأولى) — نفس فصل الاكتشاف عن التحقّق المُطبَّق
    مع SHAP أعلاه."""
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from xgboost import XGBRegressor
    train_flat = concat_splits(train)
    X = extract_feature_matrix(train_flat, feature_order=feature_order)
    y = clean_reg_target(train_flat, target)
    rf = RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_leaf=20,
                               random_state=random_state, n_jobs=-1).fit(X, y)
    gb = GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05,
                                   random_state=random_state).fit(X, y)
    xgbr = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=random_state,
                        n_jobs=-1, tree_method="hist", verbosity=0).fit(X, y)
    imp = (rf.feature_importances_ + gb.feature_importances_ + xgbr.feature_importances_) / 3.0
    order = np.argsort(imp)[::-1][:top_k]
    return [(feature_order[i], float(imp[i])) for i in order]
def make_cluster_regime_predict_fn(target, feature="close", feature_order=None, algo="kmeans",
                                    n_clusters=5, min_cluster_size=None, random_state=42):
    # حلقة يدوية بنفس نمط Matrix Profile/Ridge composite أعلاه — يحتاج
    # target صراحةً فلا يلائم بناء builder(feature_order) العام في
    # make_candidate_predict_fn. يُدرَّب على train (تجميع + متوسط عائد كل
    # عنقود)، ثم يُسقِط test على نفس العناقيد.
    def predict_fn(train, val, test):
        train_flat = concat_splits(train)
        train_series = extract_feature_series(train_flat, feature, feature_order=feature_order)
        train_targets = clean_reg_target(train_flat, target)
        test_series = extract_feature_series(test, feature, feature_order=feature_order)

        def znorm(a):
            mu = a.mean(axis=1, keepdims=True)
            sd = a.std(axis=1, keepdims=True) + 1e-9
            return (a - mu) / sd

        Ztr, Zte = znorm(train_series), znorm(test_series)
        global_mean = train_targets.mean()

        if algo == "kmeans":
            from sklearn.cluster import KMeans
            k_eff = max(2, min(n_clusters, len(Ztr) // 5))
            model = KMeans(n_clusters=k_eff, random_state=random_state, n_init=10)
            train_labels = model.fit_predict(Ztr)
            test_labels = model.predict(Zte)
        elif algo == "hdbscan":
            import hdbscan
            mcs = min_cluster_size or max(5, len(Ztr) // 20)
            model = hdbscan.HDBSCAN(min_cluster_size=mcs, prediction_data=True)
            train_labels = model.fit_predict(Ztr)
            test_labels, _ = hdbscan.approximate_predict(model, Zte)
        else:
            raise ValueError(f"algo غير مدعوم: {algo}")

        # نقاط الضجيج (label=-1 في train) لا تُشكِّل عنقوداً حقيقياً — تُستبعَد
        # من قاموس المتوسطات، فيرث أي test تُسقَط عليها (أو على عنقود غير
        # موجود في train) متوسط train العام بدل متوسط وهمي.
        cluster_mean = {lbl: train_targets[train_labels == lbl].mean()
                        for lbl in np.unique(train_labels) if lbl != -1}
        return np.array([cluster_mean.get(lbl, global_mean) for lbl in test_labels])
    return predict_fn
def make_cross_asset_predict_fn(feature="RET_1", feature_order=None, agg="mean"):
    # يحتاج test = {اسم_الأصل: قسم} (keep_asset_test_separate=True) — يبني
    # قاموس بحث لكل أصل آخر (الطابع الزمني -> قيمة feature)، ثم لكل عيّنة
    # في كل أصل يُتوسَّط قيمة بقيّة الأصول عند نفس الطابع الزمني بالضبط
    # (NaN إن لم يوجد أي أصل آخر بنفس اللحظة تماماً — تُستبعَد لاحقاً في
    # حساب IC، لا تُصفَّر).
    def predict_fn(train, val, test):
        asset_feat, asset_ts = {}, {}
        for name, split in test.items():
            asset_feat[name] = extract_feature_last_value(split, feature=feature, feature_order=feature_order)
            asset_ts[name] = np.asarray(split["last_candles"])[:, TS_COL]

        preds = []
        for name in test.keys():
            ts = asset_ts[name]
            n = len(ts)
            other_names = [o for o in test if o != name]
            other_lookup = {o: dict(zip(asset_ts[o].tolist(), asset_feat[o].tolist())) for o in other_names}
            out = np.full(n, np.nan)
            for i in range(n):
                t = ts[i]
                vals = [other_lookup[o][t] for o in other_names if t in other_lookup[o]]
                if vals:
                    out[i] = float(np.mean(vals)) if agg == "mean" else float(np.median(vals))
            preds.append(out)
        return np.concatenate(preds)
    return predict_fn


# ملاحظة تشغيل: يحتاج windows_sep = rolling_splits(dataset, ..., keep_asset_test_separate=True)
# (لا windows الافتراضية أعلاه — راجع الشرح فوق) و evaluate_candidate العام (يقبل test بأي شكل
# طالما predict_fn يتعامل معه بنفسه). النتائج الفعلية أدناه أُنتِجت على نطاق WIDEHIST (56 نافذة).
