"""
PURPOSE:  Self-tests of the axis (IC, decile, permutation, evaluate_windows, integration layer) on synthetic data: check(), PASS/FAIL, ALL_TESTS (the 38 t_* functions). Defines only; the runner notebook runs them.
TAGS:     selftests, check, PASS, FAIL, ALL_TESTS, t_ic_*, t_decile_*, t_permutation_*, t_evaluate_windows_*, t_concat_splits_*, t_extract_*, synthetic data
PITFALLS: The old notebook ran ALL_TESTS at load (and raised AssertionError on a failure); that invocation now lives in the runner notebook's self-test cell. Executed into the one shared signal_eval namespace by signal_eval/_loader.py (never imported on its own): names from other modules resolve at call time.

## ٧) اختبارات ذاتية

كل دالة أعلاه مُختبَرة ببيانات تركيبية بإشارة معروفة الحجم مُحقَنة عمداً — لا اعتماد على بيانات حقيقية أو تشغيل دفتر التحضير. شغّلها بعد أي تعديل.
"""
# @title
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  ✅ {name}")
    except Exception as e:
        FAIL.append((name, f"{type(e).__name__}: {e}"))
        print(f"  ❌ {name}: {type(e).__name__}: {str(e)[:200]}")


# ══════════════════════════════════════════════════════════════════════════
# compute_ic
# ══════════════════════════════════════════════════════════════════════════

def t_ic_recovers_known_correlation():
    rng = np.random.default_rng(0)
    n = 5000
    true_signal = rng.normal(0, 1, n)
    actuals = 0.03 * true_signal + rng.normal(0, 1, n)   # IC حقيقي صغير ~0.03 (بمعيار سبيرمان تقريباً)
    preds = true_signal + rng.normal(0, 0.1, n)
    ic = compute_ic(preds, actuals, method="spearman")
    assert 0.01 < ic < 0.06, f"IC={ic} خارج المدى المتوقَّع"


def t_ic_near_zero_on_independent_random_data():
    rng = np.random.default_rng(1)
    n = 3000
    preds = rng.normal(0, 1, n)
    actuals = rng.normal(0, 1, n)
    ic = compute_ic(preds, actuals)
    se = 1.0 / np.sqrt(n)               # خطأ معياري تقريبي لارتباط سبيرمان
    assert abs(ic) < 4 * se, f"IC={ic} أكبر من المتوقَّع للضجيج المحض"


def t_ic_ignores_nan_not_zeros_them():
    p = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    a = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    ic = compute_ic(p, a, min_samples=5)
    assert abs(ic - 1.0) < 1e-6, f"يجب أن يكون الارتباط شبه تام بعد استبعاد NaN، حصل {ic}"


def t_ic_raises_below_min_samples():
    p, a = np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])
    try:
        compute_ic(p, a, min_samples=10)
    except ValueError as e:
        assert "10" in str(e)
        return
    raise AssertionError("كان يجب أن يرفض عدداً أقل من الحدّ الأدنى")


def t_ic_spearman_robust_to_monotonic_nonlinear_transform():
    rng = np.random.default_rng(2)
    n = 2000
    p = rng.uniform(1, 10, n)
    a = np.exp(p) + rng.normal(0, 0.01, n)          # تحويل غير خطي لكن رتيب تماماً
    ic_s = compute_ic(p, a, method="spearman")
    ic_p = compute_ic(p, a, method="pearson")
    assert ic_s > 0.99, f"سبيرمان يجب أن يلتقط العلاقة الرتيبة تماماً، حصل {ic_s}"
    assert ic_p < ic_s, "بيرسون يجب أن يكون أضعف على علاقة غير خطية بهذا الحدّة"


def t_ic_zero_when_one_array_constant():
    p = np.full(20, 5.0)
    a = np.arange(20.0)
    assert compute_ic(p, a, min_samples=5) == 0.0


def t_ic_rejects_mismatched_shapes():
    try:
        compute_ic(np.zeros(5), np.zeros(6), min_samples=1)
    except ValueError:
        return
    raise AssertionError("أشكال مختلفة يجب أن تُرفَض")


def t_ic_rejects_unknown_method():
    try:
        compute_ic(np.zeros(20), np.zeros(20), method="kendall", min_samples=1)
    except ValueError as e:
        assert "kendall" in str(e)
        return
    raise AssertionError("طريقة غير معروفة يجب أن تُرفَض")


# ══════════════════════════════════════════════════════════════════════════
# decile_spread
# ══════════════════════════════════════════════════════════════════════════

def t_decile_detects_monotonic_relationship():
    rng = np.random.default_rng(3)
    n = 2000
    p = rng.normal(0, 1, n)
    a = p + rng.normal(0, 0.3, n)
    res = decile_spread(p, a, n_deciles=10, min_per_decile=20)
    assert res["monotonic"] is True
    assert res["spread"] > 0.5
    assert len(res["deciles"]) == 10
    assert res["deciles"]["n"].sum() == n


def t_decile_near_zero_spread_on_random_data():
    rng = np.random.default_rng(4)
    n = 3000
    p, a = rng.normal(0, 1, n), rng.normal(0, 1, n)
    res = decile_spread(p, a, min_per_decile=50)
    assert abs(res["spread"]) < 0.3, f"فرق العُشر {res['spread']} كبير جداً للضجيج المحض"


def t_decile_raises_on_insufficient_samples():
    p, a = np.arange(20.0), np.arange(20.0)
    try:
        decile_spread(p, a, n_deciles=10, min_per_decile=5)   # يحتاج 50، متوفر 20
    except ValueError:
        return
    raise AssertionError("عيّنات قليلة جداً لكل عُشر يجب أن تُرفَض")


def t_decile_t_stat_large_for_strong_signal_small_for_none():
    rng = np.random.default_rng(5)
    n = 1000
    p_strong = rng.normal(0, 1, n)
    a_strong = 2.0 * p_strong + rng.normal(0, 0.5, n)
    res_strong = decile_spread(p_strong, a_strong, min_per_decile=20)

    p_none, a_none = rng.normal(0, 1, n), rng.normal(0, 1, n)
    res_none = decile_spread(p_none, a_none, min_per_decile=20)

    assert abs(res_strong["top_minus_bottom_t_stat"]) > abs(res_none["top_minus_bottom_t_stat"])
    assert abs(res_strong["top_minus_bottom_t_stat"]) > 5


# ══════════════════════════════════════════════════════════════════════════
# permutation_baseline
# ══════════════════════════════════════════════════════════════════════════

def t_permutation_flags_real_signal_as_significant():
    rng = np.random.default_rng(6)
    n = 1500
    signal = rng.normal(0, 1, n)
    p = signal + rng.normal(0, 0.2, n)
    a = signal + rng.normal(0, 0.2, n)
    res = permutation_baseline(p, a, n_shuffles=300, seed=42)
    assert res["p_value"] < 0.01, f"p_value={res['p_value']} — إشارة قوية يجب أن تكون معنوية بوضوح"
    assert res["percentile"] > 95


def t_permutation_null_on_pure_random_data():
    rng = np.random.default_rng(7)
    n = 500
    p, a = rng.normal(0, 1, n), rng.normal(0, 1, n)
    res = permutation_baseline(p, a, n_shuffles=500, seed=1)
    assert res["p_value"] > 0.05, f"p_value={res['p_value']} — ضجيج محض لا يجب أن يبدو معنوياً"


def t_permutation_reproducible_with_seed():
    rng = np.random.default_rng(8)
    p, a = rng.normal(0, 1, 200), rng.normal(0, 1, 200)
    r1 = permutation_baseline(p, a, n_shuffles=100, seed=99)
    r2 = permutation_baseline(p, a, n_shuffles=100, seed=99)
    assert r1["null_mean"] == r2["null_mean"] and r1["p_value"] == r2["p_value"]


# ══════════════════════════════════════════════════════════════════════════
# evaluate_windows
# ══════════════════════════════════════════════════════════════════════════

def t_evaluate_windows_aggregates_consistent_signal():
    rng = np.random.default_rng(9)
    windows = []
    for i in range(5):
        n = 800
        signal = rng.normal(0, 1, n)
        p = signal + rng.normal(0, 1, n)
        a = 0.05 * signal + rng.normal(0, 1, n)
        windows.append((f"w{i}", p, a))
    res = evaluate_windows(windows, n_shuffles=200, seed=1, verbose=False)
    assert res["n_ok"] == 5
    assert res["mean_ic"] > 0
    assert res["consistent_sign"] is True


def t_evaluate_windows_skips_failed_window_not_whole_run():
    rng = np.random.default_rng(10)
    good = [(f"w{i}", rng.normal(0, 1, 500), rng.normal(0, 1, 500)) for i in range(3)]
    bad = [("w_bad", np.array([1.0, 2.0]), np.array([1.0, 2.0]))]   # عيّنات قليلة جداً
    res = evaluate_windows(good + bad, min_samples=10, n_shuffles=100, verbose=False)
    assert res["n_ok"] == 3
    assert (res["per_window"]["status"] == "skipped").sum() == 1


def t_evaluate_windows_no_signal_gives_low_frac_significant():
    rng = np.random.default_rng(11)
    windows = [(f"w{i}", rng.normal(0, 1, 600), rng.normal(0, 1, 600)) for i in range(5)]
    res = evaluate_windows(windows, n_shuffles=200, seed=2, verbose=False)
    assert res["frac_significant"] <= 0.4, f"ضجيج محض أعطى نوافذ معنوية كثيرة: {res['frac_significant']}"


# ══════════════════════════════════════════════════════════════════════════
# طبقة التكامل
# ══════════════════════════════════════════════════════════════════════════

def _fake_split(n=200, seed=0, feature_order=None, y_keys=("y_close_reg", "y_close_class")):
    """يحاكي الشكل الفعلي لقسم من _take (تحقّقتُه من المصدر مباشرة):
    الأهداف متداخلة تحت مفتاح فرعي واحد 'y' — لا مفاتيح y_head مباشرة على
    القسم نفسه. لا 'feature_order' (غير موجودة في أي قسم فعلياً، فقط في
    dataset الأصلي قبل التقطيع) — أي اختبار يحتاجها يمرّرها صراحةً.
    """
    rng = np.random.default_rng(seed)
    feature_order = feature_order or ["close", "RSI_14", "volume"]
    T, F = 32, len(feature_order)
    X = rng.normal(0, 1, (n, T, F)).astype("float32")
    y = {}
    for k in y_keys:
        y[k] = (np.sign(rng.normal(0, 1, n)) if k.endswith("_class")
                else rng.normal(0, 0.05, n)).astype("float32")
    return {"X_1D": X, "y": y, "base_params": np.zeros((n, 2), "float32"),
           "last_candles": np.zeros((n, 7), "float64")}


def t_concat_splits_passthrough_on_flat_dict():
    s = _fake_split()
    out = concat_splits(s)
    assert out is s


def t_concat_splits_nested_y_key_not_mistaken_for_asset_dict():
    """s['y'] قاموس أيضاً — إن اعتمد التمييز على isinstance(value, dict) فقط
    قد يُلتبَس بقاموس أصول؛ التمييز الفعلي عبر 'base_params' يتفادى هذا."""
    s = _fake_split(n=30, seed=99)
    assert isinstance(s["y"], dict)          # التبس محتمل لو اعتمدنا نوع القيمة فقط
    out = concat_splits(s)
    assert out is s


def t_concat_splits_merges_dict_of_assets():
    s1, s2 = _fake_split(n=50, seed=1), _fake_split(n=30, seed=2)
    merged = concat_splits({"AAAUSDT": s1, "BBBUSDT": s2})
    assert merged["X_1D"].shape[0] == 80
    assert merged["y"]["y_close_reg"].shape[0] == 80
    np.testing.assert_array_equal(merged["X_1D"][:50], s1["X_1D"])
    np.testing.assert_array_equal(merged["X_1D"][50:], s2["X_1D"])
    np.testing.assert_array_equal(merged["y"]["y_close_reg"][:50], s1["y"]["y_close_reg"])


def t_concat_splits_not_fooled_by_asset_names_starting_with_x():
    """يحاكي الفخّ الفعلي: اسم عملة مثل XRPUSDT قد يبدو كأنه مفتاح X_*
    لتخمين بالاسم — التمييز عبر 'base_params' لا يقع في هذا الفخّ."""
    s1, s2 = _fake_split(n=20, seed=20), _fake_split(n=15, seed=21)
    merged = concat_splits({"XRPUSDT": s1, "X_WEIRDUSDT": s2})
    assert merged["X_1D"].shape[0] == 35
    assert merged["y"]["y_close_reg"].shape[0] == 35


def t_concat_splits_raises_clear_error_when_asset_split_malformed():
    bad = {"AAAUSDT": {"some_other_key": np.zeros(5)}}      # بلا 'base_params'
    try:
        concat_splits(bad)
    except ValueError as e:
        assert "AAAUSDT" in str(e)
        return
    raise AssertionError("قسم أصل بلا 'base_params' يجب أن يُرفَض بوضوح")


def t_extract_actuals_reads_target_key():
    s = _fake_split(n=40, seed=3)
    a = extract_actuals(s, target_key="y_close_reg")
    np.testing.assert_array_equal(a, s["y"]["y_close_reg"].astype("float64"))


def t_extract_actuals_matches_with_or_without_y_prefix():
    """add_y_prefix قد يكون طُبِّق (prefix_y=True، الافتراضي) أو لا —
    يجب أن تعمل extract_actuals بالحالتين بلا تعديل من المستخدم."""
    s_prefixed = _fake_split(n=30, seed=30, y_keys=("y_close_reg",))
    a1 = extract_actuals(s_prefixed, target_key="close_reg")     # بلا بادئة
    np.testing.assert_array_equal(a1, s_prefixed["y"]["y_close_reg"].astype("float64"))

    s_unprefixed = _fake_split(n=30, seed=31, y_keys=("close_reg",))
    a2 = extract_actuals(s_unprefixed, target_key="y_close_reg")  # مع بادئة
    np.testing.assert_array_equal(a2, s_unprefixed["y"]["close_reg"].astype("float64"))


def t_extract_actuals_raises_on_missing_key():
    s = _fake_split(n=10, seed=4)
    try:
        extract_actuals(s, target_key="y_missing")
    except KeyError as e:
        assert "y_close_reg" in str(e)
        return
    raise AssertionError("مفتاح مفقود يجب أن يُرفَض بوضوح")


def t_extract_actuals_raises_clear_error_without_y_key():
    s = {"X_1D": np.zeros((5, 3, 2), "float32"), "base_params": np.zeros((5, 2), "float32"),
        "last_candles": np.zeros((5, 7))}     # بلا 'y' إطلاقاً
    try:
        extract_actuals(s)
    except ValueError as e:
        assert "'y'" in str(e)
        return
    raise AssertionError("قسم بلا مفتاح 'y' يجب أن يُرفَض بوضوح لا بخطأ غامض")


def t_extract_feature_last_diff_matches_manual():
    # feature_order غير موجودة على القسم فعلياً (مؤكَّد من الشكل الحقيقي) —
    # تُمرَّر صراحةً، كما في الاستخدام الفعلي دائماً.
    s = _fake_split(n=15, seed=5, feature_order=["a", "close", "b"])
    diff = extract_feature_last_diff(s, feature="close", tf="1D",
                                     feature_order=["a", "close", "b"])
    expected = s["X_1D"][:, -1, 1] - s["X_1D"][:, -2, 1]
    np.testing.assert_allclose(diff, expected, atol=1e-6)


def t_extract_feature_last_diff_requires_explicit_feature_order():
    """الحالة الواقعية: نوافذ rolling_splits لا تحمل feature_order إطلاقاً —
    يجب أن تُرفَض بوضوح بلا تمريرها، وتنجح عند تمريرها صراحةً."""
    s = _fake_split(n=12, seed=14, feature_order=["a", "close", "b"])
    assert "feature_order" not in s                  # الحالة الواقعية الافتراضية الآن
    try:
        extract_feature_last_diff(s, feature="close", tf="1D")
    except ValueError as e:
        assert "feature_order" in str(e)
    else:
        raise AssertionError("غياب feature_order يجب أن يُرفَض بوضوح")
    diff = extract_feature_last_diff(s, feature="close", tf="1D",
                                     feature_order=["a", "close", "b"])
    expected = s["X_1D"][:, -1, 1] - s["X_1D"][:, -2, 1]
    np.testing.assert_allclose(diff, expected, atol=1e-6)


def t_momentum_predict_fn_requires_explicit_feature_order():
    s = _fake_split(n=20, seed=15, feature_order=["close", "RSI_14", "volume"])
    preds = momentum_predict_fn(None, None, s, feature_order=["close", "RSI_14", "volume"])
    assert len(preds) == 20 and np.isfinite(preds).all()


def t_extract_feature_last_diff_raises_on_unknown_feature():
    s = _fake_split(n=10, seed=6)
    try:
        extract_feature_last_diff(s, feature="GHOST", feature_order=["close", "RSI_14", "volume"])
    except KeyError:
        return
    raise AssertionError("ميزة غير موجودة يجب أن تُرفَض")


def t_momentum_predict_fn_end_to_end_no_signal():
    s = _fake_split(n=500, seed=7)
    preds = momentum_predict_fn(None, None, s, feature_order=["close", "RSI_14", "volume"])
    assert len(preds) == 500 and np.isfinite(preds).all()


def t_evaluate_hypothesis_over_rolling_windows_detects_injected_signal():
    """محاكاة كاملة لبوّابة المرحلة ٠: زخم مُحقَن فعلياً في بيانات تركيبية
    تحاكي شكل مخرجات rolling_splits — يجب أن يُكتشَف بثبات عبر النوافذ."""
    rng = np.random.default_rng(8)
    windows = []
    for w in range(4):
        n = 600
        X = rng.normal(0, 1, (n, 32, 3)).astype("float32")
        momentum = X[:, -1, 0] - X[:, -2, 0]              # "آخر تغيّر" في close
        y = 0.04 * momentum + rng.normal(0, 1, n).astype("float32")   # استمرار ضعيف حقيقي
        test = {"X_1D": X, "y": {"y_close_reg": y}, "feature_order": ["close", "RSI_14", "volume"],
               "base_params": np.zeros((n, 2), "float32"), "last_candles": np.zeros((n, 7))}
        windows.append(({}, {}, test))
    res = evaluate_hypothesis_over_rolling_windows(
        windows, momentum_predict_fn, n_shuffles=200, seed=3, verbose=False)
    assert res["n_ok"] == 4
    assert res["mean_ic"] > 0.01
    assert res["consistent_sign"] is True


def t_evaluate_hypothesis_over_rolling_windows_no_signal_case():
    rng = np.random.default_rng(9)
    windows = []
    for w in range(4):
        n = 600
        X = rng.normal(0, 1, (n, 32, 3)).astype("float32")
        y = rng.normal(0, 1, n).astype("float32")          # عائد مستقلّ تماماً عن أي ميزة
        test = {"X_1D": X, "y": {"y_close_reg": y}, "feature_order": ["close", "RSI_14", "volume"],
               "base_params": np.zeros((n, 2), "float32"), "last_candles": np.zeros((n, 7))}
        windows.append(({}, {}, test))
    res = evaluate_hypothesis_over_rolling_windows(
        windows, momentum_predict_fn, n_shuffles=200, seed=4, verbose=False)
    assert res["frac_significant"] <= 0.5
    assert abs(res["mean_ic"]) < 0.1


def t_evaluate_hypothesis_handles_asset_separated_test():
    from functools import partial
    s1 = _fake_split(n=100, seed=11)
    s2 = _fake_split(n=80, seed=12)
    windows = [({}, {}, {"AAAUSDT": s1, "BBBUSDT": s2})]
    predict_fn = partial(momentum_predict_fn, feature_order=["close", "RSI_14", "volume"])
    res = evaluate_hypothesis_over_rolling_windows(
        windows, predict_fn, n_shuffles=50, min_samples=10, verbose=False)
    assert res["n_ok"] == 1


def t_evaluate_hypothesis_raises_on_length_mismatch():
    s = _fake_split(n=50, seed=13)
    bad_predict = lambda train, val, test: np.zeros(10)     # طول خاطئ عمداً
    try:
        evaluate_hypothesis_over_rolling_windows([({}, {}, s)], bad_predict, verbose=False)
    except ValueError as e:
        assert "طول" in str(e)
        return
    raise AssertionError("عدم تطابق الطول يجب أن يُرفَض بوضوح")


ALL_TESTS = [
    t_ic_recovers_known_correlation, t_ic_near_zero_on_independent_random_data,
    t_ic_ignores_nan_not_zeros_them, t_ic_raises_below_min_samples,
    t_ic_spearman_robust_to_monotonic_nonlinear_transform, t_ic_zero_when_one_array_constant,
    t_ic_rejects_mismatched_shapes, t_ic_rejects_unknown_method,
    t_decile_detects_monotonic_relationship, t_decile_near_zero_spread_on_random_data,
    t_decile_raises_on_insufficient_samples, t_decile_t_stat_large_for_strong_signal_small_for_none,
    t_permutation_flags_real_signal_as_significant, t_permutation_null_on_pure_random_data,
    t_permutation_reproducible_with_seed,
    t_evaluate_windows_aggregates_consistent_signal, t_evaluate_windows_skips_failed_window_not_whole_run,
    t_evaluate_windows_no_signal_gives_low_frac_significant,
    t_concat_splits_passthrough_on_flat_dict, t_concat_splits_merges_dict_of_assets,
    t_concat_splits_not_fooled_by_asset_names_starting_with_x,
    t_concat_splits_raises_clear_error_when_asset_split_malformed,
    t_concat_splits_nested_y_key_not_mistaken_for_asset_dict,
    t_extract_actuals_reads_target_key, t_extract_actuals_raises_on_missing_key,
    t_extract_actuals_matches_with_or_without_y_prefix,
    t_extract_actuals_raises_clear_error_without_y_key,
    t_extract_feature_last_diff_matches_manual, t_extract_feature_last_diff_raises_on_unknown_feature,
    t_extract_feature_last_diff_requires_explicit_feature_order,
    t_momentum_predict_fn_requires_explicit_feature_order,
    t_momentum_predict_fn_end_to_end_no_signal,
    t_evaluate_hypothesis_over_rolling_windows_detects_injected_signal,
    t_evaluate_hypothesis_over_rolling_windows_no_signal_case,
    t_evaluate_hypothesis_handles_asset_separated_test,
    t_evaluate_hypothesis_raises_on_length_mismatch,
]
