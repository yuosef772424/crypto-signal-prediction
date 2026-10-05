"""
PURPOSE:  The lab's evaluation core: clean_reg_target (unified last_close reference for high/low targets) and evaluate_candidate (one candidate through the axis: IC + decile + random baseline over rolling windows).
TAGS:     clean_reg_target, evaluate_candidate, last_close reference, own-kind reference trap, candle shape spurious correlation, axis evaluation
PITFALLS: high/low targets must use clean_reg_target (last_close reference): y_high_reg/y_low_reg reference last_high/last_low and carry the last candle's shape (spurious Spearman about +0.51). Needs the axis functions (extract_actuals, concat_splits, evaluate_windows) and LAST_COLUMNS in the namespace. Executed into the notebook's shared namespace by discovery/_loader.py (never imported on its own): names from the axis/pipeline notebooks resolve at call time. Extracted verbatim from signal_discovery_lab.ipynb cell 6 (section 3).
"""
import numpy as np


def clean_reg_target(split, target):
    """`y_{target}_reg` بمرجع `last_close` موحّد لكل الأهداف — لا
    `last_high`/`last_low` الأصليين (own-kind reference) اللذين يحملان أثر
    شكل الشمعة الأخيرة (راجع تنبيه رقم ٢٠ في crypto_data_pipeline_v6). لهدف
    `close` يعادل `y_close_reg` تماماً (نفس المرجع أصلاً) فيُقرَأ منه مباشرة."""
    if target == "close":
        return extract_actuals(split, target_key="y_close_reg")
    split = concat_splits(split)
    lc = np.asarray(split["last_candles"])
    last_close = lc[:, LAST_COLUMNS.index("last_close")]
    future_col = {"high": "future_high_max", "low": "future_low_min"}[target]
    future = lc[:, LAST_COLUMNS.index(future_col)]
    return (future - last_close) / last_close


def evaluate_candidate(predict_fn, target, windows, n_shuffles=1000, min_samples=10, seed=42, verbose=False):
    """يقيّم مرشّحاً واحداً عبر كل النوافذ — بديل رقيق لـ
    evaluate_hypothesis_over_rolling_windows يستخدم دائماً clean_reg_target
    (لا target_key خام) فلا يُمكن نسيان الحارس بالخطأ."""
    names = [f"نافذة {i + 1}" for i in range(len(windows))]
    results = []
    for name, (train, val, test) in zip(names, windows):
        test_flat = concat_splits(test)
        preds = np.asarray(predict_fn(train, val, test), dtype="float64")
        actuals = clean_reg_target(test_flat, target)
        if len(preds) != len(actuals):
            raise ValueError(f"[{name}] طول التنبؤات ({len(preds)}) ≠ طول الأهداف ({len(actuals)}).")
        results.append((name, preds, actuals))
    return evaluate_windows(results, n_shuffles=n_shuffles, min_samples=min_samples, seed=seed, verbose=verbose)
