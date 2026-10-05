"""
PURPOSE:  Model integrity diagnostics: naive baseline, lag scan (late copying), momentum baseline and directional bias.
TAGS:     run_integrity_diagnostics, cross_correlation_lag_scan, momentum_direction_accuracy, directional_bias, naive baseline, lag scan
PITFALLS: Executed into the one shared evaluation namespace by evaluation/_loader.py (never imported on its own): names from other modules resolve at call time.

## 1️⃣5️⃣ 🩺 تشخيص نزاهة النموذج (Naive Baseline × Lag Scan × الانحياز)

مُدمَج من سكربتي التشخيص المرفقين: هل النموذج "يتنبأ" فعلاً أم يستفيد فقط من الترابط الذاتي للأسعار المتتالية (persistence)؟ لتفادي إغراق المخرجات، تُرسم المنحنيات فقط لأسوأ/أفضل التركيبات (الجدول الكامل محفوظ دوماً في ملف).
"""
# ═══════════════════════════════════════════════════════════════════════════
# 🩺 تشخيص نزاهة النموذج: Naive Baseline × Lag Scan × الانحياز الموجّه
# ═══════════════════════════════════════════════════════════════════════════
#
# مُدمَج من سكربتي التشخيص المرفقين، ومُكيَّف للعمل مباشرة فوق نتائج
# test_all_assets_v4 (بدل إعادة بناء test_dict يدوياً). يجيب على: هل النموذج
# "يتنبأ" فعلاً أم يعيد إنتاج آخر سعر معروف (persistence) مستفيداً من
# الترابط الذاتي الطبيعي في الأسعار المتتالية؟
# ═══════════════════════════════════════════════════════════════════════════

def compute_point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_true) > 1e-9 and np.std(y_pred) > 1e-9 else 0.0
    return {"mae": mae, "rmse": rmse, "corr": corr}


def cross_correlation_lag_scan(y_actual: np.ndarray, y_pred: np.ndarray, max_lag: int = 5):
    """عند أي إزاحة زمنية يكون الارتباط بين actual و predicted أعلى؟ (0=صحي, ≠0=نسخ متأخر مشبوه)"""
    lags = list(range(-max_lag, max_lag + 1))
    correlations = []
    for lag in lags:
        if lag < 0:
            a, p = y_actual[:lag], y_pred[-lag:]
        elif lag > 0:
            a, p = y_actual[lag:], y_pred[:-lag]
        else:
            a, p = y_actual, y_pred
        correlations.append(float(np.corrcoef(a, p)[0, 1]) if len(a) > 1 and np.std(a) > 1e-9 and np.std(p) > 1e-9 else np.nan)
    valid = [c if not np.isnan(c) else -np.inf for c in correlations]
    best_lag = lags[int(np.argmax(valid))]
    return lags, correlations, best_lag


def momentum_direction_accuracy(y_actual: np.ndarray, entry_price: np.ndarray) -> float:
    """
    مرجع اتجاهي عادل (momentum): يتنبأ بأن الحركة القادمة تكرر آخر حركة فعلية.
    أعدل من persistence baseline الذي يعطي 0% اتجاهية بشكل مصطنع رياضياً
    (لأن حركته المتوقعة = صفر دائماً).
    """
    actual_dir = np.sign(y_actual - entry_price)
    if len(actual_dir) < 2:
        return float("nan")
    momentum_pred_dir = actual_dir[:-1]
    actual_dir_current = actual_dir[1:]
    mask = actual_dir_current != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(momentum_pred_dir[mask] == actual_dir_current[mask]))


def directional_bias(y_actual: np.ndarray, y_pred: np.ndarray) -> Dict:
    """هل أخطاء النموذج منحازة بثبات في اتجاه واحد (متفائل/متشائم دائماً) أم عشوائية؟"""
    errors = y_pred - y_actual
    bias_mean = float(np.mean(errors))
    bias_pct = float(np.mean(errors / (np.abs(y_actual) + 1e-9)) * 100)
    label = "متفائل (يبالغ بالارتفاع)" if bias_mean > 0 else "متشائم (يبالغ بالانخفاض)" if bias_mean < 0 else "متوازن"
    ref_sign = np.sign(bias_mean) if bias_mean != 0 else 0
    same_sign_pct = float(np.mean(np.sign(errors) == ref_sign)) * 100 if ref_sign != 0 else float("nan")
    return {"bias_mean": bias_mean, "bias_pct": bias_pct, "same_sign_pct": same_sign_pct, "direction_label": label}


def run_integrity_diagnostics(
    per_asset_results: pd.DataFrame,
    target_specs: List['TargetSpec'],
    max_lag: int = 5,
    plot_worst_n: int = 2,
    plot_best_n: int = 1,
    out_dir: str = DEFAULT_OUTPUT_DIR,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    يشغّل تشخيص Naive-Baseline/Lag/Bias على كل (عملة × هدف مستمر)، ويُرجع
    جدول ملخص واحد. لتفادي إغراق المخرجات، تُرسم المنحنيات (actual vs
    predicted + lag correlation) فقط لأسوأ `plot_worst_n` وأفضل `plot_best_n`
    تركيبة (كل الرسوم تُحفظ في analysis_outputs/figures/ مع ذلك).
    """
    target_specs = resolve_targets(target_specs)
    rows = []

    for _, row in per_asset_results.iterrows():
        asset = row['asset']
        for spec in target_specs:
            if spec.kind != 'continuous' or spec.name not in row or not isinstance(row[spec.name], dict):
                continue
            d = row[spec.name]
            if 'true_real' not in d:
                continue

            y_actual = np.asarray(d['true_real'], dtype=np.float64)
            y_pred = np.asarray(d['pred_real'], dtype=np.float64)
            entry = np.asarray(d['entry_price'], dtype=np.float64)
            n = len(y_actual)
            if n < 5:
                continue

            y_naive = entry  # الـ naive baseline = "لا تغيير عن آخر سعر معروف"
            model_m = compute_point_metrics(y_actual, y_pred)
            naive_m = compute_point_metrics(y_actual, y_naive)
            mae_improve = (naive_m['mae'] - model_m['mae']) / (naive_m['mae'] + 1e-12) * 100

            dir_acc_model = float(np.mean(np.sign(y_pred - entry)[np.sign(y_actual - entry) != 0] ==
                                           np.sign(y_actual - entry)[np.sign(y_actual - entry) != 0])) \
                if np.any(np.sign(y_actual - entry) != 0) else float('nan')
            dir_acc_momentum = momentum_direction_accuracy(y_actual, entry)
            bias = directional_bias(y_actual, y_pred)

            safe_max_lag = max(1, min(max_lag, n // 3))
            lags, lag_corrs, best_lag = cross_correlation_lag_scan(y_actual, y_pred, safe_max_lag)

            rows.append({
                'asset': asset, 'target': spec.name, 'n': n,
                'model_mae': model_m['mae'], 'naive_mae': naive_m['mae'], 'mae_improvement_%': mae_improve,
                'model_corr': model_m['corr'], 'naive_corr': naive_m['corr'],
                'dir_acc_model_%': dir_acc_model * 100 if not np.isnan(dir_acc_model) else np.nan,
                'dir_acc_momentum_%': dir_acc_momentum * 100 if not np.isnan(dir_acc_momentum) else np.nan,
                'best_lag': best_lag, 'bias_pct': bias['bias_pct'],
                'bias_same_sign_%': bias['same_sign_pct'], 'bias_label': bias['direction_label'],
                '_y_actual': y_actual, '_y_pred': y_pred, '_y_naive': y_naive,
                '_lags': lags, '_lag_corrs': lag_corrs,
            })

    if not rows:
        if verbose:
            print("⚠️ لا توجد أهداف مستمرة كافية لتشخيص النزاهة")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    healthy = df[
        (df['best_lag'] == 0)
        & (df['mae_improvement_%'] > 10)
        & (df['dir_acc_model_%'] > np.maximum(55, df['dir_acc_momentum_%'] + 3))
    ]

    display_df = df.drop(columns=['_y_actual', '_y_pred', '_y_naive', '_lags', '_lag_corrs'])

    if verbose:
        print("=" * 100)
        print("🩺 تشخيص نزاهة النموذج (Naive Baseline × Lag Scan × الانحياز الموجّه)")
        print("=" * 100)
        save_or_print(display_df.round(3), 'integrity_diagnostics_summary', out_dir=out_dir)

        print(f"\n✅ عدد تركيبات (عملة/هدف) اجتازت كل الفحوصات: {len(healthy)} / {len(df)}")
        if len(healthy) < len(df):
            flagged = display_df[~display_df.index.isin(healthy.index)]
            print("⚠️ تركيبات تحتاج مراجعة (قد يكون النموذج يعتمد على الترابط الذاتي بدل تنبؤ حقيقي):")
            save_or_print(
                flagged[['asset', 'target', 'mae_improvement_%', 'dir_acc_model_%', 'dir_acc_momentum_%', 'best_lag']].round(2),
                'integrity_flagged', out_dir=out_dir,
            )

        # رسم أسوأ/أفضل التركيبات فقط لتفادي إغراق المخرجات
        ranked = df.sort_values('mae_improvement_%')
        # ⚠️ لا نستخدم drop_duplicates() هنا: الإطار يحوي أعمدة مصفوفات (_y_actual/_y_pred/...)
        #    وهي غير قابلة للتجزئة (TypeError: unhashable type: 'numpy.ndarray').
        #    نُزيل التكرار بالفهرس فقط (يحافظ على الترتيب: الأسوأ ثم الأفضل).
        _plot_idx = list(dict.fromkeys(list(ranked.head(plot_worst_n).index) + list(ranked.tail(plot_best_n).index)))
        to_plot = ranked.loc[_plot_idx]
        if len(to_plot) > 0:
            print(f"\n📊 رسوم بيانية لأسوأ {plot_worst_n} وأفضل {plot_best_n} تركيبة (الباقي محفوظ في الجدول أعلاه فقط):")
        for _, r in to_plot.iterrows():
            label = f"{r['asset']} — {r['target']}"
            plot_actual_vs_predicted(r['_y_actual'], r['_y_pred'], r['_y_naive'],
                                      title=f"{label} (MAE improve: {r['mae_improvement_%']:.1f}%)",
                                      save_dir=out_dir, filename=f"integrity_{r['asset']}_{r['target']}_actual_vs_pred")
            plot_lag_correlation(r['_lags'], r['_lag_corrs'], r['best_lag'],
                                  title=f"Lag Scan — {label}", save_dir=out_dir,
                                  filename=f"integrity_{r['asset']}_{r['target']}_lag_scan")

    return display_df
