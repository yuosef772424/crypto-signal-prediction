"""
PURPOSE:  build_flat_dataframe: flattens test_all_assets_v4 results into one DataFrame (the input of the trust/calibration/pattern reports).
TAGS:     build_flat_dataframe, flat dataframe, results flattening
PITFALLS: Executed into the one shared evaluation namespace by evaluation/_loader.py (never imported on its own): names from other modules resolve at call time.

## 9️⃣ دالة مساعدة مشتركة: تسطيح النتائج إلى DataFrame واحد
"""
# ═══════════════════════════════════════════════════════════════════════════
# 🧱 دالة مساعدة مشتركة: تحويل نتائج كل الأصول إلى DataFrame مسطّح واحد
# ═══════════════════════════════════════════════════════════════════════════

def build_flat_dataframe(
    per_asset_results: pd.DataFrame,
    target_specs: List['TargetSpec'],
    continuous_only: bool = False,
) -> pd.DataFrame:
    """
    يحوّل نتائج test_all_assets_v4 (DataFrame فيه عمود لكل هدف يحوي dict)
    إلى DataFrame واحد "مسطّح" بصف لكل (عملة × هدف × عينة). تستخدمه كل
    دوال التحليل والتقارير أدناه بدل تكرار كود التجميع في كل مرة.

    الأعمدة المشتركة: asset, target, kind, confidence, uncertainty
    الأعمدة الإضافية للمستمر: pred, true, entry, abs_error, pct_error,
                              direction_correct, predicted_change, price_change,
                              p_up (احتمال صعود رأس التصنيف إن قُرئ وإلا NaN)
    الأعمدة الإضافية للفئوي: pred_label, true_label, correct, pred_proba
    """
    target_specs = resolve_targets(target_specs)
    rows = []

    for _, row in per_asset_results.iterrows():
        asset = row['asset']

        for spec in target_specs:
            if spec.name not in row or not isinstance(row[spec.name], dict):
                continue
            data = row[spec.name]
            if continuous_only and spec.kind != 'continuous':
                continue

            n = len(data.get('pred_real', data.get('pred_index', [])))
            confidence = np.asarray(data.get('confidence', [np.nan] * n), dtype=np.float64)
            uncertainty = np.asarray(data.get('uncertainty_real', [np.nan] * n), dtype=np.float64)
            aleatoric = np.asarray(data.get('aleatoric', [np.nan] * n), dtype=np.float64)
            epistemic = np.asarray(data.get('epistemic', [np.nan] * n), dtype=np.float64)
            unc_clipped = np.asarray(data.get('unc_clipped', np.zeros(n, dtype=bool)), dtype=bool)

            if spec.kind == 'continuous' and 'true_real' in data:
                pred = np.asarray(data['pred_real'], dtype=np.float64)
                true = np.asarray(data['true_real'], dtype=np.float64)
                entry = np.asarray(data['entry_price'], dtype=np.float64)
                abs_error = np.asarray(data['error_abs'], dtype=np.float64)
                pct_error = np.asarray(data['error_pct'], dtype=np.float64)
                direction_correct = np.asarray(data['direction_correct'], dtype=np.int32)
                predicted_change = pred - entry
                price_change = true - entry
                has_truth = np.asarray(data.get('has_truth', np.isfinite(true)), dtype=bool)
                ts_arr = np.asarray(data.get('timestamp', np.full(n, np.nan)), dtype=np.float64)
                p_up = np.asarray(data.get('p_up', np.full(n, np.nan)), dtype=np.float64)

                for i in range(n):
                    if not has_truth[i]:
                        continue    # عينة بلا هدف فعلي (مثل آخر شمعة في التداول الحي) — لا تدخل التحليل
                    rows.append({
                        'asset': asset, 'target': spec.name, 'kind': 'continuous',
                        'pred': pred[i], 'true': true[i], 'entry': entry[i],
                        'abs_error': abs_error[i], 'pct_error': pct_error[i],
                        'confidence': confidence[i] if i < len(confidence) else np.nan,
                        'uncertainty': uncertainty[i] if i < len(uncertainty) else np.nan,
                        'aleatoric': aleatoric[i] if i < len(aleatoric) else np.nan,
                        'epistemic': epistemic[i] if i < len(epistemic) else np.nan,
                        'unc_clipped': bool(unc_clipped[i]) if i < len(unc_clipped) else False,
                        'direction_correct': int(direction_correct[i]),
                        'correct': int(direction_correct[i]),
                        'predicted_change': predicted_change[i],
                        'price_change': price_change[i],
                        'timestamp': ts_arr[i] if i < len(ts_arr) else np.nan,
                        'p_up': p_up[i],
                        'predicted_change_pct': (predicted_change[i] / (abs(entry[i]) + 1e-7)) * 100,
                    })

            elif spec.kind == 'categorical' and 'true_label' in data:
                pred_label = data['pred_label']
                true_label = data['true_label']
                correct = data['correct']
                pred_proba = np.asarray(data.get('pred_proba', [np.nan] * n), dtype=np.float64)

                for i in range(n):
                    rows.append({
                        'asset': asset, 'target': spec.name, 'kind': 'categorical',
                        'pred_label': pred_label[i], 'true_label': true_label[i],
                        'confidence': confidence[i] if i < len(confidence) else pred_proba[i],
                        'uncertainty': uncertainty[i] if i < len(uncertainty) else np.nan,
                        'aleatoric': aleatoric[i] if i < len(aleatoric) else np.nan,
                        'epistemic': epistemic[i] if i < len(epistemic) else np.nan,
                        'correct': int(correct[i]),
                        'direction_correct': int(correct[i]),
                        'pred_proba': pred_proba[i],
                    })

    df = pd.DataFrame(rows)

    # عدم اليقين أعلاه بوحدة سعر الأصل (× |entry / reg_scale|) فلا يُقارَن عبر العملات (BTC بالدولارات مقابل عملة بالسنتات) ويصنع
    # «عنقوداً» من عملة واحدة في K-means. النسخ *_ret نسبةً لسعر الدخول (بلا وحدة) — هي ما يدخل اكتشاف الأنماط.
    if not df.empty and 'entry' in df.columns:
        _den = df['entry'].abs() + 1e-7
        for _c in ('uncertainty', 'aleatoric', 'epistemic'):
            if _c in df.columns:
                df[f'{_c}_ret'] = df[_c] / _den

    # ✅ ضمان float64 (يمنع مشاكل float16 التي واجهها الكود الأصلي)
    numeric_cols = [c for c in df.columns if df[c].dtype.kind in 'fi' and c not in ('correct', 'direction_correct')]
    for c in numeric_cols:
        df[c] = df[c].astype(np.float64)

    return df
