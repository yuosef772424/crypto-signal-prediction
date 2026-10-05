"""
PURPOSE:  Legacy uncertainty report (comprehensive_uncertainty_analysis) updated to use save_or_print.
TAGS:     comprehensive_uncertainty_analysis, legacy report, uncertainty, float16
PITFALLS: Executed into the one shared evaluation namespace by evaluation/_loader.py (never imported on its own): names from other modules resolve at call time.

## 1️⃣7️⃣ 📚 التقارير القديمة (محفوظة ومحدَّثة لاستخدام save_or_print)

`comprehensive_uncertainty_analysis` من الدفتر الأصلي — نفس المنطق تماماً، فقط الجداول الكبيرة (تحليل الاتجاه، أداء كل عملة، عتبات الثقة، إحصائيات المعايرة) تُحفظ الآن في ملف بدل طباعتها كاملة.
"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import confusion_matrix, classification_report

def comprehensive_uncertainty_analysis(per_asset_results):
    """
    ✅ تحليل شامل مع إصلاح مشكلة float16 وإضافة تحليل مفصل لاتجاه الإغلاق
    """
    print("="*100)
    print("🔬 تحليل شامل: الثقة × عدم اليقين × الأداء × اتجاه الإغلاق")
    print("="*100)

    all_data = []

    # ═══════════════════════════════════════════════════════════════════════
    # 1️⃣ جمع البيانات
    # ═══════════════════════════════════════════════════════════════════════

    for idx, row in per_asset_results.iterrows():
        asset = row['asset']

        for target in ['high', 'low', 'close']:
            data = row.get(target)
            if not isinstance(data, dict) or 'true_real' not in data:
                continue   # هدف غير مطلوب/غير متاح لهذا الأصل

            # تحويل لـ float32 فوراً
            pred = np.array(data['pred_real'], dtype=np.float32)
            true = np.array(data['true_real'], dtype=np.float32)
            entry = np.array(data['entry_price'], dtype=np.float32)
            uncertainty = np.array(data['uncertainty_real'], dtype=np.float32)
            confidence = np.array(data['confidence'], dtype=np.float32)

            # Aleatoric & Epistemic
            aleatoric = np.array(data.get('aleatoric_real', [0] * len(pred)), dtype=np.float32)
            epistemic = np.array(data.get('epistemic_real', [0] * len(pred)), dtype=np.float32)

            # حساب الأخطاء
            abs_error = np.abs(pred - true).astype(np.float32)
            rel_error = (abs_error / (np.abs(true) + 1e-10)).astype(np.float32)
            pct_error = (rel_error * 100).astype(np.float32)

            # الاتجاه - مقارنة مع سعر الدخول
            pred_direction = np.sign(pred - entry)
            true_direction = np.sign(true - entry)
            direction_correct = (pred_direction == true_direction).astype(np.int32)

            # نوع الخطأ في الاتجاه
            direction_error_type = np.where(
                (pred_direction == 1) & (true_direction == -1), 'صعود خاطئ',
                np.where(
                    (pred_direction == -1) & (true_direction == 1), 'هبوط خاطئ',
                    'صحيح'
                )
            )

            # تجميع
            for i in range(len(pred)):
                all_data.append({
                    'asset': asset,
                    'target': target,
                    'pred': float(pred[i]),
                    'true': float(true[i]),
                    'entry': float(entry[i]),
                    'abs_error': float(abs_error[i]),
                    'rel_error': float(rel_error[i]),
                    'pct_error': float(pct_error[i]),
                    'uncertainty': float(uncertainty[i]),
                    'confidence': float(confidence[i]),
                    'aleatoric': float(aleatoric[i]) if len(aleatoric) > 0 else 0.0,
                    'epistemic': float(epistemic[i]) if len(epistemic) > 0 else 0.0,
                    'pred_direction': 'صعود' if pred_direction[i] > 0 else 'هبوط' if pred_direction[i] < 0 else 'ثابت',
                    'true_direction': 'صعود' if true_direction[i] > 0 else 'هبوط' if true_direction[i] < 0 else 'ثابت',
                    'direction_correct': int(direction_correct[i]),
                    'direction_error_type': direction_error_type[i],
                    'price_change': float(true[i] - entry[i]),
                    'predicted_change': float(pred[i] - entry[i])
                })

    df = pd.DataFrame(all_data)

    # ✅ تأكيد أن كل الأعمدة float64 (ليس float16)
    numeric_cols = ['pred', 'true', 'entry', 'abs_error', 'rel_error', 'pct_error',
                    'uncertainty', 'confidence', 'aleatoric', 'epistemic',
                    'price_change', 'predicted_change']
    for col in numeric_cols:
        df[col] = df[col].astype(np.float64)

    # ═══════════════════════════════════════════════════════════════════════
    # تحليل مفصل لاتجاه الإغلاق (CLOSE فقط)
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "="*100)
    print("🎯 تحليل مفصل لاتجاه الإغلاق (CLOSE)")
    print("="*100)

    # فلترة البيانات للهدف close فقط
    df_close = df[df['target'] == 'close'].copy()

    if len(df_close) > 0:
        print(f"\n📊 إجمالي عدد توقعات الإغلاق: {len(df_close)}")

        # 1. تحليل حسب نوع التوقع
        print("\n📈 1. تحليل أداء التوقعات حسب الاتجاه المتوقع:")
        print("-"*60)

        direction_analysis = []
        for pred_dir in ['صعود', 'هبوط']:
            df_dir = df_close[df_close['pred_direction'] == pred_dir]
            if len(df_dir) > 0:
                correct = df_dir['direction_correct'].sum()
                total = len(df_dir)
                accuracy = (correct / total * 100) if total > 0 else 0

                # متوسط التغير الفعلي
                avg_actual_change = df_dir['price_change'].mean()
                avg_pred_change = df_dir['predicted_change'].mean()

                direction_analysis.append({
                    'الاتجاه المتوقع': pred_dir,
                    'عدد التوقعات': total,
                    'توقعات صحيحة': correct,
                    'دقة التوقع %': f"{accuracy:.2f}%",
                    'متوسط التغير المتوقع': f"{avg_pred_change:.4f}",
                    'متوسط التغير الفعلي': f"{avg_actual_change:.4f}",
                    'الفرق': f"{avg_pred_change - avg_actual_change:.4f}"
                })

        dir_df = pd.DataFrame(direction_analysis)
        save_or_print(dir_df, 'legacy_close_direction_analysis')

        # 2. مصفوفة الارتباك (Confusion Matrix)
        print("\n📊 2. مصفوفة الارتباك (Confusion Matrix):")
        print("-"*60)

        # إنشاء مصفوفة الارتباك
        confusion_data = []
        for pred_dir in ['صعود', 'هبوط']:
            for true_dir in ['صعود', 'هبوط']:
                count = len(df_close[(df_close['pred_direction'] == pred_dir) &
                                    (df_close['true_direction'] == true_dir)])
                confusion_data.append({
                    'المتوقع': pred_dir,
                    'الفعلي': true_dir,
                    'العدد': count,
                    'النسبة %': f"{(count/len(df_close)*100):.1f}%"
                })

        confusion_df = pd.DataFrame(confusion_data)

        # عرض بصيغة جدول
        pivot_table = confusion_df.pivot_table(
            values='العدد',
            index='المتوقع',
            columns='الفعلي',
            aggfunc='first'
        ).fillna(0)

        print("مصفوفة الارتباك:")
        print(pivot_table)

        # 3. تحليل حسب العملة
        print("\n📊 3. أداء توقعات الإغلاق حسب العملة:")
        print("-"*60)

        asset_performance = []
        for asset in df_close['asset'].unique():
            df_asset = df_close[df_close['asset'] == asset]

            for pred_dir in ['صعود', 'هبوط']:
                df_asset_dir = df_asset[df_asset['pred_direction'] == pred_dir]
                if len(df_asset_dir) > 0:
                    correct = df_asset_dir['direction_correct'].sum()
                    total = len(df_asset_dir)
                    accuracy = (correct / total * 100) if total > 0 else 0

                    asset_performance.append({
                        'العملة': asset,
                        'الاتجاه': pred_dir,
                        'عدد': total,
                        'صحيح': correct,
                        'دقة %': f"{accuracy:.2f}%",
                        'متوسط تغير فعلي': f"{df_asset_dir['price_change'].mean():.4f}"
                    })

        perf_df = pd.DataFrame(asset_performance)
        if len(perf_df) > 0:
            save_or_print(perf_df, 'legacy_asset_close_performance')

        # 4. تحليل التوقعات الخاطئة
        print("\n📊 4. تحليل التوقعات الخاطئة:")
        print("-"*60)

        wrong_predictions = df_close[df_close['direction_correct'] == 0]
        if len(wrong_predictions) > 0:
            error_analysis = wrong_predictions.groupby('direction_error_type').agg({
                'asset': 'count',
                'confidence': 'mean',
                'uncertainty': 'mean',
                'abs_error': 'mean',
                'price_change': ['min', 'max', 'mean']
            }).round(4)

            error_analysis.columns = ['عدد', 'متوسط ثقة', 'متوسط عدم يقين',
                                     'متوسط خطأ', 'أقل تغير', 'أعلى تغير', 'متوسط تغير']
            print(error_analysis)

            print(f"\n📌 ملاحظات على الأخطاء:")
            print(f"   • إجمالي التوقعات الخاطئة: {len(wrong_predictions)}")
            print(f"   • نسبة الأخطاء: {(len(wrong_predictions)/len(df_close)*100):.1f}%")

            # تحليل الثقة في التوقعات الخاطئة vs الصحيحة
            correct_conf = df_close[df_close['direction_correct'] == 1]['confidence'].mean()
            wrong_conf = df_close[df_close['direction_correct'] == 0]['confidence'].mean()
            print(f"   • متوسط الثقة في التوقعات الصحيحة: {correct_conf:.3f}")
            print(f"   • متوسط الثقة في التوقعات الخاطئة: {wrong_conf:.3f}")
            print(f"   • الفرق في الثقة: {correct_conf - wrong_conf:.3f}")

        # 5. تحسين الأداء
        print("\n📊 5. اقتراحات لتحسين الأداء:")
        print("-"*60)

        # حساب عتبة الثقة المثلى
        thresholds = [ 0.0,0.1,0.2,0.3,0.4,0.5, 0.6, 0.7, 0.8, 0.9]
        improvement_data = []

        for threshold in thresholds:
            high_conf = df_close[(df_close['confidence'] >= threshold)&(df_close['confidence'] <= (threshold+0.1))]
            if len(high_conf) > 0:
                high_conf_accuracy = high_conf['direction_correct'].mean() * 100
                coverage = len(high_conf) / len(df_close) * 100
                improvement_data.append({
                    'عتبة الثقة': threshold,
                    'دقة عالية الثقة %': f"{high_conf_accuracy:.1f}%",
                    'تغطية %': f"{coverage:.1f}%",
                    'عدد': len(high_conf)
                })

        if improvement_data:
            improve_df = pd.DataFrame(improvement_data)
            print("أداء التوقعات عالية الثقة:")
            save_or_print(improve_df, 'legacy_confidence_threshold_performance')

            # اقتراح أفضل عتبة
            best_threshold = max(improvement_data,
                               key=lambda x: float(x['دقة عالية الثقة %'].replace('%', '')))
            print(f"\n✅ أفضل عتبة: {best_threshold['عتبة الثقة']} → "
                  f"دقة: {best_threshold['دقة عالية الثقة %']} "
                  f"(تغطية: {best_threshold['تغطية %']})")
            print("   ⚠️ العتبة مختارة على نفس البيانات التي تقيسها — للحكم الصادق اخترها على val وقِسها على test "
                  "(selective_direction_report في main، القسم ٧-ب)")

    else:
        print("⚠️ لا توجد بيانات للإغلاق (close) للتحليل")

    # ═══════════════════════════════════════════════════════════════════════
    # 2️⃣ Confidence-Error Correlation (نفس الكود السابق)
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "─"*100)
    print("📈 6. Confidence-Error Correlation (CEC)")
    print("─"*100)

    corr_conf_abs, p_abs = pearsonr(df['confidence'], df['abs_error'])
    corr_conf_rel, p_rel = spearmanr(df['confidence'], df['rel_error'])

    print(f"  📊 الارتباط الكلي:")
    print(f"     Confidence ↔ Absolute Error: {corr_conf_abs:+.4f} (p={p_abs:.4e})  ← بوحدة السعر، لا يُحكم به")
    print(f"     Confidence ↔ Relative Error: {corr_conf_rel:+.4f} (p={p_rel:.4e})  ← Spearman، الحكم عليه")

    # الحكم على الخطأ النسبي: الخطأ المطلق بوحدة السعر يهيمن عليه مستوى سعر العملة لا جودة التوقع
    if corr_conf_rel < -0.3:
        print(f"     ✅ ممتاز: الثقة العالية = خطأ منخفض")
    elif corr_conf_rel < 0:
        print(f"     ⚠️  ضعيف: ارتباط سلبي لكن ضعيف")
    else:
        print(f"     ❌ مشكلة: الثقة لا ترتبط بالدقة!")

    # ═══════════════════════════════════════════════════════════════════════
    # 3️⃣ Uncertainty-Error Correlation (نفس الكود السابق)
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "─"*100)
    print("📈 7. Uncertainty-Error Correlation")
    print("─"*100)

    corr_unc_abs, p_abs = pearsonr(df['uncertainty'], df['abs_error'])
    unc_rel = df['uncertainty'] / (np.abs(df['entry']) + 1e-10)      # عدم اليقين نسبةً للسعر، كالخطأ النسبي
    corr_unc_rel, p_rel = spearmanr(unc_rel, df['rel_error'])

    print(f"  📊 الارتباط الكلي:")
    print(f"     Uncertainty ↔ Absolute Error: {corr_unc_abs:+.4f} (p={p_abs:.4e})  ← كلاهما بوحدة السعر: مرتفع لأن "
          f"العملات الغالية أخطاؤها وعدم يقينها بالدولار أكبر، لا يُحكم به")
    print(f"     Uncertainty/السعر ↔ Relative Error: {corr_unc_rel:+.4f} (p={p_rel:.4e})  ← Spearman، الحكم عليه")

    if corr_unc_rel > 0.5:
        print(f"     ✅ ممتاز: عدم اليقين يعكس الخطأ بدقة")
    elif corr_unc_rel > 0.2:
        print(f"     ⚠️  متوسط: ارتباط إيجابي لكن ضعيف")
    else:
        print(f"     ❌ مشكلة: عدم اليقين لا يعكس الخطأ!")

    # ═══════════════════════════════════════════════════════════════════════
    # 4️⃣ Calibration Analysis (نفس الكود السابق)
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "─"*100)
    print("📈 8. Calibration Quality (Binned Analysis)")
    print("─"*100)

    try:
        df['conf_bin'] = pd.qcut(
            df['confidence'].astype(np.float64),
            q=5,
            labels=['Very_Low', 'Low', 'Med', 'High', 'Very_High'],
            duplicates='drop'
        )

        calibration_stats = df.groupby('conf_bin', observed=True).agg({
            'abs_error': ['mean', 'std', 'median'],
            'pct_error': ['mean', 'std'],
            'direction_correct': 'mean',
            'confidence': ['mean', 'count']
        }).round(4)

        save_or_print(calibration_stats.reset_index(), 'legacy_calibration_stats')

        ece = 0
        for bin_name in df['conf_bin'].unique():
            if pd.isna(bin_name):
                continue
            df_bin = df[df['conf_bin'] == bin_name]
            avg_conf = float(df_bin['confidence'].mean())
            avg_acc = float(df_bin['direction_correct'].mean())
            weight = len(df_bin) / len(df)
            ece += weight * abs(avg_conf - avg_acc)

        print(f"\n  📊 Expected Calibration Error (ECE): {ece:.4f}")
        if ece < 0.05:
            print(f"     ✅ ECE < 0.05: ممتاز")
        elif ece < 0.15:
            print(f"     ⚠️  0.05 < ECE < 0.15: مقبول")
        else:
            print(f"     ❌ ECE > 0.15: سيء - النموذج غير مُعاير")

    except Exception as e:
        print(f"  ⚠️ تعذر إنشاء bins: {str(e)}")
        ece = None

    # ═══════════════════════════════════════════════════════════════════════
    # 9️⃣ Summary
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "="*100)
    print("📋 الملخص والتوصيات النهائية")
    print("="*100)

    recommendations = []

    # تحليل اتجاه الإغلاق
    if len(df_close) > 0:
        close_accuracy = df_close['direction_correct'].mean() * 100
        recommendations.append(f"📊 دقة توقعات الإغلاق: {close_accuracy:.1f}%")

        if close_accuracy > 60:
            recommendations.append("✅ أداء توقعات الإغلاق ممتاز")
        elif close_accuracy > 55:
            recommendations.append("⚠️  أداء توقعات الإغلاق مقبول")
        else:
            recommendations.append("❌ أداء توقعات الإغلاق ضعيف - يحتاج تحسين")

    # Confidence-Error
    if corr_conf_abs > -0.2:
        recommendations.append("❌ الثقة لا ترتبط بالدقة → أعد معايرة النموذج")
    elif corr_conf_abs > -0.4:
        recommendations.append("⚠️  الارتباط ضعيف → استخدم Calibration Loss")
    else:
        recommendations.append("✅ الثقة تعكس الدقة بشكل جيد")

    # Uncertainty-Error
    if corr_unc_abs < 0.3:
        recommendations.append("❌ عدم اليقين لا يعكس الخطأ → راجع lambda_reg")
    elif corr_unc_abs < 0.5:
        recommendations.append("⚠️  عدم اليقين متوسط → زد penalty_weight")
    else:
        recommendations.append("✅ عدم اليقين مُعاير ممتاز")

    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")

    # ═══════════════════════════════════════════════════════════════════════
    # Return
    # ═══════════════════════════════════════════════════════════════════════

    return {
        'full_data': df,
        'close_data': df_close if 'df_close' in locals() else pd.DataFrame(),
        'correlations': {
            'conf_abs_error': corr_conf_abs,
            'conf_rel_error': corr_conf_rel,
            'unc_abs_error': corr_unc_abs,
            'unc_rel_error': corr_unc_rel
        },
        'ece': ece,
        'close_direction_analysis': dir_df if 'dir_df' in locals() else pd.DataFrame(),
        'confusion_matrix': pivot_table if 'pivot_table' in locals() else pd.DataFrame()
    }


# # ══════════════════════════════════════════════════════════════════════════════
# # 🚀 الاستخدام
# # ══════════════════════════════════════════════════════════════════════════════

# per_asset_df = results2['per_asset_results']
# analysis = comprehensive_uncertainty_analysis(per_asset_df)
