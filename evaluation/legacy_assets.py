"""
PURPOSE:  Legacy trading-simulation reports: comprehensive_asset_analysis, compare_multiple_models, generate_detailed_report.
TAGS:     comprehensive_asset_analysis, compare_multiple_models, generate_detailed_report, legacy report
PITFALLS: Prints a banner at load, as the old notebook did. Executed into the one shared evaluation namespace by evaluation/_loader.py (never imported on its own): names from other modules resolve at call time.

## 1️⃣8️⃣ 📚 تقارير محاكاة التداول القديمة (محفوظة ومحدَّثة)

`comprehensive_asset_analysis`, `generate_detailed_report`, `compare_multiple_models` — نفس المنطق، مع حفظ الجداول الكبيرة (ملخص كل العملات، مقارنة النماذج) في ملف بدل طباعتها كاملة.

> 💡 **توصية**: للتحليلات الجديدة استخدم `run_full_analysis` أعلاه — تدعم الفئوي، تحقق فك التشفير، إدارة المخرجات، والرسوم البيانية تلقائياً.
"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

def comprehensive_asset_analysis(per_asset_results, model_name="النموذج", market_neutral=None):
    """
    ✅ تحليل شامل لكل عملة على حدة مع التركيز على تداول الإغلاق

    market_neutral: يطرح من عائد كل صفقة متوسط عوائد كل العملات في نفس الطابع الزمني. None = تلقائي من
        CHICKS_MARKET_NEUTRAL (يضبطه main مع أهداف relative، حيث متوسط العائد النسبي موجب بالانحراف فيبدو
        «الشراء» رابحاً بلا مهارة).
    """
    if market_neutral is None:
        market_neutral = bool(globals().get("CHICKS_MARKET_NEUTRAL", False))
    market_mean = {}
    if market_neutral:
        sums, counts = {}, {}
        for _, r in per_asset_results.iterrows():
            d = r.get('close')
            if not isinstance(d, dict) or 'timestamp' not in d:
                market_neutral = False
                break
            ts_a, t_a, e_a = (np.asarray(d[k], dtype=np.float64) for k in ('timestamp', 'true_real', 'entry_price'))
            ret_a = np.where(e_a != 0, (t_a - e_a) / np.where(e_a != 0, e_a, 1.0) * 100, 0.0)
            for tsv, rv in zip(ts_a, ret_a):
                sums[tsv] = sums.get(tsv, 0.0) + rv
                counts[tsv] = counts.get(tsv, 0) + 1
        market_mean = {k: sums[k] / counts[k] for k in sums} if market_neutral else {}
    print("="*100)
    print(f"📊 تحليل شامل لكل عملة - {model_name} (الإغلاق فقط)"
          + (" — الربح محايد للسوق: عائد العملة ناقص متوسط السوق في نفس اليوم" if market_neutral else ""))
    print("="*100)

    all_assets_analysis = []
    detailed_results = []

    # لكل عملة في النتائج
    for idx, row in per_asset_results.iterrows():
        asset = row['asset']
        print(f"\n{'='*60}")
        print(f"🔍 تحليل مفصل لـ {asset}")
        print(f"{'='*60}")

        # تحليل بيانات الإغلاق فقط
        if 'close' in row:
            data = row['close']

            # تحويل البيانات
            pred = np.array(data['pred_real'], dtype=np.float32)
            true = np.array(data['true_real'], dtype=np.float32)
            entry = np.array(data['entry_price'], dtype=np.float32)
            confidence = np.array(data['confidence'], dtype=np.float32) if 'confidence' in data else np.ones_like(pred) * 0.5
            ts_arr = np.asarray(data['timestamp'], dtype=np.float64) if market_neutral else None

            # إحصائيات العملة
            asset_stats = {
                'العملة': asset,
                'عدد_التوقعات': len(pred),
                'التوقعات_الصحيحة': 0,
                'التوقعات_الخاطئة': 0,
                'صفقات_الشراء': 0,
                'صفقات_الشراء_الصحيحة': 0,
                'صفقات_البيع': 0,
                'صفقات_البيع_الصحيحة': 0,
                'إجمالي_الربح_%': 0,
                'أعلى_ربح_%': float('-inf'),
                'أعلى_خسارة_%': float('inf'),
                'متوسط_الربح_%': 0,
                'متوسط_الثقة': np.mean(confidence) if len(confidence) > 0 else 0
            }

            # تحليل كل توقع
            for i in range(len(pred)):
                # سعر الدخول: سعر إغلاق الشمعة السابقة
                # سعر الخروج: سعر إغلاق الشمعة الحالية

                # حساب الربح/الخسارة بالنسبة المئوية
                if entry[i] != 0:
                    actual_return_pct = ((true[i] - entry[i]) / entry[i]) * 100
                else:
                    actual_return_pct = 0
                if market_neutral:
                    actual_return_pct -= market_mean.get(ts_arr[i], 0.0)

                # تحديد اتجاه التوقع
                predicted_direction = 'شراء' if pred[i] > entry[i] else 'بيع' if pred[i] < entry[i] else 'حياد'
                actual_direction = 'صعود' if true[i] > entry[i] else 'هبوط' if true[i] < entry[i] else 'ثابت'

                # حساب الربح بناءً على التوقع
                if predicted_direction == 'شراء':
                    profit_pct = actual_return_pct
                    asset_stats['صفقات_الشراء'] += 1
                    if actual_direction == 'صعود':  # شراء صحيح
                        asset_stats['التوقعات_الصحيحة'] += 1
                        asset_stats['صفقات_الشراء_الصحيحة'] += 1
                        asset_stats['إجمالي_الربح_%'] += profit_pct
                    else:  # شراء خاطئ
                        asset_stats['التوقعات_الخاطئة'] += 1
                        asset_stats['إجمالي_الربح_%'] += profit_pct  # profit_pct سيكون سالباً

                elif predicted_direction == 'بيع':
                    profit_pct = -actual_return_pct  # الربح عند الهبوط الصحيح
                    asset_stats['صفقات_البيع'] += 1
                    if actual_direction == 'هبوط':  # بيع صحيح
                        asset_stats['التوقعات_الصحيحة'] += 1
                        asset_stats['صفقات_البيع_الصحيحة'] += 1
                        asset_stats['إجمالي_الربح_%'] += profit_pct
                    else:  # بيع خاطئ
                        asset_stats['التوقعات_الخاطئة'] += 1
                        asset_stats['إجمالي_الربح_%'] += profit_pct  # profit_pct سيكون سالباً

                # تحديث أعلى ربح/خسارة
                if profit_pct > asset_stats['أعلى_ربح_%']:
                    asset_stats['أعلى_ربح_%'] = profit_pct
                if profit_pct < asset_stats['أعلى_خسارة_%']:
                    asset_stats['أعلى_خسارة_%'] = profit_pct

                # حفظ تفاصيل الصفقة
                detailed_results.append({
                    'النموذج': model_name,
                    'العملة': asset,
                    'التوقع': predicted_direction,
                    'الواقع': actual_direction,
                    'الربح_%': profit_pct,
                    'الربح_مطلق': true[i] - entry[i],
                    'الثقة': confidence[i] if i < len(confidence) else 0.5,
                    'سعر_الدخول': entry[i],
                    'سعر_الخروج': true[i],
                    'السعر_المتوقع': pred[i]
                })

            # حساب الإحصائيات النهائية للعملة
            if asset_stats['عدد_التوقعات'] > 0:
                asset_stats['دقة_التوقع_%'] = (asset_stats['التوقعات_الصحيحة'] / asset_stats['عدد_التوقعات']) * 100
                asset_stats['متوسط_الربح_%'] = asset_stats['إجمالي_الربح_%'] / asset_stats['عدد_التوقعات']

                if asset_stats['صفقات_الشراء'] > 0:
                    asset_stats['دقة_الشراء_%'] = (asset_stats['صفقات_الشراء_الصحيحة'] / asset_stats['صفقات_الشراء']) * 100
                else:
                    asset_stats['دقة_الشراء_%'] = 0

                if asset_stats['صفقات_البيع'] > 0:
                    asset_stats['دقة_البيع_%'] = (asset_stats['صفقات_البيع_الصحيحة'] / asset_stats['صفقات_البيع']) * 100
                else:
                    asset_stats['دقة_البيع_%'] = 0

            # عرض نتائج العملة
            print(f"📈 إحصائيات {asset}:")
            print(f"   • عدد التوقعات: {asset_stats['عدد_التوقعات']}")
            print(f"   • الدقة الإجمالية: {asset_stats.get('دقة_التوقع_%', 0):.1f}%")
            print(f"   • دقة صفقات الشراء: {asset_stats.get('دقة_الشراء_%', 0):.1f}% ({asset_stats['صفقات_الشراء_الصحيحة']}/{asset_stats['صفقات_الشراء']})")
            print(f"   • دقة صفقات البيع: {asset_stats.get('دقة_البيع_%', 0):.1f}% ({asset_stats['صفقات_البيع_الصحيحة']}/{asset_stats['صفقات_البيع']})")
            print(f"   • متوسط الربح لكل صفقة: {asset_stats.get('متوسط_الربح_%', 0):.4f}%")
            print(f"   • أعلى ربح: {asset_stats['أعلى_ربح_%']:.2f}%")
            print(f"   • أعلى خسارة: {asset_stats['أعلى_خسارة_%']:.2f}%")
            print(f"   • متوسط الثقة: {asset_stats['متوسط_الثقة']:.3f}")

            # محاكاة التداول للعملة الواحدة
            if asset_stats['عدد_التوقعات'] > 0:
                initial_capital = 10000
                capital = initial_capital

                print(f"\n💰 محاكاة التداول لـ {asset} برأس مال ${initial_capital:,}:")

                # إستراتيجية 1: جميع الإشارات
                capital_all = initial_capital
                # إستراتيجية 2: إشارات عالية الثقة (>0.7)
                capital_high_conf = initial_capital
                # إستراتيجية 3: صفقات الشراء فقط
                capital_buy_only = initial_capital

                for trade in detailed_results:
                    if trade['العملة'] == asset:
                        trade_amount = capital_all * 0.1  # 10% من رأس المال
                        profit_usd = trade_amount * (trade['الربح_%'] / 100)

                        # جميع الإشارات
                        capital_all += profit_usd

                        # إشارات عالية الثقة
                        if trade['الثقة'] > 0.7:
                            capital_high_conf += profit_usd

                        # صفقات الشراء فقط
                        if trade['التوقع'] == 'شراء':
                            capital_buy_only += profit_usd

                print(f"   • جميع الإشارات: ${capital_all:,.2f} (عائد: {((capital_all/initial_capital-1)*100):.1f}%)")
                print(f"   • عالية الثقة (>0.7): ${capital_high_conf:,.2f} (عائد: {((capital_high_conf/initial_capital-1)*100):.1f}%)")
                print(f"   • صفقات الشراء فقط: ${capital_buy_only:,.2f} (عائد: {((capital_buy_only/initial_capital-1)*100):.1f}%)")

                # إضافة نتائج المحاكاة للعملة
                asset_stats['رأس_المال_النهائي_جميع'] = capital_all
                asset_stats['رأس_المال_النهائي_عالية_ثقة'] = capital_high_conf
                asset_stats['رأس_المال_النهائي_شراء_فقط'] = capital_buy_only
                asset_stats['العائد_%_جميع'] = ((capital_all/initial_capital-1)*100)

            all_assets_analysis.append(asset_stats)

    # عرض جدول مقارنة العملات
    if all_assets_analysis:
        df_summary = pd.DataFrame(all_assets_analysis)

        # إنشاء جدول عرض بسيط
        display_cols = [
            'العملة', 'عدد_التوقعات', 'دقة_التوقع_%', 'دقة_الشراء_%',
            'دقة_البيع_%', 'متوسط_الربح_%', 'العائد_%_جميع'
        ]

        df_display = df_summary[display_cols].copy()
        df_display.columns = ['العملة', 'عدد التوقعات', 'الدقة %', 'دقة الشراء %',
                             'دقة البيع %', 'متوسط الربح %', 'العائد %']

        print(f"\n{'='*100}")
        print(f"📋 ملخص أداء جميع العملات - {model_name}")
        print(f"{'='*100}")
        save_or_print(df_display.round(2), f'legacy_assets_summary_{model_name}')

        # تحليل إجمالي
        total_trades = df_summary['عدد_التوقعات'].sum()
        weighted_accuracy = (df_summary['عدد_التوقعات'] * df_summary['دقة_التوقع_%']).sum() / total_trades
        weighted_profit = (df_summary['عدد_التوقعات'] * df_summary['متوسط_الربح_%']).sum() / total_trades

        print(f"\n📊 الإحصائيات الإجمالية - {model_name}:")
        print(f"   • إجمالي التوقعات: {total_trades:,}")
        print(f"   • متوسط الدقة المرجح: {weighted_accuracy:.2f}%")
        print(f"   • متوسط الربح المرجح: {weighted_profit:.4f}%")

        # أفضل وأسوأ عملة
        best_asset = df_summary.loc[df_summary['دقة_التوقع_%'].idxmax()]
        worst_asset = df_summary.loc[df_summary['دقة_التوقع_%'].idxmin()]
        most_profitable = df_summary.loc[df_summary['متوسط_الربح_%'].idxmax()]

        print(f"\n🏆 أفضل العملات - {model_name}:")
        print(f"   • أعلى دقة: {best_asset['العملة']} ({best_asset['دقة_التوقع_%']:.1f}%)")
        print(f"   • أعلى ربحية: {most_profitable['العملة']} ({most_profitable['متوسط_الربح_%']:.4f}%)")
        print(f"   • أقل دقة: {worst_asset['العملة']} ({worst_asset['دقة_التوقع_%']:.1f}%)")

    return {
        'تفاصيل_النتائج': detailed_results,
        'ملخص_العملات': all_assets_analysis,
        'إسم_النموذج': model_name
    }


def compare_multiple_models(models_data, model_names):
    """
    ✅ مقارنة بين عدة نماذج
    """
    print("="*100)
    print("🤝 مقارنة شاملة بين النماذج")
    print("="*100)

    all_models_results = []

    # تحليل كل نموذج
    for i, (model_data, model_name) in enumerate(zip(models_data, model_names)):
        print(f"\n📊 تحليل {model_name}...")
        model_result = comprehensive_asset_analysis(model_data, model_name)
        all_models_results.append(model_result)

    # إنشاء جدول مقارنة
    comparison_table = []

    for model_result in all_models_results:
        model_name = model_result['إسم_النموذج']
        df_summary = pd.DataFrame(model_result['ملخص_العملات'])

        if not df_summary.empty:
            total_trades = df_summary['عدد_التوقعات'].sum()
            weighted_accuracy = (df_summary['عدد_التوقعات'] * df_summary['دقة_التوقع_%']).sum() / total_trades
            weighted_profit = (df_summary['عدد_التوقعات'] * df_summary['متوسط_الربح_%']).sum() / total_trades

            # محاكاة رأس المال الإجمالي
            initial_capital = 10000
            total_capital = initial_capital

            # تجميع جميع الصفقات من كل العملات
            all_trades = [trade for trade in model_result['تفاصيل_النتائج']]

            for trade in all_trades:
                trade_amount = total_capital * 0.1
                profit_usd = trade_amount * (trade['الربح_%'] / 100)
                total_capital += profit_usd

            final_return = ((total_capital / initial_capital - 1) * 100)

            comparison_table.append({
                'النموذج': model_name,
                'عدد_الصفقات': total_trades,
                'متوسط_الدقة_%': f"{weighted_accuracy:.2f}",
                'متوسط_الربح_%': f"{weighted_profit:.4f}",
                'رأس_المال_النهائي': f"${total_capital:,.2f}",
                'العائد_%': f"{final_return:.1f}"
            })

    # عرض جدول المقارنة
    if comparison_table:
        df_comparison = pd.DataFrame(comparison_table)
        df_comparison.columns = ['النموذج', 'عدد الصفقات', 'متوسط الدقة %', 'متوسط الربح %',
                                'رأس المال النهائي', 'العائد %']

        print(f"\n{'='*100}")
        print("📊 جدول مقارنة النماذج")
        print(f"{'='*100}")
        save_or_print(df_comparison, 'legacy_models_comparison')

        # تحديد النموذج الأفضل
        if len(comparison_table) > 1:
            print(f"\n{'='*100}")
            print("🏆 التقييم النهائي")
            print(f"{'='*100}")

            # أفضل نموذج من حيث الدقة
            best_accuracy = max(comparison_table, key=lambda x: float(x['متوسط_الدقة_%']))
            # أفضل نموذج من حيث الربحية
            best_profit = max(comparison_table, key=lambda x: float(x['متوسط_الربح_%']))
            # أفضل نموذج من حيث العائد
            best_return = max(comparison_table, key=lambda x: float(x['العائد_%']))

            print(f"   • أعلى دقة: {best_accuracy['النموذج']} ({best_accuracy['متوسط_الدقة_%']}%)")
            print(f"   • أعلى ربحية: {best_profit['النموذج']} ({best_profit['متوسط_الربح_%']}%)")
            print(f"   • أعلى عائد: {best_return['النموذج']} ({best_return['العائد_%']}%)")

            # التوصية النهائية
            print(f"\n💡 التوصية:")
            if best_accuracy['النموذج'] == best_profit['النموذج'] == best_return['النموذج']:
                print(f"   ✅ {best_accuracy['النموذج']} هو النموذج الأفضل بشكل شامل")
            else:
                print(f"   📊 اختر النموذج بناءً على أولويتك:")
                print(f"      - للدقة: {best_accuracy['النموذج']}")
                print(f"      - للربحية: {best_profit['النموذج']}")
                print(f"      - للعائد المالي: {best_return['النموذج']}")

    return all_models_results


def generate_detailed_report(analysis_results):
    """
    ✅ إنشاء تقرير مفصل عن النموذج
    """
    model_name = analysis_results['إسم_النموذج']
    df_summary = pd.DataFrame(analysis_results['ملخص_العملات'])

    print("="*100)
    print(f"📄 تقرير مفصل - {model_name}")
    print("="*100)

    if not df_summary.empty:
        # 1. إحصائيات عامة
        total_trades = df_summary['عدد_التوقعات'].sum()
        total_correct = df_summary['التوقعات_الصحيحة'].sum()
        total_wrong = df_summary['التوقعات_الخاطئة'].sum()
        total_buy = df_summary['صفقات_الشراء'].sum()
        total_sell = df_summary['صفقات_البيع'].sum()

        print(f"\n📊 الإحصائيات العامة:")
        print(f"   • إجمالي التوقعات: {total_trades:,}")
        print(f"   • التوقعات الصحيحة: {total_correct:,} ({total_correct/total_trades*100:.1f}%)")
        print(f"   • التوقعات الخاطئة: {total_wrong:,} ({total_wrong/total_trades*100:.1f}%)")
        print(f"   • صفقات الشراء: {total_buy:,} ({total_buy/total_trades*100:.1f}%)")
        print(f"   • صفقات البيع: {total_sell:,} ({total_sell/total_trades*100:.1f}%)")

        # 2. تحليل الربحية
        weighted_profit = (df_summary['عدد_التوقعات'] * df_summary['متوسط_الربح_%']).sum() / total_trades
        total_profit_pct = df_summary['إجمالي_الربح_%'].sum()

        print(f"\n💰 تحليل الربحية:")
        print(f"   • متوسط الربح لكل صفقة: {weighted_profit:.4f}%")
        print(f"   • مجموع نسب الربح عبر كل الصفقات: {total_profit_pct:.2f}%  (جمع نسب لا عائد محفظة — لا يُقرأ كربح)")

        # 3. أفضل 3 عملات
        print(f"\n🏅 أفضل 3 عملات في {model_name}:")
        for i, (idx, row) in enumerate(df_summary.nlargest(3, 'دقة_التوقع_%').iterrows(), 1):
            print(f"   {i}. {row['العملة']}: {row['دقة_التوقع_%']:.1f}% دقة، {row['متوسط_الربح_%']:.4f}% ربح")

        # 4. التوصيات
        print(f"\n🎯 التوصيات العملية لـ {model_name}:")

        # توصية عامة
        overall_accuracy = total_correct / total_trades * 100
        if overall_accuracy > 70:
            print(f"   ✅ النموذج ممتاز للتداول (دقة {overall_accuracy:.1f}%)")
        elif overall_accuracy > 60:
            print(f"   ⚠️  النموذج جيد للتداول مع إدارة مخاطر (دقة {overall_accuracy:.1f}%)")
        elif overall_accuracy > 50:
            print(f"   ⚠️  النموذج مقبول للتداول الورقي فقط (دقة {overall_accuracy:.1f}%)")
        else:
            print(f"   ❌ النموذج يحتاج تحسين (دقة {overall_accuracy:.1f}%)")

        # توصية حسب العملات
        best_asset = df_summary.loc[df_summary['دقة_التوقع_%'].idxmax()]
        print(f"\n   🎯 أفضل عملة: {best_asset['العملة']}")
        print(f"      • الدقة: {best_asset['دقة_التوقع_%']:.1f}%")
        print(f"      • متوسط الربح: {best_asset['متوسط_الربح_%']:.4f}%")
        print(f"      • عدد الصفقات: {best_asset['عدد_التوقعات']}")
        print(f"      ⚠️ الأفضل من {len(df_summary)} عملة على نفس البيانات — اختيار بأثر رجعي؛ بالصدفة وحدها يتوقّع "
              f"أعلى دقة ≈ {50 + 100 * 3 * np.sqrt(0.25 / max(best_asset['عدد_التوقعات'], 1)):.0f}% مع هذا العدد من الصفقات")

        # 5. إدارة المخاطر
        print(f"\n⚠️  إدارة المخاطر (قواعد عامة ثابتة — لا تُشتق من هذه البيانات):")
        print(f"   • حجم الصفقة: 2-5% من رأس المال")
        print(f"   • وقف الخسارة: 1-2% لكل صفقة")
        print(f"   • جني الأرباح: 2-3% لكل صفقة")
        print(f"   • الحد الأقصى للخسارة اليومية: 5% من رأس المال")

        # 6. الإستراتيجية المثلى
        print(f"\n📈 الإستراتيجية المثلى لـ {model_name}:")

        # حساب أفضل إستراتيجية بناءً على المحاكاة
        avg_return_all = df_summary['العائد_%_جميع'].mean() if 'العائد_%_جميع' in df_summary.columns else 0

        if avg_return_all > 100:
            print(f"   • استخدم جميع الإشارات (عائد متوقع: {avg_return_all:.1f}%)")
        else:
            # تحليل الثقة
            if 'متوسط_الثقة' in df_summary.columns:
                avg_confidence = df_summary['متوسط_الثقة'].mean()
                if avg_confidence > 0.7:
                    print(f"   • استخدم إشارات عالية الثقة (>0.7)")
                else:
                    print(f"   • استخدم عتبة ثقة 0.5 للحصول على تغطية جيدة")

            # تحليل الاتجاهات
            avg_buy_accuracy = df_summary['دقة_الشراء_%'].mean()
            avg_sell_accuracy = df_summary['دقة_البيع_%'].mean()

            if avg_buy_accuracy > avg_sell_accuracy + 5:
                print(f"   • ركز على صفقات الشراء (دقة أعلى: {avg_buy_accuracy:.1f}% vs {avg_sell_accuracy:.1f}%)")
            elif avg_sell_accuracy > avg_buy_accuracy + 5:
                print(f"   • ركز على صفقات البيع (دقة أعلى: {avg_sell_accuracy:.1f}% vs {avg_buy_accuracy:.1f}%)")
            else:
                print(f"   • استخدم كلا نوعي الصفقات (دقة متقاربة: شراء {avg_buy_accuracy:.1f}%، بيع {avg_sell_accuracy:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# 🚀 استخدام الكود مع بياناتك
# ══════════════════════════════════════════════════════════════════════════════

print("="*100)
print("🤖 نظام تحليل النماذج المتقدم")
print("="*100)

# الحالة 1: إذا لديك نموذج واحد
# if 'results2' in locals():
#     print("\n📊 تحليل نموذج واحد...")

#     # تحليل شامل للنموذج الواحد
#     analysis_result = comprehensive_asset_analysis(
#         per_asset_results=results2['per_asset_results'],
#         model_name="النموذج الأساسي"
#     )

#     # إنشاء تقرير مفصل
#     generate_detailed_report(analysis_result)

# # الحالة 2: إذا لديك نموذجان للمقارنة
# elif all(key in locals() for key in ['results_model1', 'results_model2']):
# # if True:
#     print("\n📊 مقارنة بين نموذجين...")

#     models_data = [
#         results['per_asset_results'],
#         results2['per_asset_results']
#     ]

#     model_names = ["النموذج الأول", "النموذج الثاني"]

#     # مقارنة النماذج
#     all_results = compare_multiple_models(models_data, model_names)

#     # إنشاء تقارير مفصلة لكل نموذج
#     for result in all_results:
#         generate_detailed_report(result)

# # الحالة 3: إذا لديك بيانات مباشرة
# else:
#     print("⚠️  لم يتم العثور على بيانات النماذج.")
#     print("يرجى التأكد من وجود متغيرات:")
#     print("   - results (لنموذج واحد)")
#     print("   - results_model1 و results_model2 (لمقارنة نموذجين)")

# print("\n" + "="*100)
# print("✅ تم الانتهاء من التحليل بنجاح")
# print("="*100)

# # ══════════════════════════════════════════════════════════════════════════════
# # 💾 حفظ النتائج
# # ══════════════════════════════════════════════════════════════════════════════

# def save_analysis_results(analysis_results, filename_prefix):
#     """
#     حفظ نتائج التحليل في ملفات CSV
#     """
#     import os

#     # إنشاء مجلد للنتائج إذا لم يكن موجوداً
#     os.makedirs('analysis_results', exist_ok=True)

#     # حفظ ملخص العملات
#     if 'ملخص_العملات' in analysis_results and analysis_results['ملخص_العملات']:
#         df_summary = pd.DataFrame(analysis_results['ملخص_العملات'])
#         summary_filename = f'analysis_results/{filename_prefix}_summary.csv'
#         df_summary.to_csv(summary_filename, index=False, encoding='utf-8-sig')
#         print(f"✅ تم حفظ ملخص العملات في: {summary_filename}")

#     # حفظ التفاصيل
#     if 'تفاصيل_النتائج' in analysis_results and analysis_results['تفاصيل_النتائج']:
#         df_details = pd.DataFrame(analysis_results['تفاصيل_النتائج'])
#         details_filename = f'analysis_results/{filename_prefix}_details.csv'
#         df_details.to_csv(details_filename, index=False, encoding='utf-8-sig')
#         print(f"✅ تم حفظ تفاصيل الصفقات في: {details_filename}")

#     return True

# # حفظ النتائج إذا كان هناك تحليل
# if 'analysis_result' in locals():
#     save_analysis_results(analysis_result, 'single_model_analysis')
# elif 'all_results' in locals():
#     for i, result in enumerate(all_results):
#         save_analysis_results(result, f'model_{i+1}_analysis')
