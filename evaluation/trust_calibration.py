"""
PURPOSE:  Trust & calibration report: generate_trust_report, calibration_report (ECE/Brier before and after). The calibrator math (isotonic/Platt) lives in evaluation/calibrators.py.
TAGS:     generate_trust_report, calibration_report, ECE, Brier, reliability curve, confidence
PITFALLS: apply_confidence_calibration (evaluation/calibrators.py) is applied to test inside generate_trust_report; the calibrators themselves are fitted on val only. Executed into the one shared evaluation namespace by evaluation/_loader.py (never imported on its own): names from other modules resolve at call time.

## 🔟 🧠 تقرير: الثقة والفهم (Trust & Calibration Report)

يجيب على: **"إلى أي مدى يفهم النموذج توقعاته، وهل يمكن الوثوق بدرجة ثقته؟"** — الآن مع مخطط معايرة (Calibration Curve) لكل هدف.
"""
# ═══════════════════════════════════════════════════════════════════════════
# 🧠 تقرير الثقة والفهم (Trust & Calibration Report)
# ═══════════════════════════════════════════════════════════════════════════
#
# الهدف: تقييم "مدى فهم النموذج لتوقعاته" — أي هل ثقة النموذج (confidence)
# وعدم يقينه (uncertainty) يعكسان فعلاً احتمال صحة التوقع؟ نموذج "يفهم نفسه"
# جيداً هو نموذج تكون ثقته عالية عندما يكون صحيحاً ومنخفضة عندما يخطئ.
# ═══════════════════════════════════════════════════════════════════════════

def generate_trust_report(
    df: pd.DataFrame,
    group_by: Optional[str] = 'target',
    n_bins: int = 10,
    verbose: bool = True,
    calibrators: Optional[Dict] = None,
) -> Dict:
    """
    يبني تقرير ثقة شامل من DataFrame مسطّح (خرج build_flat_dataframe).

    المقاييس المحسوبة لكل مجموعة (كل target أو كل asset حسب group_by):
      • ECE  (Expected Calibration Error): متوسط الفرق |ثقة - دقة فعلية| مرجّحاً بالعدد
      • MCE  (Maximum Calibration Error): أسوأ فجوة في أي bin
      • Brier Score: متوسط (confidence - correct)^2 — كلما قلّ كان أفضل
      • Sharpness: متوسط عدم اليقين (uncertainty) — نموذج "حاد" يعطي عدم يقين منخفض
                   عند الصحة وعالٍ عند الخطأ، وليس ثابتاً دوماً
      • Confidence-Accuracy Correlation: هل الثقة الأعلى تعني دقة أعلى فعلاً؟
      • Coverage@1σ / @2σ: نسبة الأخطاء الواقعة ضمن 1 أو 2 انحراف معياري متوقَّع
      • Trust Score (0-100): درجة مُركّبة = 100 × (1 - ECE) × (0.5 + 0.5×correlation_scaled)
        (تُعرض المعادلة بوضوح حتى يسهل تعديل الأوزان حسب الحاجة)

    calibrators: ناتج fit_confidence_calibrators (مُلائَم على val فقط). إن مُرِّر تُحسب أيضاً معايرة post-hoc على هذا
        الـDataFrame (test) ويُقارَن ECE قبلها وبعدها (`calibration`) — القيم الخام تبقى في 'confidence' بلا مساس.

    Returns:
        dict: {'overall': {...}, 'by_group': DataFrame, 'reliability_curve': DataFrame,
               'calibration': DataFrame | None, 'calibrated_df': DataFrame | None}
    """
    def _metrics_for(sub: pd.DataFrame) -> Dict:
        sub = sub.dropna(subset=['confidence'])
        if len(sub) == 0:
            return {}

        correct = sub['correct'].astype(float).values
        conf = sub['confidence'].astype(float).values

        # --- Calibration bins ---
        try:
            bins = pd.qcut(conf, q=min(n_bins, len(np.unique(conf))), duplicates='drop')
        except Exception:
            bins = pd.cut(conf, bins=n_bins)

        bin_df = pd.DataFrame({'conf': conf, 'correct': correct, 'bin': bins})
        grouped = bin_df.groupby('bin', observed=True)

        ece, mce = 0.0, 0.0
        reliability_rows = []
        for bname, g in grouped:
            if len(g) == 0:
                continue
            avg_conf = g['conf'].mean()
            avg_acc = g['correct'].mean()
            weight = len(g) / len(bin_df)
            gap = abs(avg_conf - avg_acc)
            ece += weight * gap
            mce = max(mce, gap)
            reliability_rows.append({'bin': str(bname), 'avg_confidence': avg_conf,
                                      'avg_accuracy': avg_acc, 'gap': gap, 'count': len(g)})

        brier = float(np.mean((conf - correct) ** 2))

        # --- correlation confidence vs correctness ---
        if np.std(conf) > 1e-9 and np.std(correct) > 1e-9:
            conf_acc_corr = float(np.corrcoef(conf, correct)[0, 1])
        else:
            conf_acc_corr = 0.0

        # --- uncertainty-based metrics (إن وُجدت) ---
        sharpness = float(sub['uncertainty'].mean()) if 'uncertainty' in sub and sub['uncertainty'].notna().any() else None
        unc_error_corr = None
        if 'abs_error' in sub.columns and 'uncertainty' in sub.columns:
            common = sub.dropna(subset=['uncertainty', 'abs_error'])
            if len(common) > 2 and common['uncertainty'].std() > 1e-9:
                u, e = common['uncertainty'], common['abs_error']
                if 'entry' in common.columns:
                    # بوحدة السعر يرتبط الاثنان بمستوى السعر نفسه (BTC أخطاؤه وعدم يقينه بالدولار أكبر من عملة
                    # بسنتات) فيظهر ارتباط ~0.77 بلا أي معايرة. نسبةً لسعر الدخول، وبالرتب (مقاوم للقيم الشاذة).
                    scale = np.abs(common['entry']).replace(0, np.nan)
                    u, e = u / scale, e / scale
                unc_error_corr = float(pd.Series(u).corr(pd.Series(e), method='spearman'))

        acc_lo, acc_hi = wilson_ci(correct.sum(), len(correct))   # فاصل Wilson 95% لنسبة الصحة (n صغير ← فاصل عريض)

        # --- Trust score مُركّب (0-100) ---
        corr_scaled = (conf_acc_corr + 1) / 2  # من [-1,1] إلى [0,1]
        trust_score = 100.0 * max(0.0, (1 - ece)) * (0.5 + 0.5 * corr_scaled)

        return {
            'n_samples': len(sub),
            'ece': ece,
            'mce': mce,
            'brier_score': brier,
            'confidence_accuracy_corr': conf_acc_corr,
            'uncertainty_error_corr': unc_error_corr,
            'sharpness_avg_uncertainty': sharpness,
            'mean_confidence': float(conf.mean()),
            'mean_accuracy': float(correct.mean()),
            'accuracy_ci95_lo': acc_lo,
            'accuracy_ci95_hi': acc_hi,
            'trust_score_0_100': trust_score,
            '_reliability_rows': reliability_rows,
        }

    overall = _metrics_for(df)

    by_group_rows = []
    reliability_all = []
    if group_by and group_by in df.columns:
        for gval, sub in df.groupby(group_by):
            m = _metrics_for(sub)
            if not m:
                continue
            rel_rows = m.pop('_reliability_rows')
            for rr in rel_rows:
                rr[group_by] = gval
                reliability_all.append(rr)
            by_group_rows.append({group_by: gval, **m})

    by_group_df = pd.DataFrame(by_group_rows)
    reliability_df = pd.DataFrame(reliability_all)
    overall.pop('_reliability_rows', None)

    calibration_df = calibrated_df = None
    if calibrators:
        calibrated_df = apply_confidence_calibration(df, calibrators)
        calibration_df = calibration_report(calibrated_df, n_bins=n_bins, verbose=False)

    if verbose:
        print("=" * 100)
        print("🧠 تقرير الثقة والفهم (Trust & Calibration Report)")
        print("=" * 100)
        print(f"\n📊 التقييم العام (كل البيانات):")
        for k, v in overall.items():
            print(f"   • {k}: {v:.4f}" if isinstance(v, float) else f"   • {k}: {v}")

        _trust_verdict(overall.get('trust_score_0_100'), overall.get('ece'), overall.get('confidence_accuracy_corr'))

        if not by_group_df.empty:
            print(f"\n📋 التفصيل حسب '{group_by}':")
            cols = [group_by, 'n_samples', 'trust_score_0_100', 'ece', 'mce',
                    'confidence_accuracy_corr', 'mean_confidence', 'mean_accuracy',
                    'accuracy_ci95_lo', 'accuracy_ci95_hi']
            cols = [c for c in cols if c in by_group_df.columns]
            save_or_print(by_group_df[cols].round(4), 'trust_report_by_group', out_dir=DEFAULT_OUTPUT_DIR)
        if calibration_df is not None:
            print_calibration_report(calibration_df)

    return {'overall': overall, 'by_group': by_group_df, 'reliability_curve': reliability_df,
            'calibration': calibration_df, 'calibrated_df': calibrated_df}


def _trust_verdict(trust_score, ece, corr):
    print(f"\n💡 الحكم العام:")
    if trust_score is None:
        return
    if trust_score >= 75:
        print(f"   ✅ النموذج 'يفهم' توقعاته جيداً (Trust Score: {trust_score:.1f}/100) — يمكن الاعتماد على الثقة كمرشّح للجودة")
    elif trust_score >= 50:
        print(f"   ⚠️  فهم متوسط (Trust Score: {trust_score:.1f}/100) — الثقة مفيدة جزئياً لكنها تحتاج معايرة")
    else:
        print(f"   ❌ فهم ضعيف (Trust Score: {trust_score:.1f}/100) — لا يُنصح بالاعتماد على درجة الثقة كما هي")

    if ece is not None:
        if ece < 0.05:
            print(f"      • المعايرة (ECE={ece:.3f}) ممتازة")
        elif ece < 0.15:
            print(f"      • المعايرة (ECE={ece:.3f}) مقبولة")
        else:
            print(f"      • المعايرة (ECE={ece:.3f}) ضعيفة — النموذج مبالغ/متحفّظ في ثقته")


# ECE بصناديق كمّية (نفس طريقة generate_trust_report): التعريف الوحيد في stats/metrics.py (ece_quantile).
from stats.metrics import ece_quantile as _ece_quantile


def calibration_report(cal_df: pd.DataFrame, n_bins: int = 10, group_by: str = 'target', verbose: bool = True) -> pd.DataFrame:
    """ECE/Brier قبل وبعد المعايرة لكل هدف على cal_df (ناتج apply_confidence_calibration، أي test)."""
    rows = []
    for g, sub in cal_df.dropna(subset=['confidence_cal', 'calib_correct']).groupby(group_by):
        p0, p1, y = (sub[c].to_numpy(dtype=np.float64) for c in ('calib_raw_p', 'confidence_cal', 'calib_correct'))
        rows.append({group_by: g, 'n': int(len(sub)), 'signal': sub['calib_signal'].iloc[0],
                     'ece_before': _ece_quantile(p0, y, n_bins), 'ece_after': _ece_quantile(p1, y, n_bins),
                     'brier_before': float(np.mean((p0 - y) ** 2)), 'brier_after': float(np.mean((p1 - y) ** 2)),
                     'mean_p_before': float(p0.mean()), 'mean_p_after': float(p1.mean()), 'mean_accuracy': float(y.mean())})
    res = pd.DataFrame(rows)
    if verbose:
        print_calibration_report(res)
    return res


def print_calibration_report(res: pd.DataFrame):
    if res is None or res.empty:
        return
    print(f"\n🎚️ معايرة الثقة post-hoc (مُلائَمة على val، مُطبَّقة على test) — ECE قبل ← بعد")
    print(res.round(4).to_string(index=False))
    for _, r in res.iterrows():
        if r['signal'] == 'class_margin':
            print(f"   ⚠️ [{r['target']}] رأس الثقة بلا معلومة ← المعايرة من |p_up−0.5| لرأس التصنيف؛ ECE هنا لصحة اتجاه رأس التصنيف "
                  f"(لا رأس الانحدار). ")
        elif r['signal'] == 'base_rate':
            print(f"   ⚠️ [{r['target']}] لا إشارة ثقة مفيدة (ولا p_up): المعايرة = نسبة الصحة الثابتة على val.")
    print("   ℹ️ المعايرة تُصلح الانحياز لا التمييز: هبوط ECE لا يعني أن الثقة صارت تفرّق الصحيح من الخاطئ "
          "(راجع confidence_accuracy_corr في التقرير أعلاه).")
