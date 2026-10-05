"""
PURPOSE:  Trust & calibration report: generate_trust_report, confidence calibrators (isotonic/Platt), ECE/Brier before and after.
TAGS:     generate_trust_report, fit_confidence_calibrators, apply_confidence_calibration, calibration_report, ECE, isotonic, platt, confidence
PITFALLS: Calibrators are fitted on val only and never touch test. Executed into the one shared evaluation namespace by evaluation/_loader.py (never imported on its own): names from other modules resolve at call time.

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



# ═══════════════════════════════════════════════════════════════════════════
# 🎚️ معايرة الثقة post-hoc (بلا إعادة تدريب) — تُلائَم على val فقط وتُطبَّق على test
# ═══════════════════════════════════════════════════════════════════════════
# الفكرة: خريطة رتيبة من «إشارة ثقة» إلى احتمال صحة معايَر. الإشارة بالأولوية:
#   1) 'confidence'   : مخرَج رأس الثقة y_*_confidence، الهدف = correct (صحة اتجاه رأس الانحدار — ما يقيسه تقرير الثقة).
#   2) 'class_margin' : |p_up − 0.5| من رأس التصنيف، حين رأس الثقة ثابت/عشوائي (لا ارتباط موجب دالّ بالصحة على val).
#                       الهدف هنا صحة اتجاه **رأس التصنيف نفسه** (p_up ≥ 0.5 مقابل اتجاه السعر الفعلي) — P(صعود) الخام ←
#                       max(p_up, 1−p_up). ⚠️ مقاييس هذا الصف على هدف مختلف عن صف 'confidence' فلا تُقارَن ببعضها.
#   3) 'base_rate'    : لا إشارة مفيدة ← نسبة الصحة على val ثابتة (المعايرة تصلح الانحياز فقط).
# المعايرة تُصلح **الانحياز** (ECE) ولا تخلق **تمييزاً**: الارتباط confidence↔correct يبقى كما هو (الحكم في diag).
CALIB_MIN_RHO = 0.03        # أدنى ارتباط سبيرمان موجب على val لاعتبار رأس الثقة «مفيداً» (مع p < 0.05)
CALIB_MIN_STD = 1e-3        # انحراف معياري للثقة أدنى منه = «ثابت فعلياً»


def _ece_quantile(p, y, n_bins: int = 10) -> float:
    """ECE بصناديق كمّية (نفس طريقة generate_trust_report): Σ وزن·|متوسط p − متوسط y|."""
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if len(p) == 0:
        return float('nan')
    try:
        bins = pd.qcut(p, q=min(n_bins, len(np.unique(p))), duplicates='drop')
    except Exception:
        bins = pd.cut(p, bins=n_bins)
    g = pd.DataFrame({'p': p, 'y': y, 'b': bins}).groupby('b', observed=True)
    return float(sum(len(x) / len(p) * abs(x['p'].mean() - x['y'].mean()) for _, x in g))


def _pav_isotonic(x, y):
    """Isotonic تصاعدي بخوارزمية PAV (احتياطي حين لا يتوفر sklearn) → دالة predict(x) بإدراج خطي وقصّ الأطراف."""
    order = np.argsort(x, kind='mergesort')
    xs, ys = np.asarray(x, dtype=np.float64)[order], np.asarray(y, dtype=np.float64)[order]
    blocks = []                                   # [متوسط، وزن، أول x، آخر x]
    for xi, yi in zip(xs, ys):
        blocks.append([yi, 1.0, xi, xi])
        while len(blocks) > 1 and blocks[-2][0] >= blocks[-1][0]:
            b2, b1 = blocks.pop(), blocks.pop()
            w = b1[1] + b2[1]
            blocks.append([(b1[0] * b1[1] + b2[0] * b2[1]) / w, w, b1[2], b2[3]])
    kx = np.array([v for b in blocks for v in (b[2], b[3])])
    ky = np.array([b[0] for b in blocks for _ in (0, 1)])
    return lambda q: np.interp(np.asarray(q, dtype=np.float64), kx, ky)


def _fit_map(x, y, method: str):
    """يُلائم x → P(y=1) بـ 'isotonic' أو 'platt'؛ يُرجع دالة predict تُخرج قيماً في [0, 1]."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if method == 'isotonic':
        try:
            from sklearn.isotonic import IsotonicRegression
            m = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds='clip').fit(x, y)
            return lambda q: m.predict(np.asarray(q, dtype=np.float64))
        except ImportError:
            f = _pav_isotonic(x, y)
            return lambda q: np.clip(f(q), 0.0, 1.0)
    if method == 'platt':
        from sklearn.linear_model import LogisticRegression
        def _z(v):
            v = np.clip(np.asarray(v, dtype=np.float64), 1e-4, 1 - 1e-4)
            return np.log(v / (1 - v)).reshape(-1, 1)
        m = LogisticRegression(C=1e6, max_iter=1000).fit(_z(x), y.astype(int))
        return lambda q: m.predict_proba(_z(q))[:, 1]
    raise ValueError(f"method: 'isotonic' | 'platt' — لا {method!r}")


def _class_signal(sub: pd.DataFrame):
    """(p_raw = max(p_up, 1−p_up)، margin = |p_up−0.5|، label = اتجاه رأس التصنيف صحيح؟) أو None إن غاب p_up/الحقيقة."""
    if 'p_up' not in sub.columns or 'price_change' not in sub.columns:
        return None
    p = sub['p_up'].to_numpy(dtype=np.float64)
    chg = sub['price_change'].to_numpy(dtype=np.float64)
    ok = np.isfinite(p) & np.isfinite(chg)
    if not ok.all():
        return None
    return np.maximum(p, 1 - p), np.abs(p - 0.5), ((p >= 0.5) == (chg > 0)).astype(np.float64)


def diagnose_confidence(conf, correct) -> Dict:
    """هل رأس الثقة يحمل معلومة؟ verdict: 'constant' (لا تباين) | 'no_signal' (ارتباط غير موجب/غير دالّ) | 'informative'."""
    conf, correct = np.asarray(conf, dtype=np.float64), np.asarray(correct, dtype=np.float64)
    std = float(np.std(conf))
    rho = p = float('nan')
    if std >= CALIB_MIN_STD and np.std(correct) > 0 and len(conf) > 10:
        from scipy.stats import spearmanr
        rho, p = (float(v) for v in spearmanr(conf, correct))
    if std < CALIB_MIN_STD:
        verdict = 'constant'
    elif np.isfinite(rho) and rho >= CALIB_MIN_RHO and p < 0.05:
        verdict = 'informative'
    else:
        verdict = 'no_signal'
    return {'n': int(len(conf)), 'conf_std': std, 'conf_unique': int(len(np.unique(np.round(conf, 6)))),
            'spearman_rho': rho, 'spearman_p': p, 'verdict': verdict}


def fit_confidence_calibrators(val_df: pd.DataFrame, method: str = 'isotonic', group_by: str = 'target',
                               min_n: int = 50, verbose: bool = True) -> Dict:
    """يُلائم معايراً لكل مجموعة (هدف) من DataFrame **val** المسطّح (build_flat_dataframe). لا يلمس test أبداً.

    method: 'isotonic' (افتراضي؛ sklearn أو PAV احتياطي) | 'platt' (انحدار لوجستي على logit الإشارة؛ يتطلب sklearn).
    min_n: أقل عدد صفوف val لهدف لملاءمة خريطة (دونه 'base_rate').
    Returns: {هدف: {'method','signal','label','predict','n_val','base_rate','diag'}}."""
    out = {}
    for g, sub in val_df.dropna(subset=['confidence', 'correct']).groupby(group_by):
        correct = sub['correct'].to_numpy(dtype=np.float64)
        diag = diagnose_confidence(sub['confidence'].to_numpy(), correct)
        base = float(correct.mean())
        cls = _class_signal(sub)
        entry = {'method': method, 'n_val': int(len(sub)), 'base_rate': base, 'diag': diag}
        if len(sub) >= min_n and diag['verdict'] == 'informative':
            entry.update(signal='confidence', label='correct', predict=_fit_map(sub['confidence'], correct, method))
        elif len(sub) >= min_n and cls is not None:
            _, margin, lab = cls
            entry.update(signal='class_margin', label='class_correct', base_rate=float(lab.mean()),
                         predict=_fit_map(0.5 + margin, lab, method))
        else:
            entry.update(signal='base_rate', label='correct', predict=(lambda q, b=base: np.full(np.shape(q), b, dtype=np.float64)))
        out[g] = entry
        if verbose:
            d = diag
            warn = '' if d['verdict'] == 'informative' else (
                f"  ⚠️ رأس الثقة {'ثابت فعلياً' if d['verdict'] == 'constant' else 'بلا ارتباط موجب دالّ بالصحة'} على val "
                f"(std={d['conf_std']:.4f}، قيم فريدة={d['conf_unique']}، rho={d['spearman_rho']:.3f}) ← "
                f"{'المعايرة من |p_up−0.5| لرأس التصنيف' if entry['signal'] == 'class_margin' else 'نسبة الصحة الثابتة فقط'}")
            print(f"🎚️ معايرة [{g}] ({method}) على val n={d['n']}: الإشارة = {entry['signal']}{warn}")
    return out


def apply_confidence_calibration(df: pd.DataFrame, calibrators: Dict, group_by: str = 'target') -> pd.DataFrame:
    """نسخة من df بأعمدة إضافية (الخام 'confidence' بلا مساس): confidence_cal (الاحتمال المعايَر)، calib_raw_p (الاحتمال
    الخام في فضاء نفس الهدف)، calib_correct (الهدف)، calib_signal. مجموعة بلا معاير ← NaN."""
    out = df.copy()
    for c in ('confidence_cal', 'calib_raw_p', 'calib_correct'):
        out[c] = np.nan
    out['calib_signal'] = None
    for g, cal in calibrators.items():
        m = (out[group_by] == g).to_numpy()
        if not m.any():
            continue
        sub = out[m]
        if cal['signal'] == 'class_margin':
            cls = _class_signal(sub)
            if cls is None:
                continue
            raw_p, margin, lab = cls
            cal_p = cal['predict'](0.5 + margin)
        else:
            raw_p = sub['confidence'].to_numpy(dtype=np.float64)
            lab = sub['correct'].to_numpy(dtype=np.float64)
            cal_p = cal['predict'](raw_p)
        out.loc[m, 'calib_raw_p'] = raw_p
        out.loc[m, 'confidence_cal'] = np.asarray(cal_p, dtype=np.float64)
        out.loc[m, 'calib_correct'] = lab
        out.loc[m, 'calib_signal'] = cal['signal']
    return out


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
