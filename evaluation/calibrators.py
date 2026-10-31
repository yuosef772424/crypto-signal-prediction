"""
PURPOSE:  Post-hoc confidence calibration math (isotonic / Platt) fitted on a val flat table and applied to test: fit_confidence_calibrators, apply_confidence_calibration, diagnose_confidence (moved verbatim out of trust_calibration.py).
TAGS:     fit_confidence_calibrators, apply_confidence_calibration, diagnose_confidence, isotonic, platt, PAV, confidence calibration, class_margin, base_rate, post-hoc calibration, معايرة الثقة
PITFALLS: Calibrators are fitted on val only and never touch test. Executed into the one shared evaluation namespace by evaluation/_loader.py (never imported on its own): names from other modules (pd, np, Dict) resolve at call time.

## معايرة الثقة post-hoc (isotonic / Platt)
"""
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
