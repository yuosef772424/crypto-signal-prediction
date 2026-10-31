"""
PURPOSE:  The Calibrator component (CALIBRATORS registry: isotonic / platt) of post-hoc confidence calibration: fit(data)->state on a role-stamped val/calibration flat table, transform(state, data), dump_state/load_state (JSON). fit_confidence_calibrators / apply_confidence_calibration are thin wrappers over it; the isotonic/Platt math is unchanged (moved verbatim out of trust_calibration.py).
TAGS:     Calibrator, CALIBRATORS, CalibrationState, fit_confidence_calibrators, apply_confidence_calibration, diagnose_confidence, isotonic, platt, PAV, confidence calibration, class_margin, base_rate, post-hoc calibration, split role, role check, one component one interface, معايرة الثقة
PITFALLS: fit REFUSES (core.roles.RoleError) a table whose role stamp (`df.attrs['split_role']`, core.roles.stamp_role) is missing, train, holdout, test or otherwise not in the calibrator's allowed roles (default calibration, val: today's val-fit behaviour); put the stamp on the table right before fit (attrs do not survive groupby/merge/concat). The state remembers its fit_role; transform / load_state refuse a state fitted on a forbidden role and a raw dict without provenance. New method = a new registered calibrator, never a parallel function. CalibrationState is a dict subclass with the legacy entries (so generate_trust_report and old readers work); `predict` entries are run-time closures, the serializable form is dump_state (params). A re-exec of this module creates a fresh registry (one per namespace). Executed into the one shared evaluation namespace by evaluation/_loader.py (never imported on its own): names from other modules (pd, np, Dict) resolve at call time.

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
    f = lambda q: np.interp(np.asarray(q, dtype=np.float64), kx, ky)
    f.params = {'kind': 'pav', 'x': kx.tolist(), 'y': ky.tolist()}
    return f


def _fit_map(x, y, method: str):
    """يُلائم x → P(y=1) بـ 'isotonic' أو 'platt'؛ يُرجع دالة predict تُخرج قيماً في [0, 1]."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if method == 'isotonic':
        try:
            from sklearn.isotonic import IsotonicRegression
            m = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds='clip').fit(x, y)
            g = lambda q: m.predict(np.asarray(q, dtype=np.float64))
            g.params = {'kind': 'isotonic', 'x': m.X_thresholds_.tolist(), 'y': m.y_thresholds_.tolist()}
            return g
        except ImportError:
            f = _pav_isotonic(x, y)
            g = lambda q: np.clip(f(q), 0.0, 1.0)
            g.params = {'kind': 'pav_clip', 'x': f.params['x'], 'y': f.params['y']}
            return g
    if method == 'platt':
        from sklearn.linear_model import LogisticRegression
        def _z(v):
            v = np.clip(np.asarray(v, dtype=np.float64), 1e-4, 1 - 1e-4)
            return np.log(v / (1 - v)).reshape(-1, 1)
        m = LogisticRegression(C=1e6, max_iter=1000).fit(_z(x), y.astype(int))
        g = lambda q: m.predict_proba(_z(q))[:, 1]
        g.params = {'kind': 'platt', 'coef': float(m.coef_[0, 0]), 'intercept': float(m.intercept_[0])}
        return g
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


def _fit_confidence_groups(val_df: pd.DataFrame, method: str = 'isotonic', group_by: str = 'target',
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
            _const = lambda q, b=base: np.full(np.shape(q), b, dtype=np.float64)
            _const.params = {'kind': 'constant', 'value': base}
            entry.update(signal='base_rate', label='correct', predict=_const)
        out[g] = entry
        if verbose:
            d = diag
            warn = '' if d['verdict'] == 'informative' else (
                f"  ⚠️ رأس الثقة {'ثابت فعلياً' if d['verdict'] == 'constant' else 'بلا ارتباط موجب دالّ بالصحة'} على val "
                f"(std={d['conf_std']:.4f}، قيم فريدة={d['conf_unique']}، rho={d['spearman_rho']:.3f}) ← "
                f"{'المعايرة من |p_up−0.5| لرأس التصنيف' if entry['signal'] == 'class_margin' else 'نسبة الصحة الثابتة فقط'}")
            print(f"🎚️ معايرة [{g}] ({method}) على val n={d['n']}: الإشارة = {entry['signal']}{warn}")
    return out


def _apply_confidence_groups(df: pd.DataFrame, calibrators: Dict, group_by: str = 'target') -> pd.DataFrame:
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


# ═══════════════════════════════════════════════════════════════════════════
# 🧩 المكوّن المسجَّل: Calibrator (أول مكوّن ذي حالة متعلَّمة يفحص دور بياناته — audit C6، PR 6)
# ═══════════════════════════════════════════════════════════════════════════
# واجهة واحدة: fit(data) → state (على بيانات دورها calibration أو val فقط؛ train وholdout مرفوضان دائماً، وغير المختوم يُرفض)،
# transform(state, data)، dump_state/load_state (حالة قابلة للتسلسل JSON). كل مستهلك (run_full_analysis، الاختبارات، الحي لاحقاً)
# يستدعي هذا المكوّن؛ fit_confidence_calibrators / apply_confidence_calibration أدناه غلافان رقيقان عليه (لا تنفيذ ثانٍ).
# الرياضيات (_fit_map / _class_signal / diagnose_confidence / _fit_confidence_groups / _apply_confidence_groups) هي نفسها حرفياً.
import hashlib as _hashlib

from core import roles as _roles
from core.registry import Contract as _Contract, Registry as _Registry

#: إصدار صيغة الحالة المسلسلة (provenance['version']).
CALIBRATION_STATE_VERSION = 1


class CalibrationState(dict):
    """ناتج ``Calibrator.fit``: قاموس ``{مجموعة: {method, signal, label, predict, n_val, base_rate, diag}}`` تماماً كما أرجعته
    ``fit_confidence_calibrators`` دائماً (فكل مستهلك قديم يعمل بلا تعديل)، مع ``.method`` و``.provenance`` (fit_role, fit_rows,
    fingerprint, group_by, min_n, version). ``predict`` دوال (حالة تشغيل)؛ الصيغة القابلة للتسلسل = ``Calibrator.dump_state``."""

    def __init__(self, entries=(), *, method: str, provenance: Dict):
        super().__init__(entries)
        self.method = method
        self.provenance = dict(provenance)


def _predict_from_params(p: Dict):
    """يعيد بناء دالة predict من ``params`` المسلسَلة (نفس الصيغ التي بنتها _fit_map: isotonic sklearn، PAV، Platt، ثابت)."""
    kind = p['kind']
    if kind == 'isotonic':                                  # = IsotonicRegression.predict (out_of_bounds='clip', خطي بين العتبات)
        x, y = np.asarray(p['x'], dtype=np.float64), np.asarray(p['y'], dtype=np.float64)
        if len(x) == 1:
            return lambda q: y.repeat(np.shape(np.asarray(q, dtype=np.float64)))
        from scipy.interpolate import interp1d
        f = interp1d(x, y, kind='linear')
        return lambda q: f(np.clip(np.asarray(q, dtype=np.float64), x[0], x[-1]))
    if kind in ('pav', 'pav_clip'):
        kx, ky = np.asarray(p['x'], dtype=np.float64), np.asarray(p['y'], dtype=np.float64)
        if kind == 'pav':
            return lambda q: np.interp(np.asarray(q, dtype=np.float64), kx, ky)
        return lambda q: np.clip(np.interp(np.asarray(q, dtype=np.float64), kx, ky), 0.0, 1.0)
    if kind == 'platt':
        from scipy.special import expit
        coef, b = float(p['coef']), float(p['intercept'])

        def _platt(q):
            v = np.clip(np.asarray(q, dtype=np.float64), 1e-4, 1 - 1e-4)
            z = np.log(v / (1 - v)).reshape(-1, 1)
            return expit((z @ np.array([[coef]]).T + b).ravel())
        return _platt
    if kind == 'constant':
        return lambda q, v=float(p['value']): np.full(np.shape(q), v, dtype=np.float64)
    raise ValueError(f"calibrator params kind {kind!r} unknown (isotonic | pav | pav_clip | platt | constant)")


def _table_fingerprint(df: pd.DataFrame, group_by: str) -> str:
    """بصمة الصفوف التي دخلت الملاءمة (المجموعة + الأعمدة التي تقرؤها المعايرة)؛ تُسجَّل في الحالة."""
    cols = [c for c in (group_by, 'confidence', 'correct', 'p_up', 'price_change') if c in df.columns]
    h = pd.util.hash_pandas_object(df[cols], index=False).to_numpy().tobytes()
    return _hashlib.sha256(h).hexdigest()[:16]


class ConfidenceCalibrator:
    """معاير الثقة post-hoc (isotonic | platt) كمكوّن بواجهة واحدة. ``allowed_roles``: أدوار البيانات المسموح الملاءمة عليها
    (الافتراضي calibration وval)؛ ``train`` وholdout ممنوعان دائماً (يرفضهما عقد السجلّ عند التسجيل وfit عند الاستدعاء)."""

    def __init__(self, name: str, allowed_roles=_roles.CALIBRATOR_ROLES):
        self.name = name
        self.allowed_roles = tuple(allowed_roles)

    def fit(self, data: pd.DataFrame, *, group_by: str = 'target', min_n: int = 50, verbose: bool = True) -> CalibrationState:
        """يُلائم معايراً لكل مجموعة من الجدول المسطّح ``data`` (build_flat_dataframe) المختوم بدور مسموح (``core.roles.stamp_role``).
        يرفع ``core.roles.RoleError`` لجدول دوره train/holdout/test أو غير مختوم أو خارج ``allowed_roles``."""
        role = _roles.require_role(data, self.allowed_roles, what=f"calibrator {self.name!r}", never=_roles.NEVER_FIT_ROLES)
        entries = _fit_confidence_groups(data, method=self.name, group_by=group_by, min_n=min_n, verbose=verbose)
        used = data.dropna(subset=['confidence', 'correct'])
        prov = {'fit_role': role, 'fit_rows': int(len(used)), 'fingerprint': _table_fingerprint(used, group_by),
                'group_by': group_by, 'min_n': int(min_n), 'version': CALIBRATION_STATE_VERSION}
        return CalibrationState(entries, method=self.name, provenance=prov)

    def _check_state(self, state) -> None:
        """حالة بلا مصدر (قاموس خام) أو مُلائمة على دور غير مسموح أو بطريقة أخرى ترفضها المستهلكات."""
        if not isinstance(state, CalibrationState):
            raise _roles.RoleError("calibrator state has no provenance (not a CalibrationState from Calibrator.fit / load_state): "
                                   "a raw dict of closures cannot show what it was fitted on, so it is refused")
        if state.method != self.name:
            raise ValueError(f"state was fitted by calibrator {state.method!r}, not {self.name!r}")
        fit_role = state.provenance.get('fit_role')
        if fit_role in _roles.NEVER_FIT_ROLES or fit_role not in self.allowed_roles:
            raise _roles.RoleError(f"calibrator state was fitted on role {fit_role!r}; allowed: {list(self.allowed_roles)} "
                                   f"(never: {list(_roles.NEVER_FIT_ROLES)})")

    def transform(self, state: CalibrationState, data: pd.DataFrame, *, group_by: str = 'target') -> pd.DataFrame:
        """نسخة من ``data`` بأعمدة المعايرة (confidence_cal، calib_raw_p، calib_correct، calib_signal)؛ الخام بلا مساس."""
        self._check_state(state)
        return _apply_confidence_groups(data, state, group_by=group_by)

    def dump_state(self, state: CalibrationState) -> Dict:
        """الحالة بصيغة JSON-آمنة (بلا دوال): العتبات/المعاملات بدل ``predict``، مع المصدر. ``load_state`` يعكسها."""
        self._check_state(state)
        groups = {}
        for g, e in state.items():
            groups[str(g)] = {k: v for k, v in e.items() if k != 'predict'}
            groups[str(g)]['params'] = e['predict'].params
        return {'kind': 'confidence_calibrator', 'method': state.method, 'provenance': dict(state.provenance), 'groups': groups}

    def load_state(self, d: Dict) -> CalibrationState:
        """يعيد بناء الحالة من ``dump_state``؛ الدور المسجَّل يُفحص من جديد (حالة من train/holdout تُرفض)."""
        if d.get('kind') != 'confidence_calibrator' or d.get('method') != self.name:
            raise ValueError(f"not a {self.name!r} confidence_calibrator state: kind={d.get('kind')!r} method={d.get('method')!r}")
        entries = {}
        for g, e in d['groups'].items():
            e = dict(e)
            e['predict'] = _predict_from_params(e.pop('params'))
            entries[g] = e
        state = CalibrationState(entries, method=d['method'], provenance=d['provenance'])
        self._check_state(state)
        return state


def _check_calibrator(name, obj, meta):
    """عقد التسجيل: الاسم = obj.name، fit/transform/dump_state/load_state قابلة للنداء، وأدوار الملاءمة ⊂ ROLES بلا train/holdout."""
    problems = []
    if getattr(obj, 'name', None) != name:
        problems.append(f"obj.name {getattr(obj, 'name', None)!r} must equal the registered name {name!r}")
    for attr in ('fit', 'transform', 'dump_state', 'load_state'):
        if not callable(getattr(obj, attr, None)):
            problems.append(f"missing callable {attr}")
    allowed = tuple(getattr(obj, 'allowed_roles', ()) or ())
    if not allowed:
        problems.append("allowed_roles is empty: a calibrator declares the roles it may be fitted on")
    bad = [r for r in allowed if r not in _roles.ROLES]
    if bad:
        problems.append(f"allowed_roles {bad} are not declared roles {list(_roles.ROLES)}")
    forbidden = [r for r in allowed if r in _roles.NEVER_FIT_ROLES]
    if forbidden:
        problems.append(f"allowed_roles {forbidden}: a calibrator is never fitted on {list(_roles.NEVER_FIT_ROLES)}")
    return problems


#: السجلّ: اسم الطريقة ← معايِرها. إضافة طريقة = ConfidenceCalibrator جديد (أو صنف بالواجهة نفسها) + CALIBRATORS.register(...)؛ اسم
#: مجهول يرفع UnknownNameError (ValueError)، وتكرار الاسم يرفع DuplicateNameError.
CALIBRATORS = _Registry("calibrator", _Contract("calibrator", optional_meta=(), validator=_check_calibrator))
for _m in ('isotonic', 'platt'):
    CALIBRATORS.register(_m, ConfidenceCalibrator(_m), version=1)


def fit_confidence_calibrators(val_df: pd.DataFrame, method: str = 'isotonic', group_by: str = 'target',
                               min_n: int = 50, verbose: bool = True) -> CalibrationState:
    """غلاف على ``CALIBRATORS[method].fit``: ``val_df`` يجب أن يكون مختوماً بدور مسموح (``core.roles.stamp_role(val_flat, 'val')``)؛
    جدول غير مختوم أو دوره train/holdout/test يرفع RoleError. الناتج قاموس المعايرات كما كان + ``.provenance``."""
    return CALIBRATORS[method].fit(val_df, group_by=group_by, min_n=min_n, verbose=verbose)


def apply_confidence_calibration(df: pd.DataFrame, calibrators: CalibrationState, group_by: str = 'target') -> pd.DataFrame:
    """غلاف على ``CALIBRATORS[state.method].transform``: حالة بلا مصدر (قاموس خام) أو مُلائمة على دور غير مسموح ترفعان RoleError."""
    method = getattr(calibrators, 'method', None)
    if method is None:
        raise _roles.RoleError("calibrators has no .method/.provenance: pass the CalibrationState returned by fit_confidence_calibrators")
    return CALIBRATORS[method].transform(calibrators, df, group_by=group_by)


def stamp_calibration_table(table: pd.DataFrame, source, role: Optional[str] = None) -> pd.DataFrame:
    """يختم الجدول المسطّح ``table`` (المبنيّ من ``source`` = val_dict أو أي قسم) بدور بياناته قبل الملاءمة: ``role`` الصريح، أو دور
    ``source`` المختوم (core.roles.role_of). كلاهما غائب ← RoleError (لا دور افتراضي)؛ تعارضهما ← RoleError. هذا المكان الوحيد
    الذي يقرر دور جدول المعايرة، فلا يُعيد مستهلك (run_full_analysis…) اشتقاقه."""
    try:
        stamped = _roles.role_of(source)
    except _roles.RoleError:
        stamped = None
    if role is None and stamped is None:
        raise _roles.RoleError("calibration data has no role: pass calibration_role='val' | 'calibration' or a val_dict built from a "
                               "split stamped by data.split_data (core.roles.stamp_role); an unstamped table is never fitted on")
    if role is not None and stamped is not None and role != stamped:
        raise _roles.RoleError(f"calibration_role={role!r} contradicts the stamp {stamped!r} on the data it was built from")
    return _roles.stamp_role(table, role if role is not None else stamped)
