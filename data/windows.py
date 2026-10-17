"""
PURPOSE:  Windows and targets for one asset (prepare_single_asset), LAST_COLUMNS/last_candles, sample filters, invert_reg_predictions and decode_price_window.
TAGS:     windows, targets, prepare_single_asset, last_candles, LAST_COLUMNS, TS_COL, sample_filters, register_sample_filter, invert_reg_predictions, decode_price_window, entry_feature_table, reg_target_mode
PITFALLS: LAST_COLUMNS, TS_COL, LAST_DTYPE and TARGET_COLUMNS are defined once in core/schema.py and only re-bound here. last_candles is float64 (ns timestamps lose ~12 s in float32). Sample filters must be causal. Executed into the one shared pipeline namespace by data/_loader.py (never imported on its own): names from other modules resolve at call time.

## 11) بناء النوافذ والأهداف لأصل واحد (`windows.py` سابقاً)
"""
# @title
"""
بناء النوافذ والأهداف لأصل واحد.

لكل نافذة يُنتَج لكل هدف سعري مُفعَّل رأسان:
  * ``{t}_class`` — اتجاه 1/0 (1 إن تجاوزت القيمة المستقبلية قيمتها الحالية، وإلا 0)
  * ``{t}_reg``   — القيمة المستقبلية نفسها بعد التطبيع (تُفكّ لاحقاً بـ ``* iqr + median``)

الرؤوس المُعطَّلة في ``CONFIG['enabled_heads']`` **لا تُحسب أصلاً**.
"""

# أعمدة ``last_candles`` (LAST_COLUMNS) وفهرس الطابع الزمني (TS_COL) وdtype الصفيف (LAST_DTYPE: float64 لأن الطابع بالنانوثانية
# ~1.7e18 يفقد ~12 ثانية في float32) وأعمدة أسعار الأهداف (TARGET_COLUMNS: تبقى في البيانات حتى لو استُبعدت من مدخلات النموذج)
# مصدرها الوحيد core/schema.py؛ تُربط هنا بالاسم نفسه فيبقى نطاق خط الأنابيب (والدفاتر والاختبارات) يراها كما كانت.
from core.schema import LAST_COLUMNS, LAST_DTYPE, TARGET_COLUMNS, TS_COL


# ══════════════════════════════════════════════════════════════════════════
# فلاتر العيّنات (CONFIG['sample_filters']) — أيّ العيّنات تدخل مجموعة البيانات
# ══════════════════════════════════════════════════════════════════════════
#: دوال شروط مسجّلة بالاسم لـ ``{"fn": "<الاسم>"}``: كل دالة تأخذ إطار الفريم الأساسي الخام (كل الأعمدة المحسوبة، قبل
#: التطبيع) وتُرجع قناعاً منطقياً بطول صفوفه — True = الشمعة تصلح شمعة دخول. ⚠️ يجب أن تكون سببية: قيمة الصف t من الصفوف
#: ≤ t فقط (لا ``shift(-1)``، لا ``rolling(center=True)``) — وإلا تسرّب المستقبل إلى اختيار العيّنات.
SAMPLE_FILTER_REGISTRY: Dict[str, Callable[[pd.DataFrame], Any]] = {}

#: عمليات المقارنة في ``{"feature": ..., "op": ..., "value": ...}``.
SAMPLE_FILTER_OPS: Dict[str, Callable[[np.ndarray, Any], np.ndarray]] = {
    '>': lambda x, v: x > v, '>=': lambda x, v: x >= v,
    '<': lambda x, v: x < v, '<=': lambda x, v: x <= v,
    '==': lambda x, v: x == v, '!=': lambda x, v: x != v,
    'between': lambda x, v: (x >= v[0]) & (x <= v[1]),        # حدّان شاملان [أدنى، أعلى]
}


def register_sample_filter(name: str, fn: Callable[[pd.DataFrame], Any]) -> None:
    """يسجّل دالة شرط تُستعمل في الإعداد بـ ``{"fn": name}`` (انظر :data:`SAMPLE_FILTER_REGISTRY`)."""
    if not callable(fn):
        raise TypeError(f"register_sample_filter: {name!r} ليست دالة")
    SAMPLE_FILTER_REGISTRY[name] = fn


def _filter_entry_mask(df: pd.DataFrame, flt: dict) -> np.ndarray:
    """قناع شرط واحد على كل صفوف ``df``؛ NaN = لا يتحقّق الشرط."""
    if not isinstance(flt, dict):
        raise TypeError(f"فلتر عيّنات يجب أن يكون قاموساً، لا {flt!r}")
    kinds = [k for k in ('feature', 'expr', 'fn') if k in flt]
    allowed = {'feature': {'feature', 'op', 'value'}, 'expr': {'expr'}, 'fn': {'fn'}}
    if len(kinds) != 1 or set(flt) - allowed[kinds[0]]:
        raise ValueError(f"فلتر عيّنات غير صالح {flt!r} — أحد الأشكال: "
                         "{'feature', 'op', 'value'} | {'expr'} | {'fn'}")
    kind = kinds[0]
    if kind == 'feature':
        col, op = flt['feature'], flt.get('op')
        if col not in df.columns:
            raise ValueError(f"فلتر العيّنات: العمود {col!r} غير محسوب في الفريم الأساسي — المتاح: {sorted(df.columns)}")
        if op not in SAMPLE_FILTER_OPS:
            raise ValueError(f"فلتر العيّنات: op {op!r} غير معروف — المتاح: {list(SAMPLE_FILTER_OPS)}")
        if 'value' not in flt:
            raise ValueError(f"فلتر العيّنات {flt!r} بلا 'value'")
        x = df[col].to_numpy(dtype='float64')
        with np.errstate(invalid='ignore'):
            out = SAMPLE_FILTER_OPS[op](x, flt['value'])
    elif kind == 'expr':
        out = df.eval(flt['expr'])
    else:
        name = flt['fn']
        if name not in SAMPLE_FILTER_REGISTRY:
            raise ValueError(f"فلتر العيّنات: الدالة {name!r} غير مسجّلة — register_sample_filter أولاً. "
                             f"المسجّل: {sorted(SAMPLE_FILTER_REGISTRY)}")
        out = SAMPLE_FILTER_REGISTRY[name](df)
    out = np.asarray(out)
    if out.shape != (len(df),):
        raise ValueError(f"فلتر العيّنات {flt!r} أعاد شكلاً {out.shape} بدل ({len(df)},)")
    if out.dtype != bool:
        out = pd.Series(out).fillna(False).to_numpy().astype(bool)
    return out


def sample_filter_mask(df: pd.DataFrame, filters: Optional[List[dict]] = None,
                       config: Optional[dict] = None) -> np.ndarray:
    """قناع ``(len(df),)``: True للصفوف التي تحقّق **كل** شروط ``filters`` (افتراضياً ``config['sample_filters']``).

    يُقيَّم على إطار الفريم الأساسي الخام (القيم الفعلية لا المُطبَّعة) — والعمود يكفي أن يكون محسوباً فيه، حتى لو استُبعد من
    مدخلات النموذج عبر ``exclude_from_features`` (مثل ``ADX_14``). لا فلاتر = كل الصفوف True.

    >>> sample_filter_mask(df, [{"feature": "NATR_14", "op": ">", "value": 1.0},   # ATR% لشمعة الدخول > 1%
    ...                         {"feature": "ADX_14", "op": ">", "value": 20}])
    >>> sample_filter_mask(df, [{"expr": "NATR_14 > 1 and (ADX_14 > 20 or RSI_14 < 30)"}])
    """
    config = CONFIG if config is None else config
    filters = (config.get('sample_filters') or []) if filters is None else filters
    mask = np.ones(len(df), dtype=bool)
    for flt in filters:
        mask &= _filter_entry_mask(df, flt)
    return mask


def _sample_filters_signature(filters: List[dict]) -> List[dict]:
    """الفلاتر كما تدخل البصمة: دوال ``fn`` المسجّلة بمصدرها (تعديل جسم الدالة يُبطل نقاط الاستئناف)."""
    import hashlib
    import inspect
    out = []
    for flt in filters:
        flt = dict(flt)
        fn = SAMPLE_FILTER_REGISTRY.get(flt.get('fn')) if 'fn' in flt else None
        if fn is not None:
            try:
                flt['fn_source'] = hashlib.sha256(inspect.getsource(fn).encode('utf-8')).hexdigest()[:16]
            except (OSError, TypeError):
                flt['fn_source'] = None
        out.append(flt)
    return out


def _window_scale_params_batch(close_windows: np.ndarray, method: str = "robust") -> Tuple[np.ndarray, np.ndarray]:
    """النسخة المتّجهة من :func:`calc_scale_params` على ``(N, T)`` نافذة إغلاق ← ``(المركز، المقياس)`` لكل نافذة.
    نفس الحدّين (المطلق والنسبي) والاستيفاء، فتطابق الحلقة على calc_scale_params عددياً (يُثبته اختبار ذاتي)."""
    x = np.asarray(close_windows)
    if method == 'robust':
        q75, q50, q25 = np.percentile(x, [75, 50, 25], axis=1)
        return q50, np.maximum(np.maximum(q75 - q25, np.abs(q50) * SCALE_REL_FLOOR), SCALE_ABS_FLOOR)
    lo, hi = x.min(axis=1), x.max(axis=1)
    return lo, np.maximum(np.maximum(hi - lo, np.abs(lo) * SCALE_REL_FLOOR), SCALE_ABS_FLOOR)


def prepare_single_asset(
    dfs: Dict[str, pd.DataFrame],
    tf_order: Optional[List[str]] = None,
    window_sizes: Optional[Dict[str, int]] = None,
    targets: Optional[List[str]] = None,
    forecast_horizon: Optional[int] = None,
    stride: Optional[int] = None,
    scaler_type: Optional[str] = None,
    base_tf: Optional[str] = None,
    config: Optional[dict] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """يُرجع ``(X_tf, y_heads, base_params, last_candles)`` لأصل واحد.

    * ``X_tf``       — ``{tf: (N, T, F)}`` نوافذ مُطبَّعة.
    * ``y_heads``    — ``{head: (N,)}`` لكل رأس مُفعَّل.
    * ``base_params``— ``(N, 2)`` [المركز، المقياس] لنافذة الإغلاق (وسيط/IQR لـ robust) —
      لتطبيع الميزات فقط؛ لا مرجع لأي هدف.
    * ``last_candles``— ``(N, 7)`` انظر :data:`LAST_COLUMNS`.
    """
    config = CONFIG if config is None else config
    tf_order = config["tf_order"] if tf_order is None else tf_order
    window_sizes = config["window_sizes"] if window_sizes is None else window_sizes
    targets = config["targets"] if targets is None else targets
    forecast_horizon = config["forecast_horizon"] if forecast_horizon is None else forecast_horizon
    stride = config["stride"] if stride is None else stride
    scaler_type = config["scaler_type"] if scaler_type is None else scaler_type
    base_tf = base_tf or config.get("base_tf") or tf_order[0]
    features = feature_order(config)

    # ✅ فصل «أعمدة الأهداف» عن «أعمدة الإدخال»:
    #    الأهداف تحتاج high/low/close الخام دائماً، أمّا النموذج فقد لا يحتاجها
    #    كمدخلات (قيست 0.99 ترابطاً بينها بعد التطبيع — معلومتها الحقيقية هي
    #    الفرق بينها، وتلتقطه RANGE_rel و BODY_ratio و WICK_*).
    #    بهذا الفصل صار بالإمكان استبعادها عبر exclude_from_features بلا كسر
    #    حساب الأهداف، وهو ما كان مستحيلاً حين كان المصدر واحداً.
    dfs = {tf: df.sort_index() for tf, df in dfs.items()}
    price_df = dfs[base_tf]
    for col in ('high', 'low', 'close'):
        if col not in price_df.columns:
            raise ValueError(
                f"العمود '{col}' غير موجود في بيانات الفريم الأساسي '{base_tf}' — "
                "لا يمكن حساب الأهداف بدونه (وهو مطلوب حتى لو استُبعد من الميزات).")

    feat_dfs = {tf: dfs[tf][features] for tf in tf_order}
    # فلاتر العيّنات (CONFIG['sample_filters']): قناع شمعة الدخول على الإطار الخام للفريم الأساسي، مرة واحدة لكل عملة.
    entry_ok = sample_filter_mask(price_df, config=config)

    windows, end_times = align_multi_timeframes_time_based(
        dfs=feat_dfs, tf_order=tf_order, window_sizes=window_sizes,
        stride=stride, config=config,
    )

    N = windows[base_tf].shape[0]
    if N == 0:
        print("   ⚠️ لا توجد نوافذ صالحة بعد المحاذاة الزمنية")
        return ({}, {},
                np.empty((0, 2), dtype='float32'),
                np.empty((0, len(LAST_COLUMNS)), dtype=LAST_DTYPE))

    # مقياس النافذة يُشتق من عمود الإغلاق داخل الميزات إن وُجد، وإلا من الأسعار
    # الخام مباشرةً — فيبقى المرجع السعري واحداً في الحالتين.
    i_close_feat = features.index('close') if 'close' in features else None

    target_heads = get_target_heads(targets, config)
    y_t = {h: np.zeros(N, dtype='float32') for h in target_heads}
    bases = np.zeros((N, 2), dtype='float32')
    last = np.zeros((N, len(LAST_COLUMNS)), dtype=LAST_DTYPE)
    reg_mode = config.get('reg_target_mode', 'return')
    _reg_clip_cfg = config.get('reg_target_clip')
    reg_clip = float(_reg_clip_cfg) if _reg_clip_cfg is not None else (
        1.0 if reg_mode == 'return' else 10.0)
    # يُضرب بعد القصّ (القصّ بوحدة الهدف الأصلية) — راجع CONFIG['reg_target_scale']
    reg_scale = float(config.get('reg_target_scale', 1.0))

    T_base = len(price_df)
    win_base = window_sizes[base_tf]
    raw_high = price_df['high'].values
    raw_low = price_df['low'].values
    raw_close = price_df['close'].values
    n_valid = N
    # ✅ الهدف يُبنى من الشموع التي تبدأ فعلاً عند ts + 1..h × مدة الفريم، لا من «الصف التالي»: بعد فجوة (ساعات ناقصة
    #    من المصدر، أو صفوف حذفها dropna في add_features) كان الصف التالي شمعةً أبعد بساعات، فيختلف أفق هذه العيّنة عن
    #    أقرانها في نفس الطابع بصمت. عيّنة كهذه تُحذف (keep=False) — لا تُصحَّح، فلا شمعة حقيقية لهدفها.
    bar = pd.Timedelta(str(base_tf))
    base_index = price_df.index
    keep = np.ones(N, dtype=bool)

    for i in range(N):
        base_win_arr = windows[base_tf][i]

        end_time = end_times[i]
        try:
            end_idx = price_df.index.get_loc(end_time)
        except KeyError:
            end_idx = price_df.index.searchsorted(end_time, side='right') - 1
            end_idx = max(0, min(T_base - 1, end_idx))

        # fs/fe تُحسب مرة واحدة وتُفحص قبل أي كتابة على المصفوفات
        fs = end_idx + 1
        fe = end_idx + 1 + forecast_horizon
        if fe > T_base:
            n_valid = i
            break
        if base_index[fe - 1] - base_index[end_idx] != forecast_horizon * bar:
            keep[i] = False
            continue
        # ✅ ونافذة الإدخال أيضاً: win_base شمعة متتالية بالضبط (ts − (win−1) × مدة الفريم … ts). النوافذ تُقتطع بموضع
        #    الصف، فبعد فجوة كانت أول نوافذ بعدها تبدأ قبلها (قيس: 53 ساعة بدل 31) بينما أقرانها في الطابع نفسه 31 ساعة.
        #    الدليل: docs/research/audit/r2_02_window_spans_data_hole.py.
        if end_idx < win_base - 1 or base_index[end_idx] - base_index[end_idx - win_base + 1] != (win_base - 1) * bar:
            keep[i] = False
            continue
        if not entry_ok[end_idx]:            # شمعة الدخول لا تحقّق sample_filters
            keep[i] = False
            continue

        # ✅ قيم الشمعة الأخيرة تُقرأ من الأسعار الخام لا من مصفوفة الميزات
        last_h = raw_high[end_idx]
        last_l = raw_low[end_idx]
        last_c = raw_close[end_idx]

        ts = price_df.index[end_idx].value
        future_close = raw_close[fe - 1]
        future_high_max = raw_high[fs:fe].max()
        future_low_min = raw_low[fs:fe].min()
        last[i] = [last_h, last_l, last_c, ts,
                   future_close, future_low_min, future_high_max]

        # مقياس النافذة من الإغلاق: من الميزات إن كان موجوداً، وإلا من الخام
        if i_close_feat is not None:
            close_win = base_win_arr[:, i_close_feat]
        else:
            close_win = raw_close[max(0, end_idx - win_base + 1):end_idx + 1]
        med_p, iqr_p = calc_scale_params(close_win, scaler_type)
        # med_p=last_c
        # المركز يبقى وسيط النافذة: يخدم تطبيع الميزات وحده، ولا يُقرأ مرجعاً لأي
        # هدف (راجع 'window_scale' أدناه) — فلا يكشف موقعُ آخر سعر منه الاتجاهَ.
        bases[i] = [med_p, iqr_p]     # لتطبيع الميزات فقط الآن — راجع reg_mode أدناه لهدف الانحدار

        # كل هدف يُقارَن بنفس نوعه (high بـ high، low بـ low، close بـ close) —
        # لرأس الانحدار أيضاً في الوضعين ('return' و 'window_scale')؛ سابقاً كان reg
        # يُقارَن دائماً بآخر *إغلاق* (med_p) بصرف النظر عن نوع الهدف، فيختلف
        # مرجعه عن مرجع class لنفس الهدف على high/low — يُضعف إشارة
        # head_agreement (راجع abstention_cfg) بلا داعٍ. الآن يتّفقان دائماً.
        own_last = {'high': last_h, 'low': last_l, 'close': last_c}
        own_future = {'high': future_high_max, 'low': future_low_min,
                      'close': future_close}

        for t in targets:
            if t not in own_last:
                continue
            # ✅ الرؤوس المُعطَّلة تُتخطّى بدل محاولة الكتابة في مفتاح غير موجود
            if f'{t}_class' in y_t:
                # ترميز 1/0 لا ±1: هو ما تنتظره binary_crossentropy مباشرة فلا يحتاج
                # المستهلك تحويلاً. التعادل (مستقبلي == حالي) = 0 كما كان -1 سابقاً.
                # المستهلكون يقرؤون الصعود كـ y > 0، فيقبلون الترميزين معاً.
                y_t[f'{t}_class'][i] = 1.0 if own_future[t] > own_last[t] else 0.0
            if f'{t}_reg' in y_t:
                if reg_mode == 'return':
                    # ✅ عائد مباشر بلا مرجع خارجي: (سعر_مستقبلي/آخر_سعر) - 1،
                    # بنفس مرجع رأس التصنيف أعلاه. بلا حاجة لـ iqr_p إطلاقاً —
                    # نافذتان بتقلّب مختلف تُنتجان نفس القيمة لنفس نسبة الحركة.
                    # في الوضع القديم (أدناه) تُقسَم نفس الحركة على IQR النافذة
                    # فتكبر في نافذة هادئة وتصغر في نافذة متقلّبة — انزياح مرجع
                    # الهدف هذا بين نافذة وأخرى هو ما كان يدفع النموذج نحو
                    # الانكماش للمتوسط (أرخص حلّ حين يتذبذب معنى الصفر).
                    denom = own_last[t] if abs(own_last[t]) > EPS else (
                        med_p if abs(med_p) > EPS else EPS)
                    raw = (own_future[t] - own_last[t]) / denom
                else:                                   # 'window_scale' القديم
                    # ✅ قصّ صريح لهدف الانحدار: حتى مع الحدّ النسبي في
                    # calc_scale_params، نافذة واحدة شاذة (سكون سعري مؤقت + حركة
                    # قوية لاحقة) تكفي لإنتاج هدف بمئات الأضعاف — قيس فعلياً
                    # ([-41, +728] بدل [-3, +3] المتوقَّع).
                    # ✅ المركز = آخر سعر من نفس نوع الهدف، لا وسيط النافذة: مع
                    # الوسيط كان اتجاه الهدف معروفاً جزئياً وقت الدخول من موقع آخر
                    # سعر نسبةً إليه — قاعدة ساذجة (آخر_إغلاق > المركز) أصابت
                    # ~92%/86%/78% على high/close/low بلا أي تنبؤ. متغيّر محلي لا
                    # med_p، فلا يتغيّر مرجع تطبيع الميزات ولا الأهداف التالية.
                    center = own_last[t]
                    raw = scale_data(own_future[t], center, iqr_p)
                # ضمان أخير بغضّ النظر عن الوضع: خسارة محدودة دائماً، فلا تُفسد
                # عيّنة واحدة تدرّج الجذع المشترك لكل الرؤوس.
                y_t[f'{t}_reg'][i] = np.clip(raw, -reg_clip, reg_clip) * reg_scale

    keep[n_valid:] = False
    if not keep.all():
        for h in target_heads:
            y_t[h] = y_t[h][keep]
        bases = bases[keep]
        last = last[keep]
        for tf in tf_order:
            windows[tf] = windows[tf][keep]

    # ✅ التطبيع دفعةً واحدة لكل الفريمات: النسخة المتّجهة تُلغي حلقة بايثون
    #    المزدوجة (N نافذة × F عمود) التي كانت تستهلك أغلب زمن التجهيز.
    #    higher_tf_mode='closed': الفريمات الأعلى تُطبَّع بمركز/مقياس إغلاق نافذتها هي (نافذة 4h تمتدّ 128 ساعة فتخرج
    #    أسعارها من مقياس نافذة الفريم الأساسي وتُقصّ عند CLIP_ABS)، وbases تبقى للفريم الأساسي وحده.
    per_tf_norm = _higher_tf_closed(dict(config, tf_order=list(tf_order))) and i_close_feat is not None
    store_dtype = config.get('x_storage_dtype')
    X_tf = {}
    for tf in tf_order:
        if per_tf_norm and tf != base_tf:
            centers, scales = _window_scale_params_batch(windows[tf][:, :, i_close_feat], scaler_type)
        else:
            centers, scales = bases[:, 0], bases[:, 1]
        X_tf[tf] = process_windows(windows[tf], features, centers, scales, scaler_type, config=config)
        if store_dtype:                     # قبل الانتقال للفريم التالي: لا يتراكم float32 كامل لكل الفريمات
            X_tf[tf] = X_tf[tf].astype(store_dtype)
        windows[tf] = None                  # النافذة الخام لم تعد لازمة

    gc.collect()
    return X_tf, y_t, bases, last


def invert_reg_predictions(preds: np.ndarray, head: str,
                           last_candles: Optional[np.ndarray] = None,
                           bases: Optional[np.ndarray] = None,
                           config: Optional[dict] = None,
                           scale: Optional[float] = None) -> np.ndarray:
    """يعكس تنبؤات رأس انحدار واحد إلى سعر حقيقي، حسب ``CONFIG['reg_target_mode']``.

    ``scale``: مُعامل ``reg_target_scale`` الذي بُنيت به البيانات — يُقسَم عليه أولاً. مرّره من البيانات نفسها
    (``dataset.get('reg_target_scale', 1.0)``)؛ ``None`` يقرأ ``config['reg_target_scale']`` (افتراضياً 1.0).

    * ``'return'`` (الافتراضي): يحتاج ``last_candles`` — ``آخر_سعر_من_نفس_نوع_
      الهدف × (1 + التنبؤ)``. يُقرأ آخر سعر من عمود :data:`LAST_COLUMNS`
      المطابق (high/low/close) — لا من ``bases``، التي تخزّن مركز نافذة
      *الإغلاق* (وسيطها) بصرف النظر عن نوع الهدف (تخدم تطبيع الميزات فقط؛ راجع
      :func:`prepare_single_asset`).
    * ``'window_scale'`` (القديم): يحتاج ``last_candles`` و ``bases`` —
      ``آخر_سعر_من_نفس_النوع + التنبؤ × IQR``: نفس مركز بناء الهدف، والمقياس
      وحده من ``bases[:, 1]``. ``inverse_scale(preds, bases)`` لم يعد صالحاً هنا:
      ``bases[:, 0]`` وسيط النافذة لا مركز الهدف.

    ⚠️ **لا يعكس** تطبيعاً مقطعياً طُبِّق عبر :func:`cross_sectional_normalize`؛
    ذاك تحويل نسبي لأقران نفس اللحظة، وعكسه (لـ ``method='zscore'`` فقط) عبر
    :func:`invert_cross_sectional` باستخدام ``dataset['cs_norm_stats']``، **قبل**
    استدعاء هذه الدالة لا بعدها.
    """
    config = CONFIG if config is None else config
    target, kind = split_head_name(head)
    if kind != 'reg':
        raise ValueError(f"'{head}' ليس رأس انحدار (reg).")
    mode = config.get('reg_target_mode', 'return')

    if mode == 'window_scale' and bases is None:
        raise ValueError("mode='window_scale' يتطلب bases (median, iqr) — للـ IQR.")
    if last_candles is None:
        raise ValueError(f"mode='{mode}' يتطلب last_candles (راجع LAST_COLUMNS).")
    col = {'high': 'last_high', 'low': 'last_low', 'close': 'last_close'}.get(target)
    if col is None:
        raise ValueError(f"لا مرجع سعر معروف للهدف '{target}'.")
    last_price = np.asarray(last_candles, dtype='float64')[:, LAST_COLUMNS.index(col)]
    scale = float(config.get('reg_target_scale', 1.0) if scale is None else scale)
    preds = np.asarray(preds, dtype='float64') / scale
    if mode == 'window_scale':
        return last_price + preds * np.asarray(bases, dtype='float64')[:, 1]
    return last_price * (1.0 + preds)


def decode_price_window(data: Dict, column: str = 'close', tf: Optional[str] = None,
                        anchor: Optional[np.ndarray] = None, feature_order: Optional[List[str]] = None,
                        mode: Optional[str] = None) -> np.ndarray:
    """يفكّ عموداً سعرياً من X المُطبَّعة بوضع ``'pct_change'`` إلى أسعاره الحقيقية ``(N, T)``.

    المرساة آخر سعر حقيقي في كل نافذة: لـ high/low/close على الفريم الأساسي تُقرأ من ``last_candles``
    (``last_high``/``last_low``/``last_close``)، ولغيرها (EMA، open، فريم أعلى) تُمرَّر ``anchor`` ``(N,)`` صراحةً —
    آخر شمعة 4h بوضع 'closed' ليست آخر شمعة 1h. ثم :func:`pct_change_decode` للخلف من المرساة.

    ``data``: مجموعة البيانات كاملة، أو قسم من ``split_data`` مع ``feature_order``/``mode`` صراحةً (الأقسام لا
    تحملهما). يرفض ``ValueError`` أي وضع غير ``'pct_change'`` (الوضع القديم مقصوص ±clip_abs فلا يُفكّ بدقّة).
    """
    mode = mode if mode is not None else data.get('price_norm_mode')
    if mode != PRICE_NORM_PCT:
        raise ValueError(f"decode_price_window يتطلّب X مبنيّة بـ price_norm_mode='pct_change' — لا {mode!r} "
                         "(قسم من split_data: مرّر mode=dataset['price_norm_mode'] صراحةً)")
    features = list(feature_order if feature_order is not None else data['feature_order'])
    if column not in features:
        raise ValueError(f"العمود {column!r} ليس من ميزات X")
    if classify_feature(column) != PRICE_LEVEL:
        raise ValueError(f"العمود {column!r} نوعه {classify_feature(column)!r} لا price_level — لم يُرمَّز تغيّراً نسبياً")
    base_tf = data.get('base_tf') or (data.get('timeframes') or [None])[0]
    tf = tf or base_tf
    r = np.asarray(data[f'X_{tf}'][:, :, features.index(column)], dtype='float64')
    if anchor is None:
        if column not in ('high', 'low', 'close') or (base_tf is not None and tf != base_tf):
            raise ValueError(f"لا مرساة مخزَّنة لـ {column!r} على {tf!r} — مرّر anchor (آخر قيمة حقيقية لكل نافذة)")
        anchor = np.asarray(data['last_candles'], dtype='float64')[:, LAST_COLUMNS.index(f'last_{column}')]
    return pct_change_decode(r, anchor, anchor_at='last', axis=1)


def entry_feature_table(data: Dict, columns: List[str], load_asset_fn: Optional[Callable] = None,
                        config: Optional[dict] = None) -> pd.DataFrame:
    """قيم خام (قبل التطبيع) لأعمدة ``columns`` عند شمعة دخول كل عيّنة: جدول ``asset, ts, <columns>``.

    للتقييم بعد التدريب بفلاتر لم تُطبَّق وقت البناء (``tools/bracket_eval.py``: NATR_14 > 1، ADX_14 > 20، اتجاه SuperTrend…).
    يعيد حساب الفريم الأساسي لكل عملة بنفس ``resample_timeframes(config)`` ثم يقرأ الأعمدة عند طوابع العيّنات — فيصلح
    لأي عمود يحسبه ``add_features`` (حتى المستبعد من المدخلات)، لا للأعمدة العابرة للأصول (MKT_*) التي تحتاج الأقران.

    ``data``: مجموعة البيانات كاملة (``asset_bounds``). ``load_asset_fn(file_id, name)`` افتراضياً ``load_asset``.
    """
    config = CONFIG if config is None else config
    load_asset_fn = load_asset_fn or (lambda fid, name: load_asset(fid, name, config=config))
    base_tf = data.get('base_tf') or config['base_tf']
    ts_all = pd.to_datetime(np.asarray(data['last_candles'])[:, TS_COL].astype('int64'), utc=True)
    parts = []
    for b in data['asset_bounds']:
        frame = resample_timeframes(load_asset_fn(None, b['name']), [base_tf], config=config)[base_tf]
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise ValueError(f"entry_feature_table: {missing} غير محسوبة لـ {b['name']} على {base_tf} — "
                             f"المتاح: {sorted(frame.columns)}")
        ts = ts_all[b['start']:b['end']]
        part = frame[columns].reindex(ts).reset_index(drop=True)
        part.insert(0, 'ts', ts)
        part.insert(0, 'asset', b['name'])
        parts.append(part)
    return pd.concat(parts, ignore_index=True)
