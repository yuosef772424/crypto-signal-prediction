"""
PURPOSE:  Time-based train/val/test split with an embargo gap: global_time (default) or per_asset, holdout, leak-free and rolling splits.
TAGS:     split, split_data, embargo, holdout, split_dates, global_time, per_asset, rolling_splits, build_leak_free_split, compute_global_cutoff
PITFALLS: per_asset leaks when coins' histories differ in length; global_time is the default. The embargo is a time gap covering the largest window plus the horizon. Executed into the one shared pipeline namespace by data/_loader.py (never imported on its own): names from other modules resolve at call time.

## 16) التقسيم الزمني train/val/test (`split.py` سابقاً)
"""
# @title
"""
تقسيم زمني (بدون خلط) مع فجوة عزل (embargo).

وضعان:

``global_time`` (الافتراضي، المُوصى به)
    حدود زمنية **مطلقة مشتركة** بين كل العملات: كل عيّنة تاريخها قبل
    ``train_end`` تدخل التدريب، ومهما اختلفت أطوال تواريخ العملات يبقى
    ``test`` لاحقاً زمنياً لكل ما رآه النموذج.

``per_asset`` (السلوك القديم)
    كل عملة تُقسَّم 90/5/5 على مدى تاريخها هي.
    ⚠️ **يُسرِّب** حين تختلف أطوال تواريخ العملات: عملة مُدرجة في 2024 يقع
    تدريبها كله بعد فترة اختبار البيتكوين، فيتدرّب النموذج على مستقبلٍ
    بالنسبة لبعض عيّنات الاختبار. مُتاح للتوافق مع نتائج قديمة فقط.

في الوضعين تُطبَّق فجوة عزل، لأن النوافذ متراكبة بشدة (نافذة كبيرة، stride
صغير): بدونها تتشارك آخر عيّنة تدريب وأول عيّنة تحقق معظم شموعهما.
الفجوة الافتراضية = ``window_size(base_tf) + forecast_horizon`` شمعة.

**حدّا التقسيم في ``global_time`` يُشتقّان من العيّنات المُبقاة لا من كل العيّنات.**
فجوة العزل تُلقي عيّنات (تُهمَل) بين كل قسمين. لو اشتُقّ الحدّ الزمني من نسبة
90/5/5 على *كل* العيّنات، لقصر الزمن المتبقي بعد train عن فجوتين حين تتكدّس
العيّنات في الفترة الأخيرة (عملات كثيرة حديثة الإدراج): فيخرج val و test فارغين
— وهذا ما حدث فعلاً على بيانات ٤٢٢ عملة (train_end قبل النهاية بـ ٦٤ يوماً،
وفجوة العزل ٣٣). الآن يُبحث عن الحدّين بحيث تقترب **نِسَب العيّنات المُبقاة**
من المطلوب، وإن استحال (زمن لا يكفي فجوتين + الحدّ الأدنى للعيّنات) يُرفع
خطأ صريح بدل قسم فارغ. البديل المرفوض: تقليص فجوة العزل حتى يتّسع الزمن — يُعيد
التسرّب الذي وُجدت الفجوة لمنعه.

**holdout مختوم** (``CONFIG['holdout_start']``، تاريخ ثابت): العيّنات من ذلك التاريخ فصاعداً لا تدخل train ولا val
ولا test، فلا تصل إلى أي ملف إشارات يُصدَّر من test. بين آخر test وأول holdout فجوة العزل نفسها. لا تُقرأ إلا عبر
:func:`split_holdout` مع ``CONFIG['open_holdout']=True`` صراحةً (مرة واحدة، عند الإصدار — docs/research/audit/PROTOCOL.md).
التاريخ ثابت لا «آخر N يوماً»: حدّ نسبي إلى نهاية البيانات ينزلق مع كل إعادة جلب، فتدخل أيام كانت مختومة في test.
"""

#: بادئة مفاتيح الأهداف داخل أقسام التدريب — يتوقّعها Keras لمطابقة أسماء المخرجات.
Y_PREFIX = 'y_'

#: أوضاع التقسيم المدعومة.
SPLIT_MODES = ('global_time', 'per_asset')


def embargo_candles(data: Dict, config: Optional[dict] = None) -> int:
    """طول فجوة العزل **بالشموع** — الأساس الذي تُشتق منه بقية الصيغ."""
    config = CONFIG if config is None else config
    candles = config.get('embargo_candles')
    if candles is None:
        base_tf = data.get('base_tf', config['base_tf'])
        sizes = data.get('window_sizes', config['window_sizes'])
        window = sizes[base_tf]
        # فريمات أعلى (closed وlegacy على السواء): نافذة 4h × 32 تمتدّ 128 ساعة (أوسع من نافذة 1h ذات 32)، فالعزل بأوسع امتداد
        # بشموع الفريم الأساسي كي لا يرى مُدخل val/test (بأي فريم) هدفاً من قسم سابق. فريم واحد: window + horizon كما كانت.
        # الطور: أحدث شمعة 4h مغلقة تُغلق عند t - (t mod 4h)، فنافذتها تبدأ حتى (ratio-1) ساعة أبكر من 128h. الشبكة المشتركة
        # (stride مضاعف لمدة الفريم) تثبّت الطور عند 0 فلا زيادة (الإعداد المشحون 1h_4h: 129 كما كان)؛ وإلا تُضاف الزيادة
        # (stride لا يقبل القسمة، أو المحاذاة معطّلة، أو مرساة شبكة غير مضاعفة).
        base_dur = _tf_to_timedelta(base_tf)
        stride = int(data.get('stride', config.get('stride', 1)))
        anchor_ns = pd.Timestamp(config.get('window_grid_anchor', '1970-01-01')).as_unit('ns').value
        for tf in (data.get('timeframes') or config.get('tf_order') or []):
            if tf != base_tf and tf in sizes:
                tf_dur = _tf_to_timedelta(tf)
                ratio = max(int(tf_dur // base_dur), 1)
                span = int(sizes[tf]) * ratio
                phase_fixed = (config.get('align_windows_to_grid', True) and stride > 1 and stride % ratio == 0
                               and anchor_ns % int(tf_dur // pd.Timedelta(1, 'ns')) == 0)
                window = max(window, span if phase_fixed else span + ratio - 1)
        candles = window + data.get('forecast_horizon', config['forecast_horizon'])
    return int(candles)


def embargo_size(data: Dict, config: Optional[dict] = None) -> int:
    """فجوة العزل بعدد **العيّنات** (تُستخدم في وضع ``per_asset``)."""
    config = CONFIG if config is None else config
    stride = max(data.get('stride', config.get('stride', 1)), 1)
    return max(1, int(math.ceil(embargo_candles(data, config) / stride)))


def _tf_to_timedelta(tf: str) -> pd.Timedelta:
    """``'4h'`` → ``Timedelta('4h')``. يقبل صيغ pandas المعتادة (``1D``, ``15T``...)."""
    return pd.Timedelta(tf)


def embargo_duration(data: Dict, config: Optional[dict] = None) -> pd.Timedelta:
    """فجوة العزل كمدة **زمنية** (تُستخدم في وضع ``global_time``)."""
    config = CONFIG if config is None else config
    base_tf = data.get('base_tf', config['base_tf'])
    return embargo_candles(data, config) * _tf_to_timedelta(base_tf)


def resolve_holdout_start(config: Optional[dict] = None) -> Optional[pd.Timestamp]:
    """بداية الـholdout المختوم (UTC) من ``config['holdout_start']``، أو ``None`` إن لم يُحدَّد."""
    config = CONFIG if config is None else config
    hs = config.get('holdout_start')
    if hs is None or hs == '':
        return None
    t = pd.Timestamp(hs)
    return t.tz_convert('UTC') if t.tzinfo is not None else t.tz_localize('UTC')


def _before_holdout(data: Dict, config: dict) -> Dict:
    """نسخة خفيفة من ``data`` (طوابع وبيانات وصفية فقط) بلا العيّنات التي لا يجوز أن تدخل أي قسم: الـholdout
    وفجوة العزل قبله. تُستخدم لاشتقاق حدود التقسيم تلقائياً فلا تُحسَب نِسَب test على عيّنات مختومة."""
    hs = resolve_holdout_start(config)
    if hs is None:
        return data
    lc = np.asarray(data['last_candles'])
    keep = lc[:, TS_COL].astype('int64') < (hs - embargo_duration(data, config)).value
    if keep.all():
        return data
    view = {k: data[k] for k in ('base_tf', 'window_sizes', 'forecast_horizon', 'stride', 'timeframes',
                                 'higher_tf_mode') if k in data}
    view['last_candles'] = lc[keep]
    return view


def sample_timestamps(split_or_data: Dict) -> pd.DatetimeIndex:
    """الطابع الزمني لكل عيّنة، مأخوذاً من عمود ``timestamp`` في ``last_candles``."""
    ts = np.asarray(split_or_data['last_candles'])[:, TS_COL]
    return pd.to_datetime(ts.astype('int64'), utc=True)


def _new_accumulator(timeframes, targets) -> dict:
    return {'X': {tf: [] for tf in timeframes},
            'y': {t: [] for t in targets},
            'base_params': [], 'last_candles': []}


def _append_range(data, s, e, acc, timeframes, targets) -> None:
    if e <= s:
        return
    acc['base_params'].append(data['base_params'][s:e])
    acc['last_candles'].append(data['last_candles'][s:e])
    for tf in timeframes:
        acc['X'][tf].append(data[f'X_{tf}'][s:e])
    for t in targets:
        acc['y'][t].append(data[f'y_{t}'][s:e])


def _finalize_accumulator(acc, data, timeframes, targets) -> dict:
    out = {
        'base_params': (np.concatenate(acc['base_params'], axis=0)
                        if acc['base_params'] else np.empty((0, 2), dtype='float32')),
        'last_candles': (np.concatenate(acc['last_candles'], axis=0)
                         if acc['last_candles']
                         else np.empty((0, data['last_candles'].shape[1]),
                                       dtype=data['last_candles'].dtype)),
    }
    out['y'] = {t: (np.concatenate(acc['y'][t], axis=0) if acc['y'][t]
                    else np.empty((0,), dtype='float32')) for t in targets}
    for tf in timeframes:
        if acc['X'][tf]:
            out[f'X_{tf}'] = np.concatenate(acc['X'][tf], axis=0)
        else:
            shape = data[f'X_{tf}'].shape
            out[f'X_{tf}'] = np.empty((0, shape[1], shape[2]), dtype='float32')
    return out


def _slice(data, s, e, timeframes, targets) -> dict:
    out = {
        'base_params': data['base_params'][s:e],
        'last_candles': data['last_candles'][s:e],
        'y': {t: data[f'y_{t}'][s:e] for t in targets},
    }
    for tf in timeframes:
        out[f'X_{tf}'] = data[f'X_{tf}'][s:e]
    return out


def add_y_prefix(split: dict) -> dict:
    """يُضيف بادئة ``y_`` لمفاتيح الأهداف إن لم تكن موجودة (في المكان).

    Keras يطابق مفاتيح ``y`` بأسماء طبقات المخرجات، وهي مُسمّاة ``y_<head>``.
    """
    y = split.get('y')
    if y and not next(iter(y)).startswith(Y_PREFIX):
        split['y'] = {f'{Y_PREFIX}{t}': arr for t, arr in y.items()}
    return split


def strip_y_prefix(y: dict) -> dict:
    """يُزيل بادئة ``y_`` — يُستخدم في التقييم لمطابقة أسماء الرؤوس."""
    return {(k[len(Y_PREFIX):] if k.startswith(Y_PREFIX) else k): v
            for k, v in y.items()}


_TAKE_COUNTER = [0]


def _take_rows(arr, sel) -> np.ndarray:
    """``arr[sel]`` (قناع منطقي أو فهارس). مصفوفة عادية أو ناتج صغير (< SMALL_ARRAY_BYTES): فهرسة عادية كما كانت.
    **memmap كبير** (بيانات disk_backed): الفهرسة المتقدّمة كانت ستنسخ القسم كله إلى الرام (train ≈ 60% من X)، فيُكتب
    الناتج دفعات في ملف .npy تحت ``<scratch_dir>/splits/`` (القرص المحلي، لا مجلد المصدر: قد يكون على Drive) ويُرجَع memmap للقراءة —
    نفس القيم بالترتيب نفسه، والرام دفعة واحدة. لا تُحذَف تلقائياً (train الحيّ يقرأ منها): احذف scratch_dir بعد الانتهاء."""
    fname = getattr(arr, 'filename', None) if isinstance(arr, np.memmap) else None
    if fname is None:
        return arr[sel]
    idx = np.flatnonzero(sel) if np.asarray(sel).dtype == bool else np.asarray(sel)
    row_bytes = max(1, arr.dtype.itemsize * int(np.prod(arr.shape[1:], dtype=np.int64)))
    if len(idx) * row_bytes <= SMALL_ARRAY_BYTES:
        return np.asarray(arr[idx])
    _TAKE_COUNTER[0] += 1
    d = Path(CONFIG.get('scratch_dir') or _default_scratch_root()) / 'splits'
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"take_{os.getpid()}_{_TAKE_COUNTER[0]}_{Path(fname).stem}.npy"
    out = np.lib.format.open_memmap(path, mode='w+', dtype=arr.dtype, shape=(len(idx),) + arr.shape[1:])
    step = max(1, _COPY_CHUNK_BYTES // row_bytes)
    for s in range(0, len(idx), step):
        out[s:s + step] = arr[idx[s:s + step]]
    out.flush()
    del out
    return np.load(path, mmap_mode='r')


def _take(data: Dict, mask: np.ndarray, timeframes, targets) -> dict:
    """يقتطع عيّنات مجموعة البيانات حسب قناع منطقي (memmap كبير ← ملف على القرص لا نسخة رام: :func:`_take_rows`)."""
    out = {
        'base_params': _take_rows(data['base_params'], mask),
        'last_candles': _take_rows(data['last_candles'], mask),
        'y': {t: _take_rows(data[f'y_{t}'], mask) for t in targets},
    }
    for tf in timeframes:
        out[f'X_{tf}'] = _take_rows(data[f'X_{tf}'], mask)
    return out


def _kept_counts(ordered: np.ndarray, t1, t2, gap: int):
    """أعداد العيّنات **المُبقاة** ``(train, val, test)`` لحدّين ``t1 ≤ t2`` وفجوة ``gap``.

    كلها أعداد صحيحة بالنانوثانية. ``t2`` قد يكون مصفوفة (تقييم عدة حدود معاً).
    القواعد مطابقة لأقنعة ``_split_global_time``:
    ``train: ts ≤ t1`` | ``val: t1+gap < ts ≤ t2`` | ``test: ts > t2+gap``.
    """
    n = len(ordered)
    n_tr = np.searchsorted(ordered, t1, side='right')
    a = np.searchsorted(ordered, t1 + gap, side='right')
    b = np.searchsorted(ordered, t2, side='right')
    c = np.searchsorted(ordered, t2 + gap, side='right')
    return n_tr, np.maximum(b - a, 0), n - c


def _best_val_end(ordered, uniq, t1, gap, val_share, min_n):
    """أفضل ``val_end`` لحدّ ``t1``: حصة val من (val+test) الأقرب للهدف، وكلاهما ≥ ``min_n``."""
    cand = uniq[uniq > t1 + gap]
    if cand.size == 0:
        return None
    _, n_va, n_te = _kept_counts(ordered, t1, cand, gap)
    ok = (n_va >= min_n) & (n_te >= min_n)
    if not ok.any():
        return None
    share = n_va / np.maximum(n_va + n_te, 1)
    j = int(np.argmin(np.where(ok, np.abs(share - val_share), np.inf)))
    return int(cand[j]), int(n_va[j]), int(n_te[j])


def _resolve_by_kept_share(ordered: np.ndarray, gap: int, train_pct: float,
                           val_pct: float, min_n: int):
    """يبحث عن ``(train_end, val_end)`` بحيث تقترب نِسَب **المُبقاة** من المطلوب.

    ``ordered``: طوابع العيّنات مرتّبة (int64 ns). يُرجع
    ``(t1, t2, n_train, n_val, n_test)`` أو ``None`` إن لم يوجد حدّان يُبقيان
    ``min_n`` عيّنة على الأقل في كل قسم. بحث ثنائي على ``t1`` (كل تقييم
    ``O(U log N)``)، فيبقى سريعاً حتى مع ملايين العيّنات.
    """
    uniq = np.unique(ordered)
    if uniq.size == 0:
        return None
    test_pct = max(1.0 - train_pct - val_pct, 0.0)
    val_share = val_pct / (val_pct + test_pct) if (val_pct + test_pct) > 0 else 0.5

    def ev(i):
        t1 = int(uniq[i])
        n_tr = int(np.searchsorted(ordered, t1, side='right'))
        if n_tr < min_n:
            return "low"                       # train صغير جداً: اذهب لحدّ أحدث
        r = _best_val_end(ordered, uniq, t1, gap, val_share, min_n)
        if r is None:
            return "high"                      # لا يتّسع val/test: اذهب لحدّ أقدم
        t2, n_va, n_te = r
        return (n_tr / (n_tr + n_va + n_te), t1, t2, n_tr, n_va, n_te)

    lo, hi = 0, len(uniq) - 1
    while lo < hi:                             # أقدم حدّ يُبقي train كافياً
        mid = (lo + hi) // 2
        if ev(mid) == "low":
            lo = mid + 1
        else:
            hi = mid
    i_lo = lo
    lo, hi = i_lo, len(uniq) - 1
    while lo < hi:                             # أحدث حدّ يتّسع معه val و test
        mid = (lo + hi + 1) // 2
        if ev(mid) == "high":
            hi = mid - 1
        else:
            lo = mid
    i_hi = lo
    if isinstance(ev(i_lo), str) or isinstance(ev(i_hi), str):
        return None

    lo, hi = i_lo, i_hi                        # أول حدّ تبلغ عنده حصة train الهدف
    while lo < hi:
        mid = (lo + hi) // 2
        if ev(mid)[0] >= train_pct:
            hi = mid
        else:
            lo = mid + 1
    best = ev(lo)
    if lo > i_lo:
        prev = ev(lo - 1)
        if abs(prev[0] - train_pct) < abs(best[0] - train_pct):
            best = prev
    _, t1, t2, n_tr, n_va, n_te = best
    return t1, t2, n_tr, n_va, n_te


#: طرق اشتقاق حدّ(ي) التقسيم في split_mode='global_time'.
CUTOFF_METHODS = ('kept_share', 'raw_quantile')


def compute_global_cutoff(data: Dict, pct: float) -> pd.Timestamp:
    """حدّ زمني واحد بحيث يقع بعده أقرب عدد ممكن إلى ``pct × إجمالي العيّنات``.

    "آخر N من كل العيّنات (بصرف النظر عن عملتها)" تقارب النسبة المطلوبة
    مباشرة — استيداك (percentile) على الطابع الزمني **الخام** لكل العيّنات
    مجمّعة معاً من كل الأصول، لا لكل أصل على حدة؛ وهذا جوهر منع التسرّب بين
    العملات (حدّ واحد مشترك، مطابقاً لما يفعله ``_split_global_time`` أصلاً).

    ⚠️ يستهدف نسبة العيّنات **الخام قبل** فجوة العزل، لا **المُبقاة بعدها**
    (خلافاً لـ :func:`resolve_split_dates` بطريقتها الافتراضية ``'kept_share'``،
    التي تستهدف المُبقاة فعلياً — أدقّ لكن أعقد بحثاً). الفرق بين الطريقتين
    **يتناسب مع (عيّنات فجوة العزل المفقودة ÷ حجم القسم المستهدف)** — يصغر
    كلما كبر val_pct/test_pct×الإجمالي عن خسارة العزل، ويكبر كلما صغر: حتى
    على بيانات موزَّعة **بانتظام تام** زمنياً، val_pct صغيرة (٥٪ مثلاً) مع
    فجوة عزل معتدلة قد تُنتج قسم val أصغر فعلياً من المطلوب بعشرات بالمئة —
    قِيس فعلياً (٣٣٪ نقصان على ١٠ آلاف عيّنة، ٥ أصول، فجوة ٣٣ يوماً). وعلى
    بيانات **مكدَّسة** أيضاً (عملات كثيرة حديثة الإدراج) قد يقع الحدّ في منطقة
    خفيفة العيّنات فيتفاقم الأثر أو يفشل تماماً — وهذا بالضبط ما وقع فعلياً
    على بيانات ٤٢٢ عملة حقيقية ودفع لكتابة ``'kept_share'`` أصلاً (راجع توثيق
    ``_split_global_time``). الحماية الوحيدة هنا رفض صريح
    (``_assert_split_sizes``) لا إصلاح للانحراف نفسه — استخدم ``'kept_share'``
    (الافتراضي) إن أردتَ مطابقة دقيقة للنِّسَب المطلوبة فعلياً.
    """
    ts = extract_dates(data)
    if len(ts) == 0:
        raise ValueError("لا عيّنات لحساب حدّ فاصل منها.")
    pct = float(pct)
    if not 0.0 < pct < 1.0:
        raise ValueError(f"النسبة يجب أن تقع بين 0 و1 حصرياً (وصلت {pct}).")
    q = np.quantile(np.asarray(ts.astype('int64')), 1.0 - pct)
    return pd.Timestamp(int(round(q)), tz='UTC')


#: مرادف موثَّق لـ sample_timestamps بالاسم المطلوب — نفس الدالة تماماً، بلا
#: منطق إضافي؛ لا يوجد داعٍ لنسخة ثانية من نفس الحساب.
extract_dates = sample_timestamps


def extract_test_dates(test: Dict) -> pd.DatetimeIndex:
    """طوابع زمن عيّنات قسم ``test`` — يدعم الشكلين اللذين يُرجعهما ``split_data``:
    قاموس مسطّح واحد (``keep_asset_test_separate=False``)، أو
    ``{asset: split}`` (``keep_asset_test_separate=True``، تُجمَع طوابع كل
    الأصول معاً بالترتيب الزمني)."""
    if 'last_candles' in test:
        return extract_dates(test)
    parts = [extract_dates(v) for v in test.values() if len(v.get('last_candles', []))]
    if not parts:
        return pd.DatetimeIndex([], tz='UTC')
    return parts[0].append(parts[1:]) if len(parts) > 1 else parts[0]


def resolve_split_dates(data: Dict,
                        train_pct: float,
                        val_pct: float,
                        config: Optional[dict] = None,
                        cutoff_method: Optional[str] = None,
                        ) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """حدّا التقسيم الزمنيان المشتركان ``(train_end, val_end)``.

    إن حُدِّدت ``CONFIG['split_dates']`` تُستخدم كما هي؛ وإلا تُشتقّان بحسب
    ``cutoff_method`` (افتراضياً ``config['split_cutoff_method']``، وهو
    ``'kept_share'``):

    * ``'kept_share'`` — تقترب **نِسَب العيّنات المُبقاة بعد فجوتَي العزل** من
      ``train_pct``/``val_pct``. الحدّ الزمني واحد للجميع، وهذا جوهر منع
      التسرّب. إن لم يتّسع الزمن لقسمين بعد train (فجوتان + ``min_split_
      samples``) يُرفع ``ValueError`` بشرح الأسباب.
    * ``'raw_quantile'`` — حدّان مباشران عبر :func:`compute_global_cutoff`
      على نسبة العيّنات **الخام قبل** خصم فجوة العزل؛ أبسط لكن أقلّ دقّة على
      بيانات مكدَّسة زمنياً — راجع تحذير تلك الدالة. النتيجة تخضع لنفس فحص
      ``min_split_samples`` (رفض صريح، لا قسم فارغ صامت).
    """
    config = CONFIG if config is None else config
    dates = config.get('split_dates') or {}

    if dates.get('train_end') and dates.get('val_end'):
        return (pd.Timestamp(dates['train_end'], tz='UTC'),
                pd.Timestamp(dates['val_end'], tz='UTC'))
    data = _before_holdout(data, config)          # الحدود التلقائية تُشتقّ مما قبل الـholdout وحده

    method = cutoff_method or config.get('split_cutoff_method', 'kept_share')
    if method not in CUTOFF_METHODS:
        raise ValueError(f"split_cutoff_method غير معروفة: {method!r} — "
                         f"المتاح: {CUTOFF_METHODS}")

    if method == 'raw_quantile':
        test_pct = max(1.0 - train_pct - val_pct, 0.0)
        # حدّان مستقلّان بنفس المبدأ: train_end بحيث يقع بعده (val_pct+test_pct)
        # من كل العيّنات، وval_end بحيث يقع بعده test_pct منها فقط.
        train_end = compute_global_cutoff(data, val_pct + test_pct)
        val_end = compute_global_cutoff(data, test_pct)
        gap = embargo_duration(data, config)
        ts = sample_timestamps(data)
        sizes = {'train': int((ts <= train_end).sum()),
                'val': int(((ts > train_end + gap) & (ts <= val_end)).sum()),
                'test': int((ts > val_end + gap).sum())}
        _assert_split_sizes(sizes, config, gap, train_end, val_end)
        kept = sum(sizes.values())
        if kept and (abs(sizes['train'] / kept - train_pct) > 0.02
                     or abs(sizes['val'] / kept - val_pct) > 0.02):
            print(f"   ℹ️ النِّسَب الفعلية (raw_quantile): train={sizes['train'] / kept:.1%} | "
                  f"val={sizes['val'] / kept:.1%} | test={sizes['test'] / kept:.1%} "
                  f"(المطلوب {train_pct:.0%}/{val_pct:.0%}/{test_pct:.0%}) — فجوة العزل "
                  f"غيّرت النسب الخام؛ جرّب cutoff_method='kept_share' لمطابقة أدقّ.")
        return train_end, val_end

    # 'kept_share' (الافتراضي)
    ordered = np.sort(np.asarray(data['last_candles'])[:, TS_COL].astype('int64'))
    gap = int(embargo_duration(data, config).value)
    min_n = int(config.get('min_split_samples', 1))

    found = _resolve_by_kept_share(ordered, gap, train_pct, val_pct, min_n)
    day = 86_400 * 10 ** 9
    if found is None:
        span = (int(ordered[-1]) - int(ordered[0])) / day if len(ordered) else 0.0
        raise ValueError(
            f"❌ لا يمكن اشتقاق حدّي تقسيم يُبقيان ≥{min_n} عيّنة في train و val و test.\n"
            f"   العيّنات: {len(ordered):,} | المدى الزمني: {span:.0f} يوماً | "
            f"فجوة العزل: {gap / day:.0f} يوماً (تُفقَد مرّتين: بعد train وبعد val).\n"
            f"   الحلول: (١) حدّد CONFIG['split_dates'] يدوياً، (٢) وفّر تاريخاً أطول "
            f"أو stride أصغر، (٣) خفّض min_split_samples إن كان الحدّ مبالَغاً فيه.\n"
            f"   لا تُقلّص فجوة العزل لإنقاذ التقسيم: النوافذ المتراكبة تُسرّب الأداء.")

    t1, t2, n_tr, n_va, n_te = found
    kept = n_tr + n_va + n_te
    if abs(n_tr / kept - train_pct) > 0.02 or abs(n_va / kept - val_pct) > 0.02:
        print(f"   ℹ️ النِّسَب المُبقاة بعد العزل: train={n_tr / kept:.1%} | "
              f"val={n_va / kept:.1%} | test={n_te / kept:.1%} "
              f"(المطلوب {train_pct:.0%}/{val_pct:.0%}/{max(1 - train_pct - val_pct, 0):.0%}) — "
              f"أقرب ما يسمح به توزيع العيّنات زمنياً.")
    return pd.Timestamp(t1, tz='UTC'), pd.Timestamp(t2, tz='UTC')


def _assert_split_sizes(sizes: Dict[str, int], config: dict, gap: pd.Timedelta,
                        train_end: pd.Timestamp, val_end: pd.Timestamp) -> None:
    """يرفع خطأً صريحاً إن قلّ أي قسم عن ``min_split_samples`` (بدل قسم فارغ صامت)."""
    min_n = int(config.get('min_split_samples', 1))
    small = {k: v for k, v in sizes.items() if v < min_n}
    if small:
        raise ValueError(
            f"❌ أقسام أصغر من الحدّ الأدنى ({min_n}): {small}\n"
            f"   الحدّان: train_end={train_end:%Y-%m-%d} | val_end={val_end:%Y-%m-%d} | "
            f"عزل={gap}.\n"
            f"   إن كانا من CONFIG['split_dates'] فباعِد بينهما بما يزيد على فجوة العزل؛ "
            f"وإلا اترك split_dates=None ليُشتقّا تلقائياً.")


def _take_per_asset(data, mask, timeframes, targets):
    """``({asset: split}, [عملات بلا عيّنات])`` — تقاطع قناع الزمن مع نطاق كل عملة."""
    out: Dict[str, dict] = {}
    skipped: List[str] = []
    n = len(data['last_candles'])
    for b in data.get('asset_bounds') or []:
        asset_mask = np.zeros(n, dtype=bool)
        asset_mask[b['start']:b['end']] = True
        combined = asset_mask & mask
        name = b.get('name', b.get('asset', f"asset_{b['start']}_{b['end']}"))
        if combined.any():
            out[name] = _take(data, combined, timeframes, targets)
        else:
            skipped.append(name)
    return out, skipped


def _split_global_time(data, train_pct, val_pct, keep_asset_test_separate,
                       timeframes, targets, config):
    """تقسيم بحدود زمنية مطلقة مشتركة بين كل العملات."""
    ts = sample_timestamps(data)
    train_end, val_end = resolve_split_dates(data, train_pct, val_pct, config)
    gap = embargo_duration(data, config)

    train_mask = ts <= train_end
    val_mask = (ts > train_end + gap) & (ts <= val_end)
    test_mask = ts > val_end + gap
    hs = resolve_holdout_start(config)
    if hs is not None:
        if hs <= val_end + gap:
            raise ValueError(
                f"❌ holdout_start={hs:%Y-%m-%d %H:%M} لا يقع بعد val_end + العزل ({(val_end + gap):%Y-%m-%d %H:%M}): "
                f"الـholdout المختوم لا يجوز أن يتداخل مع val. قدّم split_dates أو أخّر holdout_start.")
        test_mask = test_mask & (ts < hs - gap)        # فجوة العزل نفسها قبل الـholdout

    train = _take(data, train_mask, timeframes, targets)
    val = _take(data, val_mask, timeframes, targets)

    print(f"📊 تقسيم زمني مشترك (عزل={gap}):")
    print(f"   train ≤ {train_end:%Y-%m-%d}          : {int(train_mask.sum()):,}")
    print(f"   val   ≤ {val_end:%Y-%m-%d}          : {int(val_mask.sum()):,}")
    print(f"   test  >  {(val_end + gap):%Y-%m-%d}"
          + (f" و< {(hs - gap):%Y-%m-%d %H:%M}" if hs is not None else "          ")
          + f" : {int(test_mask.sum()):,}")
    if hs is not None:
        n_hold = int((ts >= hs).sum())
        print(f"   🔒 holdout ≥ {hs:%Y-%m-%d}         : {n_hold:,} مختوم — خارج كل الأقسام وكل ملفات الإشارات "
              f"(يُفتح مرة واحدة عند الإصدار: open_holdout=True ثم split_holdout)")

    _assert_split_sizes({'train': int(train_mask.sum()), 'val': int(val_mask.sum()),
                         'test': int(test_mask.sum())}, config, gap, train_end, val_end)

    if not keep_asset_test_separate:
        test = _take(data, test_mask, timeframes, targets)
        return train, val, test

    # test مفصول لكل عملة — نتقاطع قناع الزمن مع نطاق كل عملة
    test, skipped = _take_per_asset(data, np.asarray(test_mask), timeframes, targets)

    print(f"   test مفصول عبر {len(test)} عملة")
    if skipped:
        # عملات تاريخها ينتهي قبل حدّ الاختبار — سلوك صحيح، لكن يستحق التنبيه
        print(f"   ℹ️ {len(skipped)} عملة بلا عيّنات اختبار (تاريخها ينتهي قبل "
              f"{(val_end + gap):%Y-%m-%d}): {skipped[:6]}"
              f"{'...' if len(skipped) > 6 else ''}")
    return train, val, test


def split_data(
    data: Dict,
    train_pct: Optional[float] = None,
    val_pct: Optional[float] = None,
    keep_asset_test_separate: Optional[bool] = None,
    prefix_y: bool = True,
    mode: Optional[str] = None,
    config: Optional[dict] = None,
) -> Tuple[dict, dict, dict]:
    """تقسيم ``train / val / test`` زمنياً مع فجوة عزل.

    Args:
        mode: ``'global_time'`` (افتراضي) حدود زمنية مشتركة تمنع التسرّب بين
            العملات مختلفة الأطوال؛ ``'per_asset'`` السلوك القديم (نسب لكل عملة).
        keep_asset_test_separate: True يُرجع ``test`` كقاموس ``{asset: split}``
            (يسمح بتقييم كل عملة على حدة)، False يدمجها في قسم واحد.
        prefix_y: True يُضيف بادئة ``y_`` لمفاتيح الأهداف لتطابق أسماء مخرجات النموذج.
    """
    config = CONFIG if config is None else config
    train_pct = config['train_pct'] if train_pct is None else train_pct
    val_pct = config['val_pct'] if val_pct is None else val_pct
    if keep_asset_test_separate is None:
        keep_asset_test_separate = config['keep_asset_test_separate']
    mode = mode or config.get('split_mode', 'global_time')
    if mode not in SPLIT_MODES:
        raise ValueError(f"split_mode غير معروف: {mode!r} — المتاح: {SPLIT_MODES}")

    if mode != 'global_time' and resolve_holdout_start(config) is not None:
        hs = resolve_holdout_start(config)
        if (sample_timestamps(data) >= hs - embargo_duration(data, config)).any():
            raise ValueError(f"❌ holdout_start={hs:%Y-%m-%d} يتطلّب split_mode='global_time' "
                             f"(حدّ زمني مشترك)؛ '{mode}' يُقسّم كل عملة على مداها فيدخل الـholdout في test.")

    if mode == 'global_time':
        train, val, test = _split_global_time(
            data, train_pct, val_pct, keep_asset_test_separate,
            data['timeframes'], data['targets'], config)
        if prefix_y:
            add_y_prefix(train)
            add_y_prefix(val)
            if isinstance(test, dict) and 'y' in test:
                add_y_prefix(test)
            else:
                for asset_split in test.values():
                    add_y_prefix(asset_split)
        return train, val, test

    timeframes: List[str] = data['timeframes']
    targets: List[str] = data['targets']
    bounds = data.get('asset_bounds')
    embargo_samples = embargo_size(data, config)

    def _bounds(s, e):
        n = e - s
        t_end = s + int(n * train_pct)
        v_start = min(t_end + embargo_samples, e)
        v_end = min(v_start + int(n * val_pct), e)
        te_start = min(v_end + embargo_samples, e)
        return (s, t_end), (v_start, v_end), (te_start, e)

    if not bounds:
        n = len(data['base_params'])
        (s_tr, e_tr), (s_va, e_va), (s_te, e_te) = _bounds(0, n)
        train = _slice(data, s_tr, e_tr, timeframes, targets)
        val = _slice(data, s_va, e_va, timeframes, targets)
        test = _slice(data, s_te, e_te, timeframes, targets)
        print(f"📊 تقسيم (بدون حدود عملات) | عزل={embargo_samples} عيّنة: "
              f"Train={e_tr - s_tr:,} | Val={e_va - s_va:,} | Test={e_te - s_te:,}")
    else:
        train_acc = _new_accumulator(timeframes, targets)
        val_acc = _new_accumulator(timeframes, targets)
        test_acc = _new_accumulator(timeframes, targets)
        test_per_asset: Dict[str, dict] = {}
        asset_stats: List[str] = []

        for b in bounds:
            s, e = b['start'], b['end']
            asset_name = b.get('name', b.get('asset', f'asset_{s}_{e}'))
            if e - s <= 0:
                continue
            (s_tr, e_tr), (s_va, e_va), (s_te, e_te) = _bounds(s, e)
            _append_range(data, s_tr, e_tr, train_acc, timeframes, targets)
            _append_range(data, s_va, e_va, val_acc, timeframes, targets)

            if keep_asset_test_separate:
                if e_te > s_te:
                    asset_split = _slice(data, s_te, e_te, timeframes, targets)
                    test_per_asset[asset_name] = asset_split
                    asset_stats.append(f"{asset_name}: {len(asset_split['base_params']):,}")
            else:
                _append_range(data, s_te, e_te, test_acc, timeframes, targets)

        train = _finalize_accumulator(train_acc, data, timeframes, targets)
        val = _finalize_accumulator(val_acc, data, timeframes, targets)

        if keep_asset_test_separate:
            test = test_per_asset
            n_te = sum(len(v['base_params']) for v in test.values())
            print(f"📊 تقسيم (عزل={embargo_samples} عيّنة، test منفصل لكل عملة):")
            print(f"   Train: {len(train['base_params']):,}")
            print(f"   Val:   {len(val['base_params']):,}")
            print(f"   Test:  {n_te:,} عبر {len(test)} عملة")
            if asset_stats:
                print(f"   التفاصيل: {', '.join(asset_stats[:5])}"
                      + ("..." if len(asset_stats) > 5 else ""))
        else:
            test = _finalize_accumulator(test_acc, data, timeframes, targets)
            print(f"📊 تقسيم (عزل={embargo_samples} عيّنة): "
                  f"Train={len(train['base_params']):,} | "
                  f"Val={len(val['base_params']):,} | "
                  f"Test={len(test['base_params']):,}")

    if prefix_y:
        add_y_prefix(train)
        add_y_prefix(val)
        if isinstance(test, dict) and 'y' in test:
            add_y_prefix(test)
        else:
            for asset_split in test.values():
                add_y_prefix(asset_split)

    return train, val, test


def split_holdout(data: Dict,
                  keep_asset_test_separate: Optional[bool] = None,
                  prefix_y: bool = True,
                  config: Optional[dict] = None):
    """عيّنات الـholdout المختوم (``ts ≥ holdout_start``) بنفس شكل ``test`` من :func:`split_data`.

    مقفلة افتراضياً: ترفع ``PermissionError`` ما لم يكن ``config['open_holdout'] is True``. الـholdout يُفتح مرة
    واحدة عند الإصدار؛ كل تقييم عليه يُسجَّل (PROTOCOL.md: «opened once, at release»)."""
    config = CONFIG if config is None else config
    hs = resolve_holdout_start(config)
    if hs is None:
        raise ValueError("لا holdout محدّد: CONFIG['holdout_start'] = None.")
    if config.get('open_holdout') is not True:
        raise PermissionError(
            f"🔒 الـholdout (≥ {hs:%Y-%m-%d}) مختوم. يُفتح مرة واحدة عند الإصدار: "
            f"update_config(open_holdout=True) ثم split_holdout(dataset).")
    if keep_asset_test_separate is None:
        keep_asset_test_separate = config['keep_asset_test_separate']
    mask = np.asarray(sample_timestamps(data) >= hs)
    timeframes, targets = data['timeframes'], data['targets']
    print(f"🔓 holdout ≥ {hs:%Y-%m-%d}: {int(mask.sum()):,} عيّنة — فتحٌ مسجَّل، لا اختيار عليه")
    if keep_asset_test_separate:
        out, _ = _take_per_asset(data, mask, timeframes, targets)
        if prefix_y:
            for sp in out.values():
                add_y_prefix(sp)
        return out
    out = _take(data, mask, timeframes, targets)
    return add_y_prefix(out) if prefix_y else out


def build_leak_free_split(data: Dict,
                          train_pct: Optional[float] = None,
                          val_pct: Optional[float] = None,
                          keep_asset_test_separate: Optional[bool] = None,
                          config: Optional[dict] = None) -> Tuple[dict, dict, dict]:
    """يُعيد بناء ``train/val/test`` بحدّ فاصل **واحد مشترك** بين كل العملات،
    باستخدام ``compute_global_cutoff``: "آخر N عيّنة" من كل العيّنات (أي
    عملة) تقارب النسبة المطلوبة مباشرة — لا نسبة كل عملة على حدة.

    غلاف رقيق حول ``split_data(mode='global_time', split_cutoff_method=
    'raw_quantile')`` — لا منطق تقطيع مستقلّ، فيبقى هناك تنفيذ واحد فقط
    للتقسيم بحدود مشتركة (``_split_global_time``)، لا نسختان قد تنحرف إحداهما
    عن الأخرى بمرور الوقت.

    ⚠️ **هذه ليست إصلاحاً إضافياً لتسرّب قائم بين العملات** — ``split_data``
    بوضعه الافتراضي (``mode='global_time'``، ``split_cutoff_method=
    'kept_share'``) يمنع هذا التسرّب أصلاً بحدّ زمني واحد مشترك؛ الفرق هنا هو
    طريقة **اختيار** ذلك الحدّ فقط (نسبة خام من كل العيّنات، لا نسبة مُبقاة
    بعد العزل) — راجع تحذير :func:`compute_global_cutoff` قبل اعتمادها
    افتراضاً على بيانات مكدَّسة زمنياً (عملات كثيرة حديثة الإدراج).
    """
    config = dict(CONFIG if config is None else config)
    config['split_cutoff_method'] = 'raw_quantile'
    return split_data(data, train_pct=train_pct, val_pct=val_pct,
                      keep_asset_test_separate=keep_asset_test_separate,
                      mode='global_time', config=config)


# ══════════════════════════════════════════════════════════════════════════
# إعادة تدريب دورية (walk-forward) — عدّة نوافذ متحرّكة بدل حدّ ثابت واحد
# ══════════════════════════════════════════════════════════════════════════
#: علامة "لم يُمرَّر" لتمييزها عن None (قيمة صحيحة في train_span/initial_train_span).
_UNSET = object()


def rolling_split_schedule(data: Dict, test_span=None, val_span=None,
                           train_span=_UNSET, initial_train_span=_UNSET,
                           step=None, max_windows: Optional[int] = None,
                           config: Optional[dict] = None) -> List[Dict[str, pd.Timestamp]]:
    """يبني جدول حدود لنوافذ تدريب/تحقق/اختبار متحرّكة (walk-forward) —
    لإعادة تدريب دورية، بدل حدّ واحد ثابت من :func:`resolve_split_dates`.

    كل نافذة: ``train`` حتى ``train_end``، ثم ``val`` حتى ``val_end`` (بعد
    فجوة عزل)، ثم ``test`` حتى ``test_end`` (بعد فجوة عزل أخرى) — نفس منطق
    العزل في ``_split_global_time`` بالضبط، مُطبَّق داخل كل نافذة على حدة.

    Args:
        test_span/val_span: مدد زمنية (``pd.Timedelta`` أو نص مثل ``'30D'``)
            لكل من الاختبار والتحقق في كل نافذة — بلا افتراض عام، حدّدهما أو
            اضبطهما في ``CONFIG['rolling_retrain']``.
        train_span: مدة تدريب ثابتة (نافذة **منزلقة**) — أو ``None`` صراحةً
            لنافذة **متمدّدة** (تبدأ من أقدم عيّنة وتكبر مع كل إعادة تدريب،
            الأشيع لأن مزيداً من التاريخ عادة لا يضرّ). غير المُمرَّر
            (``_UNSET``) يقرأ من ``CONFIG['rolling_retrain']['train_span']``.
        initial_train_span: حجم **أول** نافذة تدريب — إلزامي عند نافذة
            متمدّدة (``train_span=None``) لأنه لا حجم افتراضي صحيح عالمياً؛
            يُرفع ``ValueError`` صريح إن غاب. يُتجاهَل عند نافذة منزلقة
            (``train_span`` يحكم كل النوافذ فيها بما فيها الأولى).
        step: مقدار انزلاق كل نافذة عن سابقتها. افتراضياً = ``test_span``
            (نوافذ اختبار متتالية غير متداخلة — كل تدريب يُقيَّم مرّة واحدة).
        max_windows: أقصى عدد نوافذ (``None`` = حتى نفاد البيانات).

    Returns:
        قائمة قواميس ``{'train_start','train_end','val_end','test_end'}``
        (``train_start`` أقدم طابع في البيانات دائماً عند نافذة متمدّدة).
        مرّرها إلى :func:`rolling_splits` لتقطيع ``dataset`` فعلياً، أو
        استخدمها مباشرة لمعاينة الجدول قبل أي تقطيع.
    """
    config = CONFIG if config is None else config
    rr = config.get('rolling_retrain') or {}
    test_span = test_span if test_span is not None else rr.get('test_span')
    val_span = val_span if val_span is not None else rr.get('val_span')
    if train_span is _UNSET:
        train_span = rr.get('train_span')
    if initial_train_span is _UNSET:
        initial_train_span = rr.get('initial_train_span')
    step = step if step is not None else rr.get('step')

    if test_span is None or val_span is None:
        raise ValueError(
            "حدّد test_span وval_span، أو اضبطهما في CONFIG['rolling_retrain'].")
    test_span = pd.Timedelta(test_span)
    val_span = pd.Timedelta(val_span)
    step = pd.Timedelta(step) if step is not None else test_span

    sliding = train_span is not None
    if sliding:
        train_span = pd.Timedelta(train_span)
        first_train_len = train_span
    else:
        if initial_train_span is None:
            raise ValueError(
                "نافذة تدريب متمدّدة (train_span=None) تتطلّب initial_train_span "
                "صراحةً — حجم أول نافذة تدريب قبل أن تبدأ بالتمدّد؛ لا قيمة "
                "افتراضية صحيحة عالمياً. حدّدها أو اضبطها في CONFIG['rolling_retrain'].")
        initial_train_span = pd.Timedelta(initial_train_span)
        first_train_len = initial_train_span

    gap = embargo_duration(data, config)
    ts = sample_timestamps(_before_holdout(data, config))   # نوافذ walk-forward لا تبلغ الـholdout المختوم
    if len(ts) == 0:
        raise ValueError("dataset فارغ — لا عيّنات لبناء جدول نوافذ منه.")
    ts_min, ts_max = ts.min(), ts.max()

    schedule: List[Dict[str, pd.Timestamp]] = []
    train_end = ts_min + first_train_len
    while True:
        val_end = train_end + gap + val_span
        test_end = val_end + gap + test_span
        if test_end > ts_max:
            break
        train_start = (train_end - train_span) if sliding else ts_min
        schedule.append({'train_start': train_start, 'train_end': train_end,
                         'val_end': val_end, 'test_end': test_end})
        if max_windows and len(schedule) >= max_windows:
            break
        train_end = train_end + step

    if not schedule:
        needed = first_train_len + 2 * gap + val_span + test_span
        raise ValueError(
            f"❌ لا تتّسع البيانات لنافذة واحدة حتى: المدى الزمني المتاح "
            f"{(ts_max - ts_min).days:,} يوماً، والمطلوب لأول نافذة "
            f"{needed.days:,} يوماً (تدريب+عزلان+تحقق+اختبار). صغّر test_span/"
            f"val_span/initial_train_span أو وفّر تاريخاً أطول.")
    return schedule


def rolling_splits(data: Dict, test_span=None, val_span=None,
                   train_span=_UNSET, initial_train_span=_UNSET, step=None,
                   max_windows: Optional[int] = None,
                   keep_asset_test_separate: Optional[bool] = None,
                   prefix_y: bool = True, config: Optional[dict] = None,
                   verbose: bool = True) -> List[Tuple[dict, dict, dict]]:
    """يبني عدّة أزواج ``(train, val, test)`` بحدود زمنية متحرّكة — لإعادة
    تدريب دورية walk-forward، بدل حدّ ثابت واحد من :func:`split_data`.

    كل نافذة مُطهَّرة (purged+embargoed) بنفس منطق ``_split_global_time``
    تماماً — فجوة عزل بين كل قسمين، فلا تسرّب داخل أي نافذة على حدة.

    ⚠️ النوافذ **تتشارك بيانات تدريب** حتماً (تدريب كل نافذة يمتدّ من سابقتها
    أو يتقدّم عليها) — هذا هو المقصود بإعادة التدريب الدوري (كل نافذة =
    نموذج جديد يُدرَّب على تاريخ أحدث)، وليس تسرّباً: اختبار كل نافذة لا
    يتداخل زمنياً مع تدريبها هي فتبقى نتيجتها نظيفة على حدة.

    الوسائط كما في :func:`rolling_split_schedule`. يُرجع قائمة
    ``(train, val, test)`` بنفس شكل مخرجات ``split_data`` (``test`` قاموس
    ``{asset: split}`` إن ``keep_asset_test_separate=True``).
    """
    config = CONFIG if config is None else config
    if keep_asset_test_separate is None:
        keep_asset_test_separate = config['keep_asset_test_separate']

    schedule = rolling_split_schedule(data, test_span, val_span, train_span,
                                      initial_train_span, step, max_windows, config)
    ts = sample_timestamps(data)
    gap = embargo_duration(data, config)
    timeframes, targets = data['timeframes'], data['targets']

    out: List[Tuple[dict, dict, dict]] = []
    for i, w in enumerate(schedule, 1):
        train_mask = np.asarray((ts > w['train_start']) & (ts <= w['train_end']))
        val_mask = np.asarray((ts > w['train_end'] + gap) & (ts <= w['val_end']))
        test_mask = np.asarray((ts > w['val_end'] + gap) & (ts <= w['test_end']))

        sizes = {'train': int(train_mask.sum()), 'val': int(val_mask.sum()),
                'test': int(test_mask.sum())}
        _assert_split_sizes(sizes, config, gap, w['train_end'], w['val_end'])

        train = _take(data, train_mask, timeframes, targets)
        val = _take(data, val_mask, timeframes, targets)
        if keep_asset_test_separate:
            test: Dict[str, dict] = {}
            for b in data.get('asset_bounds') or []:
                asset_mask = np.zeros(len(ts), dtype=bool)
                asset_mask[b['start']:b['end']] = True
                combined = asset_mask & test_mask
                if combined.any():
                    name = b.get('name', b.get('asset', f"asset_{b['start']}_{b['end']}"))
                    test[name] = _take(data, combined, timeframes, targets)
            n_te = sum(len(v['base_params']) for v in test.values())
        else:
            test = _take(data, test_mask, timeframes, targets)
            n_te = sizes['test']

        if prefix_y:
            add_y_prefix(train)
            add_y_prefix(val)
            if isinstance(test, dict) and 'y' in test:
                add_y_prefix(test)
            else:
                for asset_split in test.values():
                    add_y_prefix(asset_split)

        if verbose:
            print(f"🔁 نافذة {i}/{len(schedule)}: "
                  f"train≤{w['train_end']:%Y-%m-%d} ({sizes['train']:,}) | "
                  f"val≤{w['val_end']:%Y-%m-%d} ({sizes['val']:,}) | "
                  f"test≤{w['test_end']:%Y-%m-%d} ({n_te:,})")
        out.append((train, val, test))
    return out
