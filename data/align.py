"""
PURPOSE:  Time-based multi-timeframe alignment: window end indices on a common grid, the single-TF fast path, and the 'closed' higher-TF mode (only bars closed at t).
TAGS:     alignment, align_multi_timeframes_time_based, window_end_indices, higher_tf_mode, closed bars, 4h context, align_windows_to_grid, look-ahead
PITFALLS: A higher-TF window must end on a bar already closed at t (higher_tf_offset / 'closed' mode); any change here must pass tests/test_no_lookahead.py and tests/test_multi_tf.py. Executed into the one shared pipeline namespace by data/_loader.py (never imported on its own): names from other modules resolve at call time.

## 10) محاذاة الفريمات الزمنية المتعدّدة (`align.py` سابقاً)
"""
# @title
"""
محاذاة الفريمات الزمنية المتعدّدة على أساس الوقت (وليس الفهرس).

لكل نافذة على الفريم الأصغر، تُقتطع نافذة مقابلة من كل فريم أعلى تنتهي عند
شمعة **مغلقة بالفعل** قبل زمن النهاية (``higher_tf_offset``)، منعاً لتسرّب
معلومات من المستقبل.
"""

def _higher_tf_closed(config: Optional[dict] = None) -> bool:
    """هل الفريمات الأعلى بوضع 'closed' (انظر CONFIG['higher_tf_mode'])؟ ومعناها: شموع مغلقة عند زمن العيّنة فقط."""
    config = CONFIG if config is None else config
    mode = config.get('higher_tf_mode', 'legacy')
    if mode not in ('legacy', 'closed'):
        raise ValueError(f"higher_tf_mode: 'legacy' | 'closed' — لا {mode!r}")
    return mode == 'closed' and len(config.get('tf_order') or []) > 1


def _keep_full_bars(frame: pd.DataFrame, base_index: pd.DatetimeIndex, tf: str, base_tf: str) -> pd.DataFrame:
    """يُبقي من ``frame`` (مُعاد أخذ عيّناته إلى ``tf``) الشموع التي كل شموع ``base_tf`` فيها موجودة فقط.
    ``base_index``: أزمنة شموع الفريم الأصغر **الحقيقية** (إغلاقها غير NaN) — ساعة ناقصة يُبقيها resample صفّاً فارغاً
    (حجمه 0 وأسعاره NaN) فلا يصحّ عدّ الصفوف.

    لماذا: ``resample(...).agg`` يبني شمعة 4h من ثلاث شموع 1h إن غابت الرابعة (فجوة في المصدر) — قيمتها الدائمة
    حجم ناقص ونطاق أضيق، لا تسرّب مستقبل لكنها شمعة «غير مُكوَّنة» تخالف أقرانها. فريم بلا مدة ثابتة (شهري) لا يُصفّى."""
    try:
        n_full = int(pd.Timedelta(tf) // pd.Timedelta(base_tf))
    except (ValueError, TypeError):
        return frame
    if n_full <= 1:
        return frame
    counts = pd.Series(1, index=base_index).resample(tf).sum().reindex(frame.index)   # base_index: شموع حقيقية فقط
    return frame[(counts >= n_full).to_numpy()]


def window_end_indices(index: pd.DatetimeIndex, win: int, stride: int, tf: str,
                       config: Optional[dict] = None) -> np.ndarray:
    """فهارس شموع نهاية النوافذ لأصل واحد — مصدر واحد لمسارَي المحاذاة.

    * ``stride <= 1`` أو ``align_windows_to_grid=False``: شمعة كل ``stride`` بدءاً من
      أول نافذة كاملة (السلوك القديم حرفياً).
    * ``stride > 1`` مع المحاذاة (الافتراضي): رقم شمعة النهاية منذ ``window_grid_anchor``
      (بوحدة مدة الفريم) مضاعفٌ لـ ``stride``، أي ``(ts − anchor) % (stride × مدة_الفريم)
      == 0`` لشموع على شبكة الفريم.

    **لماذا؟** العدّ من أول شمعة يربط طور النوافذ بتاريخ إدراج العملة، فلا تتشارك
    العملات طوابع النهاية (قيس على 1h بـstride=32: 30 طوراً من 32 ممكنة، ووسيط العملات
    في الطابع الواحد 2 من 83)، فينكسر كل حساب مقطعي لاحق. الشبكة الثابتة تجعل أزمنة
    النهاية واحدة لكل العملات في مداها المشترك.

    فجوة في البيانات قد تقرّب نقطتَي شبكة إلى أقل من ``stride`` شمعة؛ تُسقَط الثانية
    عندها، فيبقى بين نهايتين متتاليتين ``stride`` شمعة على الأقل كما كان (لا تراكب
    بين النوافذ أكثر مما يسمح به ``stride``).
    """
    config = CONFIG if config is None else config
    n, stride = len(index), max(int(stride), 1)
    if n < win:
        return np.empty(0, dtype=np.int64)
    if stride == 1 or not config.get('align_windows_to_grid', True):
        return np.arange(win - 1, n, stride, dtype=np.int64)

    ts_ns = index.as_unit('ns').asi8          # UTC للفهرس الواعي بالمنطقة الزمنية
    try:
        bar_ns = int(pd.Timedelta(tf) // pd.Timedelta(1, 'ns'))
    except (ValueError, TypeError):
        # فريم بلا مدة ثابتة (شهري مثلاً): وسيط الفاصل الفعلي بين الشموع.
        bar_ns = int(np.median(np.diff(ts_ns))) if n > 1 else 1
    anchor_ns = pd.Timestamp(config.get('window_grid_anchor', '1970-01-01')).as_unit('ns').value
    # قسمة سفلية: شموع مُزاحة بثابت عن الشبكة (مثل :30) تبقى متحاذية بين العملات
    # المشتركة في الإزاحة بدل أن تُسقَط كلها.
    bar_no = (ts_ns - anchor_ns) // max(bar_ns, 1)
    cand = np.flatnonzero(bar_no % stride == 0)
    cand = cand[cand >= win - 1]
    keep: List[int] = []
    for c in cand.tolist():
        if not keep or c - keep[-1] >= stride:
            keep.append(c)
    return np.asarray(keep, dtype=np.int64)


def align_multi_timeframes_time_based(
    dfs: Dict[str, pd.DataFrame],
    tf_order: Optional[List[str]] = None,
    window_sizes: Optional[Dict[str, int]] = None,
    stride: Optional[int] = None,
    config: Optional[dict] = None,
) -> Tuple[Dict[str, np.ndarray], List[pd.Timestamp]]:
    """يُرجع ``({tf: مصفوفة نوافذ (N, T, F)}, قائمة أزمنة النهاية)``.

    ✅ **مسار سريع لفريم واحد** (``len(tf_order) == 1`` — حالتك الفعلية):
    عبر ``numpy.lib.stride_tricks.sliding_window_view`` بدل نسخ كل نافذة على
    حدة في حلقة بايثون ثم تكديسها. قِيس فعلياً: ذروة الرام تنزل من ~3.16×
    حجم المصفوفة النهائية إلى ~1.13× (شبه الحدّ الأدنى النظري) على أصل بـ900
    يوم — النسختان تُنتجان **نفس المصفوفة تماماً** (تحقّقتُ رقمياً، لا فرق
    عددي إطلاقاً، فالعملية إعادة ترتيب ذاكرة لا حساب). فريمات متعدّدة
    (``len(tf_order) > 1``) تبقى على الحلقة الأصلية: محاذاة زمنية شرطية بين
    فريمات مختلفة الطول أعقد من أن تُختزل لعرض نوافذ (view) واحد بأمان.

    نهايات النوافذ في المسارين من :func:`window_end_indices` (شبكة زمنية مشتركة بين
    العملات حين ``stride > 1`` — راجع ``CONFIG['align_windows_to_grid']``).
    """
    config = CONFIG if config is None else config
    tf_order = config["tf_order"] if tf_order is None else tf_order
    window_sizes = config["window_sizes"] if window_sizes is None else window_sizes
    stride = config["stride"] if stride is None else stride

    if len(tf_order) == 1:
        return _align_single_tf_view(dfs, tf_order[0], window_sizes[tf_order[0]], stride,
                                     config=config)
    if _higher_tf_closed(dict(config, tf_order=list(tf_order))):
        return _align_closed_higher_tfs(dfs, tf_order, window_sizes, stride, config)

    offset = config.get("higher_tf_offset", 2)
    small_tf = tf_order[0]
    df_small = dfs[small_tf]
    win_small = window_sizes[small_tf]

    if not isinstance(df_small.index, pd.DatetimeIndex):
        raise ValueError(f"DataFrame للفريم {small_tf} يجب أن يكون مفهرساً بـ DatetimeIndex")
    df_small = df_small.sort_index()
    for tf in tf_order[1:]:
        if not isinstance(dfs[tf].index, pd.DatetimeIndex):
            raise ValueError(f"DataFrame للفريم {tf} يجب أن يكون مفهرساً بـ DatetimeIndex")
        dfs[tf] = dfs[tf].sort_index()

    windows: Dict[str, List[np.ndarray]] = {tf: [] for tf in tf_order}
    end_times: List[pd.Timestamp] = []
    times_small = df_small.index

    for end_idx_small in window_end_indices(times_small, win_small, stride, small_tf,
                                            config).tolist():
        start_idx_small = end_idx_small - win_small + 1
        end_time = times_small[end_idx_small]
        win_small_df = df_small.iloc[start_idx_small:end_idx_small + 1]
        current_wins: Dict[str, np.ndarray] = {small_tf: win_small_df.values}
        valid = True

        for tf in tf_order[1:]:
            df_tf = dfs[tf]
            win_tf = window_sizes[tf]
            pos = df_tf.index.searchsorted(end_time, side='right') - offset
            if pos < 0:
                valid = False
                break
            end_idx_tf = pos
            start_idx_tf = end_idx_tf - win_tf + 1
            if start_idx_tf < 0:
                valid = False
                break
            win_tf_df = df_tf.iloc[start_idx_tf:end_idx_tf + 1]
            if len(win_tf_df) != win_tf:
                valid = False
                break
            current_wins[tf] = win_tf_df.values

        if not valid:
            continue
        for tf in tf_order:
            windows[tf].append(current_wins[tf])
        end_times.append(end_time)

    out: Dict[str, np.ndarray] = {}
    for tf in tf_order:
        if windows[tf]:
            arr_tf = np.stack(windows[tf], axis=0)
        else:
            n_feat = dfs[tf].shape[1]
            arr_tf = np.zeros((0, window_sizes[tf], n_feat), dtype='float32')
        out[tf] = arr_tf.astype('float32')
    return out, end_times


def _windows_at(values: np.ndarray, starts: np.ndarray, win: int) -> np.ndarray:
    """نوافذ ``win`` صفاً تبدأ عند ``starts`` من ``values`` (T, F) ← (N, win, F): نسخة واحدة عبر عرض بلا نسخ."""
    view = np.moveaxis(np.lib.stride_tricks.sliding_window_view(values, win, axis=0), -1, 1)
    return np.ascontiguousarray(view[starts])


def _align_closed_higher_tfs(dfs: Dict[str, pd.DataFrame], tf_order: List[str],
                             window_sizes: Dict[str, int], stride: int, config: dict,
                             ) -> Tuple[Dict[str, np.ndarray], List[pd.Timestamp]]:
    """محاذاة الفريمات الأعلى بقاعدة «الشمعة المغلقة» (``CONFIG['higher_tf_mode'] == 'closed'``).

    فهرس كل شمعة = وقت **فتحها**، وإغلاقها = الفتح + مدة الفريم. لعيّنة نهاية نافذتها الأساسية t (فتح آخر شمعة أساسية):

    * الشمعة العليا الأخيرة هي أحدث شمعة إغلاقها ≤ t، أي ``فتحها ≤ t − مدة_الفريم``. لا شمعة قيد التكوّن، ولا شمعة
      إغلاقها بعد t، مهما كان طور t داخل شمعة الفريم الأعلى.
    * محافِظة عمداً بمقدار شمعة أساسية: t هنا فتح آخر شمعة أساسية لا إغلاقها (t + مدة الفريم الأساسي)، فشمعة 4h المغلقة عند
      t + 1h تُترك حين t ≡ 3 (mod 4h). على شبكة stride=8 (t ≡ 0 mod 8h) لا خسارة: إغلاق الشمعة الأخيرة = t بالضبط.
    * النافذة العليا ``window_sizes[tf]`` شمعة متتالية بالضبط (بلا فجوة)، وشمعتها الأخيرة هي الأحدث فعلاً (وإلا فقدنا
      الشمعة الأحدث ← عيّنة تُحذف لا تُملأ بشمعة أقدم بصمت). العيّنة التي لا تستوفي ذلك في أي فريم تُحذف.

    نفس مخرجات :func:`align_multi_timeframes_time_based` (قاموس نوافذ float32 + أزمنة النهاية)، وكل نافذة بنسخة واحدة."""
    small_tf = tf_order[0]
    df_small = dfs[small_tf]
    for tf in tf_order:
        if not isinstance(dfs[tf].index, pd.DatetimeIndex):
            raise ValueError(f"DataFrame للفريم {tf} يجب أن يكون مفهرساً بـ DatetimeIndex")
    df_small = df_small.sort_index()
    win_small = window_sizes[small_tf]
    ends = window_end_indices(df_small.index, win_small, stride, small_tf, config)
    end_ns = df_small.index.as_unit('ns').asi8[ends]                 # t لكل عيّنة

    frames = {tf: dfs[tf].sort_index() for tf in tf_order[1:]}
    last_pos: Dict[str, np.ndarray] = {}
    keep = np.ones(len(ends), dtype=bool)
    for tf, df_tf in frames.items():
        win = window_sizes[tf]
        try:
            dur = int(pd.Timedelta(tf) // pd.Timedelta(1, 'ns'))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"higher_tf_mode='closed' يتطلّب فريماً بمدة ثابتة، لا {tf!r}") from exc
        idx_ns = df_tf.index.as_unit('ns').asi8
        # آخر شمعة عليا فتحها ≤ t − dur ⇔ إغلاقها ≤ t
        pos = np.searchsorted(idx_ns, end_ns - dur, side='right') - 1
        start = pos - (win - 1)
        ok = start >= 0
        p, s = np.clip(pos, 0, None), np.clip(start, 0, None)
        ok &= (idx_ns[p] - idx_ns[s]) == (win - 1) * dur                       # متصلة بلا فجوة
        ok &= (end_ns - (idx_ns[p] + dur)) < dur                                # وهي الأحدث المغلقة فعلاً
        keep &= ok
        last_pos[tf] = pos

    ends = ends[keep]
    out: Dict[str, np.ndarray] = {}
    values = df_small.to_numpy(dtype='float32', copy=False)
    out[small_tf] = (_windows_at(values, ends - (win_small - 1), win_small) if len(ends)
                     else np.zeros((0, win_small, df_small.shape[1]), dtype='float32'))
    for tf, df_tf in frames.items():
        win = window_sizes[tf]
        if len(ends):
            out[tf] = _windows_at(df_tf.to_numpy(dtype='float32', copy=False),
                                  last_pos[tf][keep] - (win - 1), win)
        else:
            out[tf] = np.zeros((0, win, df_tf.shape[1]), dtype='float32')
    return out, list(df_small.index[ends])


def _align_single_tf_view(dfs: Dict[str, pd.DataFrame], tf: str, win: int,
                          stride: int, config: Optional[dict] = None,
                          ) -> Tuple[Dict[str, np.ndarray], List[pd.Timestamp]]:
    """مسار فريم واحد عبر ``sliding_window_view`` — بلا أي نسخة إلا الأخيرة."""
    df = dfs[tf]
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"DataFrame للفريم {tf} يجب أن يكون مفهرساً بـ DatetimeIndex")
    df = df.sort_index()
    n_feat = df.shape[1]

    if len(df) < win:
        return {tf: np.zeros((0, win, n_feat), dtype='float32')}, []

    ends = window_end_indices(df.index, win, stride, tf, config)
    if len(ends) == 0:
        return {tf: np.zeros((0, win, n_feat), dtype='float32')}, []

    values = df.to_numpy(dtype='float32', copy=False)          # (T, F)
    view = np.lib.stride_tricks.sliding_window_view(values, win, axis=0)  # (T-win+1, F, win) — VIEW
    view = np.moveaxis(view, -1, 1)                              # (T-win+1, win, F) — لا يزال VIEW
    starts = ends - (win - 1)
    steps = np.diff(starts)
    if len(steps) == 0 or (steps == steps[0]).all():
        # الحالة المعتادة (لا فجوات): متتالية حسابية ⇒ شريحة VIEW بلا نسخ
        sel = view[starts[0]:starts[-1] + 1:int(steps[0]) if len(steps) else 1]
    else:
        sel = view[starts]                                        # فجوات أسقطت نقاط شبكة
    arr = np.ascontiguousarray(sel)                                # نسخة واحدة فقط هنا
    end_times = list(df.index[ends])
    return {tf: arr}, end_times
