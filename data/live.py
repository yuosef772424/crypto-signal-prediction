"""
PURPOSE:  Live dataset built with the very same prepare_single_asset (Binance instead of Drive): dummy tail padding, drop_tail_per_asset, extract_last_batch.
TAGS:     live, build_dataset_live, extract_last_batch, drop_tail_per_asset, dummy tail, inference
PITFALLS: The last row per coin has real features but a dummy target; never evaluate on it. Executed into the one shared pipeline namespace by data/_loader.py (never imported on its own): names from other modules resolve at call time.

## 18) بناء بيانات حيّة — بنفس دالة `prepare_single_asset` بلا أي تكرار

هذا القسم يطبّق ما طلبتَه بالحرف: **تجهيز البيانات الحيّة مطابق تماماً**
لتجهيز بيانات التدريب — نفس `resample_fn`، ونفس `prepare_single_asset` عبر
نفس `build_dataset_from_loader` (صفر نسخ مكرّرة من منطق المحاذاة/التطبيع/الهدف).
الاختلاف الوحيد هو **مصدر الجلب**: `fetch_data` (Binance) بدل `load_asset`
(Drive) — تماماً كما طلبت.

**كيف نحصل على عيّنة بلا هدف حقيقي دون تعديل `prepare_single_asset`؟**
نُلحق بذيل البيانات المجلوبة **دفعة صورية** (candles وهمية بطول كافٍ، سعرها
= آخر إغلاق حقيقي، حجمها صفر) قبل تمريرها لخط الأنابيب، ونستدعي البناء
بـ `stride=1` بدل قيمة `CONFIG['stride']` المعتادة (لمرة هذا الاستدعاء
فقط، دون أي تعديل على `CONFIG` نفسه). هذا يمنح `prepare_single_asset` "مستقبلاً"
كافياً فلا تُسقِط آخر شمعة حقيقية بدل حسابها، فيُحسب لها **هدف صوري**
(بسعر يساوي آخر إغلاق حقيقي — قيمة تحكيمية لا معنى تنبؤياً لها) عوضاً عن
استبعادها بالكامل — تماماً كما طلبت ("بدون هدف أو بهدف افتراضي"). بعد
البناء، تُحذَف تلقائياً كل النوافذ التي **ميزاتها هي** الملوَّثة بالدفعة
الصورية (وليس فقط هدفها) عبر `drop_tail_per_asset` — فيبقى آخر صف لكل
عملة **حقيقي الميزات بالكامل** (يعتمد فقط على شموع حقيقية) بهدف صوري فقط.

**النتيجة:** لكل عملة، كل الصفوف ما عدا الأخير تملك **أهدافاً حقيقية
محسوبة فعلياً** من داخل الدفعة المجلوبة نفسها — تصلح لتقييم أداء النموذج
على الأيام الأخيرة (اختبار فوري)، أمّا الصف الأخير فهو نقطة **live**
الحقيقية (بلا هدف صالح) الجاهزة للاستدلال الفعلي. `extract_last_batch`
أدناه يستخلص هذا الصف الأخير لكل عملة من أي مجموعة بيانات مُجهَّزة.
"""
# @title
def _live_pad_length(tf_order: List[str], window_sizes: Dict[str, int],
                     forecast_horizon: int, download_interval: str) -> int:
    """طول الدفعة الصورية، بوحدة شموع ``download_interval`` — **ليس** بالضرورة
    أصغر فريم في ``tf_order`` بعد الآن: تُجلَب بيانات Binance بـ
    ``download_interval`` ثم تُجمَّع (resample) صعوداً إلى كل فريم في
    ``tf_order``، فقد يكون ``download_interval`` أدقّ منه (نموذج يومي مبنيّ
    من بيانات ساعية مثلاً) — ولهذا يُحوَّل ``forecast_horizon`` (بوحدة
    tf_order[0]) وهامش تلوّث أكبر فريم كلاهما لوحدة ``download_interval``
    قبل جمعهما، لا افتراض أنهما نفس الوحدة كما كان سابقاً.
    """
    ratios = [max(1, int(np.ceil(pd.Timedelta(tf) / pd.Timedelta(download_interval))))
             for tf in tf_order]
    base_ratio = ratios[0]      # عدد شموع download_interval في شمعة واحدة من tf_order[0]
    return forecast_horizon * base_ratio + max(ratios)


def _pad_with_dummy_tail(df: pd.DataFrame, download_interval: str, n: int) -> pd.DataFrame:
    """يُلحق ``n`` شمعة صورية بذيل ``df`` (مفهرس بـ DatetimeIndex)، بخطوة
    ``download_interval`` (فريم التحميل الفعلي من Binance، لا فريم النموذج) —
    السعر = آخر إغلاق حقيقي (ثابت)، الحجم = صفر. هذا يمنح ``prepare_single_asset``
    \"مستقبلاً\" كافياً لحساب هدف (صوري) لآخر شمعة حقيقية بدل إسقاطها، **دون
    أي تعديل على تلك الدالة**.
    """
    if len(df) == 0 or n <= 0:
        return df
    last_close = float(df['close'].iloc[-1])
    step = pd.Timedelta(download_interval if download_interval != "1D" else "1d")
    future_idx = df.index[-1] + step * np.arange(1, n + 1)
    dummy = pd.DataFrame(
        {'open': last_close, 'high': last_close, 'low': last_close,
         'close': last_close, 'volume': 0.0},
        index=future_idx)
    return pd.concat([df, dummy])


def _auto_live_limit(tf_order: List[str], window_sizes: Dict[str, int],
                     forecast_horizon: int, pad_n: int, download_interval: str) -> int:
    """عدد الشموع الحقيقية المطلوب جلبها من Binance بوحدة ``download_interval``
    (لا يشمل الدفعة الصورية).

    ✅ بلا حدّ أقصى هنا (لا ``min(..., BINANCE_MAX_LIMIT)`` كسابقاً): ``fetch_data``
    تُقسّم أي ``limit`` أكبر من ``BINANCE_MAX_LIMIT`` على عدة طلبات تلقائياً —
    تقييد الطلب هنا كان يُبطل تلك الإضافة، وهو تحديداً سبب "٣٢ شمعة فقط" في
    محاذاة النوافذ سابقاً حين يكبر ``download_interval`` نسبةً لأصغر فريم في
    ``tf_order`` (نموذج يومي يحتاج مئات الساعات من الإحماء، تتجاوز ١٥٠٠
    بسهولة). القصّ الوحيد المتبقي الآن هو التاريخ الفعلي المتاح للعملة نفسها
    (تتولّاه fetch_data بإرجاع ما توفّر لا خطأ).
    """
    ratios = {tf: max(1, int(np.ceil(pd.Timedelta(tf) / pd.Timedelta(download_interval))))
             for tf in tf_order}
    biggest_span = max(
        (window_sizes[tf] + LIVE_FEATURE_WARMUP) * ratios[tf] for tf in tf_order
    )
    base_ratio = ratios[tf_order[0]]
    return (biggest_span + forecast_horizon * base_ratio) * 2 + pad_n


def _live_contamination_len(tf_order: List[str]) -> int:
    """عدد نوافذ ``tf_order[0]`` (وحدة النموذج) المُلوَّثة في الذيل — يعتمد
    فقط على نِسَب فريمات ``tf_order`` فيما بينها، بصرف النظر تماماً عن
    ``download_interval``: فريم الجلب لا يُغيّر عدد نوافذ *النموذج* المتضرّرة،
    فقط عدد شموع *التحميل* الصورية اللازمة لتوليدها (انظر _live_pad_length).
    """
    ratios = [max(1, int(np.ceil(pd.Timedelta(tf) / pd.Timedelta(tf_order[0]))))
             for tf in tf_order]
    return max(ratios)


# ══════════════════════════════════════════════════════════════════════════
# أدوات عامة: اقتطاع/استخلاص حسب نطاقات الأصول (asset_bounds)
# ══════════════════════════════════════════════════════════════════════════
def _asset_tail_mask(dataset: Dict, n: int, keep_tail: bool) -> np.ndarray:
    """قناع منطقي ``(N,)`` يُبقي (``keep_tail=True``) أو يُسقط
    (``keep_tail=False``) آخر ``n`` عيّنة من نطاق **كل أصل** في
    ``dataset['asset_bounds']`` على حدة (لا آخر n من المصفوفة كلها)."""
    total = len(dataset['base_params'])
    bounds = dataset.get('asset_bounds') or [{'start': 0, 'end': total}]
    mask = np.zeros(total, dtype=bool) if keep_tail else np.ones(total, dtype=bool)
    for b in bounds:
        s, e = b['start'], b['end']
        cut = max(s, e - n)
        mask[cut:e] = keep_tail
    return mask


def _apply_asset_mask(dataset: Dict, mask: np.ndarray) -> Dict:
    """يقتطع ``dataset`` حسب قناع منطقي عام، ويُعيد بناء ``asset_bounds``
    بإزاحات صحيحة (كل أصل يفقد عدداً مختلفاً من عيّناته حسب القناع)."""
    array_keys = ({'base_params', 'last_candles', 'is_live'}
                  | {f'X_{tf}' for tf in dataset['timeframes']}
                  | {f'y_{h}' for h in dataset['targets']})
    out: Dict = {k: v for k, v in dataset.items() if k not in array_keys | {'asset_bounds'}}

    out['base_params'] = dataset['base_params'][mask]
    out['last_candles'] = dataset['last_candles'][mask]
    if 'is_live' in dataset:
        out['is_live'] = dataset['is_live'][mask]
    for tf in dataset['timeframes']:
        out[f'X_{tf}'] = dataset[f'X_{tf}'][mask]
    for h in dataset['targets']:
        out[f'y_{h}'] = dataset[f'y_{h}'][mask]

    new_bounds, offset = [], 0
    for b in dataset.get('asset_bounds') or []:
        s, e = b['start'], b['end']
        kept = int(mask[s:e].sum())
        if kept:
            new_bounds.append({**{k: v for k, v in b.items() if k not in ('start', 'end')},
                               'start': offset, 'end': offset + kept})
            offset += kept
    out['asset_bounds'] = new_bounds
    return out


def drop_tail_per_asset(dataset: Dict, n: int) -> Dict:
    """يُسقط آخر ``n`` عيّنة من **كل أصل** على حدة — تُستخدَم داخلياً لحذف
    النوافذ المموَّهة بالدفعة الصورية (ميزاتها لا هدفها فقط) بعد بناء بيانات
    حيّة بـ ``stride=1``. عامة الاستخدام: تصلح لأي ``dataset`` فيه
    ``asset_bounds``."""
    return _apply_asset_mask(dataset, _asset_tail_mask(dataset, n, keep_tail=False))


def extract_last_batch(dataset: Dict, n: int = 1) -> Dict:
    """يستخلص آخر ``n`` عيّنة من **كل عملة** في مجموعة بيانات مُجهَّزة (ناتج
    ``build_dataset``/``build_dataset_live`` أياً كان مصدرها) — العيّنات
    الأحدث زمنياً لكل أصل. مع ``build_dataset_live`` هذه هي عيّنة **live**
    (بميزات حقيقية بالكامل وهدف صوري يجب تجاهله)؛ مع مجموعة تاريخية عادية
    هي ببساطة آخر عيّنة حقيقية لكل عملة."""
    result = _apply_asset_mask(dataset, _asset_tail_mask(dataset, n, keep_tail=True))
    print(f"📤 استُخلصت آخر {n} عيّنة لكل عملة عبر "
          f"{len(result.get('asset_bounds') or [])} أصل "
          f"({len(result['base_params']):,} عيّنة إجمالاً)")
    return result


# ══════════════════════════════════════════════════════════════════════════
# البناء الحيّ — إعادة استخدام كاملة لـ build_dataset_from_loader
# ══════════════════════════════════════════════════════════════════════════
def build_dataset_live(
    symbols: List[str],
    limit: Optional[int] = None,
    max_workers: Optional[int] = None,
    config: Optional[dict] = None,
    **kwargs,
) -> Dict:
    """يبني مجموعة بيانات حيّة من Binance Futures — عبر **نفس**
    :func:`build_dataset_from_loader` و:func:`prepare_single_asset` المستخدمتين
    في المسار التاريخي، بلا أي تغيير عليهما. الفرق الوحيد: ``load_asset_fn``
    يجلب من Binance (:func:`fetch_data`) بدل Drive، وتُستدعى المحاذاة بـ
    ``stride=1`` مع دفعة صورية في الذيل بدل قيمة ``CONFIG['stride']``
    المعتادة (لمرة هذا الاستدعاء فقط — ``CONFIG`` لا يتغيّر).

    آخر عيّنة لكل رمز في الناتج **حقيقية الميزات بالكامل** لكن بهدف صوري —
    استخلصها بـ ``extract_last_batch(dataset, n=1)``. كل ما قبلها أهدافه
    حقيقية 100% (اختبار فوري على أحدث بيانات لم يرها التدريب).

    Args:
        symbols: رموز Binance Futures، مثل ``['BTCUSDT', 'ETHUSDT']``.
        limit: عدد الشموع **الحقيقية** المطلوب جلبها لكل رمز. ``None`` يحسبها
            تلقائياً (هامش يكفي أطول نافذة + إحماء الميزات).
        download_interval: فريم Binance الفعلي المطلوب جلبه (افتراضياً
            ``CONFIG['download_interval']``، '1h') — مستقلّ عن ``tf_order``/
            فريم النموذج؛ يُجمَّع (resample) صعوداً إليه بعد الجلب.
        **kwargs: أي معامل آخر يقبله ``build_dataset_from_loader`` (مثل
            ``targets``، ``forecast_horizon``...) يمرَّر كما هو.
    """
    config = CONFIG if config is None else config
    tf_order = kwargs.get('tf_order') or config['tf_order']
    window_sizes = kwargs.get('window_sizes') or config['window_sizes']
    forecast_horizon = kwargs.get('forecast_horizon') or config['forecast_horizon']
    download_interval = kwargs.pop('download_interval', None) or config.get('download_interval', '1h')

    pad_n = _live_pad_length(tf_order, window_sizes, forecast_horizon, download_interval)
    real_limit = limit or _auto_live_limit(tf_order, window_sizes, forecast_horizon, pad_n,
                                           download_interval)

    def _live_loader(file_id, name):
        df = fetch_data(name, download_interval, real_limit)
        if df is None:
            raise RuntimeError(f"تعذّر جلب {name} من Binance")
        df = df.set_index('timestamp').sort_index()
        return _pad_with_dummy_tail(df, download_interval, pad_n)

    configs = [{'name': s} for s in symbols]
    resample_fn = make_resample_fn(config)

    print(f"📡 بناء بيانات حيّة لـ {len(symbols)} رمز "
          f"(limit={real_limit} شمعة حقيقية + {pad_n} صورية، stride=1)...")
    raw = build_dataset_from_loader(
        configs, load_asset_fn=_live_loader, resample_fn=resample_fn,
        stride=1, config=config, max_workers=max_workers, **kwargs,
    )
    # ✅ عدد نوافذ *النموذج* (وحدة tf_order[0]) الملوَّثة في الذيل — مستقلّ
    # تماماً عن download_interval وpad_n (تينك بوحدة تحميل مختلفة قد لا تساوي
    # وحدة النموذج بعد الآن): يعتمد فقط على نِسَب فريمات tf_order فيما بينها
    # (max(ratios) — انظر _live_contamination_len)، وهو ما ينجو من كسر الحلقة
    # في prepare_single_asset رغم أن مركزه (end_idx) نفسه داخل المنطقة
    # الصورية. هذه فقط ميزاتها ملوَّثة فعلياً (مركزها شمعة صورية) ويجب حذفها؛
    # العيّنة الحقيقية المطلوبة (مركزها آخر شمعة حقيقية) هي التي تسبقها مباشرة.
    n_contaminated = _live_contamination_len(tf_order)
    dataset = drop_tail_per_asset(raw, n_contaminated)

    # ✅ شبكة أمان: الدفعة الصورية شموع مسطّحة تماماً (high=low=close) بالبناء —
    # حالة تكاد تستحيل في بيانات سوق حقيقية. إن ظهرت رغم الحذف أعلاه، هذا
    # يعني أن الصيغة الحسابية لم تُطابق حالتك (تهيئة تعدّد فريمات غير معتادة)
    # فيُحذَّر صراحة بدل إرجاع عيّنة "live" ملوَّثة الميزات بصمت.
    last = dataset['last_candles']
    for b in dataset.get('asset_bounds') or []:
        if b['end'] <= b['start']:
            continue
        i = b['end'] - 1
        if last[i, 0] == last[i, 1] == last[i, 2]:
            print(f"   ⚠️ {b['name']}: العيّنة الأخيرة ما زالت تبدو صورية "
                  f"(high=low=close) — راجع forecast_horizon/tf_order/window_sizes.")

    n_assets = len(dataset.get('asset_bounds') or [])
    print(f"✅ بيانات حيّة جاهزة: {len(dataset['base_params']):,} عيّنة عبر "
          f"{n_assets} أصل (آخر عيّنة لكل أصل = live بهدف صوري، البقية اختبار فوري)")
    return dataset
