"""
PURPOSE:  Historical data sources (mounted Drive only): asset registry, coin selection by category, coin exclusion, load_asset and resampling (make_resample_fn).
TAGS:     sources, load_asset, load_asset_registry, flatten_coins_by_category, filter_desired_coins, exclude_coins, resample_timeframes, make_resample_fn, ALL_GLOBAL, drive_raw_dir
PITFALLS: No public links / gdown: every read goes through mount_drive(). flatten_coins_by_category() without arguments uses DEFAULT_ENABLED_CATEGORIES, not the whole registry. Executed into the one shared pipeline namespace by data/_loader.py (never imported on its own): names from other modules resolve at call time.

## 14) مصادر البيانات التاريخية (`sources.py` سابقاً) — Google Drive المُركَّب حصراً

**تبسيط جوهري عن النسخة الأصلية:** الملف الأصلي كان يدعم طريقتين للتحميل
(Drive مُركَّب، أو رابط عام عبر `gdown`) مع منطق تبديل تكيّفي بينهما. هنا
**طريقة واحدة فقط**: Drive المُركَّب. سجل الأصول أيضاً يُقرأ من Drive
(`CONFIG['asset_registry_path']`) لا من رابط عام.

> ℹ️ `ALL_GLOBAL` مُعرَّفة هنا (لا في قسم الإعدادات) لأنها تنتمي أصلاً لهذا
> الملف في الحزمة الأصلية — تعني "لا تصفية فئات، خُذ سجل الأصول كاملاً".
"""
# @title
"""
مصادر البيانات التاريخية: سجل الأصول، تحميل العملات، وإعادة أخذ العينات.

هذه هي الطبقة الوحيدة التي تعرف **من أين** تأتي البيانات التاريخية؛ بقية
الحزمة تتعامل مع ``DataFrame`` مفهرس زمنياً فقط. (للبيانات **الحيّة** من
Binance، انظر القسم اللاحق "تحضير بيانات حيّة".)
"""

#: قيمة خاصة لـ categories في flatten_coins_by_category: خُذ كل عملات سجل
#: الأصول بلا أي تصفية بالفئات.
ALL_GLOBAL = 'all_global'

OHLCV_AGG = {'open': 'first', 'high': 'max', 'low': 'min',
             'close': 'last', 'volume': 'sum'}


# ══════════════════════════════════════════════════════════════════════════
# سجل الأصول
# ══════════════════════════════════════════════════════════════════════════
def load_asset_registry(path: Optional[str] = None,
                        config: Optional[dict] = None) -> List[Dict]:
    """سجل العملات ``[{'name':..., ...}, ...]`` — من CSV محلي صريح (``path``)،
    أو من Google Drive المُركَّب حسب ``CONFIG['asset_registry_path']``
    (مسار نسبي داخل ``MyDrive``). لا رابط عام في أي من الحالتين."""
    config = CONFIG if config is None else config
    if path:
        df = pd.read_csv(path)
    else:
        rel = config.get("asset_registry_path")
        if not rel:
            raise ValueError("مرّر path أو عيّن CONFIG['asset_registry_path'].")
        root = mount_drive(config=config)
        if root is None:
            raise RuntimeError("تعذّر تركيب Google Drive — تأكد من العمل داخل Colab.")
        df = pd.read_csv(root / rel)
    print(f"📒 سجل الأصول: {len(df)} عملة")
    return df.to_dict('records')


def flatten_coins_by_category(category_dict: Optional[Dict[str, List[str]]] = None,
                              categories: Optional[List[str]] = None) -> List[str]:
    """تسطيح قاموس الفئات إلى قائمة أسماء واحدة.

    Args:
        categories: الفئات المطلوبة (افتراضياً :data:`DEFAULT_ENABLED_CATEGORIES`
            — ``["Large_Caps"]`` بالإعداد الافتراضي).
            ``'all'`` أو ``[]`` → كل الفئات المُعرَّفة في :data:`COINS_BY_CATEGORY`.
            ``'all_global'`` (أي :data:`ALL_GLOBAL`) → لا تصفية إطلاقاً (تُعالَج
            في :func:`filter_desired_coins`، فتُؤخذ كل عملة في السجل حتى لو لم
            تُذكر في أي فئة).
    """
    category_dict = COINS_BY_CATEGORY if category_dict is None else category_dict
    if categories is None:
        categories = DEFAULT_ENABLED_CATEGORIES
    if categories == ALL_GLOBAL:
        print("🌍 all_global: كل عملات السجل بلا تصفية")
        return []
    if categories == 'all' or not categories:
        selected = list(category_dict.keys())
    else:
        selected = [c for c in categories if c in category_dict]
        unknown = set(categories) - set(category_dict)
        if unknown:
            print(f"⚠️ فئات غير معروفة تُتجاهَل: {sorted(unknown)}")

    flat: List[str] = []
    for cat in selected:
        flat.extend(category_dict[cat])
    print(f"🎯 الفئات المختارة: {selected} → {len(flat)} عملة")
    return flat


def filter_desired_coins(full_registry: List[Dict],
                         desired_coins: Optional[List[str]] = None,
                         verbose: bool = True) -> List[Dict]:
    """تصفية سجل الأصول على العملات المطلوبة، مع تقرير عن غير الموجود.

    ``desired_coins`` فارغة أو ``None`` → **لا تصفية**: يُؤخذ السجل كاملاً
    (سلوك ``categories=ALL_GLOBAL``).
    """
    if not desired_coins:
        usable = [c for c in full_registry if c.get('name')]
        if verbose:
            print(f"🌍 بلا تصفية: {len(usable)} عملة من السجل كاملاً")
            dropped = len(full_registry) - len(usable)
            if dropped:
                print(f"   ⚠️ {dropped} سطراً بلا عمود 'name' صالح تُتجاهَل")
        return usable

    desired_set = set(desired_coins)
    selected = [coin for coin in full_registry if coin.get('name') in desired_set]
    if verbose:
        found = {c['name'] for c in selected}
        missing = sorted(desired_set - found)
        print(f"✅ وُجد {len(selected)} من أصل {len(desired_set)} عملة مطلوبة")
        if missing:
            print(f"   ⚠️ غير موجودة في السجل: {missing[:10]}"
                  f"{'...' if len(missing) > 10 else ''}")
    return selected


# ══════════════════════════════════════════════════════════════════════════
# استبعاد عملات بعينها
# ══════════════════════════════════════════════════════════════════════════
def _coin_key(name: Any) -> str:
    """مفتاح المطابقة: أحرف كبيرة بلا مسافات طرفية — ``' btcusdt '`` = ``'BTCUSDT'``."""
    return str(name).strip().upper()


def _coin_name(item: Any) -> str:
    """اسم العملة من عنصر ``configs`` (قاموس فيه ``name``) أو من نص مباشر."""
    return str(item.get('name', '')) if isinstance(item, dict) else str(item)


def split_excluded_coins(configs: Iterable[Any],
                         excluded: Optional[Iterable[str]] = None,
                         config: Optional[dict] = None
                         ) -> Tuple[List[Any], List[str], List[str]]:
    """يفصل ``configs`` إلى ``(المُبقاة، أسماء المُستبعَدة فعلاً، أسماء لم تُوجد)``.

    Args:
        excluded: أسماء العملات المطلوب تخطّيها. ``None`` → تُؤخذ من
            ``CONFIG['excluded_coins']``؛ قائمة صريحة (حتى الفارغة) تحلّ محلّها
            ولا تُدمج معها. نصّ واحد (``'BTCUSDT'``) يُعامَل كاسم واحد لا كأحرف.

    المطابقة على **الاسم الكامل** بلا حساسية لحالة الأحرف. لا تُطابَق بالجذر
    (``'BTC'`` لا تستبعد ``BTCUSDT``) عمداً: الجذر الواحد قد يجمع عملات مختلفة
    (``BTCUSDT``/``BTCDOMUSDT``)، والاستبعاد الخاطئ الصامت أسوأ من رفض صريح.
    الأسماء التي لم تُوجد في ``configs`` تُرجَع للتحذير منها (خطأ إملائي غالباً).
    ``configs`` نفسها لا تُعدَّل.
    """
    config = CONFIG if config is None else config
    if excluded is None:
        excluded = config.get('excluded_coins')
    if excluded is None:
        excluded = []
    elif isinstance(excluded, str):
        excluded = [excluded]

    wanted: Dict[str, str] = {}
    for n in excluded:
        if str(n).strip():
            wanted.setdefault(_coin_key(n), str(n).strip())

    kept: List[Any] = []
    removed: List[str] = []
    seen = set()
    for item in configs:
        key = _coin_key(_coin_name(item))
        if key in wanted:
            removed.append(_coin_name(item))
            seen.add(key)
        else:
            kept.append(item)
    not_found = [orig for key, orig in wanted.items() if key not in seen]
    return kept, removed, not_found


def exclude_coins(configs: Iterable[Any],
                  excluded: Optional[Iterable[str]] = None,
                  config: Optional[dict] = None,
                  verbose: bool = True) -> List[Any]:
    """يُرجع ``configs`` بعد حذف العملات المُستبعَدة، فتُتخطّى في كل ما يليها.

    >>> configs = exclude_coins(configs, ['TSLAUSDT', 'XAUUSDT'])

    ``build_dataset*`` تُطبّق ``CONFIG['excluded_coins']`` تلقائياً، فلا حاجة
    لاستدعائها يدوياً إلا لتصفية قائمة قبل بناء الدفعة (أو لمرة واحدة بلا
    لمس ``CONFIG``). تفاصيل المطابقة: :func:`split_excluded_coins`.
    """
    kept, removed, not_found = split_excluded_coins(configs, excluded, config)
    if verbose:
        if removed:
            print(f"🚫 استُبعدت {len(removed)} عملة: {removed[:10]}"
                  f"{'...' if len(removed) > 10 else ''}")
        if not_found:
            print(f"   ⚠️ أسماء استبعاد غير موجودة في configs — تحقّق من الإملاء: "
                  f"{not_found[:10]}{'...' if len(not_found) > 10 else ''}")
    return kept


# ══════════════════════════════════════════════════════════════════════════
# تحميل عملة واحدة — Google Drive المُركَّب حصراً
# ══════════════════════════════════════════════════════════════════════════
def load_asset(file_id: Optional[str], name: str = "",
               config: Optional[dict] = None) -> pd.DataFrame:
    """تحميل عملة واحدة كـ ``DataFrame`` مفهرس بـ ``timestamp`` ومرتّب زمنياً —
    **حصراً** من Google Drive المُركَّب (``drive.mount``). لا رابط عام
    (``uc?id=``) ولا ``gdown`` في أي مسار من مسارات هذه الدالة.

    ``name`` (أو ``file_id`` إن لم يُحدَّد ``name``) يُستخدَم لبناء اسم الملف
    داخل ``CONFIG['drive_raw_dir']`` حسب نمط ``CONFIG['drive_raw_pattern']``.

    الإمضاء ``(file_id, name, config=None)`` مطابق عمداً لما تتوقّعه
    :func:`build_dataset_from_loader` (تستدعي ``load_asset_fn(cfg.get('file_id'), name)``)،
    حتى لو لم يعد ``file_id`` يحمل أي معنى فعلي هنا.
    """
    config = CONFIG if config is None else config
    key = name or file_id
    if not key:
        raise ValueError("يلزم name أو file_id لتحديد اسم ملف العملة.")

    raw_dir = config.get("drive_raw_dir")
    if not raw_dir:
        raise ValueError("عيّن CONFIG['drive_raw_dir'].")
    pattern = config.get("drive_raw_pattern", "{name}.csv")

    root = mount_drive(config=config)
    if root is None:
        raise RuntimeError("تعذّر تركيب Google Drive — تأكد من العمل داخل Colab.")

    path = root / raw_dir / pattern.format(name=key)
    if not path.exists():
        # ملفات fetch_history_vision_colab تُحفظ مضغوطة (<S>.csv.gz) افتراضياً: جرّب الصيغة الأخرى
        alt = (path.with_name(path.name[:-3]) if path.name.endswith(".gz")
               else path.with_name(path.name + ".gz"))
        if not alt.exists():
            raise FileNotFoundError(f"[{key}] الملف غير موجود: {path}")
        path = alt

    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()

    # ملفات history_1d تحمل ``datetime_utc`` (وقتاً مقروءاً) وقد يحمل الملف أيضاً
    # ``timestamp`` كأرقام ms — تفسيرها كنص وقت يعطي تواريخ 1970. فيُفضَّل
    # ``datetime_utc``؛ وبقاء ``timestamp`` احتياطاً يُبقي ملفات 1h القديمة تعمل.
    ts_col = 'datetime_utc' if 'datetime_utc' in df.columns else 'timestamp'
    if ts_col not in df.columns:
        raise ValueError(
            f"[{key}] لا عمود 'datetime_utc' ولا 'timestamp' — "
            f"الأعمدة المتاحة: {list(df.columns)}")

    df['timestamp'] = pd.to_datetime(df[ts_col], utc=True, errors='coerce')
    df = df.dropna(subset=['timestamp']).set_index('timestamp').sort_index()
    return df


#: اسم متوافق مع الكود القديم الذي كان يستدعي load_asset2 صراحةً — نفس الدالة
#: الوحيدة الآن.
load_asset2 = load_asset


# ══════════════════════════════════════════════════════════════════════════
# إعادة أخذ العينات
# ══════════════════════════════════════════════════════════════════════════
def resample_timeframes(df: pd.DataFrame,
                        target_tfs: Optional[List[str]] = None,
                        base_tf: Optional[str] = None,
                        with_features: bool = True,
                        config: Optional[dict] = None) -> Dict[str, pd.DataFrame]:
    """تحويل ``df`` إلى كل الفريمات المطلوبة، مع إضافة المؤشرات لكل فريم.

    Args:
        target_tfs: الفريمات المطلوبة بصيغة pandas offset (``['1h','4h','1D']``).
        base_tf: فريم ``df`` الأصلي؛ إن لم يُعطَ يُفترض أنه أصغر فريم مطلوب.
        with_features: True يستدعي :func:`add_features` على كل فريم — اتركها True
            وإلا لن تطابق الأعمدةُ ``feature_order``.

    ``CONFIG['higher_tf_mode'] == 'closed'``: الفريمات الأعلى تُبقي الشموع الكاملة فقط (:func:`_keep_full_bars`) —
    الشمعة الأخيرة الناقصة في نهاية البيانات وشموع الفجوات تُحذف قبل حساب المؤشرات، فكل مؤشر على 4h محسوب من شموع
    مُكوَّنة ومغلقة فقط (ومحاذاتها عند العيّنة في :func:`_align_closed_higher_tfs`). الفريم الأصغر لا يُصفّى.
    """
    config = CONFIG if config is None else config
    target_tfs = config["tf_order"] if target_tfs is None else target_tfs
    df = df.sort_index()

    smallest = target_tfs[0]
    if base_tf != smallest:
        df = df.resample(smallest).agg(OHLCV_AGG).dropna(how='all')
        base_tf = smallest

    closed = _higher_tf_closed(dict(config, tf_order=list(target_tfs)))
    dfs: Dict[str, pd.DataFrame] = {}
    for tf in target_tfs:
        frame = df if tf == base_tf else df.resample(tf).agg(OHLCV_AGG).dropna(how='all')
        if closed and tf != base_tf:
            frame = _keep_full_bars(frame, df.index[df['close'].notna()], tf, base_tf)
        dfs[tf] = add_features(frame.copy(), config=config) if with_features else frame.copy()
    return dfs


def make_resample_fn(config: Optional[dict] = None):
    """يُنتج ``resample_fn`` مربوطة بـ ``config`` — جاهزة لـ ``build_dataset_from_loader``
    ولخط الأنابيب الحيّ أيضاً (القسم اللاحق)."""
    config = CONFIG if config is None else config

    def _fn(df: pd.DataFrame, tf_order: List[str]) -> Dict[str, pd.DataFrame]:
        return resample_timeframes(df, tf_order, config=config)

    return _fn
