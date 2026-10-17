"""
PURPOSE:  Technical indicator generation via pandas_ta_classic (add_features) and automatic derivation of the feature column list (infer_feature_columns), feature exclusion and price indices.
TAGS:     features, add_features, pandas_ta, indicators, infer_feature_columns, exclude_features, available_features, price_indices, OHLCV
PITFALLS: The feature list is never written by hand: it is derived by running add_features on synthetic data with the current CONFIG. Requires pandas_ta_classic (or pandas_ta) at load time. Executed into the one shared pipeline namespace by data/_loader.py (never imported on its own): names from other modules resolve at call time.

## 8) توليد الميزات الفنية (`features.py` سابقاً)

يستخدم `pandas_ta_classic` (أو `pandas_ta`) لحساب المؤشرات المُعرَّفة في
`CONFIG['indicator_settings']`، ثم يستدعي ميزات custom.py، ثم ينظّف القيم
المفقودة/غير المنتهية.
"""
"""
توليد الميزات الفنية واشتقاق قائمة الأعمدة تلقائياً.

المبدأ الأساسي: قائمة الميزات **لا تُكتب يدوياً أبداً** — تُشتق بتشغيل
:func:`add_features` على بيانات تركيبية بنفس ``CONFIG`` الحالي، فتبقى مطابقة
لمخرجات ``pandas_ta`` الفعلية مهما عُدّلت ``indicator_settings``.
"""

# تسجيل ملحق df.ta — ``pandas_ta_classic`` هو الحزمة المتوافقة مع pandas الحديثة،
# مع تراجع تلقائي إلى ``pandas_ta`` الأصلية.
try:                                              # pragma: no cover - يعتمد على البيئة
    import pandas_ta_classic as ta                # noqa: F401
except ImportError:                               # pragma: no cover
    try:
        import pandas_ta as ta                    # noqa: F401
    except ImportError as exc:                    # pragma: no cover
        raise ImportError(
            "يلزم تثبيت pandas_ta_classic (أو pandas_ta):\n"
            "    pip install pandas_ta_classic"
        ) from exc

OHLCV = ("open", "high", "low", "close", "volume")


def add_features(data, config: Optional[dict] = None) -> pd.DataFrame:
    """إضافة كل المؤشرات المُعرَّفة في ``config['indicator_settings']``.

    المؤشرات الثابتة (``static_indicators``) تُحسب **مرة واحدة بعد** حلقة
    المؤشرات المُعامَلية، ولكل منها معاملاته الخاصة من ``static_indicator_params``.
    """
    config = CONFIG if config is None else config
    df = pd.DataFrame(data).copy()
    indicator_settings = config["indicator_settings"]
    static_indicators = config.get("static_indicators", [])
    static_params = config.get("static_indicator_params", {})

    for ind, values in indicator_settings.items():
        if ind == 'stoch':
            for fast_k, slow_k in values:
                df.ta.stoch(fast_k=fast_k, slow_k=slow_k, append=True)
        elif ind == 'stochrsi':
            for rsi_len, stoch_len in values:
                df.ta.stochrsi(rsi_length=rsi_len, stoch_length=stoch_len, append=True)
        elif ind == 'macd':
            for f, s, sig in values:
                df.ta.macd(fast=f, slow=s, signal=sig, append=True)
        elif ind == 'ppo':
            for f, s, sig in values:
                df.ta.ppo(fast=f, slow=s, signal=sig, append=True)
        elif ind == 'donchian':
            for length in values:
                df.ta.donchian(lower_length=length, upper_length=length, append=True)
        else:
            for v in values:
                if hasattr(df.ta, ind):
                    getattr(df.ta, ind)(length=v, append=True)
                else:
                    print(f"⚠️ المؤشر غير موجود: {ind}")

    # ✅ الميزات المخصّصة الساكنة تُضاف قبل الحذف النهائي لصفوف NaN
    if config.get("use_custom_features", True):
        df = add_custom_features(df, config=config)

    for ind in static_indicators:
        if ind == 'cross_value':
            df.ta.cross_value(value=50, append=True)
            continue
        if not hasattr(df.ta, ind):
            print(f"⚠️ المؤشر غير موجود: {ind}")
            continue
        getattr(df.ta, ind)(append=True, **static_params.get(ind, {}))

    df.dropna(inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


def infer_feature_columns(config: Optional[dict] = None,
                          min_rows: int = 600) -> List[str]:
    """اشتقاق قائمة الميزات الحقيقية بتشغيل :func:`add_features` على بيانات تركيبية.

    يمنع أي تعارض بين الأسماء المتوقَّعة والأسماء التي تُنتجها ``pandas_ta`` فعلياً.
    """
    config = CONFIG if config is None else config
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 0.5, min_rows))
    high = close + rng.random(min_rows) * 1.5
    low = close - rng.random(min_rows) * 1.5
    open_ = close + rng.normal(0, 0.3, min_rows)
    volume = rng.random(min_rows) * 1000 + 100
    idx = pd.date_range("2024-01-01", periods=min_rows, freq="h")
    sample = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    processed = add_features(sample, config=config)
    exclude = set(config.get("exclude_from_features", []))
    cols = [c for c in processed.columns if c not in exclude]
    # ✅ أعمدة السياق السوقي (MKT_) لا تُنتجها add_features (مصدرها عملة
    # أخرى، لا هذا الإطار نفسه) — تُدرَج هنا فقط لتظهر في feature_order/
    # X_tf تلقائياً؛ قيمها الفعلية تُملأ لاحقاً بـ add_market_context.
    cols += [c for c in market_context_columns(config) if c not in exclude]
    # ✅ معدّل التمويل/الفائدة المفتوحة (add_funding_oi_features) — مصدرها
    # أرشيف Drive الخاص بكل رمز، لا هذا الإطار نفسه، كـMKT_ أعلاه بالضبط.
    cols += [c for c in funding_oi_feature_columns(config) if c not in exclude]
    # ✅ رتبة الزخم المقطعية (MOM_RANK_) — مصدرها كل الأصول معاً (build_dataset)
    # لا هذا الإطار وحده، كـMKT_/funding أعلاه بالضبط.
    cols += [c for c in momentum_rank_columns(config) if c not in exclude]
    # ✅ نسبة كفاءة كوفمان الداخل-يومية (EFF_RATIO_) — أرشيف خارجي محسوب من
    # بيانات ساعية، لا هذا الإطار نفسه، كـFUND_/OI_ أعلاه بالضبط.
    cols += [c for c in intraday_efficiency_columns(config) if c not in exclude]
    # ✅ انحراف VWAP الداخل-يومي (VWAP_DEVIATION_) — نفس مصدر EFF_RATIO_ أعلاه بالضبط.
    cols += [c for c in intraday_vwap_columns(config) if c not in exclude]
    # ✅ تركّز الحجم الساعي (VOL_CONC_) — نفس مصدر EFF_RATIO_/VWAP_DEVIATION_ أعلاه بالضبط.
    cols += [c for c in intraday_volume_concentration_columns(config) if c not in exclude]
    # ✅ المرحلة ٢: أعمدة ITD_ من شموع 15m، ونسب long/short/taker من futures_metrics
    # (FUND_sum_1d/3d تُضاف أعلاه ضمن funding_oi_feature_columns).
    cols += [c for c in intraday_15m_columns(config) if c not in exclude]
    cols += [c for c in futures_metrics_columns(config) if c not in exclude]
    # ✅ اتساق السوق (MKT_BREADTH_) — مصدرها كل الأصول معاً (build_dataset)
    # لا هذا الإطار وحده، كـMOM_RANK_ أعلاه بالضبط.
    cols += [c for c in market_breadth_columns(config) if c not in exclude]
    cols += [c for c in momentum_orth_natr_columns(config) if c not in exclude]
    return cols


def available_features(config: Optional[dict] = None, min_rows: int = 600) -> List[str]:
    """كل أعمدة ``add_features`` الممكنة **قبل** ``exclude_from_features``.

    مرجع الأسماء لـ :func:`exclude_features` (وما يمكنك كتابته في أنماطها).
    """
    config = CONFIG if config is None else config
    return infer_feature_columns({**config, "exclude_from_features": []},
                                 min_rows=min_rows)


def exclude_features(names: Any,
                     config: Optional[dict] = None,
                     refresh: bool = True,
                     verbose: bool = True) -> List[str]:
    """يستبعد ميزات من **مدخلات النموذج** بالاسم أو بنمط، ويُحدّث ``feature_order``.

    >>> exclude_features(['RSI_14', 'BB*', 'TIME_*'])

    * ``names``: أسماء أعمدة أو أنماط ``*``/``?`` (بلا حساسية لحالة الأحرف).
      أسماؤك تُطابَق على :func:`available_features`، فما لا يوجد فيها (خطأ إملائي
      غالباً) يُحذَّر منه ولا يُضاف بصمت.
    * تُضاف الأسماء إلى ``CONFIG['exclude_from_features']`` — المصدر الوحيد للحقيقة —
      فتبقى الأعمدة محسوبة داخلياً (للأهداف والمقياس) وتغيب عن المدخلات فقط.
      لإلغاء الاستبعاد: ``update_config(exclude_from_features=[...])`` بالقائمة المطلوبة.
    * استدعِها **قبل** ``build_dataset``: بيانات بُنيت سابقاً تحمل الأبعاد القديمة.
    * ترفع ``ValueError`` (دون تعديل شيء) إن كان الاستبعاد سيُفرغ كل الميزات.

    Returns:
        الأسماء التي أُضيفت فعلاً في هذا الاستدعاء (فارغة إن لم يتغيّر شيء).

    لإيقاف **عائلة مؤشر كاملة** من الحساب أصلاً (لا مجرد إخفائها) عدّل
    ``indicator_settings`` أو ``static_indicators`` — الاستبعاد هنا أسرع تجريباً
    وأأمن لأنه لا يمسّ الميزات التي تعتمد عليها الأهداف.
    """
    import fnmatch

    config = CONFIG if config is None else config
    if isinstance(names, str):
        names = [names]
    patterns = [str(n).strip() for n in names if str(n).strip()]

    universe = available_features(config)
    current = list(config.get("exclude_from_features") or [])
    taken = set(current)
    added: List[str] = []
    already: List[str] = []
    unknown: List[str] = []

    for pat in patterns:
        hits = [c for c in universe if fnmatch.fnmatchcase(c.lower(), pat.lower())]
        if not hits:
            unknown.append(pat)
            continue
        for c in hits:
            if c in taken:
                if c not in already:
                    already.append(c)
            else:
                taken.add(c)
                added.append(c)

    if not [c for c in universe if c not in taken]:
        raise ValueError("❌ هذا الاستبعاد سيُزيل كل الميزات — لم يُغيَّر شيء.")

    if added:
        config["exclude_from_features"] = current + added
        if refresh:
            refresh_features(config)
    if verbose:
        if added:
            print(f"🚫 استُبعدت {len(added)} ميزة: {added[:12]}{'...' if len(added) > 12 else ''}")
        if already:
            print(f"   ℹ️ مُستبعَدة مسبقاً: {already[:8]}{'...' if len(already) > 8 else ''}")
        if unknown:
            print(f"   ⚠️ لا ميزة بهذه الأسماء/الأنماط (تحقّق من الإملاء أو استعرض "
                  f"available_features()): {unknown}")
    return added


def extract_features(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """انتقاء أعمدة الميزات المتاحة فقط، مع تحذير عن المفقود بدل ``KeyError``."""
    available = [c for c in features if c in df.columns]
    missing = [c for c in features if c not in df.columns]
    if missing:
        print(f"⚠️ أعمدة مفقودة ({len(missing)}) — تُتجاهَل: {missing[:8]}"
              f"{'...' if len(missing) > 8 else ''}")
    return df[available]


def price_indices(features: Optional[List[str]] = None,
                  config: Optional[dict] = None) -> dict:
    """فهارس أعمدة السعر داخل مصفوفة الميزات.

    تُستخدم بدل تثبيت أرقام الأعمدة (0/1/2)، فتبقى صحيحة مهما تغيّرت
    ``exclude_from_features`` أو ترتيب المؤشرات.
    """
    config = CONFIG if config is None else config
    if features is None:
        features = feature_order(config)
    idx = {}
    for col in ("high", "low", "close", "open", "volume"):
        if col in features:
            idx[col] = features.index(col)
    for required in ("high", "low", "close"):
        if required not in idx:
            raise ValueError(
                f"العمود '{required}' غير موجود في feature_order — "
                "لا يمكن حساب الأهداف. راجع exclude_from_features.")
    return idx
