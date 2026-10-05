"""
PURPOSE:  Funding-rate / open-interest archives turned into features, and the intraday archive features (efficiency ratio, VWAP deviation, volume concentration).
TAGS:     funding rate, open interest, add_funding_oi_features, intraday efficiency, vwap, volume concentration, availability flags
PITFALLS: Missing archive periods are filled neutrally with an *_available flag, never dropped (dropna would silently cut years of history). Executed into the one shared pipeline namespace by data/_loader.py (never imported on its own): names from other modules resolve at call time.

## 17-ب) دمج معدّل التمويل والفائدة المفتوحة كميزات

بروح `add_market_context` (المحاذاة بـ `reindex(..., method='ffill')`)، لكن كل أصل يحمل **أرشيفه الخاص** (`load_funding_open_interest`) لا مرجعاً مشتركاً واحداً. تُدمَج فقط إن `CONFIG['funding_rate']['as_feature']` أو `CONFIG['open_interest']['as_feature']` مفعَّلة (الافتراض الآن: كلاهما True) — الجلب/الأرشفة (`enabled`) والإدماج في المدخلات (`as_feature`) مفتاحان منفصلان عمداً، فيمكن أرشفة بيانات دون دمجها فوراً.


**المرحلة ٢ (أرشيف Binance الكامل):** مع `CONFIG['phase2_data']['use_futures_metrics']` (تلقائياً True إن وُجدت `funding_rate/`/`open_interest/`/`futures_metrics/` على Drive) تُحسب هذه الأعمدة عبر `tools/intraday_features.py` بمحاذاة على **إغلاق** الشمعة (آخر قيمة متاحة قبله، بحدّ عمر يوم)، وتُضاف `FUND_sum_1d/3d` و`LSR_*`/`TAKER_LSR_1d`/`MET_available`. ومع `use_intraday_15m` (مجلد `history_15m/`) تُملأ فتحات `EFF_RATIO_24H`/`VWAP_DEVIATION`/`VOL_CONC_HHI` من شموع 15m الحقيقية + أعمدة `ITD_*`. التفاصيل: `docs/research/phase2_data.md`.
"""
# @title
"""
دمج معدّل التمويل والفائدة المفتوحة كميزات — المرحلة ١ من خطة اكتشاف الإشارة.

بخلاف السياق السوقي (add_market_context، مرجع واحد مشترك بين كل الأصول)،
هنا كل أصل يحمل أرشيفه الخاص المحفوظ عبر save_funding_open_interest. القيمة
الخام غير آمنة كميزة مباشرة على أكثر من محور:

  * open_interest مستوى مطلق يختلف بأوامر عشرية بين الأصول (BTC مقابل عملة
    صغيرة) — بالضبط نفس مشكلة OHLCV الخام التي حُلّت بـ RET_/RANGE_ في
    custom.py؛ الحلّ هنا مماثل: تغيّر نسبي (لوغاريتمي) لا مستوى.
  * الأرشيف يبدأ من أول تشغيل فعلي لـsave_funding_open_interest — فترات
    أقدم (شائعة على بيانات تاريخية بمدى سنوات) بلا أي عيّنة محفوظة أصلاً.
    تركها NaN ثم dropna() في add_features كان سيُسقط كل ذلك التاريخ بصمت؛
    ملء صفر بلا تمييز كان سيخلط "لا تغيّر حقيقي" بـ"لا بيانات إطلاقاً" على
    النموذج. الحل: قيمة محايدة (صفر) + عمود علم توفّر صريح
    (FUND_available/OI_available) يميّز الحالتين.
"""


def _funding_rate_frame(raw: Optional[pd.DataFrame],
                        config: Optional[dict] = None) -> Optional[pd.DataFrame]:
    """يحوّل أرشيف funding_rate الخام (عمودا timestamp، funding_rate) إلى
    إطار ميزات على فترات التمويل الأصلية (كل ٨ ساعات عادة) — غير مُحاذى بعد
    على فريم أي أصل (تتولّاه add_funding_oi_features). ``None`` إن كان
    ``raw`` فارغاً أو ``None`` (لا أرشيف محفوظ بعد لهذا الرمز).

    ``FUND_rate_z``: درجة معيارية متدحرجة (``zscore_window`` شمعة تمويل) —
    تُشغّل مباشرة فرضية "funding مرتفع جداً = تموضع مفرط" من كتالوج
    الفرضيات (قسم ٢ في خطة المشروع)، بمرجع كل عملة على تاريخها هي، لا حدّاً
    مطلقاً يتجاهل اختلاف مستويات funding الطبيعية بين الأصول.
    """
    if raw is None or raw.empty:
        return None
    config = CONFIG if config is None else config
    window = int((config.get('funding_rate') or {}).get('zscore_window', 90))
    s = raw.set_index('timestamp')['funding_rate'].astype('float64').sort_index()
    min_p = max(5, window // 3)
    mu = s.rolling(window, min_periods=min_p).mean()
    sd = s.rolling(window, min_periods=min_p).std()
    z = ((s - mu) / sd.replace(0, np.nan)).fillna(0.0)     # إحماء الرولينغ: محايد لا NaN
    return pd.DataFrame({'FUND_rate': s, 'FUND_rate_z': z})


def _oi_frame(raw: Optional[pd.DataFrame],
             config: Optional[dict] = None) -> Optional[pd.DataFrame]:
    """يحوّل أرشيف open_interest الخام إلى تغيّر نسبي لوغاريتمي — لا مستوى
    مطلق (يختلف بأوامر عشرية بين الأصول). أول نقطة في الأرشيف بلا "سابقة"
    فيُنتج diff() لها NaN، تُترَك لتُملأ محايدة صفر لاحقاً في المُحاذاة —
    حدّ إحماء طبيعي، لا خطأ."""
    if raw is None or raw.empty:
        return None
    config = CONFIG if config is None else config
    period = int((config.get('open_interest') or {}).get('change_period', 1))
    s = raw.set_index('timestamp')['open_interest'].astype('float64').sort_index()
    chg = np.log(s.clip(lower=EPS)).diff(period)
    return pd.DataFrame({f'OI_chg_{period}': chg})


def funding_oi_feature_columns(config: Optional[dict] = None) -> List[str]:
    """أسماء أعمدة معدّل التمويل/الفائدة المفتوحة — قائمة ثابتة من الإعدادات
    وحدها (بلا بيانات فعلية)، لتُدرَج في feature_order تلقائياً — بنفس دور
    market_context_columns لأعمدة MKT_."""
    config = CONFIG if config is None else config
    cols: List[str] = []
    fr = config.get('funding_rate') or {}
    fut2 = _phase2_on(config, 'use_futures_metrics')
    if fr.get('enabled', False) and fr.get('as_feature', False):
        cols += (['FUND_rate', 'FUND_rate_z', 'FUND_sum_1d', 'FUND_sum_3d', 'FUND_available'] if fut2
                 else ['FUND_rate', 'FUND_rate_z', 'FUND_available'])
    oi = config.get('open_interest') or {}
    if oi.get('enabled', False) and oi.get('as_feature', False):
        period = int(oi.get('change_period', 1))
        cols += [f'OI_chg_{period}', 'OI_available']
    return cols


def add_funding_oi_features(dfs: Dict[str, pd.DataFrame], symbol: str,
                            config: Optional[dict] = None) -> Dict[str, pd.DataFrame]:
    """يُلحق أعمدة معدّل التمويل/الفائدة المفتوحة (المُفعَّلتان فقط) إلى
    ``dfs`` (لكل فريم) — محاذاة بـ ``reindex(..., method='ffill')`` من
    أرشيف Drive الخاص بـ ``symbol`` (خلاف add_market_context: كل أصل هنا
    يحمل أرشيفه الخاص، لا مرجعاً مشتركاً). لا شيء مُفعَّل ⇒ ``dfs`` تُعاد
    كما هي بلا نسخ إضافي.
    """
    config = CONFIG if config is None else config
    fr = config.get('funding_rate') or {}
    oi = config.get('open_interest') or {}
    use_fr = bool(fr.get('enabled', False) and fr.get('as_feature', False))
    use_oi = bool(oi.get('enabled', False) and oi.get('as_feature', False))
    if _phase2_on(config, 'use_futures_metrics'):
        # المرحلة ٢: أرشيف Drive الكامل، محاذاة على إغلاق الشمعة + futures_metrics
        return _add_funding_oi_features_phase2(dfs, symbol, config, use_fr, use_oi)
    if not use_fr and not use_oi:
        return dfs

    fr_frame = (_funding_rate_frame(load_funding_open_interest(symbol, 'funding_rate', config), config)
               if use_fr else None)
    oi_period = int(oi.get('change_period', 1))
    oi_col = f'OI_chg_{oi_period}'
    oi_frame = (_oi_frame(load_funding_open_interest(symbol, 'open_interest', config), config)
               if use_oi else None)

    out: Dict[str, pd.DataFrame] = {}
    for tf, df in dfs.items():
        d = df.copy()
        if use_fr:
            if fr_frame is not None:
                aligned = fr_frame.reindex(d.index, method='ffill')
                d['FUND_available'] = aligned['FUND_rate'].notna().astype('float64')
                d['FUND_rate'] = aligned['FUND_rate'].fillna(0.0)
                d['FUND_rate_z'] = aligned['FUND_rate_z'].fillna(0.0)
            else:
                d['FUND_rate'] = 0.0
                d['FUND_rate_z'] = 0.0
                d['FUND_available'] = 0.0
        if use_oi:
            if oi_frame is not None:
                aligned = oi_frame.reindex(d.index, method='ffill')
                d['OI_available'] = aligned[oi_col].notna().astype('float64')
                d[oi_col] = aligned[oi_col].fillna(0.0)
            else:
                d[oi_col] = 0.0
                d['OI_available'] = 0.0
        out[tf] = d
    return out


def _intraday_efficiency_frame(raw: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """يحوّل أرشيف نسبة كفاءة كوفمان الخام (عمودا date، efficiency_ratio_24h)
    إلى إطار مفهرس بالتاريخ (UTC) — غير مُحاذى بعد على فريم أي أصل (تتولّاه
    add_intraday_efficiency_features). ``None`` إن كان ``raw`` فارغاً أو
    ``None`` (لا أرشيف محسوب بعد لهذا الرمز)."""
    if raw is None or raw.empty:
        return None
    date_idx = pd.to_datetime(raw['date'], utc=True)
    s = pd.Series(raw['efficiency_ratio_24h'].astype('float64').to_numpy(),
                  index=date_idx).sort_index()
    return pd.DataFrame({'EFF_RATIO_24H': s})


def intraday_efficiency_columns(config: Optional[dict] = None) -> List[str]:
    """أسماء أعمدة نسبة كفاءة كوفمان — قائمة ثابتة من الإعدادات وحدها (بلا
    بيانات فعلية)، بنفس دور funding_oi_feature_columns بالضبط."""
    config = CONFIG if config is None else config
    ie = config.get('intraday_efficiency') or {}
    if (ie.get('enabled', False) and ie.get('as_feature', False)) or \
            _phase2_on(config, 'use_intraday_15m'):
        return ['EFF_RATIO_24H', 'EFF_RATIO_available']
    return []


def load_intraday_efficiency_archive(symbol: str, config: Optional[dict] = None) -> Optional[pd.DataFrame]:
    """يقرأ أرشيف نسبة كفاءة كوفمان لرمز — من ``intraday_efficiency.data``
    (قاموس {symbol: DataFrame} جاهز في الذاكرة، للاختبار المباشر بلا Drive)
    إن وُجد، وإلا من ``intraday_efficiency.archive_dir`` (ملف
    ``{symbol}_efficiency.csv`` بعمودي date، efficiency_ratio_24h). ``None``
    إن لم يتوفّر أيّهما لهذا الرمز."""
    config = CONFIG if config is None else config
    ie = config.get('intraday_efficiency') or {}
    preloaded = ie.get('data')
    if preloaded is not None:
        df = preloaded.get(symbol)
        return df.copy() if df is not None else None
    archive_dir = ie.get('archive_dir')
    if not archive_dir:
        return None
    path = Path(archive_dir) / f'{symbol}_efficiency.csv'
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=['date'])


def add_intraday_efficiency_features(dfs: Dict[str, pd.DataFrame], symbol: str,
                                     config: Optional[dict] = None) -> Dict[str, pd.DataFrame]:
    """يُلحق أعمدة نسبة كفاءة كوفمان (EFF_RATIO_) إلى ``dfs`` (لكل فريم) —
    محاذاة بـ ``reindex(..., method='ffill')`` من أرشيف نسبة الكفاءة الخاص
    بـ``symbol``، بنفس بنية add_funding_oi_features بالضبط (أرشيف يومي واحد
    يُبنى مرّة، لا مرجعاً مشتركاً كـadd_market_context). لا شيء مُفعَّل ⇒
    ``dfs`` تُعاد كما هي بلا نسخ إضافي."""
    config = CONFIG if config is None else config
    cols = intraday_efficiency_columns(config)
    if not cols:
        return dfs
    if _phase2_on(config, 'use_intraday_15m'):     # المرحلة ٢: من history_15m الحقيقي (يتقدّم على الأرشيف)
        return _phase2_slot_features(dfs, symbol, config, 'EFF_RATIO_24H', 'EFF_RATIO_available')

    frame = _intraday_efficiency_frame(load_intraday_efficiency_archive(symbol, config))

    out: Dict[str, pd.DataFrame] = {}
    for tf, df in dfs.items():
        d = df.copy()
        if frame is not None:
            aligned = frame.reindex(d.index, method='ffill')
            d['EFF_RATIO_available'] = aligned['EFF_RATIO_24H'].notna().astype('float64')
            d['EFF_RATIO_24H'] = aligned['EFF_RATIO_24H'].fillna(0.0)
        else:
            d['EFF_RATIO_24H'] = 0.0
            d['EFF_RATIO_available'] = 0.0
        out[tf] = d
    return out


def _intraday_vwap_frame(raw: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """يحوّل أرشيف انحراف VWAP الخام (عمودا date، vwap_deviation) إلى إطار
    مفهرس بالتاريخ (UTC) — بنفس بنية _intraday_efficiency_frame بالضبط."""
    if raw is None or raw.empty:
        return None
    date_idx = pd.to_datetime(raw['date'], utc=True)
    s = pd.Series(raw['vwap_deviation'].astype('float64').to_numpy(),
                  index=date_idx).sort_index()
    return pd.DataFrame({'VWAP_DEVIATION': s})


def intraday_vwap_columns(config: Optional[dict] = None) -> List[str]:
    """أسماء أعمدة انحراف VWAP — بنفس دور intraday_efficiency_columns بالضبط."""
    config = CONFIG if config is None else config
    iv = config.get('intraday_vwap') or {}
    if (iv.get('enabled', False) and iv.get('as_feature', False)) or \
            _phase2_on(config, 'use_intraday_15m'):
        return ['VWAP_DEVIATION', 'VWAP_DEVIATION_available']
    return []


def load_intraday_vwap_archive(symbol: str, config: Optional[dict] = None) -> Optional[pd.DataFrame]:
    """يقرأ أرشيف انحراف VWAP لرمز — بنفس بنية load_intraday_efficiency_archive بالضبط."""
    config = CONFIG if config is None else config
    iv = config.get('intraday_vwap') or {}
    preloaded = iv.get('data')
    if preloaded is not None:
        df = preloaded.get(symbol)
        return df.copy() if df is not None else None
    archive_dir = iv.get('archive_dir')
    if not archive_dir:
        return None
    path = Path(archive_dir) / f'{symbol}_vwap.csv'
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=['date'])


def add_intraday_vwap_features(dfs: Dict[str, pd.DataFrame], symbol: str,
                               config: Optional[dict] = None) -> Dict[str, pd.DataFrame]:
    """يُلحق أعمدة انحراف VWAP (VWAP_DEVIATION_) إلى ``dfs`` — بنفس بنية
    add_intraday_efficiency_features بالضبط."""
    config = CONFIG if config is None else config
    cols = intraday_vwap_columns(config)
    if not cols:
        return dfs
    if _phase2_on(config, 'use_intraday_15m'):     # المرحلة ٢: من history_15m الحقيقي (يتقدّم على الأرشيف)
        return _phase2_slot_features(dfs, symbol, config, 'VWAP_DEVIATION', 'VWAP_DEVIATION_available')

    frame = _intraday_vwap_frame(load_intraday_vwap_archive(symbol, config))

    out: Dict[str, pd.DataFrame] = {}
    for tf, df in dfs.items():
        d = df.copy()
        if frame is not None:
            aligned = frame.reindex(d.index, method='ffill')
            d['VWAP_DEVIATION_available'] = aligned['VWAP_DEVIATION'].notna().astype('float64')
            d['VWAP_DEVIATION'] = aligned['VWAP_DEVIATION'].fillna(0.0)
        else:
            d['VWAP_DEVIATION'] = 0.0
            d['VWAP_DEVIATION_available'] = 0.0
        out[tf] = d
    return out


def _intraday_volume_concentration_frame(raw: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """يحوّل أرشيف تركّز الحجم الخام (عمودا date، volume_concentration_hhi)
    إلى إطار مفهرس بالتاريخ (UTC) — بنفس بنية _intraday_efficiency_frame بالضبط."""
    if raw is None or raw.empty:
        return None
    date_idx = pd.to_datetime(raw['date'], utc=True)
    s = pd.Series(raw['volume_concentration_hhi'].astype('float64').to_numpy(),
                  index=date_idx).sort_index()
    return pd.DataFrame({'VOL_CONC_HHI': s})


def intraday_volume_concentration_columns(config: Optional[dict] = None) -> List[str]:
    """أسماء أعمدة تركّز الحجم — بنفس دور intraday_efficiency_columns بالضبط."""
    config = CONFIG if config is None else config
    vc = config.get('intraday_volume_concentration') or {}
    if (vc.get('enabled', False) and vc.get('as_feature', False)) or \
            _phase2_on(config, 'use_intraday_15m'):
        return ['VOL_CONC_HHI', 'VOL_CONC_HHI_available']
    return []


def load_intraday_volume_concentration_archive(symbol: str, config: Optional[dict] = None) -> Optional[pd.DataFrame]:
    """يقرأ أرشيف تركّز الحجم لرمز — بنفس بنية load_intraday_efficiency_archive بالضبط."""
    config = CONFIG if config is None else config
    vc = config.get('intraday_volume_concentration') or {}
    preloaded = vc.get('data')
    if preloaded is not None:
        df = preloaded.get(symbol)
        return df.copy() if df is not None else None
    archive_dir = vc.get('archive_dir')
    if not archive_dir:
        return None
    path = Path(archive_dir) / f'{symbol}_volconc.csv'
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=['date'])


def add_intraday_volume_concentration_features(dfs: Dict[str, pd.DataFrame], symbol: str,
                                               config: Optional[dict] = None) -> Dict[str, pd.DataFrame]:
    """يُلحق أعمدة تركّز الحجم (VOL_CONC_) إلى ``dfs`` — بنفس بنية
    add_intraday_efficiency_features بالضبط."""
    config = CONFIG if config is None else config
    cols = intraday_volume_concentration_columns(config)
    if not cols:
        return dfs
    if _phase2_on(config, 'use_intraday_15m'):     # المرحلة ٢: من history_15m الحقيقي (يتقدّم على الأرشيف)
        return _phase2_slot_features(dfs, symbol, config, 'VOL_CONC_HHI', 'VOL_CONC_HHI_available')

    frame = _intraday_volume_concentration_frame(load_intraday_volume_concentration_archive(symbol, config))

    out: Dict[str, pd.DataFrame] = {}
    for tf, df in dfs.items():
        d = df.copy()
        if frame is not None:
            aligned = frame.reindex(d.index, method='ffill')
            d['VOL_CONC_HHI_available'] = aligned['VOL_CONC_HHI'].notna().astype('float64')
            d['VOL_CONC_HHI'] = aligned['VOL_CONC_HHI'].fillna(0.0)
        else:
            d['VOL_CONC_HHI'] = 0.0
            d['VOL_CONC_HHI_available'] = 0.0
        out[tf] = d
    return out
