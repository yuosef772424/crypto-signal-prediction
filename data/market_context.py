"""
PURPOSE:  Cross-asset market context: features of a reference coin (BTC by default) added as MKT_* columns to every other coin, time-aligned; zero for the reference itself.
TAGS:     market context, MKT_, reference coin, BTC, add_market_context, build_market_context, cross-asset
PITFALLS: The reference coin's own MKT_ values are zero by design (a non-zero value would leak the target into itself). Executed into the one shared pipeline namespace by data/_loader.py (never imported on its own): names from other modules resolve at call time.

## 14-ب) سياق سوقي عابر للأصول (Market Context)
"""
# @title
"""
سياق سوقي عابر للأصول (Cross-Asset Market Context).

كل نافذة كانت تُبنى بمعزل تام عن باقي السوق — لا ميزة تخبر النموذج "هل BTC
يرتفع أم ينخفض الآن؟" رغم أن حركة أغلب العملات مرتبطة به جزئياً. الحل هنا
مستوحى من ``include_corr_pairlist`` في FreqAI (إطار تعلّم آلي لبوت التداول
مفتوح المصدر freqtrade): تُضاف ميزات عملة مرجعية (BTC افتراضياً) كأعمدة إدخال
إضافية (بادئة ``MKT_``) لكل عملة أخرى، بعد محاذاتها زمنياً.

**للعملة المرجعية نفسها**: القيمة صفر دائماً — عائد BTC نسبة لنفسه صفر
بالتعريف، وأي قيمة أخرى كانت ستُسرِّب معلومة دائرية (الهدف يتنبأ بنفسه جزئياً
عبر ميزة مطابقة تقريباً).
"""


def market_context_columns(config: Optional[dict] = None) -> List[str]:
    """أسماء أعمدة السياق السوقي — قائمة ثابتة تُحسب من الإعدادات وحدها
    (بلا بيانات فعلية)، لتُدرَج في feature_order تلقائياً عبر infer_feature_columns."""
    config = CONFIG if config is None else config
    mc = config.get('market_context') or {}
    if not mc.get('enabled', False):
        return []
    periods = mc.get('return_periods') or [1]
    cols = [f'MKT_ret_{p}' for p in periods]
    # ── نظام الارتباط (Correlation Regime) — مُعطَّل افتراضياً (corr_windows=[]) ──
    # الارتباط المتحرك بين عائد هذا الأصل وعائد العملة المرجعية (BTC) خلال
    # نافذة w — مختلف جوهرياً عن MKT_ret (عائد BTC نفسه، مرفوض سابقاً): هنا
    # السؤال ليس "هل BTC صاعد؟" بل "هل هذا الأصل يتحرّك مع BTC الآن أم
    # بمعزل عنه؟". الأدبيات (راجع docs/research/market_correlation_regime.md)
    # توثّق ارتفاع الارتباط حاداً أثناء الانهيارات الجماعية ("risk-off"/
    # عدوى)، وانخفاضه في فترات المضاربة الفردية الهادئة — نظام سوق مختلف
    # جوهرياً عن الزخم (RET) الذي اختُبِر سابقاً كمصفٍّ ولم يحسم الأمر.
    # يتطلّب وجود MKT_ret_1 (فترة 1 ضمن return_periods) لحساب الارتباط —
    # الإعداد الافتراضي [1] يحقّق هذا الشرط تلقائياً.
    corr_windows = mc.get('corr_windows') or []
    if corr_windows and 1 in periods:
        cols += [f'MKT_CORR_{w}' for w in corr_windows]
    # ── بيتا السوق (Rolling Beta، CAPM) — مُعطَّلة افتراضياً (beta_windows=[]) ──
    # مختلفة جوهرياً عن MKT_CORR (قوّة/اتجاه الترابط الخطّي، محصورة [-1,1]):
    # بيتا تقيس *مقدار* تضخيم حركة الأصل نسبةً لحركة المرجع (BTC) — أصل عالي
    # الارتباط قد يكون بيتا 0.5 (يتحرّك بهدوء مع السوق) أو 2.0 (يضخّم حركته)
    # بنفس درجة الارتباط. أوّل اختبار فعلي حقيقي لـMKT_beta في هذا المشروع —
    # الاختبارات السابقة (قبل هذه الجلسة) استخدمت market_context معطَّلة
    # بالخطأ فأنتجت IC=0.0 صفراً حرفياً (خلل صامت موثَّق في
    # discovery_lab_phase1_2.md، لا نتيجة رفض حقيقية). يتطلّب MKT_ret_1
    # كـMKT_CORR تماماً.
    beta_windows = mc.get('beta_windows') or []
    if beta_windows and 1 in periods:
        cols += [f'MKT_BETA_{w}' for w in beta_windows]
    # ── انتشار متأخّر (Lead-Lag Spillover) — مُعطَّلة افتراضياً (lag_windows=[]) ──
    # فرضية مختلفة جوهرياً عن MKT_ret (المعاصر): هل عائد BTC قبل k يوماً
    # (لا اليوم نفسه) يحمل معلومة عن هذا الأصل؟ موثَّقة في أدبيات "BTC
    # يقود، العملات البديلة تتبع بتأخّر" (attention/liquidity spillover) —
    # راجع docs/research. سببية بالكامل (shift إلى الماضي فقط).
    lag_windows = mc.get('lag_windows') or []
    cols += [f'MKT_LAG_RET_{k}' for k in lag_windows]
    # ── نظام ارتداد ما بعد الانهيار على مستوى السوق (مُعطَّلة افتراضياً) ──────
    # نشأت من تشخيص مستقلّ: RSI_14 وVWAP_DEVIATION ينكسر اتساق إشارتهما في
    # نفس نافذة اختبار محدَّدة (ما بعد انهيار FTX). محاولة أولى بتعريف على
    # مستوى الأصل الفردي (custom_settings.crash_rebound_regime) أعطت نتيجة
    # مختلطة (منتشرة جداً — 73% من العيّنات). هذه محاولة ثانية على مستوى
    # السوق ككل عبر العملة المرجعية (BTC): هل هبطت البورصة المرجعية ذاتها
    # ≥threshold عن قمّتها خلال high_lookback يوماً، في أي لحظة من آخر
    # recent_window يوماً؟ راجع docs/research. None يعني مُعطَّلة تماماً.
    if mc.get('crash_rebound_regime') is not None:
        cols += ['MKT_CRASH_REBOUND_REGIME']
    return cols


def _market_return_frame(price_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """عوائد الإغلاق للعملة المرجعية على فتراتها المُعدَّة — إطار خفيف
    (عمود واحد لكل فترة) يُحاذى لاحقاً بفهرس كل عملة أخرى."""
    mc = config.get('market_context') or {}
    periods = mc.get('return_periods') or [1]
    close = price_df['close']
    out = pd.DataFrame(index=price_df.index)
    for p in periods:
        out[f'MKT_ret_{p}'] = close.pct_change(int(p))
    lag_windows = mc.get('lag_windows') or []
    if lag_windows:
        ref_ret_1 = close.pct_change(1)
        for k in lag_windows:
            out[f'MKT_LAG_RET_{k}'] = ref_ret_1.shift(int(k))
    crr = mc.get('crash_rebound_regime')
    if crr is not None:
        high_lb = int(crr.get('high_lookback', 60))
        recent_w = int(crr.get('recent_window', 30))
        threshold = float(crr.get('drawdown_threshold', -0.30))
        rolling_high = close.rolling(high_lb, min_periods=max(2, high_lb // 2)).max()
        drawdown = close / rolling_high - 1.0
        out['MKT_CRASH_REBOUND_REGIME'] = (
            drawdown.rolling(recent_w, min_periods=1).min() <= threshold
        ).astype('float64').fillna(0.0)
    return out


def build_market_context(load_asset_fn: Optional[Callable] = None,
                         resample_fn: Optional[Callable] = None,
                         data: Optional[Dict] = None,
                         config: Optional[dict] = None) -> Optional[Dict[str, pd.DataFrame]]:
    """يبني (مرّة واحدة قبل حلقة الأصول المتوازية) عوائد العملة المرجعية لكل
    فريم في ``tf_order`` — تُمرَّر بعدها إلى :func:`add_market_context` لكل
    عملة. ``None`` إن كان ``market_context`` معطَّلاً، أو تعذّر تحميل المرجع
    (يُحذَّر ولا يُوقِف البناء — كل العملات تحصل على سياق صفري بدلاً منه).

    مرّر ``data`` للمصدر المُجهَّز مسبقاً، أو ``load_asset_fn``+``resample_fn``
    لتحميل المرجع مباشرة من Binance/الملفات الخام — بصرف النظر عمّا إذا كانت
    العملة المرجعية ضمن ``configs`` المطلوبة فعلياً أم لا.
    """
    config = CONFIG if config is None else config
    mc = config.get('market_context') or {}
    if not mc.get('enabled', False):
        return None
    ref = mc.get('reference_symbol', 'BTCUSDT')
    tf_order = config['tf_order']
    try:
        if data is not None:
            if ref not in data:
                raise KeyError(f"'{ref}' غير موجودة في data المُجهَّزة مسبقاً")
            raw = {tf: data[ref][tf] for tf in tf_order if tf in data[ref]}
        else:
            if load_asset_fn is None or resample_fn is None:
                raise ValueError("يلزم load_asset_fn+resample_fn أو data")
            df = load_asset_fn(None, ref)
            raw = resample_fn(df, tf_order)
        return {tf: _market_return_frame(raw[tf], config) for tf in tf_order if tf in raw}
    except Exception as exc:                                   # noqa: BLE001
        print(f"   ⚠️ تعذّر بناء السياق السوقي من '{ref}': {exc} — "
              f"سيُترَك بلا سياق سوقي (أعمدة MKT_ = صفر للجميع).")
        return None


def add_market_context(dfs: Dict[str, pd.DataFrame],
                       market_dfs: Optional[Dict[str, pd.DataFrame]],
                       is_reference: bool,
                       config: Optional[dict] = None) -> Dict[str, pd.DataFrame]:
    """يُلحق أعمدة MKT_ إلى ``dfs`` (لكل فريم) — محاذاة زمنية بـ
    ``reindex(..., method='ffill')`` على فهرس كل عملة، فلا تحتاج العملتان
    نفس الطول أو نفس أوقات الشموع بالضبط. عدم توفّر ``market_context`` أصلاً
    (أعمدته فارغة) يُعيد ``dfs`` كما هي بلا نسخ إضافي.

    ``MKT_CORR_{w}`` (إن مُفعَّلة عبر ``corr_windows``) تُحسَب هنا لا في
    :func:`_market_return_frame`: تحتاج عائد **هذا الأصل نفسه** (غير
    متاح عند بناء ``market_dfs`` مرّة واحدة لكل السوق)، مُحسوبة كـ
    ``rolling(w).corr`` بين عائد الأصل وعائد المرجع المُحاذى — ``rolling``
    بلا ``center`` فسببية بالكامل (لا تسرّب مستقبلي).
    """
    cols = market_context_columns(config)
    if not cols:
        return dfs
    config = CONFIG if config is None else config
    corr_windows = (config.get('market_context') or {}).get('corr_windows') or []
    corr_cols = {f'MKT_CORR_{w}' for w in corr_windows}
    beta_windows = (config.get('market_context') or {}).get('beta_windows') or []
    beta_cols = {f'MKT_BETA_{w}' for w in beta_windows}
    self_computed_cols = corr_cols | beta_cols
    out = {}
    for tf, df in dfs.items():
        d = df.copy()
        if is_reference or market_dfs is None or tf not in market_dfs:
            for c in cols:
                d[c] = 0.0
        else:
            aligned = market_dfs[tf].reindex(d.index, method='ffill')
            for c in cols:
                if c in self_computed_cols:
                    continue
                d[c] = aligned[c].fillna(0.0) if c in aligned.columns else 0.0
            if self_computed_cols and 'MKT_ret_1' in aligned.columns:
                ret_self = d['close'].astype('float64').pct_change(1)
                ret_ref = aligned['MKT_ret_1']
                for w in corr_windows:
                    d[f'MKT_CORR_{w}'] = ret_self.rolling(int(w)).corr(ret_ref).fillna(0.0)
                for w in beta_windows:
                    cov = ret_self.rolling(int(w)).cov(ret_ref)
                    var = ret_ref.rolling(int(w)).var()
                    d[f'MKT_BETA_{w}'] = (cov / var.replace(0, np.nan)).fillna(0.0)
            else:
                for c in self_computed_cols:
                    d[c] = 0.0
        out[tf] = d
    return out
