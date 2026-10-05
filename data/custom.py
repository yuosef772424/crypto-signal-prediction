"""
PURPOSE:  Static scale-free custom features (log returns, relative ranges, candle structure, time features, fractals, volatility estimators, trend age, ...) and their column names.
TAGS:     custom features, add_custom_features, custom_feature_names, DEFAULT_CUSTOM_SETTINGS, returns, fractal, supertrend, psar, hurst, variance ratio, ath distance, TIME_hour
PITFALLS: Runs its _test_*() causality checks at load time (module level). Every feature must be causal: row t may only use rows <= t (tests/test_no_lookahead.py). Executed into the one shared pipeline namespace by data/_loader.py (never imported on its own): names from other modules resolve at call time.

## 7) الميزات المخصّصة الساكنة (`custom.py` سابقاً)

ميزات خالية من المقياس بالبناء (عوائد لوغاريتمية، مدى نسبي، بنية الشمعة...)
تُعمِّم عبر كل الأصول بغضّ النظر عن مستوى سعرها.
"""
"""
ميزات مخصّصة للعملات الرقمية — ساكنة (stationary) وخالية من المقياس.

**لماذا هذه ولم تكفِ مؤشرات ``pandas_ta``؟**

النموذج يتدرّب على عشرات العملات بمستويات أسعار تختلف بمقدار ستة أوامر عشرية
(بيتكوين 60,000 مقابل عملة بـ 0.00003). المؤشرات الكلاسيكية إمّا بوحدة السعر
(فتحتاج تطبيعاً لكل نافذة يمحو جزءاً من معلومتها) أو مذبذبات محصورة تكرّر بعضها.
الميزات هنا **خالية من المقياس بالبناء**: نِسَب ولوغاريتمات عوائد، فتعني الشيء
نفسه على أي عملة وفي أي حقبة سعرية — وهذا شرط أساسي للتعميم عبر الأصول.

قيس فعلياً على الميزات القديمة: ``high``/``low``/``close`` مترابطة بـ 0.98–0.99
بعد التطبيع، لأن معلومتها الحقيقية هي **الفرق بينها** لا مستواها. لذا نستبدلها
هنا بترميز غير مكرَّر: مدى نسبي + موضع الإغلاق داخل الشمعة.

العائلات:

============ ==========================================================
البادئة       المعلومة
============ ==========================================================
``RET_``     عوائد لوغاريتمية على آفاق متعددة — أقوى عائلة ساكنة
``RANGE_``   المدى النسبي (تقلب داخل الشمعة) بالنسبة المئوية
``BODY_``    نسبة جسم الشمعة إلى مداها — بنية الشمعة
``WICK_``    الظلال العليا/السفلى — ضغط الرفض
``POS_``     موضع الإغلاق داخل نطاق N شمعة — ارتداد/اختراق
``VOLR_``    لوغاريتم نسبة التقلب القصير للطويل — كاشف النظام السعري
``VOLZ_``    درجة معيارية للحجم — الحجم غير الاعتيادي
``TIME_``    موسمية يومية/أسبوعية (السوق يعمل 24/7 وله أنماط حقيقية)
``FRACTAL_`` نقاط تطرّف محلي مؤكَّدة (فركتال ويليامز)، سببية بالكامل عبر
             إزاحة نصف النافذة — مُعطَّلة افتراضياً (``fractal_windows=[]``)
============ ==========================================================
"""

EPS = 1e-12

#: الإعدادات الافتراضية للميزات المخصّصة.
DEFAULT_CUSTOM_SETTINGS: Dict[str, object] = {
    "returns": [1, 3, 6, 12, 24],      # آفاق العوائد اللوغاريتمية (بالشموع)
    "range_windows": [14, 50],         # نوافذ الموضع داخل النطاق
    "vol_windows": [(6, 24), (12, 48)],  # (قصير، طويل) لنِسَب التقلب
    "volume_window": 20,               # نافذة درجة الحجم المعيارية
    "candle_structure": True,          # جسم/ظلال الشمعة
    "time_features": True,             # موسمية الساعة/اليوم
    # نوافذ فركتال ويليامز (فردية فقط) — [] يعني مُعطَّلة تماماً (لا أعمدة
    # FRACTAL_* تُنتَج). راجع docstring _add_fractal_features لسبب التعطيل
    # الافتراضي وشرح الإزاحة السببية (٢٤ سبتمبر ٢٠٢٦، مرشّح H003 المفتوح).
    "fractal_windows": [],
    # RVI (Relative Vigor Index) وFisher Transform — [] يعني مُعطَّلتان تماماً.
    # مرشّحان "ضعيفان موثَّقان" من docs/research/pandas_ta_deferred_candidates.md:
    # عبر محور IC الخطّي المفرد أظهرا frac_significant مرتفعاً نسبياً بمعزل
    # (RVI_14/close=58%، FISHERT_14/low=58%) لكن consistent_sign=False — دلالة
    # جزئية بلا اتجاه ثابت لا تكفي لقبول IC وحده. أُضيفا هنا كميزتين اختياريتين
    # ليُختبَر أثرهما الفعلي داخل تدريب NIG-TimeNet v2 كامل (تفاعل غير خطّي مع
    # ميزات أخرى قد يستغلّه النموذج حيث فشل IC المنفرد) — لا تنفيذ تلقائي بلا
    # تفعيل صريح (٢٥ سبتمبر ٢٠٢٦).
    "rvi_windows": [],
    "fisher_windows": [],
    # مقدّرات تقلّب/سيولة من الأدبيات الكمّية الكلاسيكية (Parkinson 1980،
    # Garman-Klass 1980، Rogers-Satchell 1991، Yang-Zhang 2000، Amihud 2002،
    # Roll 1984) — ليست في pandas_ta أصلاً. [] يعني مُعطَّلة تماماً. راجع
    # docs/research/new_volatility_liquidity_estimators.md للتحقّق الأوّلي
    # والمنهجية الكاملة (٢٥ سبتمبر ٢٠٢٦).
    "vol_estimator_windows": [],
    # Hurst exponent (مقياس التذكّر الطويل/الانعكاس) وVariance Ratio
    # (اختبار لو-ماكينلي 1988 المباشر للانعكاس مقابل الاستمرار) — لا شيء
    # منهما في pandas_ta أصلاً. [] يعني مُعطَّلة تماماً. راجع
    # docs/research/hurst_variance_ratio_features.md للمنهجية والتحقّق
    # (٢٥ سبتمبر ٢٠٢٦).
    "mean_reversion_windows": [],
    # نسبة التقلّب "القفزي" (Barndorff-Nielsen & Shephard 2004، Bipower
    # Variation): يفصل التباين المُحقَّق الكلّي إلى مكوّن مستمرّ (انتشار
    # عادي) ومكوّن قفزي (صدمات مفاجئة) — سؤال مختلف كلياً عن حجم التقلّب
    # (NATR) أو اتجاهه (Hurst/VR): "هل هذا التقلّب انتشار سلس أم قفزات؟"
    # ليست في pandas_ta أصلاً. [] يعني مُعطَّلة تماماً. راجع
    # docs/research/jump_ratio_feature.md (٢٥ سبتمبر ٢٠٢٦).
    "jump_windows": [],
    # SuperTrend (باراميترات 10/3.0 الكلاسيكية الموصى بها في مصادر الاستخدام
    # الصحيح — راجع docs/research/proper_indicator_signals.md): الإشارة
    # الصحيحة هي **الاتجاه** (فوق/تحت الخط) لا قيمة الخط السعرية نفسها —
    # اتّجاه استمراري (trend-following)، عكس فرضية الانعكاس في H003. غائبة
    # عن مجموعة الميزات الافتراضية (خلافاً لـRSI/MACD/ADX/BBANDS). []
    # يعني مُعطَّلة تماماً.
    "supertrend_windows": [],
    # مقياس تباعد RSI/السعر (تقريب لـ"Divergence" — الميزة التي وصفها
    # وايلدر نفسه بأنها **الأقوى** في RSI، لا عتبتَي ذروة الشراء/البيع
    # الخام المُختبَرتين ضمنياً مسبقاً كـIC خام على RSI_14). تقريب مبسَّط:
    # فرق درجة معيارية بين تغيّر السعر وتغيّر RSI على نفس الأفق (لا مطابقة
    # قمم/قيعان حرفية) — راجع التوثيق للتفاصيل والقيود. [] يعني مُعطَّلة.
    "rsi_divergence_windows": [],
    # Parabolic SAR (وايلدر 1978، إعدادات af0=af=0.02/max_af=0.2 القياسية
    # الموصى بها في مصادره الأصلية — راجع docs/research/proper_indicator_
    # signals.md): الإشارة الصحيحة اتجاه النقاط (تحت السعر=صاعد، فوقه=
    # هابط)، لا القيمة السعرية للنقاط. غائب كلياً عن مجموعة الميزات
    # الافتراضية لهذا المشروع (خلافاً لـRSI/MACD/ADX/BBANDS/Stochastic).
    # False يعني مُعطَّلة تماماً.
    "psar_enabled": False,
    # عمر الاتجاه (Trend Age) — عدد الشموع منذ آخر قمة/قاع فركتالي مؤكَّد
    # (يعتمد على _fractal_columns الموجودة أصلاً). فرضية استنفاد الاتجاه
    # موثَّقة في الأدبيات (LuxAlgo, EBC): كلما طال غياب قمة/قاع جديد، كلما
    # اقترب الاتجاه الحالي من الاستنفاد — راجع docs/research/trend_age_
    # and_correlation_regime.md. [] يعني مُعطَّلة تماماً.
    "trend_age_windows": [],
    # اتساق الاتجاه اليومي (Directional Consistency) — نسبة الأيام التي
    # وافق فيها اتجاه العائد اتجاه اليوم السابق مباشرة (استمرارية بسيطة،
    # لا انحدار Hurst/VR الإحصائي الذي فشل سابقاً على نوافذ قصيرة بسبب
    # الضجيج التقديري — راجع docs/research). 0=تذبذب تام يوم بيوم،
    # 0.5=عشوائي، 1=اتجاه تام. [] يعني مُعطَّلة تماماً.
    "direction_consistency_windows": [],
    # عدم تناظر التقلّب (Volatility Asymmetry / Semi-Variance Skew) — من
    # لديه التقلّب الآن: الحركات الصاعدة أم الهابطة؟ سؤال مختلف عن حجم
    # التقلّب (NATR) أو اتجاهه (Hurst/VR) أو بنيته (Jump Ratio) — مبني على
    # شبه-التباين الكلاسيكي (Markowitz downside risk)، وموثَّق في أدبيات
    # الكريبتو بأن بعض العملات "معكوسة" مقارنةً بالأسهم التقليدية (تقلّب
    # صاعد أكبر، لا هابط) — راجع docs/research. [] يعني مُعطَّلة تماماً.
    "vol_asymmetry_windows": [],
    # نظام "ارتداد ما بعد الانهيار" (Crash Rebound Regime) — تعريف مباشر (لا
    # تصنيف نسبي بالزخم كما في القسم ٢٨ السابق، الذي كان غير حاسم): هل هبط
    # الأصل ≥30% عن قمّته خلال آخر 60 يوماً في أي لحظة من الثلاثين يوماً
    # الأخيرة؟ نشأت هذه الفرضية من تشخيص مستقلّ لسبب انعكاس اتجاه RSI_14
    # وVWAP_DEVIATION في نفس نوافذ الاختبار تحديداً (ما بعد مايو 2021،
    # Terra/Celsius يونيو 2022، وFTX نوفمبر 2022) — راجع docs/research.
    # None يعني مُعطَّلة تماماً (توافق خلفي).
    "crash_rebound_regime": None,
    # زخم مُعدَّل بالمخاطرة (Risk-Adjusted / Volatility-Scaled Momentum) —
    # متوسط العائد اليومي مقسوماً على انحرافه المعياري خلال نفس النافذة
    # (نسبة شبيهة بشارب، لا الاتجاه الخام كـRET_w ولا حجم التقلّب وحده
    # كـNATR، بل "جودة" الاتجاه: عائد ثابت بضجيج منخفض يُعطي قيمة عالية،
    # عائد كبير لكن متذبذب جداً يُعطي قيمة منخفضة). مبنية على أدبيات "الزخم
    # المُدار بالمخاطرة" (Barroso & Santa-Clara 2015) — راجع docs/research.
    # [] يعني مُعطَّلة تماماً.
    "risk_adj_momentum_windows": [],
    # المسافة عن القمّة التاريخية (Distance from All-Time High) — نافذة
    # متوسّعة (expanding) لا نافذة ثابتة كـBARS_SINCE_HIGH الفركتالية (ذاكرة
    # قصيرة المدى، w أيام فقط). "% تحت القمّة التاريخية" مقياس سلوكي شهير في
    # الكريبتو (ترسيخ نفسي بسعر مرجعي بارز — أدبيات disposition effect)،
    # لم يُختبَر من قبل في هذا المشروع. False يعني مُعطَّلة تماماً.
    "ath_distance_enabled": False,
    # البنية الزمنية للتقلّب (Volatility Term Structure) — نسبة تقلّب قصير
    # المدى إلى طويل المدى (NATR_short/NATR_long، محسوبة يدوياً هنا بمعزل عن
    # NATR_14 القياسية في الفهرس الافتراضي، لا اعتماداً عليها). سؤال مختلف
    # عن حجم التقلّب المطلق (H003) أو بنيته القفزية (Jump Ratio): هل التقلّب
    # يتسارع (short>long، "انضغاط قبل انفجار" — أدبيات تداول تركيب التقلّب
    # في الخيارات) أم يتباطأ؟ [] يعني مُعطَّلة تماماً.
    "vol_term_structure_pairs": [],
}


def _hour_features_meaningful(config: Optional[dict] = None) -> bool:
    """هل لميزات الساعة (``TIME_hour_*``) معنى في هذه الإعدادات؟

    الساعة داخل اليوم لا تتغيّر على فريم يومي فما فوق (كل شمعة تبدأ عند 00:00
    UTC)، فتصير ``TIME_hour_sin`` و``TIME_hour_cos`` ثابتتين بلا أي معلومة —
    وهذا ما كشفه ``audit_normalization`` ودفتر التدقيق (§٦ و§٩) على
    ``base_tf='1D'``. المعيار **أصغر** فريم في ``tf_order``: إن وُجد فريم
    داخل اليوم بين الفريمات تبقى الميزتان (تحملان معلومة عليه، وعلى الفريمات
    الأكبر تُصفَّران تلقائياً في ``process_windows`` لأنها ثابتة داخل نافذتها).
    """
    config = CONFIG if config is None else config
    tfs = [tf for tf in (config.get("tf_order") or [config.get("base_tf")]) if tf]
    try:
        smallest = min(pd.Timedelta(tf) for tf in tfs)
    except (ValueError, TypeError):
        return True        # فريم لم نفهمه: لا نحذف ميزة بصمت
    return smallest < pd.Timedelta(days=1)


def add_custom_features(data: pd.DataFrame,
                        settings: Optional[dict] = None,
                        config: Optional[dict] = None) -> pd.DataFrame:
    """يُضيف الميزات المخصّصة إلى ``DataFrame`` بأعمدة OHLCV.

    كل الميزات **سببية**: تعتمد على الشمعة الحالية والسابقة فقط، ولا تستخدم أي
    قيمة مستقبلية (``shift`` موجب فقط، و``rolling`` بلا ``center``).
    """
    config = CONFIG if config is None else config
    s = {**DEFAULT_CUSTOM_SETTINGS, **(settings or config.get("custom_settings") or {})}

    df = pd.DataFrame(data).copy()
    close = df['close'].astype('float64')
    high = df['high'].astype('float64')
    low = df['low'].astype('float64')
    open_ = df['open'].astype('float64') if 'open' in df else close.shift(1).bfill()

    # ── العوائد اللوغاريتمية (بالنسبة المئوية) ───────────────────────────
    log_close = np.log(close.clip(lower=EPS))
    ret_1 = log_close.diff()
    for h in s["returns"]:
        df[f'RET_{h}'] = (log_close - log_close.shift(h)) * 100.0

    # ── المدى النسبي ─────────────────────────────────────────────────────
    hl_range = (high - low)
    df['RANGE_rel'] = (hl_range / close.clip(lower=EPS)) * 100.0

    # ── بنية الشمعة ──────────────────────────────────────────────────────
    if s["candle_structure"]:
        denom = hl_range.replace(0, np.nan)
        df['BODY_ratio'] = ((close - open_) / denom).fillna(0.0)
        df['WICK_upper'] = ((high - np.maximum(close, open_)) / denom).fillna(0.0)
        df['WICK_lower'] = ((np.minimum(close, open_) - low) / denom).fillna(0.0)

    # ── الموضع داخل النطاق ───────────────────────────────────────────────
    for w in s["range_windows"]:
        lo = low.rolling(w, min_periods=w).min()
        hi = high.rolling(w, min_periods=w).max()
        df[f'POS_{w}'] = ((close - lo) / (hi - lo).replace(0, np.nan)).fillna(0.5)

    # ── نظام التقلب ──────────────────────────────────────────────────────
    for short, long in s["vol_windows"]:
        v_s = ret_1.rolling(short, min_periods=short).std()
        v_l = ret_1.rolling(long, min_periods=long).std()
        df[f'VOLR_{short}_{long}'] = np.log(
            (v_s / v_l.replace(0, np.nan)).clip(lower=EPS)).fillna(0.0)

    # ── الحجم غير الاعتيادي ──────────────────────────────────────────────
    if 'volume' in df:
        w = int(s["volume_window"])
        lv = np.log1p(df['volume'].clip(lower=0).astype('float64'))
        mu = lv.rolling(w, min_periods=w).mean()
        sd = lv.rolling(w, min_periods=w).std()
        df[f'VOLZ_{w}'] = ((lv - mu) / sd.replace(0, np.nan)).fillna(0.0)

    # ── الموسمية ─────────────────────────────────────────────────────────
    if s["time_features"] and isinstance(df.index, pd.DatetimeIndex):
        hour = df.index.hour.values + df.index.minute.values / 60.0
        dow = df.index.dayofweek.values
        if _hour_features_meaningful(config):
            df['TIME_hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
            df['TIME_hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
        df['TIME_dow_sin'] = np.sin(2 * np.pi * dow / 7.0)
        df['TIME_dow_cos'] = np.cos(2 * np.pi * dow / 7.0)

    # ── فركتالات ويليامز (مُعطَّلة افتراضياً، fractal_windows=[]) ──────────
    for w in s["fractal_windows"]:
        df[f'FRACTAL_high_{w}'], df[f'FRACTAL_low_{w}'] = _fractal_columns(high, low, w)

    # ── RVI/Fisher Transform (مُعطَّلتان افتراضياً، rvi_windows/fisher_windows=[]) ──
    # استدعاء مباشر لـpandas_ta (يعتمد على تسجيل ملحق df.ta في الخلية التالية
    # في ترتيب تشغيل الدفتر؛ add_custom_features تُستدعى من add_features بعد
    # تسجيل الملحق فعلياً، فلا مشكلة رغم أن هذه الدالة مُعرَّفة قبله نصّياً).
    for w in s["rvi_windows"]:
        df.ta.rvi(length=w, append=True)
    for w in s["fisher_windows"]:
        df.ta.fisher(length=w, append=True)

    # ── مقدّرات تقلّب/سيولة كلاسيكية (مُعطَّلة افتراضياً) ──────────────────
    # Parkinson/Garman-Klass/Rogers-Satchell/Yang-Zhang: مقدّرات تقلّب مبنية
    # على OHLC كامل (لا close-to-close فقط) — نِسَب لوغاريتمية ضمن الشمعة
    # نفسها (H/O, L/O, C/O) فخالية من المقياس بالبناء تماماً كـRET_1، لا
    # تحتاج تطبيعاً إضافياً رغم عدم توفّر معالجة إحصائية شاملة كباقي الميزات.
    # Amihud (2002)/Roll (1984): مقياسا سيولة/انطباع سعري كلاسيكيان — ليسا
    # في pandas_ta. تحقّق أوّلي رخيص على 50 أصلاً حقيقياً: Parkinson/GK/RS/YZ
    # شبه مطابقة لـNATR_14 (IC متقارب جداً — على الأرجح تكرار للمعلومة
    # نفسها لا جديدة)، بينما Amihud/Roll أضعف لكن مختلفان (IC أصغر بكثير) —
    # راجع docs/research/new_volatility_liquidity_estimators.md.
    for w in s["vol_estimator_windows"]:
        u = np.log(high.clip(lower=EPS) / open_.clip(lower=EPS))
        dn = np.log(low.clip(lower=EPS) / open_.clip(lower=EPS))
        cc = np.log(close.clip(lower=EPS) / open_.clip(lower=EPS))
        overnight = np.log(open_.clip(lower=EPS) / close.shift(1).clip(lower=EPS))
        hl_log = np.log(high.clip(lower=EPS) / low.clip(lower=EPS))

        parkinson_bar = (1.0 / (4.0 * np.log(2))) * hl_log ** 2
        gk_bar = 0.5 * hl_log ** 2 - (2 * np.log(2) - 1) * cc ** 2
        rs_bar = u * (u - cc) + dn * (dn - cc)

        df[f'PARKINSON_{w}'] = np.sqrt(parkinson_bar.rolling(w, min_periods=w).mean()).fillna(0.0)
        df[f'GK_{w}'] = np.sqrt(gk_bar.clip(lower=0).rolling(w, min_periods=w).mean()).fillna(0.0)
        df[f'RS_{w}'] = np.sqrt(rs_bar.clip(lower=0).rolling(w, min_periods=w).mean()).fillna(0.0)

        k = 0.34 / (1.34 + (w + 1) / (w - 1))
        overnight_var = overnight.rolling(w, min_periods=w).var()
        openclose_var = cc.rolling(w, min_periods=w).var()
        rs_mean = rs_bar.rolling(w, min_periods=w).mean()
        yz_var = (overnight_var + k * openclose_var + (1 - k) * rs_mean).clip(lower=0)
        df[f'YZ_{w}'] = np.sqrt(yz_var).fillna(0.0)

        if 'volume' in df:
            dollar_vol = (df['volume'].clip(lower=0).astype('float64') * close).replace(0, np.nan)
            amihud_bar = (ret_1.abs() / dollar_vol) * 1e6
            df[f'AMIHUD_{w}'] = amihud_bar.rolling(w, min_periods=w).mean().fillna(0.0)

        dp = close.diff()
        roll_cov = dp.rolling(w, min_periods=w).apply(
            lambda x: np.cov(x[:-1], x[1:])[0, 1] if len(x) > 2 else np.nan, raw=True)
        df[f'ROLL_SPREAD_{w}'] = np.sqrt(np.clip(-roll_cov, 0, None)).fillna(0.0)

    # ── Hurst / Variance Ratio (مُعطَّلة افتراضياً، mean_reversion_windows=[]) ──
    # Hurst (Hurst 1951, عبر مقياس تدرّج التباين): Var(مجموع τ عوائد) ~ τ^(2H)؛
    # الميل نصف انحدار log(Var) على log(τ) عبر عدّة قيَم τ يقدّر H مباشرة —
    # H=0.5 مسار عشوائي، H<0.5 انعكاس (نفس موضوع H003)، H>0.5 استمرار/زخم.
    # Variance Ratio (Lo-MacKinlay 1988): VR(q) = Var(عائد q فترات)/(q×Var(عائد
    # فترة واحدة))؛ نفس المنطق لكن اختبار إحصائي مباشر بدل انحدار. كلاهما
    # غائب تماماً عن pandas_ta (تحقّق مباشر عبر ta.Category). راجع
    # docs/research/hurst_variance_ratio_features.md.
    for w in s["mean_reversion_windows"]:
        q = max(2, w // 4)
        taus = tuple(t for t in (1, 2, 4, 8) if t < w) or (1,)
        df[f'VR_{w}'] = ret_1.rolling(w, min_periods=w).apply(
            lambda x: _variance_ratio_stat(x, q), raw=True).fillna(1.0)
        df[f'HURST_{w}'] = ret_1.rolling(w, min_periods=w).apply(
            lambda x: _hurst_variance_scaling(x, taus), raw=True).fillna(0.5)

    # ── نسبة التقلّب القفزي (مُعطَّلة افتراضياً، jump_windows=[]) ──────────
    # Bipower Variation (Barndorff-Nielsen & Shephard 2004): يقدّر المكوّن
    # المستمرّ فقط من التباين المُحقَّق (RV) عبر حاصل ضرب |عائد| متتاليين
    # (لا مربّع عائد واحد كما في RV) — القفزات المفاجئة لا تُسهم في هذا
    # التقدير لأن احتمال قفزتين متتاليتين متقاربتي الحجم ضئيل جداً. الفرق
    # RV-BV (مُقصوص عند الصفر) هو تقدير مكوّن القفزات، ونسبته لـRV
    # (JUMP_RATIO) خالية من المقياس بالبناء ومحصورة [0, 1].
    for w in s["jump_windows"]:
        rv = (ret_1 ** 2).rolling(w, min_periods=w).sum()
        bv = ret_1.rolling(w, min_periods=w).apply(_bipower_variation, raw=True)
        jump = (rv - bv).clip(lower=0)
        df[f'JUMP_RATIO_{w}'] = (jump / rv.replace(0, np.nan)).clip(0.0, 1.0).fillna(0.0)

    # ── SuperTrend (مُعطَّلة افتراضياً، supertrend_windows=[]) ─────────────
    # مؤشّر استمراري (trend-following) شائع جداً عملياً لكن غائب عن مجموعة
    # الميزات الافتراضية لهذا المشروع. الاستخدام الصحيح الموثَّق (LuxAlgo/
    # Mudrex/forex.com، راجع docs/research/proper_indicator_signals.md):
    # الإشارة الفعلية هي **اتجاه الخط** (فوق السعر=هابط، تحته=صاعد)، لا قيمة
    # الخط السعرية المطلقة — SUPERTd الجاهزة من pandas_ta تُعطي هذا مباشرة.
    # SUPERT_STRETCH يضيف مقياس "مدى امتداد الاتجاه" (نسبة مئوية، خالية من
    # المقياس): المسافة بين الإغلاق وخطّ SuperTrend.
    for w in s["supertrend_windows"]:
        st = df.ta.supertrend(length=w, multiplier=3.0)
        line_col, dir_col = f'SUPERT_{w}_3.0', f'SUPERTd_{w}_3.0'
        df[f'SUPERT_DIR_{w}'] = st[dir_col].fillna(0.0)
        df[f'SUPERT_STRETCH_{w}'] = (
            (close - st[line_col]) / close.clip(lower=EPS) * 100.0
        ).fillna(0.0)

    # ── مقياس تباعد RSI/السعر (مُعطَّلة افتراضياً، rsi_divergence_windows=[]) ──
    # تقريب مبسَّط لمفهوم "Divergence" — وايلدر نفسه وصفه بأنه الميزة الأقوى
    # في RSI (أقوى من عتبتَي 70/30 التقليديتين). التقريب هنا: فرق الدرجة
    # المعيارية بين تغيّر السعر وتغيّر RSI على نفس الأفق L (لا مطابقة قمم/
    # قيعان مؤكَّدة حرفياً كما في التعريف الأصلي — قيد موثَّق في التوثيق
    # المرفق). قيمة موجبة كبيرة = السعر ارتفع أكثر ممّا يدعمه الزخم (تباعد
    # هابط، انعكاس محتمل للأسفل)؛ قيمة سالبة كبيرة = العكس (تباعد صاعد).
    if s["rsi_divergence_windows"]:
        rsi_14 = df['RSI_14'] if 'RSI_14' in df.columns else df.ta.rsi(length=14)
    for L in s["rsi_divergence_windows"]:
        norm_w = max(4 * L, 20)
        price_chg = log_close - log_close.shift(L)
        rsi_chg = rsi_14 - rsi_14.shift(L)
        z_price = (price_chg - price_chg.rolling(norm_w, min_periods=norm_w).mean()) /             price_chg.rolling(norm_w, min_periods=norm_w).std().replace(0, np.nan)
        z_rsi = (rsi_chg - rsi_chg.rolling(norm_w, min_periods=norm_w).mean()) /             rsi_chg.rolling(norm_w, min_periods=norm_w).std().replace(0, np.nan)
        df[f'RSI_DIVERGENCE_{L}'] = (z_price - z_rsi).fillna(0.0)

    # ── عمر الاتجاه (مُعطَّلة افتراضياً، trend_age_windows=[]) ─────────────
    # عدد الشموع منذ آخر قمة/قاع فركتالي مؤكَّد (window). القيمة تُقصّ عند
    # 10×w (سقف معقول يمنع تطرّفاً غير محدود في بدايات السلسلة النادرة بلا
    # فركتال بعد)، وتُملأ بـw عند الغياب الكامل (قيمة "حديثة نسبياً" محايدة
    # بدل صفر أو NaN مضلِّلين).
    for w in s["trend_age_windows"]:
        fh, fl = _fractal_columns(high, low, w)
        df[f'BARS_SINCE_HIGH_{w}'] = _bars_since_last_true(fh == 1.0, cap=10 * w, default=w)
        df[f'BARS_SINCE_LOW_{w}'] = _bars_since_last_true(fl == 1.0, cap=10 * w, default=w)

    # ── اتساق الاتجاه اليومي (مُعطَّلة افتراضياً، direction_consistency_windows=[]) ──
    # نسبة بسيطة (لا انحدار لوغاريتمي) — بديل أكثر استقراراً إحصائياً من
    # Hurst/VR على نوافذ قصيرة: متوسط توافق إشارة عائد كل يوم مع اليوم
    # السابق مباشرة، محسوب مباشرة عبر rolling.mean بلا تقدير معاملات.
    same_sign = (np.sign(ret_1) == np.sign(ret_1.shift(1))).astype('float64')
    for w in s["direction_consistency_windows"]:
        df[f'DIR_CONSISTENCY_{w}'] = same_sign.rolling(w, min_periods=w).mean().fillna(0.5)

    # ── عدم تناظر التقلّب (مُعطَّلة افتراضياً، vol_asymmetry_windows=[]) ────
    # شبه-تباين هابط (متوسط مربّعات العوائد السالبة فقط) مقابل شبه-تباين
    # صاعد (الموجبة فقط) خلال نافذة w — النسبة (هابط-صاعد)/(هابط+صاعد)
    # محصورة [-1, 1] بالبناء: +1 كل التقلّب هابط، -1 كل التقلّب صاعد، 0
    # تناظر تام. سببية (rolling بلا center)، خالية من المقياس (نسبة).
    ret_sq = ret_1 ** 2
    down_sq = ret_sq.where(ret_1 < 0, 0.0)
    up_sq = ret_sq.where(ret_1 > 0, 0.0)
    for w in s["vol_asymmetry_windows"]:
        down_var = down_sq.rolling(w, min_periods=w).mean()
        up_var = up_sq.rolling(w, min_periods=w).mean()
        total = (down_var + up_var).replace(0, np.nan)
        df[f'VOL_ASYMMETRY_{w}'] = ((down_var - up_var) / total).fillna(0.0)

    # ── Parabolic SAR (مُعطَّلة افتراضياً، psar_enabled=False) ─────────────
    # مؤشّر اتجاهي/انعكاسي قديم (وايلدر 1978، سابق لـSuperTrend وأبسط
    # حسابياً). الاستخدام الصحيح الموثَّق: **اتجاه** النقاط لا قيمتها
    # السعرية — pandas_ta يُخرج عمودين منفصلين (PSARl عند الاتجاه الصاعد،
    # PSARs عند الهابط، أحدهما NaN دائماً بحسب الحالة)؛ PSAR_DIR يدمجهما
    # في إشارة اتجاه واحدة (+1/-1)، بنفس منطق SUPERT_DIR تماماً.
    if s["psar_enabled"]:
        psar = df.ta.psar(af0=0.02, af=0.02, max_af=0.2)
        long_col = next(c for c in psar.columns if c.startswith('PSARl_'))
        short_col = next(c for c in psar.columns if c.startswith('PSARs_'))
        df['PSAR_DIR'] = np.where(psar[long_col].notna(), 1.0,
                                  np.where(psar[short_col].notna(), -1.0, 0.0))

    # ── نظام "ارتداد ما بعد الانهيار" (مُعطَّلة افتراضياً، crash_rebound_regime=None) ──
    # هل هبط الأصل ≥threshold عن أعلى قمّة في آخر high_lookback يوماً، في أي
    # لحظة من آخر recent_window يوماً؟ سببي بالكامل (rolling بلا center).
    crr = s["crash_rebound_regime"]
    if crr is not None:
        high_lb = int(crr.get("high_lookback", 60))
        recent_w = int(crr.get("recent_window", 30))
        threshold = float(crr.get("drawdown_threshold", -0.30))
        rolling_high = close.rolling(high_lb, min_periods=max(2, high_lb // 2)).max()
        drawdown = close / rolling_high - 1.0
        df['CRASH_REBOUND_REGIME'] = (
            drawdown.rolling(recent_w, min_periods=1).min() <= threshold
        ).astype('float64').fillna(0.0)

    # ── زخم مُعدَّل بالمخاطرة (مُعطَّلة افتراضياً، risk_adj_momentum_windows=[]) ──
    # متوسط/انحراف معياري لعائد يوم واحد خلال نافذة w — نسبة شبيهة بشارب،
    # سببية بالكامل (rolling بلا center). محصورة عملياً بلا حدّ صريح؛
    # القسمة على صفر (سلسلة ثابتة تماماً، لا تحدث في بيانات حقيقية) تُعطي
    # NaN تُملأ صفراً (محايد).
    for w in s["risk_adj_momentum_windows"]:
        mu = ret_1.rolling(w, min_periods=w).mean()
        sigma = ret_1.rolling(w, min_periods=w).std()
        df[f'RISK_ADJ_MOM_{w}'] = (mu / sigma.replace(0, np.nan)).fillna(0.0)

    # ── المسافة عن القمّة التاريخية (مُعطَّلة افتراضياً، ath_distance_enabled=False) ──
    # ath_close: أعلى إغلاق حتى الآن شاملاً (expanding().max())، سببي بالكامل
    # (لا يعتمد إلا على الماضي والحاضر). PCT_FROM_ATH دائماً ≤0 بالبناء.
    # DAYS_SINCE_ATH: عدد الشموع منذ آخر قمّة تاريخية جديدة (إعادة استخدام
    # _bars_since_last_true، بلا سقف/cap هنا فالمفهوم نفسه غير محدود زمنياً).
    if s["ath_distance_enabled"]:
        ath_close = close.expanding(min_periods=1).max()
        df['PCT_FROM_ATH'] = (close / ath_close - 1.0).fillna(0.0)
        is_new_ath = (close >= ath_close)
        df['DAYS_SINCE_ATH'] = _bars_since_last_true(is_new_ath, cap=np.inf, default=0.0)

    # ── البنية الزمنية للتقلّب (مُعطَّلة افتراضياً، vol_term_structure_pairs=[]) ──
    # NATR محسوبة يدوياً (True Range الكلاسيكي: أقصى من المدى اليومي، أو
    # الفرق المطلق عن إغلاق الأمس) — مستقلّة عن NATR_14 القياسية عمداً، فلا
    # يتأثّر هذا الاختبار بأي تغيير مستقبلي على إعدادات pandas_ta الافتراضية.
    if s["vol_term_structure_pairs"]:
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(),
                        (low - prev_close).abs()], axis=1).max(axis=1)
        for short_w, long_w in s["vol_term_structure_pairs"]:
            natr_short = 100.0 * tr.rolling(short_w, min_periods=short_w).mean() / close
            natr_long = 100.0 * tr.rolling(long_w, min_periods=long_w).mean() / close
            df[f'VOL_TERM_{short_w}_{long_w}'] = (
                natr_short / natr_long.replace(0, np.nan)
            ).fillna(1.0)

    return df


def _fractal_columns(high: pd.Series, low: pd.Series, window: int):
    """فركتال ويليامز — سببي بالكامل رغم استخدام ``center=True`` داخلياً.

    ``high.rolling(window, center=True).max()`` عند الموضع ``t`` يحتاج فعلياً
    بيانات حتى ``t + window//2`` — قيمة مستقبلية غير معروفة عند اتخاذ القرار
    في ``t`` (بالضبط فخّ DPO الموثَّق في خطة المشروع، قسم "تقطير الرؤية
    المتأخّرة"؛ راجع أيضاً حاشية 24 سبتمبر 2026 هناك). الإصلاح هنا **إزاحة**
    الناتج الخام بمقدار ``window//2`` إلى الأمام (``shift``، لا حذف/تصفير)،
    فتصبح القيمة عند ``t`` هي "هل كان ``t - window//2`` قمّة/قاعاً مؤكَّدة؟" —
    وهذا يحتاج فقط بيانات حتى ``t`` نفسها (``(t-window//2)+window//2 = t``)،
    أي **سببي تماماً** عند كل موضع بما في ذلك آخر شمعة في أي نافذة، لا حاجة
    لتصفير آخر خطوتين يدوياً: الإزاحة تُغني عن ذلك لأنها تُعيد توسيم كل نقطة
    بزمنها الحقيقي (زمن التأكيد) بدل تركها بزمن الرصد الخام غير المُتاح بعد.
    فقط بداية كامل السجل (قبل توفّر أول ``window`` شمعة) تبقى NaN→0 (إحماء
    عادي، لا تسرّب)."""
    half = window // 2
    raw_high = (high == high.rolling(window, center=True).max()).astype('float64')
    raw_low = (low == low.rolling(window, center=True).min()).astype('float64')
    return raw_high.shift(half).fillna(0.0), raw_low.shift(half).fillna(0.0)


def _variance_ratio_stat(x, q: int) -> float:
    """إحصائية Variance Ratio للو-ماكينلي (1988): Var(عائد q فترات، بتراكب
    متداخل)/(q×Var(عائد فترة واحدة)). =1 تحت المسار العشوائي، <1 انعكاس،
    >1 استمرار/زخم. ``x`` مصفوفة عوائد لوغاريتمية (نافذة واحدة، raw=True)."""
    x = np.asarray(x, dtype='float64')
    n = len(x)
    if n <= q:
        return np.nan
    var1 = np.var(x, ddof=1)
    if var1 <= EPS:
        return np.nan
    qsums = pd.Series(x).rolling(q).sum().dropna().to_numpy()
    if len(qsums) < 2:
        return np.nan
    varq = np.var(qsums, ddof=1)
    return varq / (q * var1)


def _hurst_variance_scaling(x, taus) -> float:
    """أسّ هيرست عبر تدرّج التباين: Var(مجموع τ عوائد) ~ τ^(2H)، فيُقدَّر H
    بنصف ميل انحدار log(Var) على log(τ) عبر قيَم ``taus`` المُعطاة. ``x``
    مصفوفة عوائد لوغاريتمية (نافذة واحدة، raw=True)."""
    x = np.asarray(x, dtype='float64')
    n = len(x)
    log_taus, log_vars = [], []
    for tau in taus:
        if tau >= n:
            continue
        s = pd.Series(x).rolling(tau).sum().dropna().to_numpy()
        if len(s) < 2:
            continue
        var = np.var(s, ddof=1)
        if var <= EPS:
            continue
        log_taus.append(np.log(tau))
        log_vars.append(np.log(var))
    if len(log_taus) < 2:
        return np.nan
    slope = np.polyfit(log_taus, log_vars, 1)[0]
    return slope / 2.0


def _bars_since_last_true(mask: pd.Series, cap: float, default: float) -> pd.Series:
    """عدد الصفوف منذ آخر ``True`` في ``mask`` (شاملاً الصف نفسه: 0 = ``True``
    عند نفس الصف) — سببي بالكامل (``np.maximum.accumulate`` عند أي نقطة
    يعتمد فقط على القيم حتى تلك النقطة). ``default`` يملأ الغياب الكامل
    (قبل أوّل ``True`` في السلسلة)، ``cap`` يقصّ القيم المتطرّفة."""
    idx = np.arange(len(mask))
    last_true_idx = np.where(mask.to_numpy(), idx, -1)
    last_true_idx = np.maximum.accumulate(last_true_idx)
    bars_since = np.where(last_true_idx < 0, default, idx - last_true_idx).astype('float64')
    return pd.Series(np.clip(bars_since, 0, cap), index=mask.index)


def _bipower_variation(x) -> float:
    """تباين ثنائي القوّة (Barndorff-Nielsen & Shephard 2004): تقدير مُتّسق
    للمكوّن المستمرّ فقط من التباين المُحقَّق، عبر حاصل ضرب |عائد| متتاليين
    بدل مربّع عائد واحد — قفزة مفاجئة تُسهم بمربّعها الكامل في RV لكن بجداء
    ضعيف مع جارتيها هنا (احتمال قفزتين متتاليتين ضئيل). ``x`` مصفوفة عوائد
    لوغاريتمية (نافذة واحدة، raw=True)."""
    x = np.asarray(x, dtype='float64')
    n = len(x)
    if n < 2:
        return np.nan
    abs_x = np.abs(x)
    return (np.pi / 2.0) * (n / (n - 1)) * np.sum(abs_x[1:] * abs_x[:-1])


def custom_feature_names(settings: Optional[dict] = None,
                         config: Optional[dict] = None) -> List[str]:
    """أسماء الميزات المخصّصة التي ستُنتَج بالإعدادات الحالية."""
    config = CONFIG if config is None else config
    s = {**DEFAULT_CUSTOM_SETTINGS, **(settings or config.get("custom_settings") or {})}
    names = [f'RET_{h}' for h in s["returns"]] + ['RANGE_rel']
    if s["candle_structure"]:
        names += ['BODY_ratio', 'WICK_upper', 'WICK_lower']
    names += [f'POS_{w}' for w in s["range_windows"]]
    names += [f'VOLR_{a}_{b}' for a, b in s["vol_windows"]]
    names += [f'VOLZ_{int(s["volume_window"])}']
    if s["time_features"]:
        if _hour_features_meaningful(config):
            names += ['TIME_hour_sin', 'TIME_hour_cos']
        names += ['TIME_dow_sin', 'TIME_dow_cos']
    for w in s["fractal_windows"]:
        names += [f'FRACTAL_high_{w}', f'FRACTAL_low_{w}']
    for w in s["rvi_windows"]:
        names += [f'RVI_{w}']
    for w in s["fisher_windows"]:
        names += [f'FISHERT_{w}_1', f'FISHERTs_{w}_1']
    for w in s["vol_estimator_windows"]:
        names += [f'PARKINSON_{w}', f'GK_{w}', f'RS_{w}', f'YZ_{w}', f'ROLL_SPREAD_{w}']
        names += [f'AMIHUD_{w}']
    for w in s["mean_reversion_windows"]:
        names += [f'VR_{w}', f'HURST_{w}']
    for w in s["jump_windows"]:
        names += [f'JUMP_RATIO_{w}']
    for w in s["supertrend_windows"]:
        names += [f'SUPERT_DIR_{w}', f'SUPERT_STRETCH_{w}']
    for L in s["rsi_divergence_windows"]:
        names += [f'RSI_DIVERGENCE_{L}']
    if s["psar_enabled"]:
        names += ['PSAR_DIR']
    for w in s["trend_age_windows"]:
        names += [f'BARS_SINCE_HIGH_{w}', f'BARS_SINCE_LOW_{w}']
    for w in s["direction_consistency_windows"]:
        names += [f'DIR_CONSISTENCY_{w}']
    for w in s["vol_asymmetry_windows"]:
        names += [f'VOL_ASYMMETRY_{w}']
    if s["crash_rebound_regime"] is not None:
        names += ['CRASH_REBOUND_REGIME']
    for w in s["risk_adj_momentum_windows"]:
        names += [f'RISK_ADJ_MOM_{w}']
    if s["ath_distance_enabled"]:
        names += ['PCT_FROM_ATH', 'DAYS_SINCE_ATH']
    for short_w, long_w in s["vol_term_structure_pairs"]:
        names += [f'VOL_TERM_{short_w}_{long_w}']
    return names



def _test_fractal_features_causal():
    """اختبار ذاتي لـ_fractal_columns/add_custom_features: (أ) مُعطَّلة
    افتراضياً (fractal_windows=[]) فلا تُغيّر أي سلوك قديم، (ب) عند التفعيل
    كلّ قيمة سببية فعلاً (تقطيع السلسلة عند أي نقطة يُعطي نفس آخر قيمة التي
    تُنتجها السلسلة الكاملة عند تلك النقطة — أي لا تعتمد على أي بيانات بعدها)،
    (ج) قمّة/قاعاً صناعيين واضحين يُكتشَفان عند الموضع المتوقَّع بالضبط."""
    import numpy as np, pandas as pd
    n = 60
    idx = pd.date_range('2024-01-01', periods=n, freq='D')
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))
    high = pd.Series(close + rng.random(n) * 0.5, index=idx)
    low = pd.Series(close - rng.random(n) * 0.5, index=idx)
    df = pd.DataFrame({'open': close, 'high': high.values, 'low': low.values,
                       'close': close, 'volume': rng.random(n) * 100}, index=idx)

    # (أ) مُعطَّلة افتراضياً
    out_default = add_custom_features(df, config={'custom_settings': {}})
    assert not any(c.startswith('FRACTAL_') for c in out_default.columns),         'fractal_windows=[] يجب ألا يُنتج أي عمود FRACTAL_ (توافق خلفي)'

    # (ب) السببية: تقطيع عند أي نقطة يُطابق القيمة المقابلة من السلسلة الكاملة
    fh_full, fl_full = _fractal_columns(high, low, window=5)
    for k in (20, 40, 59, 60):
        fh_k, fl_k = _fractal_columns(high.iloc[:k], low.iloc[:k], window=5)
        assert fh_k.iloc[-1] == fh_full.iloc[k - 1], f'تسرّب معلومة مستقبلية عند k={k} (high)'
        assert fl_k.iloc[-1] == fl_full.iloc[k - 1], f'تسرّب معلومة مستقبلية عند k={k} (low)'

    # (ج) قمّة/قاعاً صناعيان عند مواضع معروفة (window=5 → half=2)
    h2 = high.copy()
    h2.iloc[30] = h2.iloc[25:36].max() + 10.0  # قمّة واضحة عند 30
    l2 = low.copy()
    l2.iloc[30] = l2.iloc[25:36].min() - 10.0  # قاع واضح عند نفس الموضع (يُلغي بعضه)
    fh2, _ = _fractal_columns(h2, low, window=5)
    assert fh2.iloc[32] == 1.0, 'القمّة الصناعية عند 30 يجب أن تظهر مُؤكَّدة عند 30+half=32'
    assert fh2.iloc[31] == 0.0 and fh2.iloc[33] == 0.0, 'لا تأكيد قبل/بعد نقطة التأكيد مباشرة'

    print('✅ _test_fractal_features_causal: مُعطَّلة افتراضياً + سببية مُتحقَّقة + كشف صحيح لقمّة/قاع صناعيين')


_test_fractal_features_causal()


def _test_rvi_fisher_features():
    """اختبار ذاتي لـRVI/Fisher الاختياريتين: (أ) مُعطَّلتان افتراضياً فلا
    تُغيّران أي سلوك قديم، (ب) عند التفعيل تُنتجان نفس القيم بالضبط التي
    يُنتجها استدعاء pandas_ta مباشرةً (لا استنساخ منطق خاطئ)، (ج) الأعمدة
    المتوقَّعة في custom_feature_names تطابق ما يُنتجه add_custom_features
    فعلياً. يستورد pandas_ta محلياً لأن هذه الخلية تُنفَّذ (وهذا الاختبار
    الذاتي أسفلها) قبل خلية `import pandas_ta_classic as ta` في ترتيب
    الدفتر — تسجيل ملحق df.ta عملية عامة على العملية بأكملها، فاستيراده هنا
    مبكراً لا يتعارض مع ذلك الاستيراد لاحقاً (نفس الوحدة، الاستيراد الثاني
    مجرّد إرجاع من الذاكرة المخبَّأة)."""
    try:
        import pandas_ta_classic as ta  # noqa: F401
    except ImportError:
        import pandas_ta as ta  # noqa: F401
    import numpy as np, pandas as pd
    n = 60
    idx = pd.date_range('2024-01-01', periods=n, freq='D')
    rng = np.random.default_rng(1)
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))
    high = close + rng.random(n) * 0.5
    low = close - rng.random(n) * 0.5
    open_ = close + rng.normal(0, 0.2, n)
    volume = rng.random(n) * 100
    df = pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close,
                       'volume': volume}, index=idx)

    # (أ) مُعطَّلتان افتراضياً
    out_default = add_custom_features(df, config={'custom_settings': {}})
    assert not any(c.startswith(('RVI_', 'FISHERT_')) for c in out_default.columns),         'rvi_windows/fisher_windows=[] يجب ألا يُنتجا أي عمود (توافق خلفي)'

    # (ب) عند التفعيل: نفس قيم pandas_ta المباشرة تماماً
    settings = {'rvi_windows': [14], 'fisher_windows': [14]}
    out = add_custom_features(df, config={'custom_settings': settings})
    direct_rvi = df.copy(); direct_rvi.ta.rvi(length=14, append=True)
    direct_fisher = df.copy(); direct_fisher.ta.fisher(length=14, append=True)
    pd.testing.assert_series_equal(out['RVI_14'], direct_rvi['RVI_14'], check_names=False)
    pd.testing.assert_series_equal(out['FISHERT_14_1'], direct_fisher['FISHERT_14_1'], check_names=False)
    pd.testing.assert_series_equal(out['FISHERTs_14_1'], direct_fisher['FISHERTs_14_1'], check_names=False)

    # (ج) أسماء الميزات المتوقَّعة تطابق الأعمدة الفعلية
    expected = set(custom_feature_names(settings=settings))
    produced = set(out.columns) - set(df.columns)
    assert expected.issubset(produced), f'أعمدة متوقَّعة غائبة: {expected - produced}'

    print('✅ _test_rvi_fisher_features: مُعطَّلتان افتراضياً + قيم مطابقة لـpandas_ta المباشر + أسماء صحيحة')


_test_rvi_fisher_features()


def _test_vol_estimator_features():
    """اختبار ذاتي لمقدّرات التقلّب/السيولة الجديدة: (أ) مُعطَّلة افتراضياً،
    (ب) القيم غير سالبة وغير NaN بعد فترة الإحماء، (ج) Parkinson/GK/RS/YZ
    مرتبطة موجباً بمدى الشمعة (تحقّق منطقي بسيط: مدى أوسع ↔ تقلّب مُقدَّر
    أعلى)، (د) لا قيمة مستقبلية (rolling بلا center، shift موجب فقط —
    فحص بصري كافٍ هنا؛ التحقّق الرياضي الكامل تمّ يدوياً عبر مقارنة صيغة كل
    مقدّر بمرجعه الأكاديمي الأصلي)."""
    import numpy as np, pandas as pd
    n = 60
    idx = pd.date_range('2024-01-01', periods=n, freq='D')
    rng = np.random.default_rng(2)
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))
    high = close + rng.random(n) * 0.5
    low = close - rng.random(n) * 0.5
    open_ = close + rng.normal(0, 0.2, n)
    volume = rng.random(n) * 100 + 10
    df = pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close,
                       'volume': volume}, index=idx)

    out_default = add_custom_features(df, config={'custom_settings': {}})
    assert not any(c.startswith(('PARKINSON_', 'GK_', 'RS_', 'YZ_', 'AMIHUD_', 'ROLL_SPREAD_'))
                  for c in out_default.columns), 'vol_estimator_windows=[] يجب ألا ينتج أي عمود (توافق خلفي)'

    settings = {'vol_estimator_windows': [14]}
    out = add_custom_features(df, config={'custom_settings': settings})
    for col in ('PARKINSON_14', 'GK_14', 'RS_14', 'YZ_14', 'AMIHUD_14', 'ROLL_SPREAD_14'):
        assert col in out.columns, f'{col} غائب رغم التفعيل'
        assert out[col].notna().all(), f'{col} يحوي NaN رغم fillna'
        assert (out[col].iloc[14:] >= 0).all(), f'{col} يجب أن يكون غير سالب (جذر تربيعي)'

    wide_range = df.copy()
    wide_range.loc[wide_range.index[30:], 'high'] = wide_range.loc[wide_range.index[30:], 'high'] + 5.0
    out_wide = add_custom_features(wide_range, config={'custom_settings': settings})
    assert out_wide['PARKINSON_14'].iloc[-1] > out['PARKINSON_14'].iloc[-1],         'مدى أوسع يجب أن يرفع Parkinson المُقدَّر'

    expected = set(custom_feature_names(settings=settings))
    produced = set(out.columns) - set(df.columns)
    assert expected.issubset(produced), f'أعمدة متوقَّعة غائبة: {expected - produced}'

    print('✅ _test_vol_estimator_features: مُعطَّلة افتراضياً + قيم غير سالبة + حسّاسة لاتساع المدى + أسماء صحيحة')


_test_vol_estimator_features()


def _test_mean_reversion_features():
    """اختبار ذاتي لـHurst/Variance Ratio: (أ) مُعطَّلتان افتراضياً، (ب) على
    سلاسل AR(1) صناعية معروفة الخاصية إحصائياً (عشوائية/مُنعكسة/مُتّجهة)
    تُرتَّب القيم بالاتجاه الصحيح نظرياً: HURST(انعكاس) < HURST(عشوائي) <
    HURST(اتجاه)، وVR(انعكاس) < 1 < VR(اتجاه) — هذا تحقّق أقوى من مجرّد
    الحساسية لأنه يقيس مطابقة تعريف إحصائي معروف بدل سلوك نوعي فقط،
    (ج) عند التفعيل داخل add_custom_features لا NaN وأسماء الأعمدة صحيحة."""
    import numpy as np, pandas as pd
    rng = np.random.default_rng(3)
    n = 4000

    def ar1_returns(phi, n, rng):
        eps = rng.normal(0, 1.0, n)
        r = np.zeros(n)
        for t in range(1, n):
            r[t] = phi * r[t - 1] + eps[t]
        return r

    mean_revert = ar1_returns(-0.5, n, rng)
    random_walk = ar1_returns(0.0, n, rng)
    trending = ar1_returns(0.5, n, rng)

    taus = (1, 2, 4, 8)
    q = 50
    h_rev = _hurst_variance_scaling(mean_revert, taus)
    h_rnd = _hurst_variance_scaling(random_walk, taus)
    h_trend = _hurst_variance_scaling(trending, taus)
    assert h_rev < h_rnd < h_trend, \
        f'ترتيب Hurst خاطئ: انعكاس={h_rev:.3f} عشوائي={h_rnd:.3f} اتجاه={h_trend:.3f}'
    assert abs(h_rnd - 0.5) < 0.15, \
        f'Hurst للمسار العشوائي بعيد جداً عن 0.5: {h_rnd:.3f}'

    vr_rev = _variance_ratio_stat(mean_revert, q)
    vr_rnd = _variance_ratio_stat(random_walk, q)
    vr_trend = _variance_ratio_stat(trending, q)
    assert vr_rev < 1.0 < vr_trend, \
        f'ترتيب Variance Ratio خاطئ حول 1: انعكاس={vr_rev:.3f} اتجاه={vr_trend:.3f}'
    assert abs(vr_rnd - 1.0) < 0.3, \
        f'VR للمسار العشوائي بعيد جداً عن 1: {vr_rnd:.3f}'

    idx = pd.date_range('2024-01-01', periods=200, freq='D')
    close = 100 + np.cumsum(rng.normal(0, 0.3, 200))
    high = close + rng.random(200) * 0.5
    low = close - rng.random(200) * 0.5
    open_ = close + rng.normal(0, 0.2, 200)
    volume = rng.random(200) * 100 + 10
    df = pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close,
                       'volume': volume}, index=idx)

    out_default = add_custom_features(df, config={'custom_settings': {}})
    assert not any(c.startswith(('VR_', 'HURST_')) for c in out_default.columns), \
        'mean_reversion_windows=[] يجب ألا ينتج أي عمود (توافق خلفي)'

    settings = {'mean_reversion_windows': [40]}
    out = add_custom_features(df, config={'custom_settings': settings})
    for col in ('VR_40', 'HURST_40'):
        assert col in out.columns, f'{col} غائب رغم التفعيل'
        assert out[col].notna().all(), f'{col} يحوي NaN رغم fillna'

    expected = set(custom_feature_names(settings=settings))
    produced = set(out.columns) - set(df.columns)
    assert expected.issubset(produced), f'أعمدة متوقَّعة غائبة: {expected - produced}'

    print('✅ _test_mean_reversion_features: مُعطَّلتان افتراضياً + ترتيب Hurst/VR '
          'مطابق للتعريف الإحصائي على AR(1) صناعية + أسماء صحيحة')


_test_mean_reversion_features()


def _test_jump_ratio_feature():
    """اختبار ذاتي لـJUMP_RATIO: (أ) مُعطَّلة افتراضياً، (ب) محصورة [0,1]
    دائماً، (ج) الاختبار الأقوى: حقن قفزة سعرية صناعية ضخمة واحدة داخل نافذة
    عوائد صغيرة ومستقرة يجب أن يرفع JUMP_RATIO بشكل واضح (قريب من 1) مقارنةً
    بنافذة بلا قفزة (قريبة من 0) — تحقّق مباشر من تعريف Bipower Variation
    الإحصائي، لا مجرّد حساسية نوعية."""
    import numpy as np, pandas as pd
    rng = np.random.default_rng(4)
    n = 60
    small_returns = rng.normal(0, 0.001, n)  # انتشار مستمرّ صغير جداً، بلا قفزات
    assert abs(_bipower_variation(small_returns) -
               np.sum((small_returns[1:] ** 2 + small_returns[:-1] ** 2)) * 0) >= 0  # sanity: لا استثناء

    rv_no_jump = np.sum(small_returns ** 2)
    bv_no_jump = _bipower_variation(small_returns)
    jump_ratio_no_jump = max(rv_no_jump - bv_no_jump, 0) / rv_no_jump

    with_jump = small_returns.copy()
    with_jump[30] = 0.15  # قفزة صناعية ضخمة (15%) وسط عوائد <0.1% نموذجياً
    rv_jump = np.sum(with_jump ** 2)
    bv_jump = _bipower_variation(with_jump)
    jump_ratio_with_jump = max(rv_jump - bv_jump, 0) / rv_jump

    assert jump_ratio_with_jump > jump_ratio_no_jump, \
        f'القفزة الصناعية يجب أن ترفع JUMP_RATIO: بلا قفزة={jump_ratio_no_jump:.3f} مع قفزة={jump_ratio_with_jump:.3f}'
    assert jump_ratio_with_jump > 0.5, \
        f'قفزة مهيمنة بهذا الحجم يجب أن تعطي JUMP_RATIO مرتفعاً جداً: {jump_ratio_with_jump:.3f}'

    idx = pd.date_range('2024-01-01', periods=200, freq='D')
    close = 100 + np.cumsum(rng.normal(0, 0.3, 200))
    high = close + rng.random(200) * 0.5
    low = close - rng.random(200) * 0.5
    open_ = close + rng.normal(0, 0.2, 200)
    volume = rng.random(200) * 100 + 10
    df = pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close,
                       'volume': volume}, index=idx)

    out_default = add_custom_features(df, config={'custom_settings': {}})
    assert not any(c.startswith('JUMP_RATIO_') for c in out_default.columns), \
        'jump_windows=[] يجب ألا ينتج أي عمود (توافق خلفي)'

    settings = {'jump_windows': [40]}
    out = add_custom_features(df, config={'custom_settings': settings})
    assert 'JUMP_RATIO_40' in out.columns, 'JUMP_RATIO_40 غائب رغم التفعيل'
    assert out['JUMP_RATIO_40'].notna().all(), 'JUMP_RATIO_40 يحوي NaN رغم fillna'
    assert out['JUMP_RATIO_40'].between(0.0, 1.0).all(), 'JUMP_RATIO_40 يجب أن يكون محصوراً [0,1]'

    expected = set(custom_feature_names(settings=settings))
    produced = set(out.columns) - set(df.columns)
    assert expected.issubset(produced), f'أعمدة متوقَّعة غائبة: {expected - produced}'

    print('✅ _test_jump_ratio_feature: مُعطَّلة افتراضياً + محصورة [0,1] + '
          'قفزة صناعية ترفع JUMP_RATIO كما يقتضي تعريف Bipower Variation + أسماء صحيحة')


_test_jump_ratio_feature()


def _test_supertrend_rsi_divergence_features():
    """اختبار ذاتي لـSuperTrend وRSI_DIVERGENCE: (أ) مُعطَّلتان افتراضياً،
    (ب) SUPERT_DIR يطابق SUPERTd الخام من pandas_ta تماماً (لا استنساخ منطق
    خاطئ)، (ج) على اتجاه صاعد صناعي واضح SUPERT_DIR=1 وSTRETCH>0 (السعر فوق
    الخط)، (د) الاختبار الأقوى: سيناريو تباعد هابط صناعي صريح (سعر قمّة
    أعلى + RSI قمّة أدنى في نفس الفترة، تعريف وايلدر بالضبط) يُنتج
    RSI_DIVERGENCE موجباً واضحاً، والعكس (تباعد صاعد) يُنتج قيمة سالبة."""
    try:
        import pandas_ta_classic as ta  # noqa: F401
    except ImportError:
        import pandas_ta as ta  # noqa: F401
    import numpy as np, pandas as pd
    n = 100
    idx = pd.date_range('2024-01-01', periods=n, freq='D')
    rng = np.random.default_rng(5)

    # (ب)+(ج) اتجاه صاعد صناعي واضح: سعر يرتفع باطّراد بلا انعكاسات كبيرة
    close_up = 100 + np.cumsum(np.full(n, 0.8) + rng.normal(0, 0.05, n))
    high_up = close_up + rng.random(n) * 0.3
    low_up = close_up - rng.random(n) * 0.3
    open_up = close_up + rng.normal(0, 0.1, n)
    volume = rng.random(n) * 100 + 10
    df_up = pd.DataFrame({'open': open_up, 'high': high_up, 'low': low_up,
                          'close': close_up, 'volume': volume}, index=idx)

    out_default = add_custom_features(df_up, config={'custom_settings': {}})
    assert not any(c.startswith(('SUPERT_DIR_', 'SUPERT_STRETCH_', 'RSI_DIVERGENCE_'))
                  for c in out_default.columns), 'مُعطَّلتان افتراضياً يجب ألا تُنتِجا أي عمود (توافق خلفي)'

    settings = {'supertrend_windows': [10]}
    out_up = add_custom_features(df_up, config={'custom_settings': settings})
    direct_st = df_up.copy()
    direct_line = direct_st.ta.supertrend(length=10, multiplier=3.0)
    pd.testing.assert_series_equal(
        out_up['SUPERT_DIR_10'], direct_line['SUPERTd_10_3.0'].fillna(0.0),
        check_names=False)
    assert out_up['SUPERT_DIR_10'].iloc[-1] == 1.0,         'اتجاه صاعد صناعي واضح يجب أن يُعطي SUPERT_DIR=1 (صاعد)'
    assert out_up['SUPERT_STRETCH_10'].iloc[-1] > 0,         'في اتجاه صاعد السعر يجب أن يكون فوق خطّ SuperTrend (STRETCH>0)'

    expected = set(custom_feature_names(settings=settings))
    produced = set(out_up.columns) - set(df_up.columns)
    assert expected.issubset(produced), f'أعمدة SuperTrend المتوقَّعة غائبة: {expected - produced}'

    # (د) سيناريوهات تباعد صناعية صريحة
    def make_divergence_df(bearish: bool):
        # قمّة/قاع 1 عند idx=79 (صعود/هبوط حادّ 20 شمعة)، ثمّ تصحيح جزئي،
        # ثمّ قمّة/قاع 2 عند idx=129 **أكثر تطرّفاً سعرياً** لكن أبطأ زخماً
        # (30 شمعة بدل 20 لنفس مدى الحركة تقريباً) — هذا بالضبط تعريف
        # وايلدر: سعر يصنع قمّة/قاعاً أعلى/أدنى، لكنّ RSI (يقيس سرعة/زخم
        # التغيّر لا مداه) يصنع قمّة/قاعاً أضعف (تباعد).
        n2 = 200
        idx2 = pd.date_range('2024-01-01', periods=n2, freq='D')
        rng2 = np.random.default_rng(6)
        close2 = np.full(n2, 100.0)
        close2[:60] += np.cumsum(rng2.normal(0, 0.05, 60))
        sign = 1.0 if bearish else -1.0
        close2[60:80] = close2[59] + sign * np.linspace(0, 10, 20)
        close2[80:100] = close2[79] - sign * np.linspace(0, 5, 20)
        close2[100:130] = close2[99] + sign * np.linspace(0, 12, 30)
        close2[130:] = close2[129]
        high2 = close2 + rng2.random(n2) * 0.1
        low2 = close2 - rng2.random(n2) * 0.1
        open2 = close2 + rng2.normal(0, 0.05, n2)
        vol2 = rng2.random(n2) * 100 + 10
        return pd.DataFrame({'open': open2, 'high': high2, 'low': low2,
                             'close': close2, 'volume': vol2}, index=idx2)

    div_settings = {'rsi_divergence_windows': [14]}
    bearish_df = make_divergence_df(bearish=True)
    bullish_df = make_divergence_df(bearish=False)
    out_bear = add_custom_features(bearish_df, config={'custom_settings': div_settings})
    out_bull = add_custom_features(bullish_df, config={'custom_settings': div_settings})
    assert out_bear['RSI_DIVERGENCE_14'].iloc[120:130].mean() > 0,         'تباعد هابط صناعي صريح (قمّة سعرية أعلى + RSI أضعف) يجب أن يُعطي RSI_DIVERGENCE موجباً'
    assert out_bull['RSI_DIVERGENCE_14'].iloc[120:130].mean() < 0,         'تباعد صاعد صناعي صريح (قاع سعري أدنى + RSI أضعف) يجب أن يُعطي RSI_DIVERGENCE سالباً'

    expected_div = set(custom_feature_names(settings=div_settings))
    produced_div = set(out_bear.columns) - set(bearish_df.columns)
    assert expected_div.issubset(produced_div), f'أعمدة RSI_DIVERGENCE المتوقَّعة غائبة: {expected_div - produced_div}'

    print('✅ _test_supertrend_rsi_divergence_features: مُعطَّلتان افتراضياً + '
          'SUPERT_DIR مطابق لـpandas_ta + اتجاه/امتداد صحيحان + '
          'سيناريو تباعد هابط/صاعد صناعي يُميَّز بشكل صحيح')


_test_supertrend_rsi_divergence_features()


def _test_psar_feature():
    """اختبار ذاتي لـPSAR_DIR: (أ) مُعطَّلة افتراضياً، (ب) على اتجاه صاعد
    صناعي واضح PSAR_DIR=1 دائماً (النقاط تحت السعر)، وعلى اتجاه هابط
    صناعي واضح PSAR_DIR=-1 دائماً، (ج) الاتجاهان متعاكسان كما يقتضي
    تعريف Parabolic SAR (لا قيمة وسيطة أو NaN غير معالَجة)."""
    import numpy as np, pandas as pd
    n = 80
    idx = pd.date_range('2024-01-01', periods=n, freq='D')
    rng = np.random.default_rng(7)

    def trending_df(sign):
        close = 100 + np.cumsum(np.full(n, sign * 0.8) + rng.normal(0, 0.05, n))
        high = close + rng.random(n) * 0.3
        low = close - rng.random(n) * 0.3
        open_ = close + rng.normal(0, 0.1, n)
        volume = rng.random(n) * 100 + 10
        return pd.DataFrame({'open': open_, 'high': high, 'low': low,
                             'close': close, 'volume': volume}, index=idx)

    out_default = add_custom_features(trending_df(1), config={'custom_settings': {}})
    assert 'PSAR_DIR' not in out_default.columns, 'psar_enabled=False يجب ألا ينتج أي عمود (توافق خلفي)'

    settings = {'psar_enabled': True}
    out_up = add_custom_features(trending_df(1), config={'custom_settings': settings})
    out_down = add_custom_features(trending_df(-1), config={'custom_settings': settings})
    assert out_up['PSAR_DIR'].iloc[-10:].eq(1.0).all(),         'اتجاه صاعد صناعي واضح يجب أن يُعطي PSAR_DIR=1 باستمرار قرب نهاية السلسلة'
    assert out_down['PSAR_DIR'].iloc[-10:].eq(-1.0).all(),         'اتجاه هابط صناعي واضح يجب أن يُعطي PSAR_DIR=-1 باستمرار قرب نهاية السلسلة'

    expected = set(custom_feature_names(settings=settings))
    produced = set(out_up.columns) - set(trending_df(1).columns)
    assert expected.issubset(produced), f'أعمدة PSAR المتوقَّعة غائبة: {expected - produced}'

    print('✅ _test_psar_feature: مُعطَّلة افتراضياً + اتجاه صاعد/هابط صناعي '
          'يُميَّز بشكل صحيح ومتعاكس + أسماء صحيحة')


_test_psar_feature()


def _test_trend_age_feature():
    """اختبار ذاتي لعمر الاتجاه: (أ) مُعطَّلة افتراضياً، (ب) قمّة صناعية
    عند موضع معروف يجب أن تُصفِّر BARS_SINCE_HIGH بالضبط عند لحظة التأكيد
    السببي (نفس إزاحة _fractal_columns)، ثمّ يتزايد خطياً 1،2،3... بعدها
    حتى القمّة التالية، (ج) سببية: تقطيع السلسلة لا يُغيّر القيمة عند أي
    نقطة سابقة."""
    import numpy as np, pandas as pd
    n = 80
    idx = pd.date_range('2024-01-01', periods=n, freq='D')
    rng = np.random.default_rng(9)
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))
    high = close + rng.random(n) * 0.3
    low = close - rng.random(n) * 0.3
    high[30] = high[25:36].max() + 5.0  # قمّة صناعية واضحة عند 30 (window=5 → تأكيد عند 32)
    open_ = close + rng.normal(0, 0.1, n)
    volume = rng.random(n) * 100 + 10
    df = pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close,
                       'volume': volume}, index=idx)

    out_default = add_custom_features(df, config={'custom_settings': {}})
    assert not any(c.startswith('BARS_SINCE_') for c in out_default.columns), \
        'trend_age_windows=[] يجب ألا ينتج أي عمود (توافق خلفي)'

    settings = {'trend_age_windows': [5]}
    out = add_custom_features(df, config={'custom_settings': settings})
    assert out['BARS_SINCE_HIGH_5'].iloc[32] == 0.0, 'يجب أن يُصفَّر عند لحظة تأكيد القمّة (30+half=32)'
    assert out['BARS_SINCE_HIGH_5'].iloc[33] == 1.0
    assert out['BARS_SINCE_HIGH_5'].iloc[35] == 3.0

    for k in (40, 60, n):
        out_k = add_custom_features(df.iloc[:k], config={'custom_settings': settings})
        assert out_k['BARS_SINCE_HIGH_5'].iloc[-1] == out['BARS_SINCE_HIGH_5'].iloc[k - 1], \
            f'تسرّب معلومة مستقبلية عند k={k}'

    expected = set(custom_feature_names(settings=settings))
    produced = set(out.columns) - set(df.columns)
    assert expected.issubset(produced), f'أعمدة متوقَّعة غائبة: {expected - produced}'

    print('✅ _test_trend_age_feature: مُعطَّلة افتراضياً + تصفير صحيح عند '
          'لحظة التأكيد + تزايد خطّي صحيح + سببية مُتحقَّقة')


_test_trend_age_feature()


def _test_direction_consistency_feature():
    """اختبار ذاتي لاتساق الاتجاه: (أ) مُعطَّلة افتراضياً، (ب) على اتجاه
    صاعد صناعي بلا انعكاسات تقترب القيمة من 1.0 (اتساق شبه تام)، وعلى
    سلسلة متذبذبة صناعية (تعاكس الإشارة كل يوم بالضبط) تكون القيمة 0.0
    بالضبط، (ج) على ضجيج عشوائي بحت تقترب من 0.5 (لا اتساق ولا تذبذب
    منتظم) — يتحقّق من تطابق التعريف الإحصائي المباشر، لا سلوك نوعي فقط."""
    import numpy as np, pandas as pd
    n = 100
    idx = pd.date_range('2024-01-01', periods=n, freq='D')
    rng = np.random.default_rng(10)

    def make_df(close):
        high = close + 0.1
        low = close - 0.1
        open_ = close
        volume = rng.random(n) * 100 + 10
        return pd.DataFrame({'open': open_, 'high': high, 'low': low,
                             'close': close, 'volume': volume}, index=idx)

    settings = {'direction_consistency_windows': [20]}

    close_trend = 100 + np.cumsum(np.full(n, 1.0))  # اتجاه صاعد تامّ بلا انعكاس
    out_trend = add_custom_features(make_df(close_trend), config={'custom_settings': settings})
    assert out_trend['DIR_CONSISTENCY_20'].iloc[-1] > 0.95, 'اتجاه تامّ يجب أن يُعطي اتساقاً قريباً من 1.0'

    alt = np.empty(n); alt[::2] = 1.0; alt[1::2] = -1.0  # يعاكس كل يوم بالضبط
    close_oscillating = 100 + np.cumsum(alt)
    out_osc = add_custom_features(make_df(close_oscillating), config={'custom_settings': settings})
    assert out_osc['DIR_CONSISTENCY_20'].iloc[-1] == 0.0, 'تذبذب تامّ يجب أن يُعطي اتساقاً صفرياً بالضبط'

    close_random = 100 + np.cumsum(rng.normal(0, 1, n))
    out_rand = add_custom_features(make_df(close_random), config={'custom_settings': settings})
    assert 0.3 < out_rand['DIR_CONSISTENCY_20'].iloc[-1] < 0.7, 'ضجيج عشوائي يجب أن يقترب من 0.5'

    out_default = add_custom_features(make_df(close_trend), config={'custom_settings': {}})
    assert not any(c.startswith('DIR_CONSISTENCY_') for c in out_default.columns), \
        'direction_consistency_windows=[] يجب ألا ينتج أي عمود (توافق خلفي)'

    expected = set(custom_feature_names(settings=settings))
    produced = set(out_trend.columns) - set(make_df(close_trend).columns)
    assert expected.issubset(produced), f'أعمدة متوقَّعة غائبة: {expected - produced}'

    print('✅ _test_direction_consistency_feature: مُعطَّلة افتراضياً + '
          'اتجاه تامّ→~1.0 + تذبذب تامّ→0.0 بالضبط + عشوائي→~0.5 + أسماء صحيحة')


_test_direction_consistency_feature()


def _test_vol_asymmetry_feature():
    """اختبار ذاتي لعدم تناظر التقلّب: (أ) مُعطَّلة افتراضياً، (ب) محصورة
    [-1,1] دائماً، (ج) سلسلة بحركات هابطة كبيرة وصاعدة صغيرة فقط تُعطي قيمة
    موجبة قريبة من 1 (تقلّب هابط مهيمن)، والعكس تُعطي قيمة سالبة قريبة من
    -1، (د) عوائد متناظرة تماماً (نفس المقدار، إشارة متعاكسة بالتناوب)
    تُعطي 0.0 بالضبط — تحقّق مباشر من تعريف شبه-التباين، لا سلوك نوعي فقط."""
    import numpy as np, pandas as pd
    n = 60
    idx = pd.date_range('2024-01-01', periods=n, freq='D')
    rng = np.random.default_rng(11)

    def make_df_from_close(close):
        high = close + np.abs(rng.normal(0, 0.05, n))
        low = close - np.abs(rng.normal(0, 0.05, n))
        open_ = close
        volume = rng.random(n) * 100 + 10
        return pd.DataFrame({'open': open_, 'high': high, 'low': low,
                             'close': close, 'volume': volume}, index=idx)

    def make_df(rets):
        close = 100 * np.cumprod(1 + rets / 100.0)
        return make_df_from_close(close)

    settings = {'vol_asymmetry_windows': [20]}

    # هابط مهيمن: حركات هابطة كبيرة (-3%)، صاعدة صغيرة (+0.3%)
    rets_down = np.where(np.arange(n) % 2 == 0, -3.0, 0.3)
    out_down = add_custom_features(make_df(rets_down), config={'custom_settings': settings})
    assert out_down['VOL_ASYMMETRY_20'].iloc[-1] > 0.8, 'تقلّب هابط مهيمن يجب أن يُعطي قيمة موجبة قريبة من 1'

    # صاعد مهيمن: العكس
    rets_up = np.where(np.arange(n) % 2 == 0, 3.0, -0.3)
    out_up = add_custom_features(make_df(rets_up), config={'custom_settings': settings})
    assert out_up['VOL_ASYMMETRY_20'].iloc[-1] < -0.8, 'تقلّب صاعد مهيمن يجب أن يُعطي قيمة سالبة قريبة من -1'

    # متناظر تماماً: عوائد لوغاريتمية متعاكسة بالضبط بالتناوب (لا نسبة
    # حسابية بسيطة — log(1+x) وlog(1-x) غير متماثلين رياضياً حول الصفر)
    log_rets_sym = np.where(np.arange(n) % 2 == 0, 0.02, -0.02)
    close_sym = 100 * np.exp(np.cumsum(log_rets_sym))
    out_sym = add_custom_features(make_df_from_close(close_sym), config={'custom_settings': settings})
    assert abs(out_sym['VOL_ASYMMETRY_20'].iloc[-1]) < 1e-9, 'تناظر تامّ يجب أن يُعطي 0.0 بالضبط'

    assert out_down['VOL_ASYMMETRY_20'].between(-1.0, 1.0).all(), 'يجب أن تكون محصورة [-1,1] دائماً'

    out_default = add_custom_features(make_df(rets_down), config={'custom_settings': {}})
    assert not any(c.startswith('VOL_ASYMMETRY_') for c in out_default.columns), \
        'vol_asymmetry_windows=[] يجب ألا ينتج أي عمود (توافق خلفي)'

    expected = set(custom_feature_names(settings=settings))
    produced = set(out_down.columns) - set(make_df(rets_down).columns)
    assert expected.issubset(produced), f'أعمدة متوقَّعة غائبة: {expected - produced}'

    print('✅ _test_vol_asymmetry_feature: مُعطَّلة افتراضياً + محصورة [-1,1] '
          '+ هابط/صاعد مهيمن يُعطي ±1 تقريباً + تناظر تامّ→0.0 بالضبط')


_test_vol_asymmetry_feature()


def _test_crash_rebound_regime_feature():
    """اختبار ذاتي لنظام ارتداد ما بعد الانهيار: (أ) مُعطَّلة افتراضياً
    (None)، (ب) سلسلة بها هبوط صناعي حادّ (٥٠٪) عن قمّة واضحة يجب أن تُفعِّل
    العلم (=1.0) خلال نافذة recent_window التالية، (ج) سلسلة مستقرّة بلا أي
    هبوط كبير يجب أن يبقى العلم صفراً طوال الوقت، (د) سببية: تقطيع السلسلة
    لا يُغيّر القيمة عند أي نقطة سابقة."""
    import numpy as np, pandas as pd
    n = 150
    idx = pd.date_range('2024-01-01', periods=n, freq='D')
    rng = np.random.default_rng(12)

    def make_df(close):
        high = close + rng.random(n) * 0.1
        low = close - rng.random(n) * 0.1
        open_ = close
        volume = rng.random(n) * 100 + 10
        return pd.DataFrame({'open': open_, 'high': high, 'low': low,
                             'close': close, 'volume': volume}, index=idx)

    settings = {'crash_rebound_regime': {'high_lookback': 60, 'recent_window': 30,
                                         'drawdown_threshold': -0.30}}

    out_default = add_custom_features(make_df(100 + np.cumsum(rng.normal(0, 0.3, n))),
                                      config={'custom_settings': {}})
    assert 'CRASH_REBOUND_REGIME' not in out_default.columns, \
        'crash_rebound_regime=None يجب ألا ينتج أي عمود (توافق خلفي)'

    # هبوط حادّ صناعي: صعود مستقرّ إلى قمّة عند اليوم 60، ثمّ هبوط 50% حادّ
    close_crash = np.concatenate([
        100 + np.arange(60) * 0.5,                              # صعود إلى ~130
        np.linspace(130, 65, 20),                                # هبوط حادّ (-50%)
        65 + np.cumsum(rng.normal(0, 0.2, n - 80)),              # استقرار/ارتداد بعدها
    ])
    out_crash = add_custom_features(make_df(close_crash), config={'custom_settings': settings})
    # خلال الثلاثين يوماً التالية للقاع مباشرة يجب أن يكون العلم مفعَّلاً
    assert out_crash['CRASH_REBOUND_REGIME'].iloc[85:100].mean() == 1.0, \
        'يجب أن يُفعَّل العلم بعد هبوط ≥30% مباشرة'
    # قبل الهبوط بوقت طويل (اليوم 30) يجب أن يكون العلم صفراً
    assert out_crash['CRASH_REBOUND_REGIME'].iloc[30] == 0.0, \
        'يجب أن يبقى العلم صفراً قبل حدوث أي هبوط'

    # سلسلة مستقرّة بلا أي هبوط كبير (تذبذب ±2% فقط حول الاتجاه)
    close_stable = 100 + np.cumsum(rng.normal(0, 0.3, n))
    out_stable = add_custom_features(make_df(close_stable), config={'custom_settings': settings})
    assert (out_stable['CRASH_REBOUND_REGIME'] == 0.0).all(), \
        'سلسلة مستقرّة بلا هبوط كبير يجب ألا تُفعِّل العلم إطلاقاً'

    for k in (90, 120, n):
        out_k = add_custom_features(pd.DataFrame(make_df(close_crash)).iloc[:k],
                                    config={'custom_settings': settings})
        assert out_k['CRASH_REBOUND_REGIME'].iloc[-1] == out_crash['CRASH_REBOUND_REGIME'].iloc[k - 1], \
            f'تسرّب معلومة مستقبلية عند k={k}'

    expected = set(custom_feature_names(settings=settings))
    produced = set(out_crash.columns) - set(make_df(close_crash).columns)
    assert expected.issubset(produced), f'أعمدة متوقَّعة غائبة: {expected - produced}'

    print('✅ _test_crash_rebound_regime_feature: مُعطَّلة افتراضياً + '
          'تفعيل صحيح بعد هبوط ≥30% + صفر عند سلسلة مستقرّة + سببية مُتحقَّقة')


_test_crash_rebound_regime_feature()


def _test_risk_adj_momentum_feature():
    """اختبار ذاتي للزخم المُعدَّل بالمخاطرة: (أ) مُعطَّلة افتراضياً، (ب)
    عائد ثابت تماماً (بلا ضجيج) يُعطي قيمة كبيرة جداً (تباين شبه صفري
    بالمقام)، (ج) نفس متوسط العائد لكن ضجيج أكبر يُعطي قيمة أصغر (نفس
    الاتجاه، جودة أقلّ)، (د) عائد بمتوسط صفر بالضبط (يتناوب +x/-x) يُعطي
    قيمة صفرية تقريباً — تحقّق مباشر من تعريف شارب، لا سلوك نوعي فقط."""
    import numpy as np, pandas as pd
    n = 80
    idx = pd.date_range('2024-01-01', periods=n, freq='D')
    rng = np.random.default_rng(14)

    def make_df(rets):
        close = 100 * np.cumprod(1 + rets / 100.0)
        high = close + np.abs(rng.normal(0, 0.05, n))
        low = close - np.abs(rng.normal(0, 0.05, n))
        open_ = close
        volume = rng.random(n) * 100 + 10
        return pd.DataFrame({'open': open_, 'high': high, 'low': low,
                             'close': close, 'volume': volume}, index=idx)

    settings = {'risk_adj_momentum_windows': [20]}

    out_default = add_custom_features(make_df(np.full(n, 0.5)), config={'custom_settings': {}})
    assert not any(c.startswith('RISK_ADJ_MOM_') for c in out_default.columns), \
        'risk_adj_momentum_windows=[] يجب ألا ينتج أي عمود (توافق خلفي)'

    # عائد شبه ثابت (ضجيج ضئيل جداً): جودة زخم عالية جداً
    rets_clean = np.full(n, 0.5) + rng.normal(0, 1e-4, n)
    out_clean = add_custom_features(make_df(rets_clean), config={'custom_settings': settings})
    val_clean = out_clean['RISK_ADJ_MOM_20'].iloc[-1]
    assert val_clean > 100, f'عائد شبه ثابت يجب أن يُعطي قيمة كبيرة جداً، حصلت على {val_clean}'

    # نفس المتوسط بالضبط (0.5)، تذبذب أكبر حتماً (لا عشوائي، لتجنّب أن
    # يُغيّر الضجيج العشوائي متوسط النافذة الفعلي بالصدفة): تناوب حتمي
    # +2.5/-1.5 حول نفس المتوسط 0.5 — جودة أقلّ (قيمة أصغر لكن نفس الإشارة)
    rets_noisy = np.where(np.arange(n) % 2 == 0, 2.5, -1.5)
    assert np.isclose(rets_noisy.mean(), 0.5), 'يجب أن يتطابق المتوسط تماماً مع الحالة النظيفة'
    out_noisy = add_custom_features(make_df(rets_noisy), config={'custom_settings': settings})
    val_noisy = out_noisy['RISK_ADJ_MOM_20'].iloc[-1]
    assert 0 < val_noisy < val_clean, 'ضجيج أكبر بنفس المتوسط يجب أن يُعطي قيمة أصغر (إيجابية) من الحالة النظيفة'

    # متوسط صفر بالضبط: يتناوب +x/-x بدقّة
    rets_zero_mean = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    out_zero = add_custom_features(make_df(rets_zero_mean), config={'custom_settings': settings})
    assert abs(out_zero['RISK_ADJ_MOM_20'].iloc[-1]) < 0.05, 'متوسط عائد صفري بالضبط يجب أن يُعطي قيمة قريبة من صفر'

    expected = set(custom_feature_names(settings=settings))
    produced = set(out_clean.columns) - set(make_df(rets_clean).columns)
    assert expected.issubset(produced), f'أعمدة متوقَّعة غائبة: {expected - produced}'

    print('✅ _test_risk_adj_momentum_feature: مُعطَّلة افتراضياً + عائد شبه '
          'ثابت→قيمة كبيرة + ضجيج أكبر→قيمة أصغر (نفس الاتجاه) + متوسط '
          'صفري→~0 + أسماء صحيحة')


_test_risk_adj_momentum_feature()


def _test_ath_distance_feature():
    """اختبار ذاتي للمسافة عن القمّة التاريخية: (أ) مُعطَّلة افتراضياً، (ب)
    سلسلة صاعدة تماماً (قمّة جديدة كل يوم) تُعطي PCT_FROM_ATH=0.0 بالضبط
    وDAYS_SINCE_ATH=0.0 دائماً، (ج) قمّة عند موضع معروف ثمّ هبوط يُعطي
    PCT_FROM_ATH سالباً متزايداً وDAYS_SINCE_ATH يتزايد خطّياً 1،2،3...، (د)
    سببية: تقطيع السلسلة لا يُغيّر القيمة عند أي نقطة سابقة."""
    import numpy as np, pandas as pd
    n = 60
    idx = pd.date_range('2024-01-01', periods=n, freq='D')
    rng = np.random.default_rng(15)

    def make_df(close):
        high = close + 0.1
        low = close - 0.1
        open_ = close
        volume = rng.random(n) * 100 + 10
        return pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close,
                             'volume': volume}, index=idx)

    out_default = add_custom_features(make_df(100 + np.arange(n, dtype='float64')),
                                      config={'custom_settings': {}})
    assert 'PCT_FROM_ATH' not in out_default.columns and 'DAYS_SINCE_ATH' not in out_default.columns, \
        'ath_distance_enabled=False يجب ألا ينتج أي عمود (توافق خلفي)'

    settings = {'ath_distance_enabled': True}

    close_uptrend = 100 + np.arange(n, dtype='float64')  # قمّة جديدة كل يوم بالضبط
    out_up = add_custom_features(make_df(close_uptrend), config={'custom_settings': settings})
    assert (out_up['PCT_FROM_ATH'] == 0.0).all(), 'صعود متواصل يجب أن يُبقي PCT_FROM_ATH=0.0 دائماً'
    assert (out_up['DAYS_SINCE_ATH'] == 0.0).all(), 'صعود متواصل يجب أن يُبقي DAYS_SINCE_ATH=0.0 دائماً'

    close_peak = np.concatenate([100 + np.arange(30, dtype='float64'), 128 - np.arange(30, dtype='float64')])
    out_peak = add_custom_features(make_df(close_peak), config={'custom_settings': settings})
    assert out_peak['PCT_FROM_ATH'].iloc[29] == 0.0, 'يوم القمّة نفسه يجب أن يُعطي مسافة صفرية'
    assert out_peak['PCT_FROM_ATH'].iloc[35] < out_peak['PCT_FROM_ATH'].iloc[30] < 0.0, \
        'المسافة يجب أن تزداد سلبيةً كلّما ابتعدنا عن القمّة'
    assert list(out_peak['DAYS_SINCE_ATH'].iloc[29:35]) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], \
        'يجب أن يتزايد خطّياً 0،1،2... بعد القمّة'

    for k in (40, 50, n):
        out_k = add_custom_features(pd.DataFrame(make_df(close_peak)).iloc[:k],
                                    config={'custom_settings': settings})
        assert np.isclose(out_k['PCT_FROM_ATH'].iloc[-1], out_peak['PCT_FROM_ATH'].iloc[k - 1]), \
            f'تسرّب معلومة مستقبلية (PCT_FROM_ATH) عند k={k}'
        assert out_k['DAYS_SINCE_ATH'].iloc[-1] == out_peak['DAYS_SINCE_ATH'].iloc[k - 1], \
            f'تسرّب معلومة مستقبلية (DAYS_SINCE_ATH) عند k={k}'

    expected = set(custom_feature_names(settings=settings))
    produced = set(out_peak.columns) - set(make_df(close_peak).columns)
    assert expected.issubset(produced), f'أعمدة متوقَّعة غائبة: {expected - produced}'

    print('✅ _test_ath_distance_feature: مُعطَّلة افتراضياً + صعود متواصل→0.0 دائماً '
          '+ تزايد صحيح بعد القمّة + سببية مُتحقَّقة')


_test_ath_distance_feature()


def _test_vol_term_structure_feature():
    """اختبار ذاتي للبنية الزمنية للتقلّب: (أ) مُعطَّلة افتراضياً، (ب)
    سلسلة تقلّبها يتسارع فعلياً (نطاق يومي ثابت طويلاً ثمّ يتّسع فجأة قرب
    النهاية) يجب أن تُعطي VOL_TERM > 1 (قصير المدى > طويل المدى)، (ج) العكس
    (نطاق واسع طويلاً ثمّ يضيق) يُعطي VOL_TERM < 1، (د) سببية: تقطيع السلسلة
    لا يُغيّر القيمة عند نقطة سابقة."""
    import numpy as np, pandas as pd
    n = 100
    idx = pd.date_range('2024-01-01', periods=n, freq='D')
    rng = np.random.default_rng(17)

    def make_df(ranges):
        close = 100 + np.cumsum(rng.normal(0, 0.1, n))
        high = close + ranges / 2
        low = close - ranges / 2
        open_ = close
        volume = rng.random(n) * 100 + 10
        return pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close,
                             'volume': volume}, index=idx)

    out_default = add_custom_features(make_df(np.full(n, 1.0)), config={'custom_settings': {}})
    assert not any(c.startswith('VOL_TERM_') for c in out_default.columns), \
        'vol_term_structure_pairs=[] يجب ألا ينتج أي عمود (توافق خلفي)'

    settings = {'vol_term_structure_pairs': [(5, 20)]}

    # تقلّب متسارع: نطاق ثابت 0.5 ثمّ يتّسع فجأة إلى 5.0 قرب النهاية
    ranges_accel = np.concatenate([np.full(n - 10, 0.5), np.full(10, 5.0)])
    df_accel = make_df(ranges_accel)
    out_accel = add_custom_features(df_accel, config={'custom_settings': settings})
    assert out_accel['VOL_TERM_5_20'].iloc[-1] > 1.5, 'تقلّب متسارع يجب أن يُعطي VOL_TERM > 1 بوضوح'

    # تقلّب متباطئ: نطاق واسع 5.0 ثمّ يضيق فجأة إلى 0.5 قرب النهاية
    ranges_decel = np.concatenate([np.full(n - 10, 5.0), np.full(10, 0.5)])
    out_decel = add_custom_features(make_df(ranges_decel), config={'custom_settings': settings})
    assert out_decel['VOL_TERM_5_20'].iloc[-1] < 0.7, 'تقلّب متباطئ يجب أن يُعطي VOL_TERM < 1 بوضوح'

    # سببية: تُستخدَم نفس df_accel (لا استدعاء make_df جديد يُغيّر السلسلة
    # العشوائية) لضمان أن k:70 فعلاً بادئة من نفس الأصل، لا سلسلة أخرى.
    for k in (70, 90, n):
        out_k = add_custom_features(df_accel.iloc[:k].copy(), config={'custom_settings': settings})
        assert np.isclose(out_k['VOL_TERM_5_20'].iloc[-1], out_accel['VOL_TERM_5_20'].iloc[k - 1]), \
            f'تسرّب معلومة مستقبلية عند k={k}'

    expected = set(custom_feature_names(settings=settings))
    produced = set(out_accel.columns) - set(df_accel.columns)
    assert expected.issubset(produced), f'أعمدة متوقَّعة غائبة: {expected - produced}'

    print('✅ _test_vol_term_structure_feature: مُعطَّلة افتراضياً + '
          'تقلّب متسارع→>1 + تقلّب متباطئ→<1 + سببية مُتحقَّقة')


_test_vol_term_structure_feature()
