"""
PURPOSE:  Window normalisation driven by a semantic kind per feature (FEATURE_KINDS prefixes): process_windows, price_norm_mode ('window_scale' | 'pct_change'), pct_change encode/decode, audit_normalization.
TAGS:     normalization, normalize, FEATURE_KINDS, classify_feature, process_windows, price_norm_mode, pct_change, clip_abs, audit_normalization, التطبيع
PITFALLS: Classification is by column-name prefix (longest prefix wins): a new feature with an unknown prefix falls back to DEFAULT_KIND silently — register its kind in FEATURE_KINDS. Executed into the one shared pipeline namespace by data/_loader.py (never imported on its own): names from other modules resolve at call time.

## 9) التطبيع (`normalize.py` سابقاً)

تطبيع كل ميزة حسب تصنيفها الدلالي (مستوى سعري، مذبذب محصور، تراكمي...) بدل
تطبيع موحّد يخلط بين ما يجب تمركزه وما لا يجوز.
"""
# @title
"""
التطبيع الموحّد للنوافذ — مبني على **تصنيف دلالي** لكل ميزة.

المبدأ: لكل ميزة *وحدة قياس* و*مرجع صفر*، والتطبيع الصحيح يعتمد عليهما:

============= ================================================ ==========================
النوع          الدلالة                                          التحويل
============= ================================================ ==========================
price_level   مستوى سعري (close, EMA, نطاقات بولنجر)            ``(x - c) / s``
price_scale   كمية بوحدة السعر لكن ليست مستوى (ATR, MACD, STDEV) ``x / s``  ← بلا تمركز
percent       نسبة مئوية (ROC, PPO, NATR)                       ``x / 10``
osc_0_100     مذبذب محصور [0, 100] (RSI, STOCH, ADX)            ``(x - 50) / 50``
osc_sym_100   مذبذب متماثل [-100, 100] (CMO, AROONOSC)          ``x / 100``
osc_willr     مذبذب [-100, 0] (WILLR)                           ``(x + 50) / 50``
unit_sym      محصور [-1, 1] (BOP, CMF)                          ``x``
unit_0_1      محصور [0, 1] (ER, BBP)                            ``(x - 0.5) * 2``
zscore        درجة معيارية أصلاً (ZS)                            ``x / 3``
sign_robust   مذبذب غير محصور حول الصفر (TSI, TRIX, CCI, FISHER) ``x / (1.4826·med|x|)``
cumulative    تراكمي بمرجع اعتباطي (OBV, AD, KVO, EFI)          تقييس داخل النافذة
log_volume    الحجم                                             ``log1p`` ثم تقييس
log_centered  لوغاريتم مستواه يختلف بين الأصول (ITD_TRADES_LOG)  ``x - وسيط النافذة``
binary_flag   علم توفّر 0/1 (``*_available``)                   ``x * 2 - 1`` (لا يُصفَّر حين يثبت)
============= ================================================ ==========================

**لماذا هذا التمييز ضروري** — ثلاثة أخطاء كانت قائمة وقيست فعلياً:

1. ``ATRr_14`` كان مُصنَّفاً ``price_level``، فيُطرح منه *سعر الإغلاق*. على
   بيتكوين بسعر 61,000 و ATR=350 كانت النتيجة **-60.7** بينما ``close`` نفسه
   ينتج ``±1`` — أي أن عموداً واحداً أكبر بـ 61× ويبتلع الطبقة الأولى.
2. ``volume`` كان ``log1p`` فقط بلا تمركز، فمتوسطه **13.0** لا 0.
3. أعمدة مداها ``[-1, 1]`` (``CMF``, ``BOP``, ``BBP``, ``UI``, ``ZS``) كانت
   تُقسَم على 100، فصار انحرافها المعياري ~0.002 — **معلومة معدومة عملياً**.

كذلك المذبذبات حول الصفر (``MACD``, ``TSI``, ``TRIX``) كانت تُقيَّس داخل النافذة
بطرح وسيطها، فتضيع **إشارتها** — و«MACD فوق الصفر» معلومة حقيقية لا يجوز محوها.

**وضع تطبيع الأعمدة السعرية** (``CONFIG['price_norm_mode']``، يخصّ ``price_level`` وحده):

* ``'window_scale'`` (الافتراضي، السلوك السابق حرفياً) — ``(x - c) / s`` بمركز/مقياس نافذة الإغلاق، مقصوص ±``clip_abs``.
* ``'pct_change'`` — تغيّر نسبي **هندسي** عن الشمعة السابقة لنفس العمود بالنقاط المئوية:
  ``100 × (x_t / x_{t-1} − 1)`` (مثلاً ``-1``، ``+5``، ``-0.4``). أول صف في النافذة ``0`` (لا سابق له داخلها).
  مقصوص ±``price_pct_clip`` (لا ``clip_abs``: حركة +7% حقيقية لا تُقصّ عند 5). **قابل للفكّ** بالضرب التراكمي من مرساة
  واحدة: :func:`pct_change_decode` (وفي مجموعة البيانات :func:`decode_price_window` بمرساة ``last_candles``).
"""

# ══════════════════════════════════════════════════════════════════════════
# أنواع الميزات
# ══════════════════════════════════════════════════════════════════════════
PRICE_LEVEL = 'price_level'
PRICE_SCALE = 'price_scale'
PERCENT = 'percent'
OSC_0_100 = 'osc_0_100'
OSC_SYM_100 = 'osc_sym_100'
OSC_WILLR = 'osc_willr'
UNIT_SYM = 'unit_sym'
UNIT_0_1 = 'unit_0_1'
ZSCORE = 'zscore'
SIGN_ROBUST = 'sign_robust'
CUMULATIVE = 'cumulative'
LOG_VOLUME = 'log_volume'
#: علم توفّر 0/1 (FUND_available/OI_available) → -1/+1. ثباته داخل النافذة هو المعلومة نفسها
#: («لا بيانات» مقابل «بيانات موجودة»)، فلا يُصفَّر حين يثبت.
BINARY_FLAG = 'binary_flag'
#: معدّل التمويل الخام (0.0001 = 0.01% لكل 8 ساعات) → ×1000: مستوى مطلق (0.01% → 0.1، 0.05% → 0.5)،
#: لا نسبي للنافذة — تمويل ثابت عند 0.05% أسابيع معلومة مختلفة عن ثابت عند 0.01%.
FUNDING_RATE = 'funding_rate'
#: أنواع مقياسها ثابت ومعناها في قيمتها المطلقة — لا تُصفَّر حين تثبت داخل نافذتها.
NEVER_ZEROED_KINDS = frozenset({BINARY_FLAG, FUNDING_RATE})
#: لوغاريتم كمية مستواها يختلف بين الأصول لكن وحدته معلومة (log عدد الصفقات اليومي) → ``x − وسيط النافذة``
#: بلا قسمة. **لماذا لا CUMULATIVE؟** السلسلة يومية داخل نافذة ساعات، أي دالّة درجية بقيمتين أو ثلاث: IQR النافذة
#: إمّا الدرجة نفسها (فيخرج 1 لكل حجم درجة) أو ~0 (فتُقصّ عند ±5) — يضيع حجم التغيّر وهو المعلومة. الفرق
#: اللوغاريتمي نفسه خالٍ من المقياس (0.1 = +10.5% صفقات) فلا يحتاج مقياساً من النافذة.
#: الدليل: docs/research/audit/r2_04_step_feature_saturation.py.
LOG_CENTERED = 'log_centered'

#: النوع الافتراضي لأي عمود غير مُصنَّف — تقييس داخل النافذة (آمن دائماً).
DEFAULT_KIND = CUMULATIVE

#: أقصى قيمة مطلقة بعد التطبيع. القصّ يمنع انفجار التدرّجات من عيّنة شاذة
#: واحدة، ويُبقي 99%+ من القيم بلا مساس (انظر ``audit_normalization``).
CLIP_ABS = 5.0

#: أوضاع تطبيع أعمدة ``price_level`` (CONFIG['price_norm_mode']) — انظر رأس الخلية.
PRICE_NORM_WINDOW = 'window_scale'
PRICE_NORM_PCT = 'pct_change'
PRICE_NORM_MODES = (PRICE_NORM_WINDOW, PRICE_NORM_PCT)
#: قصّ أعمدة 'pct_change' بالنقاط المئوية (CONFIG['price_pct_clip']): ±100 لا يمسّ حركة شمعة حقيقية فيبقى الفكّ دقيقاً.
PRICE_PCT_CLIP = 100.0

#: تصنيف الميزات ببادئة الاسم. الترتيب مهم: تُفحص البادئات الأطول أولاً
#: تلقائياً في :func:`classify_feature`، فلا يلتقط ``BB`` ما هو لـ ``BBP``.
FEATURE_KINDS: Dict[str, str] = {
    # ── مستويات سعرية ────────────────────────────────────────────────────
    **{p: PRICE_LEVEL for p in (
        'open', 'high', 'low', 'close', 'hl2', 'HLC3', 'OHLC4', 'WCP',
        'SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 'TRIMA', 'HMA', 'RMA', 'VWMA',
        'VWAP', 'ALMA', 'T3_', 'ZL_EMA', 'ZLMA', 'KAMA', 'FWMA', 'SINWMA',
        'SSF', 'SWMA', 'PWMA', 'LINREG', 'MIDPOINT', 'MIDPRICE',
        'BBL', 'BBM', 'BBU', 'DCL', 'DCM', 'DCU', 'KCL', 'KCB', 'KCU',
        'SUPERT', 'PSAR', 'ICHIMOKU', 'ISA_', 'ISB_', 'ITS_', 'IKS_',
    )},
    # ── كميات بوحدة السعر لكنها ليست مستوى (لا تُمركَز أبداً) ─────────────
    **{p: PRICE_SCALE for p in (
        'ATR', 'ATRr', 'TRUERANGE', 'TR_', 'STDEV', 'VAR_', 'MAD_',
        'MACD', 'MOM', 'SLOPE', 'MIDP', 'DPO', 'ABER',
    )},
    # ── نِسَب مئوية ───────────────────────────────────────────────────────
    **{p: PERCENT for p in (
        'ROC', 'PPO', 'NATR', 'BBB', 'CHOP', 'MASSI', 'PVR',
    )},
    # ── مذبذبات [0, 100] ─────────────────────────────────────────────────
    **{p: OSC_0_100 for p in (
        'RSI', 'STOCHk', 'STOCHd', 'STOCHh', 'STOCHRSIk', 'STOCHRSId',
        'ADX', 'ADXR', 'DMP', 'DMN', 'AROOND', 'AROONU', 'MFI', 'RVI',
        'UO_', 'CRSI', 'PSL', 'RVGI', 'KST',
    )},
    # ── مذبذبات [-100, 100] ──────────────────────────────────────────────
    **{p: OSC_SYM_100 for p in ('CMO', 'AROONOSC', 'CFO', 'CTI')},
    # ── مذبذب [-100, 0] ──────────────────────────────────────────────────
    **{p: OSC_WILLR for p in ('WILLR',)},
    # ── محصورة [-1, 1] ───────────────────────────────────────────────────
    **{p: UNIT_SYM for p in ('BOP', 'CMF', 'QS_', 'INC_', 'DEC_')},
    # ── محصورة [0, 1] ────────────────────────────────────────────────────
    **{p: UNIT_0_1 for p in ('ER_', 'BBP', 'SQZ_ON', 'SQZ_OFF', 'SQZ_NO',
                             'UI_')},
    # ── درجات معيارية ────────────────────────────────────────────────────
    **{p: ZSCORE for p in ('ZS_', 'SKEW', 'KURT')},
    # ── مذبذبات غير محصورة حول الصفر (الإشارة معلومة) ────────────────────
    **{p: SIGN_ROBUST for p in (
        'CCI', 'TSI', 'TRIX', 'FISHERT', 'LOGRET', 'PCTRET', 'PVO', 'PVI',
        'NVI', 'BIAS', 'COPC', 'EOM', 'SMI', 'STC', 'QQE', 'TD_SEQ',
    )},
    # ── تراكمية بمرجع اعتباطي ────────────────────────────────────────────
    # ENTP مداها ~log(length) (≈3.3 لطول 10) لا [0,1] — تقييس داخل النافذة
    **{p: CUMULATIVE for p in ('OBV', 'AD', 'ADOSC', 'KVO', 'EFI', 'PVT',
                               'CMF_SUM', 'VP_', 'ENTP')},
    # ── الحجم ────────────────────────────────────────────────────────────
    **{p: LOG_VOLUME for p in ('volume', 'VOL_')},

    # ══ الميزات المخصّصة (custom.py) ══════════════════════════════════════
    # كلها ساكنة وخالية من المقياس بالبناء، فتصنيفها مباشر لا اجتهاد فيه.
    **{p: SIGN_ROBUST for p in ('RET_', 'VOLR_')},   # عوائد ونِسَب لوغاريتمية
    # ── سياق سوقي عابر للأصول (add_market_context) ─────────────────────────
    **{p: SIGN_ROBUST for p in ('MKT_ret_',)},       # عوائد العملة المرجعية
    **{p: PERCENT for p in ('RANGE_',)},             # مدى نسبي بالنسبة المئوية
    **{p: UNIT_SYM for p in ('BODY_', 'TIME_')},     # [-1, 1] أصلاً
    **{p: UNIT_0_1 for p in ('WICK_', 'POS_')},      # [0, 1] أصلاً
    **{p: ZSCORE for p in ('VOLZ_',)},               # درجة معيارية أصلاً

    # ══ ميزات اختيارية أُضيفت لاحقاً — تصنيف صريح إلزامي ═══════════════════
    # ⚠️ التصنيف بالبادئة (أطول بادئة تفوز) كان يلتقط هذه الأسماء خطأً بصمت:
    #   VWAP_DEVIATION → 'VWAP' (مستوى سعري): (x - مركز_السعر)/مقياس_السعر ≈
    #     -سعر/IQR، يُقصّ عند -5 لأغلب العيّنات — القيمة المستخرَجة لم تكن
    #     الانحراف عن VWAP إطلاقاً بل نسبة سعر/تقلّب (قيس فعلياً: وسيط -5.0).
    #   SUPERT_DIR/SUPERT_STRETCH/PSAR_DIR → 'SUPERT'/'PSAR' (مستوى سعري): نفس
    #     الخلل لعلَم اتجاه ±1 ولنسبة مئوية.
    #   MOM_RANK → 'MOM' (وحدة سعرية): رتبة [0,1] مقسومة على IQR السعر
    #     المطلق — الترتيب عبر الأصول يصير فعلياً حسب مستوى سعر العملة.
    #   VOL_ASYMMETRY/VOL_TERM/VOL_CONC → 'VOL_' (حجم لوغاريتمي): log1p بعد
    #     قصّ السالب إلى صفر — VOL_ASYMMETRY (محصورة [-1,1]) تفقد نصفها السالب.
    # كل هذه الميزات مُعطَّلة افتراضياً، فهذا التصحيح لا يُغيّر أي سلوك
    # افتراضي — فقط ما يراه النموذج/محور التقييم حين تُفعَّل. التصحيح مُقفَل
    # باختبار ذاتي (t_optional_feature_kinds_explicit).
    **{p: SIGN_ROBUST for p in ('VWAP_DEVIATION', 'MKT_LAG_RET_')},  # عائد-شبيه، كـRET_/MKT_ret_
    **{p: UNIT_SYM for p in ('SUPERT_DIR_', 'PSAR_DIR', 'VOL_ASYMMETRY_',
                             'MKT_CORR_', 'PCT_FROM_ATH', 'MOM_ORTH_NATR')},            # محصورة [-1, 1] أصلاً
    **{p: PERCENT for p in ('SUPERT_STRETCH_',)},                       # مسافة بالنسبة المئوية
    **{p: UNIT_0_1 for p in ('MOM_RANK_', 'MKT_BREADTH_', 'DIR_CONSISTENCY_',
                             'EFF_RATIO_', 'VOL_CONC_', 'CRASH_REBOUND_REGIME',
                             'MKT_CRASH_REBOUND_REGIME')},             # [0, 1] أصلاً
    **{p: ZSCORE for p in ('VOL_TERM_', 'RISK_ADJ_MOM_', 'MKT_BETA_')},  # نِسَب حول 0/1، قيمها المعتادة داخل ±3
    # ✅ ميزات التمويل/الفائدة المفتوحة — كانت غير مُصنَّفة فتسقط في CUMULATIVE (تقييس داخل النافذة):
    # علم التوفّر ثابت داخل كل نافذة تقريباً فيُصفَّر دائماً (النموذج لا يرى «لا بيانات» أبداً)،
    # وFUND_rate_z يُقيَّس مرة ثانية، وFUND_rate يفقد مستواه المطلق.
    **{p: BINARY_FLAG for p in ('FUND_available', 'OI_available')},
    **{p: FUNDING_RATE for p in ('FUND_rate',)},     # FUND_rate_z أطول بادئة ← ZSCORE أدناه
    **{p: ZSCORE for p in ('FUND_rate_z',)},
    **{p: SIGN_ROBUST for p in ('OI_chg_',)},        # تغيّر لوغاريتمي كالعوائد
    # ── المرحلة ٢ (tools/intraday_features.py) — تصنيف صريح لكل اسم ──
    **{p: FUNDING_RATE for p in ('FUND_sum_',)},     # مجموع أحداث التمويل (1d/3d): مستوى مطلق كـFUND_rate
    **{p: BINARY_FLAG for p in ('ITD_available', 'MET_available')},
    **{p: UNIT_0_1 for p in ('ITD_TAKER_BUY_RATIO', 'ITD_VOL_TOPK')},   # حصص [0, 1]
    **{p: LOG_CENTERED for p in ('ITD_TRADES_LOG',)},  # log عدد الصفقات (يومي): مستواه يختلف بين الأصول ← − وسيط النافذة
    **{p: ZSCORE for p in ('ITD_TRADES_Z',)},        # درجة معيارية مقابل تاريخ العملة نفسها
    **{p: PERCENT for p in ('ITD_RVOL',)},           # تقلّب محقَّق يومي بالنسبة المئوية (كـNATR)
    **{p: UNIT_SYM for p in ('LSR_', 'TAKER_LSR_')}, # log نِسَب long/short: حول 0، مستواها معلومة
    **{p: SIGN_ROBUST for p in ('LSR_GLOBAL_chg_',)},
}

#: البادئات مرتّبة تنازلياً بالطول — تضمن أن ``BBP`` تسبق ``BB``، و``ATRr`` تسبق ``ATR``.
_SORTED_PREFIXES: List[str] = sorted(FEATURE_KINDS, key=len, reverse=True)


#: لاحقة أعلام التوفّر: كل ``*_available`` علم 0/1 ← BINARY_FLAG مهما كانت بادئته. كانت EFF_RATIO_/VWAP_DEVIATION_/
#: VOL_CONC_available تُلتقط ببادئات UNIT_0_1، فيُصفَّر العلم الثابت داخل نافذته (أغلب النوافذ) ويتساوى «مغطّى» و«غير
#: مغطّى» عند 0 — العلم ميّت. الدليل: docs/research/audit/r2_03_dead_availability_flags.py.
FLAG_SUFFIX = '_available'


def classify_feature(col_name: str,
                     kinds: Optional[Dict[str, str]] = None,
                     default: str = DEFAULT_KIND) -> str:
    """نوع الميزة الدلالي حسب بادئة اسمها (أطول بادئة مطابِقة تفوز)؛ أعلام ``*_available`` ← BINARY_FLAG دائماً
    في الجدول الافتراضي."""
    if kinds is None and col_name.endswith(FLAG_SUFFIX):
        return BINARY_FLAG
    table = FEATURE_KINDS if kinds is None else kinds
    prefixes = _SORTED_PREFIXES if kinds is None else sorted(table, key=len, reverse=True)
    for p in prefixes:
        if col_name.startswith(p):
            return table[p]
    return default


#: اسم متوافق مع الكود القديم.
classify_column = classify_feature


#: حدّ أدنى مطلق للمقياس — يمنع القسمة على صفر حرفي فقط.
SCALE_ABS_FLOOR = 1e-8
#: حدّ أدنى **نسبي** لمستوى السعر (0.1%). أصل بسعر مرتفع في نافذة شبه مسطّحة
#: يكون IQR لها صغيراً بالمطلق دون أن يلمس ``SCALE_ABS_FLOOR`` فعلياً، فينفجر
#: أي هدف/ميزة يُطبَّع بها إلى مئات الأضعاف عند أي حركة سعرية لاحقة — قيس فعلياً:
#: ``y_high_reg`` بمدى ``[-41, +728]`` بدل ``[-3, +3]`` المتوقَّع لمقياس سليم.
SCALE_REL_FLOOR = 1e-3


# ══════════════════════════════════════════════════════════════════════════
# أدوات القياس
# ══════════════════════════════════════════════════════════════════════════
def calc_scale_params(data: np.ndarray, method: str = "robust") -> Tuple[float, float]:
    """``(المركز، المقياس)`` — وسيط/IQR لـ ``robust``، أو min/range لـ ``minmax``.

    المقياس محدود بحدّين معاً: مطلق (``SCALE_ABS_FLOOR``) لمنع القسمة على
    صفر حرفي، ونسبي لمستوى السعر (``SCALE_REL_FLOOR``) لمنع انفجار الهدف/الميزة
    حين يكون IQR صغيراً بالمطلق نسبةً لسعر مرتفع — انظر التعليق أعلاه.
    """
    if method == 'robust':
        q75, q50, q25 = np.percentile(data, [75, 50, 25])
        rel_floor = abs(q50) * SCALE_REL_FLOOR
        return q50, max(q75 - q25, rel_floor, SCALE_ABS_FLOOR)
    lo, hi = data.min(), data.max()
    rel_floor = abs(lo) * SCALE_REL_FLOOR
    return lo, max(hi - lo, rel_floor, SCALE_ABS_FLOOR)


def scale_data(data: np.ndarray, center: float, scale: float) -> np.ndarray:
    return (data - center) / scale


def inverse_scale(preds: np.ndarray, bases: np.ndarray) -> np.ndarray:
    """عكس التطبيع السعري: ``preds * iqr + median`` لكل عيّنة."""
    if preds.ndim == 2:
        return preds * bases[:, 1:2] + bases[:, 0:1]
    return preds * bases[:, 1] + bases[:, 0]


def _robust_magnitude(x: np.ndarray, eps: float) -> float:
    """مقياس حجم حول الصفر يحفظ الإشارة (لا يُمركِز)."""
    mag = 1.4826 * np.median(np.abs(x))
    if mag < eps:                      # نافذة شبه ثابتة عند الصفر
        mag = max(np.std(x), eps)
    return float(mag)


def _standardize(x: np.ndarray, method: str, eps: float) -> np.ndarray:
    """تقييس داخل النافذة — للأعمدة التي مرجعها اعتباطي."""
    if method == "minmax":
        lo, hi = np.nanmin(x), np.nanmax(x)
        return (x - lo) / (hi - lo + eps)
    c, s = calc_scale_params(x, method)
    return (x - c) / s


# ══════════════════════════════════════════════════════════════════════════
# التغيّر النسبي الهندسي (price_norm_mode='pct_change') — ترميز وفكّ
# ══════════════════════════════════════════════════════════════════════════
def price_norm_mode(config: Optional[dict] = None) -> str:
    """وضع تطبيع أعمدة ``price_level`` من ``config['price_norm_mode']``؛ قيمة غير معروفة ترفع ``ValueError``."""
    config = CONFIG if config is None else config
    mode = config.get('price_norm_mode', PRICE_NORM_WINDOW)
    if mode not in PRICE_NORM_MODES:
        raise ValueError(f"price_norm_mode: {' | '.join(map(repr, PRICE_NORM_MODES))} — لا {mode!r}")
    return mode


def price_pct_clip(config: Optional[dict] = None) -> float:
    """قصّ أعمدة 'pct_change' بالنقاط المئوية (``config['price_pct_clip']``)؛ يجب أن يكون موجباً."""
    config = CONFIG if config is None else config
    clip = float(config.get('price_pct_clip', PRICE_PCT_CLIP))
    if not clip > 0:
        raise ValueError(f"price_pct_clip يجب أن يكون > 0، لا {clip!r}")
    return clip


def pct_change_encode(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """تغيّر نسبي هندسي بالنقاط المئوية عن القيمة السابقة على المحور ``axis``: ``100 × (x_t / x_{t-1} − 1)``.

    سعر 100 ← 99 ← 103.95 ← 103.53 يعطي ``[0, -1, +5, -0.4]``: أول عنصر ``0`` (لا سابق له)، وأي تغيّر غير منتهٍ أو
    سابق غير موجب (سعر صفري/سالب لا معنى لنسبته) ``0``. يُحسب بـ float64 (نسبة سعرين متقاربين تفقد دقّتها في float32).
    العكس: :func:`pct_change_decode`.
    """
    x = np.asarray(x, dtype='float64')
    x = np.moveaxis(x, axis, -1)
    out = np.zeros_like(x)
    prev, cur = x[..., :-1], x[..., 1:]
    with np.errstate(divide='ignore', invalid='ignore'):
        r = 100.0 * (cur / prev - 1.0)
    out[..., 1:] = np.where(np.isfinite(r) & (prev > 0), r, 0.0)
    return np.moveaxis(out, -1, axis)


def pct_change_decode(r: np.ndarray, anchor, anchor_at: str = 'last', axis: int = 0) -> np.ndarray:
    """فكّ :func:`pct_change_encode`: يُعيد المستويات من التغيّرات المئوية ``r`` ومرساة واحدة بالضرب التراكمي.

    Args:
        r: تغيّرات مئوية (``r[0]`` على المحور مُهمَل — هو تغيّر ما قبل النافذة).
        anchor: القيمة الحقيقية عند طرف النافذة — شكلها شكل ``r`` بلا محور الزمن (أو قابل للبثّ إليه).
        anchor_at: ``'last'`` (افتراضياً: المرساة آخر قيمة، كـ ``last_close`` في ``last_candles``) فيُفكّ للخلف
            ``x_{t-1} = x_t / (1 + r_t/100)``؛ أو ``'first'`` فيُفكّ للأمام ``x_t = x_{t-1} × (1 + r_t/100)``.

    الفكّ دقيق ما دامت ``r`` غير مقصوصة (``|r| < price_pct_clip``) ولم تُخزَّن بدقّة float16.
    """
    if anchor_at not in ('last', 'first'):
        raise ValueError(f"anchor_at: 'last' | 'first' — لا {anchor_at!r}")
    g = 1.0 + np.moveaxis(np.asarray(r, dtype='float64'), axis, -1) / 100.0
    a = np.asarray(anchor, dtype='float64')[..., None]
    ones = np.ones(g.shape[:-1] + (1,))
    if anchor_at == 'last':
        # S_t = ∏_{k=t+1}^{T-1} g_k (و S_{T-1} = 1) ← x_t = المرساة / S_t
        tail = np.cumprod(g[..., :0:-1], axis=-1)[..., ::-1]
        x = a / np.concatenate([tail, ones], axis=-1)
    else:
        x = a * np.concatenate([ones, np.cumprod(g[..., 1:], axis=-1)], axis=-1)
    return np.moveaxis(x, -1, axis)


def _pct_change_columns(raw: np.ndarray, axis: int) -> np.ndarray:
    """:func:`pct_change_encode` بعد استبدال غير المنتهي بوسيط نافذته (نفس معاملة بقية الأعمدة في التطبيع)."""
    x = np.asarray(raw, dtype='float64')
    finite = np.isfinite(x)
    if not finite.all():
        med = np.nanmedian(np.where(finite, x, np.nan), axis=axis, keepdims=True)
        x = np.where(finite, x, np.nan_to_num(med, nan=0.0))
    return pct_change_encode(x, axis=axis)


# ══════════════════════════════════════════════════════════════════════════
# التطبيع
# ══════════════════════════════════════════════════════════════════════════
def normalize_column(x: np.ndarray, kind: str, center: float, scale: float,
                     method: str = "robust", eps: float = 1e-8,
                     price_mode: str = PRICE_NORM_WINDOW) -> np.ndarray:
    """يُطبّع عموداً واحداً حسب نوعه الدلالي.

    Args:
        center / scale: مركز ومقياس نافذة **الإغلاق** — يُستخدمان للأعمدة
            السعرية فقط، فتبقى كلها على مرجع مشترك واحد.
        price_mode: وضع أعمدة ``price_level`` (:data:`PRICE_NORM_MODES`)؛ ``'pct_change'`` يتجاهل center/scale.
    """
    if kind == PRICE_LEVEL:
        if price_mode == PRICE_NORM_PCT:
            return pct_change_encode(x)
        if price_mode != PRICE_NORM_WINDOW:
            raise ValueError(f"price_mode: {' | '.join(map(repr, PRICE_NORM_MODES))} — لا {price_mode!r}")
        return (x - center) / scale
    if kind == PRICE_SCALE:
        # ✅ قسمة بلا تمركز: ATR=350 على سعر 61,000 يعطي 0.35 لا -60.7
        return x / scale
    if kind == PERCENT:
        return x / 10.0
    if kind == OSC_0_100:
        return (x - 50.0) / 50.0
    if kind == OSC_SYM_100:
        return x / 100.0
    if kind == OSC_WILLR:
        return (x + 50.0) / 50.0
    if kind == UNIT_SYM:
        return x
    if kind == UNIT_0_1:
        return (x - 0.5) * 2.0
    if kind == ZSCORE:
        return x / 3.0
    if kind == SIGN_ROBUST:
        # ✅ يحفظ الإشارة: 'MACD فوق الصفر' معلومة حقيقية لا تُمحى
        return x / _robust_magnitude(x, eps)
    if kind == BINARY_FLAG:
        return x * 2.0 - 1.0
    if kind == FUNDING_RATE:
        return x * 1000.0
    if kind == LOG_VOLUME:
        return _standardize(np.log1p(np.clip(x, 0, None)), method, eps)
    if kind == LOG_CENTERED:
        return x - np.median(x)
    # CUMULATIVE والافتراضي
    return _standardize(x, method, eps)


def process_window(w: np.ndarray, columns: list, med_p: float, iqr_p: float,
                   method: str = "robust", config: Optional[dict] = None,
                   clip_abs: Optional[float] = None) -> np.ndarray:
    """تطبيع نافذة ``(T, F)`` عموداً عموداً حسب النوع الدلالي لكل عمود."""
    config = CONFIG if config is None else config
    if not isinstance(columns, (list, tuple)):
        columns = list(columns)
    eps = config.get("eps", 1e-8)
    clip_abs = config.get("clip_abs", CLIP_ABS) if clip_abs is None else clip_abs
    price_mode = price_norm_mode(config)

    out = np.asarray(w, dtype='float64').copy()
    for j in range(out.shape[1]):
        x = out[:, j]
        finite = np.isfinite(x)
        if not finite.all():
            x = np.where(finite, x, np.nanmedian(x[finite]) if finite.any() else 0.0)

        lo, hi = np.nanmin(x), np.nanmax(x)
        kind = classify_feature(columns[j])
        if not np.isfinite(lo) or not np.isfinite(hi) or (hi == lo and kind not in NEVER_ZEROED_KINDS):
            out[:, j] = 0.0
            continue

        out[:, j] = normalize_column(x, kind, med_p, iqr_p, method, eps, price_mode)

    out = np.nan_to_num(out, nan=0.0, posinf=clip_abs, neginf=-clip_abs)
    # 'pct_change': أعمدة price_level بنقاط مئوية تُقصّ بـ price_pct_clip لا clip_abs (تُحفظ قبل القصّ العام وتُعاد بعده)
    pct_idx = ([j for j in range(out.shape[1]) if classify_feature(columns[j]) == PRICE_LEVEL]
               if price_mode == PRICE_NORM_PCT else [])
    pct_vals = out[:, pct_idx].copy() if pct_idx else None
    if clip_abs:
        np.clip(out, -clip_abs, clip_abs, out=out)
    if pct_idx:
        pc = price_pct_clip(config)
        out[:, pct_idx] = np.clip(pct_vals, -pc, pc)
    return out.astype('float32')


def process_windows(windows: np.ndarray, columns: List[str],
                    centers: np.ndarray, scales: np.ndarray,
                    method: str = "robust", config: Optional[dict] = None,
                    clip_abs: Optional[float] = None) -> np.ndarray:
    """تطبيع **كل النوافذ دفعةً واحدة** — النسخة المتّجهة من :func:`process_window`.

    :func:`process_window` تعالج نافذة واحدة بحلقة بايثون على الأعمدة، فتصير
    التكلفة ``N × F`` دورة بايثون (≈9,600 دورة لأصل واحد بـ 246 نافذة و39 ميزة).
    هنا نُجمّع الأعمدة حسب نوعها الدلالي **مرة واحدة**، ثم نُطبّق تحويل كل نوع
    على كل النوافذ معاً بعمليات numpy متّجهة. النتيجة مطابقة عددياً.

    ✅ **float32 من البداية للنهاية** (كانت ترفع إلى float64 ثم تُخصّص ``out``
    منفصلة، فتحمل ذروتها ~7.77× حجم المصفوفة النهائية — قِيس فعلياً؛ الحساب
    هنا يكتب في ``X`` نفسها بلا مصفوفة ``out`` موازية، فتنزل الذروة إلى ~2.64×
    (رقم حقيقي مُقاس، لا نظري). الفرق العددي عن النسخة القديمة ≤5e-7 — أصغر
    من دقّة float32 نفسها، لا يُغيّر شيئاً عملياً.

    Args:
        windows: ``(N, T, F)`` نوافذ خام.
        centers / scales: ``(N,)`` مركز ومقياس نافذة الإغلاق لكل عيّنة.
    """
    config = CONFIG if config is None else config
    eps = config.get("eps", 1e-8)
    clip_abs = config.get("clip_abs", CLIP_ABS) if clip_abs is None else clip_abs
    price_mode = price_norm_mode(config)

    raw = windows                       # 'pct_change' يقرأ الأسعار الخام بدقّتها (float64 عادة) لا نسختها float32
    X = np.asarray(windows, dtype='float32')
    n, _, n_cols = X.shape
    c = np.asarray(centers, dtype='float32').reshape(n, 1, 1)
    s = np.asarray(scales, dtype='float32').reshape(n, 1, 1)

    # استبدال غير المنتهي بوسيط العمود داخل نافذته (مطابق للنسخة المفردة)
    finite = np.isfinite(X)
    if not finite.all():
        med = np.nanmedian(np.where(finite, X, np.nan), axis=1, keepdims=True)
        X = np.where(finite, X, np.nan_to_num(med, nan=0.0)).astype('float32', copy=False)

    # الأعمدة الثابتة داخل نافذتها لا تحمل معلومة → تُصفَّر لاحقاً (عدا NEVER_ZEROED_KINDS: ثباتها معلومة)
    lo = X.min(axis=1)
    hi = X.max(axis=1)
    keep_const = np.array([classify_feature(nm) in NEVER_ZEROED_KINDS for nm in columns[:n_cols]])
    degenerate = ((hi == lo) & ~keep_const[None, :]) | ~np.isfinite(lo) | ~np.isfinite(hi)   # (N, F)

    # تجميع الأعمدة حسب النوع مرة واحدة بدل تصنيفها لكل نافذة
    by_kind: Dict[str, List[int]] = {}
    for j, name in enumerate(columns[:n_cols]):
        by_kind.setdefault(classify_feature(name), []).append(j)

    # ✅ نكتب في X نفسها بدل تخصيص out منفصلة: مجموعات by_kind أعمدتها
    # متفرّقة لا تتداخل أبداً (كل عمود ينتمي لنوع واحد فقط)، فالكتابة في
    # X[:, :, idx] لا تُفسد قراءة مجموعة أخرى لاحقة — هذا ما يوفّر مصفوفة
    # float32 كاملة إضافية بالمقارنة مع النسخة القديمة (out = np.empty_like).
    for kind, idx in by_kind.items():
        sub = X[:, :, idx]

        if kind == PRICE_LEVEL and price_mode == PRICE_NORM_PCT:
            res = _pct_change_columns(np.asarray(raw)[:, :, idx], axis=1)
        elif kind == PRICE_LEVEL:
            res = (sub - c) / s
        elif kind == PRICE_SCALE:
            res = sub / s
        elif kind == PERCENT:
            res = sub / 10.0
        elif kind == OSC_0_100:
            res = (sub - 50.0) / 50.0
        elif kind == OSC_SYM_100:
            res = sub / 100.0
        elif kind == OSC_WILLR:
            res = (sub + 50.0) / 50.0
        elif kind == UNIT_SYM:
            res = sub
        elif kind == UNIT_0_1:
            res = (sub - 0.5) * 2.0
        elif kind == ZSCORE:
            res = sub / 3.0
        elif kind == SIGN_ROBUST:
            mag = 1.4826 * np.median(np.abs(sub), axis=1, keepdims=True)
            fallback = np.std(sub, axis=1, keepdims=True)
            mag = np.where(mag < eps, np.maximum(fallback, eps), mag)
            res = sub / mag
        elif kind == BINARY_FLAG:
            res = sub * 2.0 - 1.0
        elif kind == FUNDING_RATE:
            res = sub * 1000.0
        elif kind == LOG_VOLUME:
            res = _standardize_batch(np.log1p(np.clip(sub, 0, None)), method, eps)
        elif kind == LOG_CENTERED:
            res = sub - np.median(sub, axis=1, keepdims=True)
        else:                                   # CUMULATIVE والافتراضي
            res = _standardize_batch(sub, method, eps)

        X[:, :, idx] = res

    X[np.broadcast_to(degenerate[:, None, :], X.shape)] = 0.0
    X = np.nan_to_num(X, nan=0.0, posinf=clip_abs, neginf=-clip_abs, copy=False)
    # 'pct_change': أعمدة price_level تُقصّ بـ price_pct_clip لا clip_abs (كما في process_window)
    pct_idx = by_kind.get(PRICE_LEVEL, []) if price_mode == PRICE_NORM_PCT else []
    pct_vals = X[:, :, pct_idx].copy() if pct_idx else None
    if clip_abs:
        np.clip(X, -clip_abs, clip_abs, out=X)
    if pct_idx:
        pc = price_pct_clip(config)
        X[:, :, pct_idx] = np.clip(pct_vals, -pc, pc)
    return X


def _standardize_batch(sub: np.ndarray, method: str, eps: float) -> np.ndarray:
    """تقييس متّجه داخل كل نافذة على حدة (المحور الزمني)."""
    if method == "minmax":
        lo = sub.min(axis=1, keepdims=True)
        hi = sub.max(axis=1, keepdims=True)
        return (sub - lo) / (hi - lo + eps)
    q25, q50, q75 = np.percentile(sub, [25, 50, 75], axis=1, keepdims=True)
    return (sub - q50) / np.maximum(q75 - q25, 1e-8)


# ══════════════════════════════════════════════════════════════════════════
# التدقيق
# ══════════════════════════════════════════════════════════════════════════
def describe_features(features: List[str]) -> Dict[str, List[str]]:
    """يُجمّع الميزات حسب نوعها الدلالي — لمراجعة التصنيف قبل التدريب."""
    groups: Dict[str, List[str]] = {}
    for f in features:
        groups.setdefault(classify_feature(f), []).append(f)
    return dict(sorted(groups.items()))


def audit_normalization(windows: np.ndarray, features: List[str],
                        verbose: bool = True, config: Optional[dict] = None) -> pd.DataFrame:
    """يقيس التوزيع الفعلي لكل عمود بعد التطبيع ويكشف الأعمدة المُعطَّلة.

    Args:
        windows: مصفوفة ``(N, T, F)`` **بعد** :func:`process_window`.
        config: يُقرأ منه ``price_norm_mode`` — أعمدة price_level بوضع 'pct_change' نقاط مئوية حدّها
            ``price_pct_clip`` لا ``CLIP_ABS``.

    يُرجع جدولاً بأعمدة: النوع، المتوسط، الانحراف، أقصى قيمة مطلقة، وحكم.
    """
    config = CONFIG if config is None else config
    pct_limit = price_pct_clip(config) if price_norm_mode(config) == PRICE_NORM_PCT else None
    rows = []
    for j, name in enumerate(features):
        kind = classify_feature(name)
        is_pct = pct_limit is not None and kind == PRICE_LEVEL     # تشتّت النقاط المئوية معلومة لا خلل
        limit = pct_limit if is_pct else CLIP_ABS
        # float16 يُرفع قبل أي حساب: numpy يُجمّع float16 بدقة float16 فيفيض مجموع المربّعات (> 65504) وتصير std = inf
        v = np.asarray(windows[:, :, j]).ravel().astype(np.float32, copy=False)
        v = v[np.isfinite(v)]
        if not v.size:
            continue
        std = float(v.std())
        absmax = float(np.abs(v).max())
        if std < 0.01:
            verdict = "🚨 بلا معلومة (شبه ثابت)"
        elif absmax > limit * 1.5:
            verdict = "🚨 متطرف"
        elif std > 3.0 and not is_pct:
            verdict = "⚠️ تشتت عالٍ"
        else:
            verdict = "✅"
        rows.append({
            'feature': name, 'kind': kind,
            'mean': float(v.mean()), 'std': std,
            'p01': float(np.percentile(v, 1)), 'p99': float(np.percentile(v, 99)),
            'absmax': absmax, 'verdict': verdict,
        })

    table = pd.DataFrame(rows)
    if verbose and len(table):
        bad = table[~table.verdict.str.startswith("✅")]
        print(f"🔬 تدقيق التطبيع: {len(table)} ميزة | "
              f"سليمة: {len(table) - len(bad)} | تحتاج مراجعة: {len(bad)}")
        if len(bad):
            print(bad[['feature', 'kind', 'mean', 'std', 'absmax', 'verdict']]
                  .to_string(index=False))
    return table


#: خريطة الأنواع القديمة (رقمية) — للرجوع فقط، لم تعد مستخدَمة.
LEGACY_NORM_TYPES = {0: PRICE_LEVEL, 1: OSC_0_100, 2: OSC_SYM_100,
                     3: CUMULATIVE, 4: LOG_VOLUME}
