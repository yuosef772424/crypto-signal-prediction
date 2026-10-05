"""
PURPOSE:  DEFAULT_CONFIG — the single source of truth for every pipeline/model/training setting — plus COINS_BY_CATEGORY, STATIC_INDICATORS_FULL and DEFAULT_ENABLED_CATEGORIES.
TAGS:     DEFAULT_CONFIG, defaults, settings, config keys, COINS_BY_CATEGORY, coin categories, indicator_settings, phase2_data, module_dirs, split_dates, holdout_start, الإعدادات
PITFALLS: A new key's default must reproduce the old behaviour exactly (checkpoints and recorded results stay valid). Never edit DEFAULT_CONFIG at run time: use update_config() on CONFIG. Executed into the one shared pipeline namespace by data/_loader.py (never imported on its own): names from other modules resolve at call time.

## 4) ⚙️ الإعدادات الافتراضية (`defaults.py` سابقاً)

**المصدر الوحيد للحقيقة** لكل إعدادات النظام — منقول هنا **حرفياً** كما ورفعته
(لا تخمين هذه المرة). لا تُعدّل `DEFAULT_CONFIG` مباشرة أثناء العمل — استخدم
`update_config()` بعد بناء `CONFIG` في القسم التالي.

أضفتُ **مفتاحَين فقط** لم يكونا في الأصل (معلَّمين أدناه بوضوح) لأن الدفتر
يعتمد حصراً على Google Drive **المُركَّب** لا على `gdown`:
`drive_mount_point` و`asset_registry_path`. المفتاحان الأصليان
`asset_registry_file_id`/`preprocessed_data_file_id` (لمسار gdown القديم)
بقيا في القاموس للتوثيق فقط — **غير مُستخدَمين** فعلياً في هذا الدفتر.
"""
"""
الإعدادات الافتراضية الموحّدة لكل النظام.

هذا الملف هو **المصدر الوحيد للحقيقة**: كل مرحلة (تجهيز البيانات، بناء النموذج،
التدريب، الفحص والتحقق) تقرأ إعداداتها من هنا عبر ``CONFIG`` (القسم التالي).
لا تُعدّل هذا القاموس مباشرة أثناء العمل — استخدم ``update_config()`` أو حمّل
ملف JSON عبر ``load_config()``.
"""

DEFAULT_CONFIG: dict = {
    # ══════════════════════════════════════════════════════════════════════
    # 1) الفريمات الزمنية والنوافذ
    # ══════════════════════════════════════════════════════════════════════
    "tf_order": ["1D"],
    "base_tf": "1D",
    "window_sizes": {"1D": 32},
    "stride": 1,
    # ✅ نهايات النوافذ على شبكة زمنية ثابتة (أثرها مع stride > 1 فقط): نهاية كل نافذة
    # تحقّق (ts − window_grid_anchor) % (stride × مدة الفريم) == 0، فتتشارك كل العملات
    # نفس أزمنة النهاية أيّاً كان تاريخ إدراجها. بدونها يعدّ كل أصل من أول شمعة له،
    # فتتوزّع العملات على أطوار مختلفة (قيس على 1h بـstride=32: وسيط العملات في الطابع
    # الواحد 2 من 83)، فينكسر كل حساب مقطعي (لوحة cross_asset، هدف relative،
    # cross_sectional_normalize). مع stride=1 كل شمعة نهايةُ نافذة أصلاً ⇒ لا أثر.
    "align_windows_to_grid": True,
    "window_grid_anchor": "1970-01-01",   # UTC
    "higher_tf_offset": 2,
    # 🕓 الفريمات الأعلى من الفريم الأساسي (4h فوق 1h مثلاً). "legacy" = القصّ بموضع الصف (higher_tf_offset) كما كان
    # حرفياً لكل إعداد قائم. "closed" = نافذة الفريم الأعلى لعيّنة نهايتها t لا تحوي إلا شموعاً *مغلقة* عند t (إغلاق
    # الشمعة = فتحها + مدتها ≤ t)، وكاملة (كل شموع الفريم الأصغر فيها موجودة)، ومتصلة بلا فجوة، ومن أحدث شمعة مغلقة
    # (لا شمعة أقدم بعد فقدان الأحدث)؛ وتُطبَّع أسعارها بمركز/مقياس نافذتها هي لا نافذة الفريم الأساسي (نافذة 4h تمتدّ
    # 128 ساعة، فتنفلت أسعارها من مقياس نافذة 1h ذات 32 ساعة). أثره فقط حين len(tf_order) > 1.
    "higher_tf_mode": "legacy",
    # تخزين X: None = float32 كما هي. "float16" يُنصّف ذاكرة X في البناء والحفظ والتقسيم (القيم مُطبَّعة ومقصوصة ±5،
    # فخطأ التقريب ≤ 2.5e-3)؛ كل مستهلك يرفعها إلى float32 عند تكوين الدفعة (make_shuffled_dataset في main، PanelSplit.assemble).
    "x_storage_dtype": None,
    # 💾 بناء مدعوم بالقرص (رام محدودة على Colab): كل عملة تُكتب فور معالجتها إلى scratch_dir (.npy غير مضغوطة) ثم تُحرَّر من الرام،
    # والدمج على القرص (open_memmap)، والناتج memmap للقراءة. ذروة الرام لا تتبع عدد العملات ولا حجم البيانات. الإعدادات الجاهزة لفريم
    # الساعة (HOURLY_PRESET وHOURLY_W32_S8_OVERRIDES وامتدادها 4h) تفعّله؛ False هنا = السلوك القديم حرفياً (قوائم + np.concatenate).
    "disk_backed": False,
    # مجلد scratch: None → /content/pipeline_scratch (أو tempdir خارج Colab). يُنشأ تحته مجلد باسم بصمة الإعدادات لكل بناء.
    "scratch_dir": None,
    # True: بعد نجاح save_data_to_drive بصيغة المجلد يُحذَف مجلد scratch الذي تسكنه مصفوفات memmap (تبقى قابلة للقراءة في هذه
    # الجلسة على لينكس؛ أعد التحميل من Drive بعد إعادة التشغيل). False (الافتراضي) يُبقيه لإعادة الاستخدام.
    "scratch_cleanup_after_save": False,
    # صيغة save_data_to_drive: "auto" (مجلد .npy لمجموعة memmap وإلا .pkl.gz) | "pkl.gz" (القديمة؛ تحتاج كل X في الرام) | "npy_dir".
    "save_format": "auto",
    # True: save_data_to_drive يكتب نسخة واحدة فقط ({اسم}_latest، ما يقرؤه main) بدل نسخة مؤرّخة + latest — نصف المساحة ووقت
    # الرفع إلى Drive. False (الافتراضي) = نسختان كالسابق.
    "save_single_copy": False,
    # مستوى ضغط gzip لصيغة pkl.gz (1 أسرع بكثير، 9 أصغر قليلاً). 9 (الافتراضي) = سلوك gzip.open السابق حرفياً. قيس على بيانات
    # 1h_s8 pct (2.7 GB في الرام): المستوى 1 أنتج 544 MB في 76 ثانية.
    "pkl_compresslevel": 9,
    # dtype إطارات high/low/close في المرور الأول للميزات العابرة للأصول (مسار التحميل): "float64" (الافتراضي، مطابق بايتاً) أو
    # "float32" (نصف الذاكرة، يغيّر الميزات العابرة في ~1e-7). ≈ 88 MB لـ 83 عملة 1h بـ float64 — لا يُذكر أمام ذروة البناء.
    "cross_asset_dtype": "float64",

    # ══════════════════════════════════════════════════════════════════════
    # 2) الأهداف
    # ══════════════════════════════════════════════════════════════════════
    "targets": ["high", "low", "close"],
    "forecast_horizon": 1,
    # "direction" → أهداف 1/0 ، "regression" → سعر مُطبَّع.
    # ملاحظة: رؤوس '_class' تُدرَّب دائماً كاتجاه 1/0 (1 = صعود)، ورؤوس '_reg' كسعر مُطبَّع،
    # لذا هذا المفتاح يؤثر فقط على الأدوات القديمة أحادية الرأس.
    "target_mode": "direction",

    # ══════════════════════════════════════════════════════════════════════
    # 3) التحكم في الرؤوس (المخرجات) — مفتاح إطفاء/تشغيل لكل رأس على حدة
    # ══════════════════════════════════════════════════════════════════════
    # لكل هدف سعري رأسان: '_class' (تصنيف اتجاه 1/0) و '_reg' (سعر مُطبَّع).
    # ضع False لأي رأس لا تحتاجه، فيُحذف بالكامل من: تحضير البيانات (لا يُحسب
    # أصلاً)، بناء النموذج (لا تُبنى طبقة Dense له)، الخسارة، التنبؤ، والتقييم.
    "enabled_heads": {
        "high_class": True,
        "high_reg": True,
        "low_class": True,
        "low_reg": True,
        "close_class": True,
        "close_reg": True,
    },

    # أوزان الخسارة لكل هدف سعري (يُطبَّق نفس الوزن على رأسي الهدف).
    # 0.0 يُبقي الرأس موجوداً كمخرج لكنه لا يؤثر في التدريب إطلاقاً.
    "target_loss_weights": {"high": 1.0, "low": 1.0, "close": 1.0},

    # ══════════════════════════════════════════════════════════════════════
    # 4) طبقة الامتناع (Selective Prediction / Learning to Reject)
    # ══════════════════════════════════════════════════════════════════════
    # قابلة للإطفاء الكامل بمفتاح واحد ("enabled": False) دون تغيير أي كود.
    "abstention": {
        "enabled": True,

        # (أ) رأس عدم اليقين داخل النموذج (المسار البديل build_classification_heads):
        # True  → رؤوس '_class' تُبنى بـ MC-dropout وتُخرج epistemic حقيقية.
        # False → رؤوس Dense(tanh) بسيطة، والشكوك تُعامَل كأصفار.
        # ⚠️ تغيير هذه القيمة يُغيّر بنية النموذج ⇒ يتطلب إعادة تدريب.
        # ✅ مُعطَّلة: predict_batch لم يعد يُشغّل MC-Dropout وقت الاستدلال إطلاقاً
        # (عدم اليقين يأتي تحليلياً من نموذج NIG)، فتفعيلها يبني طاقة Dropout
        # بلا أي استفادة فعلية منها.
        "use_uncertainty_head": False,
        "head_hidden_dim": 128,
        "head_dropout": 0.15,

        # (ب) إشارات الثقة المُدمَجة (Meta-Confidence) — كل إشارة قابلة للإطفاء:
        "signals": {
            "margin": True,          # هامش |tanh| بعيداً عن الصفر
            "uncertainty": True,     # epistemic + aleatoric
            "head_agreement": True,  # تطابق اتجاه رأسي class و reg لنفس الهدف
        },

        # (ج) عتبات القرار (تُعاير على val عبر calibrate_abstention_thresholds):
        "min_margin": 0.20,
        "max_uncertainty_ratio": 1.0,
        "require_head_agreement": True,
        "min_coverage": 0.05,

        # (د) عتبة ديناميكية حسب التقلب:
        #     العتبة الفعلية = min_margin * (1 + k*(vol_ratio-1))
        "dynamic_by_volatility": False,
        "volatility_sensitivity": 0.5,

        # (هـ) تكاليف اقتصادية حقيقية (نسبة مئوية من السعر):
        "cost_per_trade_pct": 0.06,
        "slippage_pct": 0.02,
        "opportunity_cost_pct": 0.0,

        # (و) الرأس الذي تُبنى عليه قرارات الصفقات:
        "decision_head": "close_class",
    },

    # ══════════════════════════════════════════════════════════════════════
    # 5) التطبيع والتقسيم
    # ══════════════════════════════════════════════════════════════════════
    "scaler_type": "robust",          # "robust" أو "minmax"
    "scaling_window": 32,
    "eps": 1e-8,

    # وضع التقسيم:
    #   "global_time" (مُوصى به) — حدود زمنية مطلقة مشتركة بين كل العملات.
    #   "per_asset" — السلوك القديم (نسب لكل عملة على حدة). للتوافق فقط.
    "split_mode": "global_time",
    "split_dates": None,
    # holdout مختوم (PROTOCOL.md: يُفتح مرة واحدة عند الإصدار): تاريخ ثابت (مثل "2026-07-15") — العيّنات من بعده خارج
    # train/val/test وكل ملفات الإشارات، بفجوة عزل قبله. None = لا holdout (السلوك السابق). إعداد 1h_s8 يضبطه.
    "holdout_start": None,
    # split_holdout() يرفض ما لم يكن True صراحةً — لا تُفعّله إلا لتقييم الإصدار الأخير.
    "open_holdout": False,
    "train_pct": 0.80,
    "val_pct": 0.10,
    "keep_asset_test_separate": True,
    "embargo_candles": None,          # None → window_size(base_tf) + forecast_horizon

    # أقل عدد عيّنات مقبول في كلٍّ من train و val و test. تحت هذا الحدّ يرفع
    # split_data خطأً صريحاً بدل إرجاع قسم فارغ بصمت (كان val و test يخرجان
    # فارغتين حين يقصر الزمن المتبقي بعد train عن فجوتَي العزل).
    "min_split_samples": 64,
    # 🔧 فريم التحميل الحيّ (Binance) — مستقلّ الآن عن فريم عمل النموذج
    # (tf_order/base_tf): يُجلَب دائماً بهذا الفريم ثم يُجمَّع (resample) صعوداً
    # إلى كل فريم في tf_order. كان قبلاً مُثبَّتاً "1h" بصمت داخل fetch_data
    # بصرف النظر عمّا يُطلَب فعلاً — الآن صريح ومُعدَّل هنا فقط.
    "download_interval": "1h",

    # 🔧 سياق سوقي عابر للأصول: عائد عملة مرجعية (BTC افتراضياً) يُضاف كميزة
    # إدخال (بادئة MKT_) لكل عملة أخرى — تُشتقّ آلياً في feature_order، لا حاجة
    # لتضمينها يدوياً. للعملة المرجعية نفسها القيمة صفر دائماً (لا معنى لعائدها
    # نسبة لنفسها). التوسعة لاحقاً (قيمة سوقية إجمالية، هيمنة BTC...) تحتاج
    # مصدر بيانات آخر غير Binance — أضف مفتاحاً موازياً هنا حين يتوفّر.
    "market_context": {
        "enabled": True,
        "reference_symbol": "BTCUSDT",
        "return_periods": [1],        # MKT_ret_1 مستبعد من المدخلات (فشل) لكنه لازم داخلياً
        "corr_windows": [20, 60],     # MKT_CORR_20/60 (rel/low 74%/77%)
        "beta_windows": [20],         # MKT_BETA_20 (rel/low 93%، 66/68؛ 95% خارج العيّنة)
    },

    # رتبة الزخم المقطعية (Cross-Sectional Momentum Rank): موضع أداء كل
    # أصل نسبةً لكل الأصول الأخرى عند نفس اللحظة بالضبط (0=الأسوأ أداءً،
    # 1=الأفضل) — مختلف عن market_context (متوسط/عائد مرجع واحد فقط).
    # مُعطَّلة افتراضياً (توافق خلفي)؛ مدعومة فقط عبر build_dataset (لا
    # build_dataset_from_loader — تحتاج رؤية كل الأصول معاً دفعة واحدة).
    "momentum_rank": {
        "enabled": False,
        "horizons": [6, 24],
    },

    # اتساع السوق (Market Breadth): نسبة الأصول الصاعدة في نفس اللحظة —
    # قيمة واحدة مشتركة لكل الأصول (خلافاً لـmomentum_rank، رتبة مختلفة لكل
    # أصل). تحتاج رؤية كل الأصول معاً: build_dataset_from_preloaded يراها في data،
    # وbuild_dataset_from_loader يحمّل high/low/close لكل العملات في مرور أول
    # (_load_cross_asset_frames) قبل المعالجة.
    "market_breadth": {
        "enabled": True,              # MKT_BREADTH_24 (abs/close 77%، 74% خارج العيّنة)
        "horizons": [24],
    },

    # ✅ الزخم المتعامد مع التقلّب (MOM_ORTH_NATR) — أول ميزة مقبولة في اختبار
    # خارج العيّنة مسجَّل مسبقاً (38/38 نافذة، 2023-07 → 2026-09). مفعَّلة
    # افتراضياً. تحتاج كل الأصول معاً في نفس اللحظة (كـmomentum_rank)، ويوفّرها
    # المساران كلاهما (build_dataset_from_loader عبر مرور تحميل أول لكل العملات).
    # تصحّ بعدد أصول كافٍ (min_assets لكل لحظة، وعملياً عشرات الأصول).
    "momentum_orth_natr": {
        "enabled": True,
        "horizons": [1, 3, 6, 12, 24],
        "natr_length": 14,
        "min_assets": 5,
    },

    # 🔧 إعادة تدريب دورية (walk-forward) — قيم افتراضية لـ rolling_splits().
    # None يعني "يجب تحديدها صراحة عند الاستدعاء"؛ لا قيمة واحدة صحيحة عامة
    # لكل مشروع. مثال: {"test_span": "30D", "val_span": "15D",
    # "initial_train_span": "180D"} (نافذة تدريب متمدّدة) أو أضف "train_span"
    # لنافذة منزلقة بحجم ثابت. راجع توثيق rolling_split_schedule.
    "rolling_retrain": {
        "test_span": None,
        "val_span": None,
        "train_span": None,
        "initial_train_span": None,
        "step": None,
    },

    # 🔧 معدّل التمويل والفائدة المفتوحة (عقود Binance الآجلة) — تُجلَب وتُخزَّن
    # على Drive بدالتَي fetch_funding_rate/fetch_open_interest_hist و
    # save_funding_open_interest، ثم تُدمَج كميزات فعلية عبر
    # add_funding_oi_features (بروح add_market_context، لكن بأرشيف خاص بكل
    # عملة لا مرجعاً مشتركاً) — "as_feature" يتحكّم بالإدماج بمعزل عن
    # "enabled" (الجلب/الأرشفة)، فيمكن أرشفة بلا إدماج (كان السلوك الوحيد
    # سابقاً) أو العكس نظرياً (إدماج بلا enabled يُعيد أعمدة محايدة دائماً).
    # ⚠️ الفائدة المفتوحة محدودة من Binance نفسها بآخر ٣٠ يوماً فقط مهما
    # طُلب — راجع تحذير fetch_open_interest_hist. معدّل التمويل بلا هذا القيد،
    # لكن الأرشيف المحفوظ فعلياً يبدأ من أول تشغيل لـsave_funding_open_interest؛
    # فترات أقدم منه (شائعة في بيانات تاريخية طويلة) تُملأ بقيمة محايدة مع علم
    # توفّر صريح (FUND_available/OI_available) — راجع add_funding_oi_features.
    "funding_rate": {"enabled": True, "drive_dir": "funding_rate",
                     "as_feature": True, "zscore_window": 90},
    "open_interest": {"enabled": True, "drive_dir": "open_interest", "period": "1h",
                      "as_feature": True, "change_period": 1},

    # 🔧 نسبة كفاءة كوفمان الداخل-يومية (Kaufman's Efficiency Ratio) — أول
    # ميزة في المشروع مصدرها بيانات **ساعية** حقيقية لا OHLCV اليومي وحده
    # (غير متاحة سابقاً). محسوبة خارجياً (خارج هذا الدفتر — نافذة 24 ساعة
    # متدحرجة في مساحة اللوغاريتم، |إزاحة صافية|/إجمالي مسافة المسار) وتُدمَج
    # هنا كأرشيف خارجي جاهز، بنفس روح add_funding_oi_features بالضبط (أرشيف
    # خاص بكل رمز، محاذاة ffill + علم توفّر EFF_RATIO_available). "data" يقبل
    # قاموساً {symbol: DataFrame(date, efficiency_ratio_24h)} جاهزاً في
    # الذاكرة (للاختبار)، أو "archive_dir" مساراً يحوي {symbol}_efficiency.csv.
    "intraday_efficiency": {"enabled": False, "as_feature": False,
                            "archive_dir": None, "data": None},

    # 🔧 انحراف VWAP الداخل-يومي (Volume-Weighted Average Price Deviation) —
    # ثاني ميزة من بيانات ساعية حقيقية (بعد نسبة كفاءة كوفمان). VWAP اليوم
    # (متوسط السعر النموذجي [عالٍ+منخفض+إغلاق]/3 مرجَّحاً بالحجم الساعي) هو
    # "السعر العادل" المؤسسي الفعلي لليوم — الانحراف عنه ((إغلاق-VWAP)/VWAP)
    # يقيس ضغط شراء/بيع فعلياً حدث، لا افتراضاً من مستويات OHLC فقط. محسوب
    # خارجياً (كـintraday_efficiency بالضبط)، بنفس بنية الأرشيف الخارجي.
    "intraday_vwap": {"enabled": False, "as_feature": False,
                      "archive_dir": None, "data": None},

    # 🔧 تركّز الحجم الساعي الداخل-يومي (Intraday Volume Concentration) —
    # ثالث ميزة من بيانات ساعية حقيقية. مؤشّر هيرفندال-هيرشمان (HHI) لتوزّع
    # الحجم على الـ٢٤ ساعة: قيمة قريبة من 1/24 تعني حجماً موزَّعاً بالتساوي
    # (يوم تداول عادي)، وقيمة أعلى تعني تركّزاً حاداً في ساعات قليلة (يوم
    # مدفوع بخبر/حدث) — أدبيات البنية الدقيقة للسوق (Admati-Pfleiderer 1988)
    # تربط تركّز الحجم بأحداث معلوماتية. محسوبة خارجياً كـintraday_efficiency.
    "intraday_volume_concentration": {"enabled": False, "as_feature": False,
                                      "archive_dir": None, "data": None},

    # 🔧 المرحلة ٢ — أرشيف Binance الكامل من tools/fetch_history_vision_colab.py على Drive:
    #   history_15m/<S>.csv.gz (شموع 15m + taker + trades)، funding_rate/، open_interest/،
    #   futures_metrics/ (OI + نسب long/short + taker، ساعياً منذ 2024-01-01).
    # المفتاحان (USE_INTRADAY_15M / USE_FUTURES_METRICS): True/False صراحةً، أو "auto" = True فقط إن
    # وُجد المجلد تحت الجذر (غير فارغ) ووُجدت tools/intraday_features.py — فبلا هذه الملفات يبقى
    # المسار اليومي كما هو حرفياً (نفس feature_order ونفس القيم). يُثبَّت "auto" في بداية
    # build_dataset* (resolve_phase2_toggles). مع USE_INTRADAY_15M تُملأ الفتحات القائمة
    # EFF_RATIO_24H / VWAP_DEVIATION / VOL_CONC_HHI من الشموع الحقيقية + أعمدة ITD_ جديدة؛ ومع
    # USE_FUTURES_METRICS تُحاذى FUND_/OI_ على **إغلاق** الشمعة + FUND_sum_1d/3d + LSR_/TAKER_LSR_1d.
    # الشرح والأسماء: docs/research/phase2_data.md.
    "phase2_data": {
        "use_intraday_15m": "auto",      # USE_INTRADAY_15M
        "use_futures_metrics": "auto",   # USE_FUTURES_METRICS
        "data_root": None,               # None = /content/drive/MyDrive ؛ أو مسار يحوي المجلدات أدناه
        "intraday_dir": "history_15m",
        "intraday_pattern": "{name}.csv.gz",   # .csv يُجرَّب تلقائياً إن لم يوجد
        "metrics_dir": "futures_metrics",      # funding/OI: funding_rate.drive_dir / open_interest.drive_dir
        "min_coverage": 0.75,            # يوم بأقل من 75% من شموع 15m المتوقَّعة = NaN + ITD_available=0
        "trades_z_window": 30,           # ITD_TRADES_Z: درجة معيارية مقابل الأيام السابقة فقط
        "topk_bars": 4,                  # ITD_VOL_TOPK: حصة أعلى 4 شموع 15m من حجم اليوم
        "max_age": "1D",                 # قيمة أقدم من هذا عند إغلاق الشمعة = مفقودة (لا ffill بلا حدّ)
        # أين يُبحث عن tools/intraday_features.py (إضافةً إلى مجلد العمل الحالي):
        "module_dirs": ["/content/crypto-signal-prediction", "/content/drive/MyDrive/crypto"],
    },


    # 🔧 كيف يُشتقّ حدّ (حدّا) التقسيم الزمني في split_mode='global_time':
    #   'kept_share'   (افتراضي، القائم) — يبحث عن الحدّين بحيث تقترب نِسَب
    #                  العيّنات **المُبقاة بعد فجوتَي العزل** من train_pct/val_pct.
    #                  الأدقّ، ويتعامل بنجاح مع بيانات مكدَّسة زمنياً (عملات كثيرة
    #                  حديثة الإدراج) — انظر توثيق resolve_split_dates.
    #   'raw_quantile' — حدّ مباشر: "آخر N من كل العيّنات (أي عملة)" يقارب
    #                  train_pct/val_pct **الخام قبل** خصم فجوة العزل — أبسط،
    #                  لكن الفرق عن 'kept_share' يتناسب مع (عيّنات العزل
    #                  المفقودة ÷ حجم القسم المستهدف)، فقد ينحرف بعشرات
    #                  بالمئة حتى على بيانات منتظمة إن كان val_pct/test_pct
    #                  صغيراً؛ وعلى بيانات مكدَّسة قد يُنتج أقساماً أصغر من
    #                  min_split_samples (يُرفَض بخطأ صريح حينها، لا قسم فارغ
    #                  صامت). راجع compute_global_cutoff و build_leak_free_split.
    "split_cutoff_method": "kept_share",

    # يُستبعد من **مدخلات النموذج** فقط — تبقى الأعمدة في البيانات لحساب
    # الأهداف والمؤشرات.
    #   open / high / low : ترابط 0.99 مع close بعد التطبيع؛ ومعلومتها الحقيقية
    #                       (المدى) تلتقطها RANGE_rel.
    #   الباقي: مخرجات فرعية لمؤشرات مُبقاة فشلت في فرز الميزات
    #   (docs/research/pipeline_feature_selection.md) — مؤشرها الأم يبقى لأن
    #   أحد مخرجاته الأخرى عبر القاعدة (ADX → DMP/DMN، BBANDS → BBB، SUPERTREND
    #   → STRETCH، MKT_ret_1 لازم داخلياً لحساب MKT_CORR/MKT_BETA).
    # ✅ volume أُعيد إلى المدخلات: عبر قاعدة الاختيار (abs/high: 87% معنوية،
    #    65/68 نافذة) بدرجة أعلى من VOLZ_20 نفسه، رغم ترابطهما 0.96.
    "exclude_from_features": ["open", "high", "low",
                              "ADX_14", "BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0", "BBP_20_2.0",
                              "SUPERT_DIR_10", "MKT_ret_1"],

    # تُملأ تلقائياً عبر refresh_features() — لا تكتبها يدوياً.
    "feature_order": None,

    # ══════════════════════════════════════════════════════════════════════
    # 6) المؤشرات الفنية
    # ══════════════════════════════════════════════════════════════════════
    # ✅ مُختارة بفرز تجريبي (docs/research/pipeline_feature_selection.md): كل ميزة قيست
    # منفردة على 50 أصلاً × 68 نافذة شهرية (30 داخل العيّنة + 38 خارجها،
    # 2021-01 → 2026-09) مقابل high/low/close المطلق والسوقي-المحايد. تبقى
    # الميزة إن عبرت معيار القبول على كل النوافذ، أو كانت قريبة منه
    # (معنوية × اتّساق إشارة ≥ 0.60) وتأكّدت خارج العيّنة (معنوية ≥ 60%).
    # حُذف: ema، stoch، macd، cmf (وكذلك RET_*، POS_14، VOLR_*، بنية الشمعة،
    # الوقت، ADX_14، BBL/BBM/BBU/BBP، MKT_ret_1).
    "indicator_settings": {
        'adx': [14],                 # DMP_14 (rel/low 78%) و DMN_14 (abs/close 74%)؛ ADX_14 مستبعد
        'rsi': [14],                 # abs/close 74%، 71% خارج العيّنة
        'natr': [14],                # ✅ H003 — مقبولة: 100%، 68/68 نافذة
        'bbands': [20],              # ✅ BBB_20 (العرض) مقبولة: rel/low 100%، 68/68؛ الباقي مستبعد
        'mfi': [14],                 # rel/low 68%، 63% خارج العيّنة
        # ── مؤشرات أُسقطت بالفرز (أعدها هنا إن احتجتها — FEATURES تُشتق تلقائياً) ──
        # 'ema': [9, 26], 'stoch': [(14, 3)], 'macd': [(12, 26, 9)], 'cmf': [20],
        # 'sma': [20], 'ppo': [(12,26,9)], 'mom': [10], 'roc': [10],
        # 'cci': [20],   ← أُسقط: ترابط 0.963 مع BBP
        # 'aroon': [14], 'willr': [14], 'er': [10], 'tsi': [13],
        # 'trix': [14], 'fisher': [9], 'slope': [14], 'chop': [14],
        # 'atr': [14], 'stdev': [14], 'zscore': [20], 'ui': [14],
        # 'efi': [13], 'adosc': [3], 'qstick': [14], 'kurtosis': [20],
    },
    # مؤشرات بلا معامل طول. OBV تراكمي يكمّل CMF/MFI بمنظور مختلف.
    "static_indicators": [],   # كان ['obv'] — أُوقف بقرارك؛ أعده بإضافة 'obv' هنا
    "static_indicator_params": {
        "log_return": {"length": 14},
        "percent_return": {"length": 14},
    },

    # ══════════════════════════════════════════════════════════════════════
    # 6-ب) الميزات المخصّصة (ساكنة وخالية من المقياس — انظر add_custom_features)
    # ══════════════════════════════════════════════════════════════════════
    "use_custom_features": True,
    "custom_settings": {
        "returns": [],                       # RET_* فشلت منفردة؛ MOM_ORTH_NATR يحسب عوائده بنفسه
        "range_windows": [50],               # POS_50 (abs/close 74%)؛ POS_14 حُذف
        "vol_windows": [],                   # VOLR_* حُذفت
        "volume_window": 20,                 # VOLZ_20 (abs/high 84%)
        "candle_structure": False,           # BODY_ratio/WICK_* حُذفت (RANGE_rel يبقى دائماً: ✅ مقبولة 68/68)
        "time_features": False,              # TIME_dow_* حُذفت
        "supertrend_windows": [10],          # SUPERT_STRETCH_10 (abs/close 71%)؛ SUPERT_DIR_10 مستبعد
        "vol_term_structure_pairs": [(3, 30)],  # VOL_TERM_3_30 (abs/low 74%، 82% خارج العيّنة)
    },

    # أقصى قيمة مطلقة بعد التطبيع (قصّ يمنع انفجار التدرّجات من شذوذ واحد).
    "clip_abs": 5.0,

    # 🔧 تطبيع الأعمدة السعرية (price_level: open/high/low/close، المتوسطات، نطاقات بولنجر...) داخل كل نافذة:
    #   'window_scale' (الافتراضي = السلوك السابق حرفياً) — (x − وسيط إغلاق النافذة) / IQR، مقصوص ±clip_abs.
    #   'pct_change'   — تغيّر نسبي هندسي عن الشمعة السابقة لنفس العمود بالنقاط المئوية: 100 × (x_t / x_{t−1} − 1)
    #                    (مثلاً −1، +5، −0.4). أول صف في النافذة 0. قابل للفكّ: pct_change_decode، وفي مجموعة
    #                    البيانات decode_price_window (مرساة آخر سعر من last_candles). يُحفظ في dataset['price_norm_mode'].
    # بقية الأنواع (ATR/MACD بوحدة السعر، المذبذبات...) لا تتغيّر في الوضعين.
    "price_norm_mode": "window_scale",
    # قصّ أعمدة 'pct_change' بالنقاط المئوية بدل clip_abs: ±100 لا يمسّ حركة شمعة حقيقية فيبقى الفكّ دقيقاً.
    "price_pct_clip": 100.0,

    # 🔧 فلاتر العيّنات: تُبقي فقط العيّنات التي تحقّق **كل** الشروط عند شمعة الدخول (آخر شمعة في نافذة الفريم الأساسي)، على
    # القيم الخام قبل التطبيع. العمود يكفي أن يكون محسوباً (حتى لو مستبعداً من المدخلات، مثل ADX_14). [] (الافتراضي) = كل العيّنات.
    #   {"feature": "NATR_14", "op": ">", "value": 1.0}        ATR% لشمعة الدخول > 1%   (op: > >= < <= == != between)
    #   {"feature": "ADX_14", "op": ">", "value": 20}
    #   {"expr": "NATR_14 > 1 and (ADX_14 > 20 or RSI_14 < 30)"}   تعبير pandas (DataFrame.eval)؛ اسم بنقطة بين `…`
    #   {"fn": "my_rule"}                                       دالة مسجّلة بـ register_sample_filter (سببية إلزاماً)
    "sample_filters": [],

    # 🔧 كيف يُحسب هدف الانحدار {t}_reg — مستوحى من عائد Qlib المباشر
    # (Ref($close,-2)/Ref($close,-1)-1) بدل سعر مُعاد تسويته بإحصاء خارجي:
    #   'return'       (افتراضي جديد) — عائد مباشر: (السعر_المستقبلي /
    #                  آخر_سعر_من_نفس_نوع_الهدف) - 1. بلا مرجع خارجي (لا وسيط
    #                  نافذة ولا IQR)، فنفس نسبة الحركة تُنتج نفس القيمة بصرف
    #                  النظر عن أي نافذة سقطت فيها العيّنة أو أي عملة. راجع
    #                  سجلّ التعديلات في رأس الدفتر لتفصيل الأثر على مشكلة
    #                  الانكماش نحو المتوسط.
    #   'window_scale' (القديم) — (السعر_المستقبلي - آخر_سعر_من_نفس_النوع) /
    #                  IQR نافذة الإغلاق. للمقارنة أو التراجع. المركز ليس وسيط
    #                  النافذة: ذاك يكشف جزءاً من الاتجاه وقت الدخول (تسرّب).
    "reg_target_mode": "return",
    # قصّ هدف الانحدار بعد حسابه. None يختار افتراضاً حسب reg_target_mode
    # (1.0 أي ±100% لـ'return'، أو 10.0 كسابقاً لـ'window_scale') — الرقمان
    # بوحدتين مختلفتين تماماً فلا يصحّ ترك رقم واحد ثابتاً بين الوضعين. حدّده
    # صراحةً فقط إن أردتَ تجاوز اختيار الوضع التلقائي.
    "reg_target_clip": None,
    # مُعامل ضرب هدف الانحدار بعد القصّ: y = clip(الخام, ±reg_target_clip) × reg_target_scale — القصّ بوحدة
    # الهدف الأصلية (العائد) ثم الضرب. لماذا: عوائد 1h صغيرة (انحراف ≈ 0.012) فتُماثل أرضيات رأس NIG في
    # model_v2 (beta_min=0.01، nu_min…) وتهيمن على الخسارة؛ 100 يجعل الهدف بنقاط مئوية. يُحفَظ في مجموعة
    # البيانات (dataset['reg_target_scale']) فيقرؤه كل عكس منها لا من CONFIG: invert_reg_predictions(scale=…)،
    # export_signals في cross_asset، ودفتر main. 1.0 = السلوك القديم (والقيمة المفترضة لملفات بلا المفتاح).
    "reg_target_scale": 1.0,

    # 🔧 تطبيع مقطعي اختياري لأهداف الانحدار عبر كل الأصول عند نفس اللحظة —
    # بروح Qlib CSZScoreNorm/CSRankNorm على حقل label (وليس محاكاة حرفية لكودها).
    # كل عيّنة تُقاس نسبة إلى توزيع عوائد أقرانها في نفس اللحظة، لا نسبة إلى
    # ماضيها هي — فتصير قابلة للمقارنة عبر عملات مختلفة التقلّب في نفس اللحظة،
    # خلاف العائد الخام وحده (reg_target_mode='return') الذي يبقى ثابت المعنى
    # عبر الزمن لكن ليس عبر عملات مختلفة القوة في نفس اللحظة. يُطبَّق صراحةً
    # بعد build_dataset عبر cross_sectional_normalize() — ليس تلقائياً، لأنه
    # يُحوِّل دلالة الرأس من عائد مطلق إلى أداء نسبي مقابل الأقران (راجع
    # توثيق الدالة قبل استخدامه).
    "cs_norm": {
        "method": "zscore",     # 'zscore' (افتراضي، قابل للعكس) أو 'rank' ([-1,1] ترتيبي)
        "min_assets": 5,        # أقل عدد أصول باللحظة الواحدة ليُحسب التطبيع؛ وإلا يبقى العائد الخام
        "clip": 5.0,            # قصّ بعد 'zscore' فقط ('rank' محصورة أصلاً)
    },

    # ══════════════════════════════════════════════════════════════════════
    # 7) مصادر البيانات
    # ══════════════════════════════════════════════════════════════════════
    # ⚠️ القيمتان التاليتان من الأصل (مسار gdown القديم) — غير مُستخدَمتين في
    # هذا الدفتر (Drive المُركَّب فقط)، أُبقيتا للتوثيق.
    "asset_registry_file_id": "1Ozei2b9z8uyEszutke7LV_8Mpomi-W2X",  # files_info.csv الحالي (القديم 1hITH2… قديم المعرّفات)
    "preprocessed_data_file_id": "1EWTFjth2swnnTbnJ4gV_doFDfsJ4iqzP",

    # مسار ملفات العملات الخام على Drive **المُركَّب** (لا تنزيل عام).
    "drive_raw_dir": "history_1d",
    "drive_raw_pattern": "{name}.csv",
    # 🔧 إضافة جديدة (غير موجودة في الأصل): مسار سجل الأصول على Drive
    # المُركَّب — بديل asset_registry_file_id بعد إزالة مسار gdown بالكامل.
    "asset_registry_path": "crypto_data/asset_registry.csv",
    # 🔧 إضافة جديدة: نقطة تركيب Drive (يستخدمها mount_drive أعلاه).
    "drive_mount_point": "/content/drive",

    # 🔧 إضافة جديدة: أسماء عملات تُتخطّى كلياً في build_dataset* — قبل محاولة
    # التحميل أصلاً. المطابقة بلا حساسية لحالة الأحرف ولا للمسافات الطرفية.
    # تسري أيضاً على build_dataset_live. انظر exclude_coins.
    "excluded_coins": [],

    # 🔧 إضافة جديدة: الحدّ الأقصى لعدد طلبات Binance في الدقيقة في الجلب الحيّ.
    # يُطبَّق على **كل** طلب futures_klines (كل صفحة وكل إعادة محاولة) ومشترك بين كل
    # الخيوط بنافذة منزلقة. 0 أو None يعطّل الحدّ. ⚠️ هذا حدّ على *عدد* الطلبات؛
    # Binance تقيّد بـ«الوزن» لكل IP، وتكلفة الطلب تكبر مع limit (راجع rateLimits
    # في exchangeInfo أو الترويسة x-mbx-used-weight-1m).
    "live_max_requests_per_minute": 2000,

    # ══════════════════════════════════════════════════════════════════════
    # 8) بنية النموذج (RoPE + GQA + MoE Transformer)
    # ══════════════════════════════════════════════════════════════════════
    "model_tf": "4h",          # الفريم المُغذّى فعلياً للنموذج
    "d_model": 64,
    "num_layers": 3,
    "num_heads": 4,
    "num_kv_heads": 2,
    "kv_latent": None,         # None → d_model // 4
    "expansion": 8 / 3,
    "model_dropout": 0.15,
    "model_name": "ModernTST",

    "use_decomposition": True,
    "decomp_kernels": None,

    "moe_every": 0,
    "n_shared": 2,
    "n_routed": 4,
    "top_k": 2,

    # ══════════════════════════════════════════════════════════════════════
    # 9) التدريب
    # ══════════════════════════════════════════════════════════════════════
    "learning_rate": 5e-4,
    "gradient_clipnorm": 1.0,
    "epochs": 100,
    "batch_size": 64,
    "use_class_weights": True,
    "class_weight_smoothing": 0.5,
    "early_stopping_patience": 30,
    "reduce_lr_patience": 10,
    "reduce_lr_factor": 0.5,

    # ══════════════════════════════════════════════════════════════════════
    # 10) التقييم
    # ══════════════════════════════════════════════════════════════════════
    "eval_batch_size": 256,
    "eval_range_frac": 0.2,
    "eval_n_display": 5,

    # ══════════════════════════════════════════════════════════════════════
    # 11) التخزين (تُحلّ تلقائياً حسب البيئة: Colab / محلي)
    # ══════════════════════════════════════════════════════════════════════
    "project_name": "crypto_model",
    "workspace_dir": None,     # None → يُكتشف تلقائياً
    "use_drive": True,         # يُتجاهَل تلقائياً خارج Colab
}

#: قائمة المؤشرات الثابتة الكاملة (غير مُفعَّلة افتراضياً — انظر التعليق أعلاه).
STATIC_INDICATORS_FULL = [
    'obv', 'ad', 'kvo', 'true_range', 'log_return',
    'percent_return', 'bop', 'uo', 'hlc3', 'wcp',
]

#: كون العملات الافتراضي — يُستخدم في صفحة تجهيز البيانات.
COINS_BY_CATEGORY = {
    "Large_Caps": [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
        "ADAUSDT", "AVAXUSDT", "DOTUSDT", "DOGEUSDT", "TRXUSDT",
        "LINKUSDT", "LTCUSDT", "ATOMUSDT", "UNIUSDT", "TONUSDT",
        "ICPUSDT", "ETCUSDT", "FILUSDT", "APTUSDT", "XLMUSDT",
        "INJUSDT", "NEARUSDT", "XMRUSDT", "OPUSDT",
    ],
    "DeFi": [
        "AAVEUSDT", "COMPUSDT", "SNXUSDT", "SUSHIUSDT",
        "LDOUSDT", "RUNEUSDT", "FXSUSDT", "PENDLEUSDT", "DYDXUSDT",
    ],
    "AI_BigData": [
        "TAOUSDT", "FETUSDT", "RNDRUSDT", "AKTUSDT", "NMRUSDT",
        "GRTUSDT", "PHBUSDT", "AIUSDT",
    ],
    "Gaming_Metaverse": [
        "SANDUSDT", "MANAUSDT", "IMXUSDT", "GALAUSDT", "AXSUSDT",
        "ENJUSDT", "YGGUSDT", "MAGICUSDT", "PORTALUSDT", "PIXELUSDT",
    ],
    "Meme": [
        "1000SHIBUSDT", "1000PEPEUSDT",
    ],
    "Layer1_Layer2": [
        "SUIUSDT", "SEIUSDT", "ALGOUSDT", "EGLDUSDT", "ARBUSDT",
        "STRKUSDT", "ZETAUSDT", "KASUSDT", "TIAUSDT", "HBARUSDT",
        "EOSUSDT", "XTZUSDT",
    ],
    "Various_Services": [
        "VETUSDT", "THETAUSDT", "QNTUSDT", "CHZUSDT", "HOTUSDT",
        "BATUSDT", "ZILUSDT", "ONTUSDT", "IOTXUSDT", "ONEUSDT",
        "CELOUSDT", "CFXUSDT", "ANKRUSDT", "STORJUSDT", "DENTUSDT",
        "GASUSDT", "NEOUSDT", "DASHUSDT",
    ],
}

#: الفئات المُفعَّلة افتراضياً.
DEFAULT_ENABLED_CATEGORIES = ["Large_Caps"]
