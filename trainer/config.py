"""
PURPOSE:  TRAINER_DEFAULT_CONFIG (alias DEFAULT_CONFIG; the trainer's single source of truth), deep_update, build_config, config_fingerprint, model_signature.
TAGS:     TRAINER_DEFAULT_CONFIG, DEFAULT_CONFIG, build_config, deep_update, config_fingerprint, model_signature, VALID_TRAIN_MODES, trainer
          config
PITFALLS: Unknown config keys raise in build_config; the fingerprint decides whether a saved run may resume. Code here must use TRAINER_DEFAULT_CONFIG, never the bare DEFAULT_CONFIG: data/defaults.py binds a different DEFAULT_CONFIG (the pipeline's ~100 keys) and the two meet in main.ipynb's namespace. Executed
          into the one shared trainer namespace by trainer/_loader.py (never imported on its own): names from other
          modules resolve at call time.

## 2) قاموس الإعدادات (Config) — المصدر الوحيد للحقيقة

**فلسفة التصميم**: كل ما يخص "مهمتك" الخاصة يُعرَّف بالكامل داخل `config`. لا
يوجد داخل كود الإطار (الخلايا التالية) أي اسم هدف أو مفتاح مخرج مكتوب صراحة.
هذا يعني عمليًا:

- تغيير عدد الأهداف وأسمائها بحرية تامة، بلا حدود.
- خلط أهداف `evidential` (مع عدم يقين) مع `regression` بسيطة ومع
  `classification` في نفس النموذج وبنفس التدريب.
- إضافة/حذف جدول معامل (scheduled hyperparameter) بإضافة/حذف سطر واحد في
  `config['loss']['schedules']` دون لمس أي كلاس.
- إضافة نوع مهمة جديد بالكامل (`task_type`) عبر `register_task_type(...)`
  (القسم التالي) دون تعديل `GenericTrainer` إطلاقًا.

`DEFAULT_CONFIG` أدناه هو مجرد **هيكل افتراضي** (قيم منطقية آمنة + توثيق لكل
مفتاح) — أنت تكتب `config` مشروعك الخاص وتمرره لـ `build_config()` الذي يدمجه
فوق الافتراضي.
"""
# @title 2) قاموس الإعدادات الافتراضي + دالة الدمج
VALID_TRAIN_MODES = ("auto", "new", "resume", "warm_start")

TRAINER_DEFAULT_CONFIG: Dict[str, Any] = {
    # ─────────────────────────────────────────────────────────────
    # إعدادات التشغيل العامة
    # ─────────────────────────────────────────────────────────────
    "run": {
        "run_dir": "/content/drive/MyDrive/training_runs/default_run",  # المسار الدائم (يُفضّل Drive)
        "mirror_dir": None,      # مسار احتياطي اختياري (مثلاً Drive) إن كان run_dir محليًا سريعًا
        "mirror_every": 1,       # كل كم حقبة يتم نسخ run_dir بالكامل إلى mirror_dir
        "epochs": 40,
        "batch_size": 64,
        "seed": 42,
        "verbose": 1,

        # ── طريقة البدء (جديد) ───────────────────────────────────────────────
        #  "auto"       : يستأنف من آخر checkpoint إن وُجد، وإلا يبدأ من الصفر (السلوك القديم)
        #  "new"        : يبدأ من الصفر دائمًا. أي حالة سابقة في run_dir تُنقل إلى _archive/ (لا تُحذف)
        #  "resume"     : يستأنف الحالة الكاملة (أوزان + optimizer + جداول). يفشل بوضوح إن لم يجد checkpoint
        #  "warm_start" : أوزان نموذج موجود + optimizer جديد، وتستمر الجداول من الحقبة run.warm_start.epochs_done
        "train_mode": "auto",
        "on_existing": "archive",   # عند new/warm_start وفي run_dir حالة سابقة: "archive" (انقلها) | "error" (توقّف)
        "warm_start": {
            "weights_path": None,        # ملف أوزان النموذج الأساسي (.weights.h5) — إلزامي في وضع warm_start
            "epochs_done": None,         # كم حقبة تدرّبها هذا النموذج فعلًا — إلزامي (يحدد موضع الجداول)
            "lr_rewarmup_epochs": 0,     # تسخين قصير لمعدل التعلم لأن optimizer جديد (0 = استمرار الجدول كما هو)
        },
        "strict_epoch_guard": True,   # يفشل fit() إن كان initial_epoch لا يطابق عدد الحقب المنجزة فعلًا
    },

    # ─────────────────────────────────────────────────────────────
    # الأهداف (targets) — عرّف هنا كل مخرج تريد تدريب النموذج عليه.
    # كل هدف قاموس مستقل بالكامل عن غيره. مثال (احذفه/بدّله بمشروعك):
    #
    # "targets": {
    #     "my_target": {
    #         "true_key": "y_my_target",        # المفتاح داخل y الحقيقي (dict)
    #         "task_type": "evidential",         # evidential | regression | classification | (نوعك الخاص)
    #         "output_keys": {                   # مطابقة أسماء مخرجات نموذجك (dict outputs)
    #             "mu": "y_my_target", "nu": "y_my_target_nu",
    #             "alpha": "y_my_target_alpha", "beta": "y_my_target_beta",
    #             "confidence": "y_my_target_confidence",
    #         },
    #         "loss_weight": 1.0,                # وزن ابتدائي (قابل للتحديث التلقائي لاحقًا)
    #         "use_calibration_loss": True,       # فقط لـ evidential
    #         "lambda_reg_var": "lambda_reg",     # اسم المتغير المجدول المستخدم في هذا الهدف
    #         "lambda_calib_var": "lambda_calib",
    #
    #         # ── خيارات evidential الإضافية (كلها افتراضيًا = السلوك القديم تمامًا) ──
    #         "normalize_reg": False,             # Meinert 2023 Eq.(11): قسمة الباقي على w_St داخل منظِّم الأدلة
    #         "normalize_reg_stop_grad": True,    # قطع التدرج عن w_St في المنظِّم (اختيار تصميم — الورقة لا تحدده)
    #         "beta_nll": 0.0,                    # Seitzer 2022: وزن NLL بـ (w_St²)^beta_nll — جرّب 0.5 (تجريبي على NIG)
    #         "huber_weight": 0.0,                # خسارة Huber مباشرة على mu كي لا "يفسّر" عدم اليقين الخطأ
    #         "huber_delta": 1.0,                 # ⚠️ يجب أن يناسب مقياس هدفك (بعد التطبيع)
    #         "alpha_floor_straight_through": False,  # مرّر التدرج عبر قصّ alpha<1 بدل قتله
    #
    #         # ── مقاومة الحفظ (أي نوع مهمة؛ كلها افتراضيًا معطّلة = السلوك القديم) ──
    #         "label_smoothing": 0.0,             # classification فقط: y → y·(1−ε) + ε/K. تسمية اتجاه عائد شبه صفري
    #                                             # ضجيج خالص؛ بلا تنعيم يدفع BCE النموذجَ لحفظه بثقة كاملة
    #         "flood_level": None,                # Flooding (Ishida 2020): أثناء التدريب فقط L → |L − b| + b —
    #                                             # حين تهبط خسارة المهمة تحت b ينعكس التدرّج فلا ينزل التدريب
    #                                             # لما دون مستوى يستحيل بلوغه بلا حفظ. التحقق/الاختبار بلا تعديل
    #     },
    # },
    "targets": {},

    # ─────────────────────────────────────────────────────────────
    # الخسارة (loss)
    # ─────────────────────────────────────────────────────────────
    "loss": {
        "use_uncertainty_weighting": True,   # موازنة تلقائية بين المهام (Kendall et al.)
        # أنواع مهام تُستثنى من موازنة Kendall وتُجمَع بوزنها الثابت (loss_weight). خسارة evidential
        # هي NLL كاملة قد تكون سالبة، ومعها تنحدر 0.5·exp(-s)·L + 0.5·s بلا حدّ (s → -∞) فتنفجر الخسارة
        # الكلية نحو -∞ — راجع ملاحظة القسم 4. أفرِغ القائمة ([]) فقط إن كنت متأكداً أن كل خسائرك موجبة.
        "uncertainty_weighting_exclude_task_types": ["evidential"],
        "schedules": {
            # اسم تختاره أنت: {"start":.., "end":.., "warmup_epochs":.., "schedule": "linear|cosine|exponential|sqrt"}
            # "lambda_reg":   {"start": 0.0, "end": 0.0, "warmup_epochs": 20, "schedule": "linear"},
            # "lambda_calib": {"start": 0.0, "end": 0.0, "warmup_epochs": 20, "schedule": "cosine"},
        },
        "constraints": [
            # قيود منطقية/فيزيائية اختيارية بين المخرجات، مثال:
            # {"name": "high_low_order", "fn": my_constraint_fn, "weight_var": "penalty_weight"},
        ],
    },

    # ─────────────────────────────────────────────────────────────
    # المُحسِّن (optimizer)
    # ─────────────────────────────────────────────────────────────
    "optimizer": {
        "name": "adamw",              # adamw | adam
        "lr_initial": 5e-5,
        "lr_min": 5e-7,
        "lr_warmup_epochs": 3,
        "lr_schedule": {"type": "cosine_restarts", "cycle_length": 10, "cycle_mult": 1.5},
        # أنواع lr_schedule المتاحة: "constant" | "cosine" (مرة واحدة حتى total_epochs) | "cosine_restarts"
        # ⚠️ AdamW في Keras يطبّق كل خطوة w ← w − lr·weight_decay·w (مُتحقَّق منه في مصدر Keras:
        # base_optimizer._apply_weight_decay). مع lr_initial=5e-5 تعطي 1e-4 انكماشاً 5e-9 لكل خطوة — أي
        # **لا تنظيم فعلياً**. قيم الممارسة الشائعة مع AdamW (0.01–0.1) هي المعنى المقصود؛ إعداد مقاومة
        # الحفظ في main يستخدم قيمة فعّالة. تُركت 1e-4 هنا كي لا يتغيّر سلوك تشغيلات قائمة.
        "weight_decay": 1e-4,
        # متغيرات لا يُطبَّق عليها weight_decay (تعابير نمطية تُبحث في اسم المتغير — Keras 3: "bias"، Keras 2:
        # "dense/bias:0"): الانحيازات ومعاملات التطبيع (gamma/beta/scale) والتضمين الموضعي وانحياز الانتباه
        # النسبي، و**log_var موازنة Kendall** — تقليصها نحو 0 يغيّر أوزان المهام نفسها لا تعقيد النموذج.
        # [] = السلوك القديم (تقليص الكل).
        "weight_decay_exclude": [r"(^|/)(bias|gamma|beta|scale|pos_emb|rel_bias)(:\d+)?$", r"(^|/)log_var_"],
        "clip_norm": 1.0,
        "use_ema": True,
        "ema_momentum": 0.999,
        "ema_warmup": False,     # True: زخم EMA = min(0.999, (1+t)/(10+t)) — ضروري حين الخطوات قليلة (enable_ema_warmup)
        "ema_window_epochs": None,  # مع ema_warmup: نافذة EMA بالحقب (1.0 ≈ متوسط آخر حقبة) بدل ~1000 خطوة ثابتة (EmaWindow)
        "use_xla": False,   # فعّلها True لتسريع GPU عبر XLA (جرّبها أولاً على مشروعك)
    },

    # ─────────────────────────────────────────────────────────────
    # الكولباكس (callbacks)
    # ─────────────────────────────────────────────────────────────
    "callbacks": {
        # مصدر واحد لـ"أفضل نسخة" + التوقف المبكر (BestModelTracker):
        "early_stopping": {
            "monitor": "val_loss",      # أي مقياس يظهر في logs (مثل val_a_mae أو val_b_accuracy، أو val_class_loss:
                                        # مجموع خسائر رؤوس التصنيف وحدها — DerivedMetrics)
            "mode": "min",              # "min" للخسارة/الخطأ، "max" للدقة
            "patience": 15,
            "min_delta": 1e-4,
            "smoothing": "window",      # طريقة تنعيم *المقياس* قبل اختيار الأفضل: "none" | "window" | "ema"
            "smoothing_window": 3,      # لـ "window": متوسط آخر N حقب
            "ema_beta": 0.7,            # لـ "ema": s_t = beta*s_(t-1) + (1-beta)*x_t  (≈ نافذة 1/(1-beta) حقبة)
            "restore_best_weights": True,
            # أي *أوزان* تُحفَظ عند كل تحسّن: "raw" (أوزان تلك الحقبة كما هي) | "ema_weights" (متوسط EMA
            # المتراكم داخل optimizer عبر optimizer['use_ema'] — أكثر استقرارًا، لكنه يتطلب use_ema=True).
            # هذا مستقل تمامًا عن "smoothing" أعلاه (ذاك ينعّم *مقياس المقارنة*، هذا يغيّر *ماذا نحفظ*).
            "weights_snapshot": "raw",
        },
        "task_weight_update_frequency": 0,   # >0 لتفعيل تحديث يدوي (فقط عند تعطيل uncertainty_weighting)
        "metrics_log_every": 5,
        "snapshots": {"epochs": [], "dir": "snapshots"},   # اتركها [] لتعطيل حفظ اللقطات الإضافية
    },

    # ─────────────────────────────────────────────────────────────
    # نظام الحفظ/الاستئناف (checkpoint)
    # ─────────────────────────────────────────────────────────────
    "checkpoint": {
        "save_every": 1,             # احفظ كل epoch (الوحدة الدنيا لضمان الاستئناف)
        "max_to_keep": 3,
        "save_best_weights": True,   # ملف best.weights.h5 لأفضل أوزان (يُحفظ ذرّيًا ويبقى صحيحًا عبر الاستئناف)
    },
}


#: الاسم القديم لـ TRAINER_DEFAULT_CONFIG (القاموس نفسه). خط الأنابيب (data/) يعرّف أيضاً DEFAULT_CONFIG (~100 مفتاحاً)،
#: فالاسم المجرّد ملتبس في نطاق يضمّ الحزمتين (main.ipynb)؛ كود هذه الحزمة يستعمل TRAINER_DEFAULT_CONFIG دائماً.
DEFAULT_CONFIG = TRAINER_DEFAULT_CONFIG


def deep_update(base: dict, override: dict) -> dict:
    """دمج عميق: override تُكتب فوق base دون فقدان أي مفتاح غير مذكور في override"""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = v
    return result


def build_config(user_config: dict) -> dict:
    """يدمج إعدادات المستخدم فوق إعدادات الإطار الافتراضية، ويتحقق من الحد الأدنى اللازم"""
    cfg = deep_update(TRAINER_DEFAULT_CONFIG, user_config)
    if not cfg["targets"]:
        raise ValueError("❌ يجب تعريف هدف واحد على الأقل داخل config['targets']")
    for name, tcfg in cfg["targets"].items():
        for required in ("true_key", "task_type", "output_keys"):
            if required not in tcfg:
                raise ValueError(f"❌ الهدف '{name}': المفتاح المطلوب '{required}' غير موجود")

    run = cfg["run"]
    if str(run["train_mode"]).lower() not in VALID_TRAIN_MODES:
        raise ValueError(f"❌ run.train_mode='{run['train_mode']}' غير صالح — اختر واحدًا من {VALID_TRAIN_MODES}")
    run["train_mode"] = str(run["train_mode"]).lower()
    if run["on_existing"] not in ("archive", "error"):
        raise ValueError("❌ run.on_existing يجب أن يكون 'archive' أو 'error'")

    es = cfg["callbacks"]["early_stopping"]
    if es["smoothing"] not in ("none", "window", "ema"):
        raise ValueError("❌ early_stopping.smoothing يجب أن يكون 'none' أو 'window' أو 'ema'")
    if es["mode"] not in ("min", "max"):
        raise ValueError("❌ early_stopping.mode يجب أن يكون 'min' أو 'max'")
    if not 0.0 <= float(es["ema_beta"]) < 1.0:
        raise ValueError("❌ early_stopping.ema_beta يجب أن يكون في [0, 1)")
    return cfg


# ── بصمة الإعدادات: تُستخدم لمنع إعادة استخدام مدرّب قديم في الذاكرة بعد تغيير config أو النموذج ──
# مفاتيح run التي لا تغيّر «هوية» المدرّب (تغييرها مشروع أثناء الجلسة: رفع epochs مثلًا)
_FINGERPRINT_IGNORED_RUN_KEYS = ("epochs", "verbose", "batch_size", "seed", "train_mode", "on_existing", "warm_start")


def _canonical(o):
    if isinstance(o, dict):
        return {str(k): _canonical(v) for k, v in sorted(o.items(), key=lambda kv: str(kv[0]))}
    if isinstance(o, (list, tuple)):
        return [_canonical(v) for v in o]
    if callable(o):
        return f"<fn:{getattr(o, '__module__', '')}.{getattr(o, '__qualname__', repr(o))}>"
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o if isinstance(o, (int, float, str, bool, type(None))) else repr(o)


def config_fingerprint(cfg: dict) -> str:
    c = {k: v for k, v in cfg.items() if k != "run"}
    c["run"] = {k: v for k, v in cfg["run"].items() if k not in _FINGERPRINT_IGNORED_RUN_KEYS}
    return hashlib.sha256(json.dumps(_canonical(c), sort_keys=True).encode("utf-8")).hexdigest()[:16]


def model_signature(model: tf.keras.Model) -> list:
    """توقيع بنيوي للنموذج: نوع كل طبقة + دالة التنشيط + أشكال الأوزان (لا نستخدم أسماء الطبقات لأن Keras يُرقّمها تلقائيًا)"""
    sig = []
    for layer in model.layers:
        act = getattr(layer, "activation", None)
        shapes = tuple(tuple(int(s) for s in w.shape) for w in layer.weights)
        sig.append((type(layer).__name__, getattr(act, "__name__", None), shapes))
    return sig
