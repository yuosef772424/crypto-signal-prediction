"""
PURPOSE:  TRAINER_DEFAULT_CONFIG (alias DEFAULT_CONFIG; the trainer's single source of truth), deep_update, build_config (merge + validate_config, the ONE place that validates), config_fingerprint, model_signature.
TAGS:     TRAINER_DEFAULT_CONFIG, DEFAULT_CONFIG, build_config, validate_config, _CONFIG_SCHEMA, strict config, deep_update, config_fingerprint, model_signature, VALID_TRAIN_MODES, trainer
          config
PITFALLS: build_config validates through validate_config (aggregated ValueError): an unknown key at any level (with a did-you-mean), a non-finite / negative / non-number loss_weight, an unregistered task_type, a key of another task type, a schedule name (lambda_*_var, constraint weight_var) not defined in loss.schedules, an early-stopping monitor the trainer never logs, a knob that would be silently ignored. Every key of TRAINER_DEFAULT_CONFIG must have a rule in _CONFIG_SCHEMA (tests/test_config_strictness.py fails otherwise); keys a task type reads are declared at register_task_type(config_keys=...), never as a second list here; the keys of the per-head auxiliary loss terms (ic_weight, var_weight, huber_*) come from core.loss_plan.HEAD_LOSS_TERMS (_head_term_keys), so TASK_REGISTRY content (tests/golden_registry_contents.json) is unchanged. A default value is never changed by validation (tests/golden_trainer_configs.json pins the configs and fingerprints). The fingerprint decides whether a saved run may resume. Code here must use TRAINER_DEFAULT_CONFIG, never the bare DEFAULT_CONFIG: data/defaults.py binds a different DEFAULT_CONFIG (the pipeline's ~100 keys) and the two meet in main.ipynb's namespace. Executed
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
import difflib as _difflib
import inspect as _inspect
import math as _math
import re as _re

from core.loss_plan import HEAD_LOSS_TERMS as _HEAD_LOSS_TERMS, SCHEDULE_CURVES as _SCHEDULE_CURVES
from core.registry import Contract as _Contract, Registry as _Registry

VALID_TRAIN_MODES = ("auto", "new", "resume", "warm_start")

def _check_task_entry(name, entry, meta):
    """The task-type contract, checked AT REGISTRATION (register_task_type): a malformed registration fails here, not at trainer
    build / training-step trace time. Returns the list of problems (empty = fine)."""
    problems = []
    fn = entry.get("fn") if isinstance(entry, dict) else None
    if not callable(fn):
        return [f"fn must be callable, got {fn!r}"]
    try:
        _inspect.signature(fn).bind(None, None, None, None)
    except TypeError:
        problems.append("fn must accept (y_true, outputs, target_cfg, scheduled_vars) positionally")
    except ValueError:
        pass                                    # a builtin without an introspectable signature: nothing to check
    sk = meta["stat_keys"]
    if not isinstance(sk, list) or not sk or not all(isinstance(k, str) and k.strip() for k in sk) or len(set(sk)) != len(sk):
        problems.append(f"stat_keys must be a non-empty list of distinct non-empty strings, got {sk!r}")
    ck = meta["config_keys"]
    if not isinstance(ck, tuple) or not all(isinstance(k, str) and k.strip() for k in ck) or len(set(ck)) != len(ck):
        problems.append(f"config_keys must be a tuple of distinct non-empty strings, got {ck!r}")
    elif set(ck) & set(_COMMON_TARGET_KEYS):
        problems.append(f"config_keys {sorted(set(ck) & set(_COMMON_TARGET_KEYS))} are common target keys (declared once, not per task type)")
    if not isinstance(meta["supports_sample_weight"], bool):
        problems.append(f"supports_sample_weight must be a bool, got {meta['supports_sample_weight']!r}")
    elif meta["supports_sample_weight"]:
        try:
            _inspect.signature(fn).bind(None, None, None, None, sample_weight=None)
        except TypeError:
            problems.append("supports_sample_weight=True but fn does not accept a `sample_weight` keyword")
        except ValueError:
            pass
    for k in ("stat_keys", "supports_sample_weight", "config_keys"):
        if entry.get(k) != meta[k]:
            problems.append(f"entry[{k!r}] differs from the registered metadata")
    return problems


#: name -> {"fn", "stat_keys", "supports_sample_weight", "config_keys"}: filled by register_task_type (trainer/tasks.py). A core.registry
#: Registry (duplicate name raises, an unknown name raises with a did-you-mean, the contract above is checked at registration);
#: reads keep the dict pattern (TASK_REGISTRY[name]["stat_keys"], name in TASK_REGISTRY, .get / .items / .values). Created here, in the
#: module that loads before tasks.py, because build_config must know the registered task types (a forward reference to tasks.py
#: would be an implicit dependency).
TASK_REGISTRY = _Registry("task", _Contract(
    "task", required_meta=("stat_keys", "supports_sample_weight", "config_keys"), optional_meta=(), validator=_check_task_entry))

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
    """يدمج إعدادات المستخدم فوق إعدادات الإطار الافتراضية ثم يتحقّق منها كلها في مكان واحد (validate_config):
    مفتاح مجهول أو خاطئ الكتابة، وزن غير منتهٍ/سالب، نوع مهمة غير مسجَّل، مرجع جدول لا وجود له، مقياس لا يسجّله المدرّب... كلها ترفع ValueError."""
    cfg = deep_update(TRAINER_DEFAULT_CONFIG, user_config)
    return validate_config(cfg)


# ───────────────────────────── التحقّق الصارم (مكان واحد) ─────────────────────────────
# كل مفتاح في الإعداد له قاعدة هنا؛ مفتاح بلا قاعدة = خطأ (مع أقرب اسم معروف). build_config يستدعي validate_config وحدها،
# فلا تُوزَّع فحوص متناثرة في الوحدات الأخرى. إضافة مفتاح افتراضي جديد في TRAINER_DEFAULT_CONFIG بلا قاعدة هنا يُفشل
# tests/test_config_strictness.py (التغطية)، فلا يعود «مفتاح مقبول بلا معنى» ممكناً.
def _is_int(v):
    return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))


def _is_real(v):
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


def _r_int(ge=None):
    def rule(v):
        if not _is_int(v):
            return f"must be an int, got {v!r}"
        if ge is not None and v < ge:
            return f"must be an int >= {ge}, got {v!r}"
    return rule


def _r_real(ge=None, gt=None, lt=None):
    def rule(v):
        if not _is_real(v) or not _math.isfinite(float(v)):
            return f"must be a finite number, got {v!r}"
        if ge is not None and v < ge:
            return f"must be >= {ge}, got {v!r}"
        if gt is not None and v <= gt:
            return f"must be > {gt}, got {v!r}"
        if lt is not None and v >= lt:
            return f"must be < {lt}, got {v!r}"
    return rule


def _r_bool(v):
    if not isinstance(v, (bool, np.bool_)):
        return f"must be a bool, got {v!r}"


def _r_str(v):
    if not isinstance(v, str) or not v:
        return f"must be a non-empty str, got {v!r}"


def _r_path(v):
    if not isinstance(v, (str, os.PathLike)) or not str(v):
        return f"must be a non-empty path, got {v!r}"


def _r_enum(*options):
    def rule(v):
        if v not in options:
            return f"must be one of {list(options)}, got {v!r}"
    return rule


def _r_opt(inner):
    """None أو القاعدة `inner`."""
    return lambda v: None if v is None else inner(v)


def _r_regex_list(v):
    if not isinstance(v, (list, tuple)):
        return f"must be a list of regex strings, got {v!r}"
    for p in v:
        try:
            _re.compile(p)
        except (TypeError, _re.error) as e:
            return f"invalid regex {p!r}: {e}"


def _r_int_list(v):
    if not isinstance(v, (list, tuple, set)) or not all(_is_int(e) and e >= 1 for e in v):
        return f"must be a list of epoch numbers (ints >= 1), got {v!r}"


#: المخططات: {مفتاح: قاعدة | مخطط متداخل}. القسم الكامل dict؛ المفاتيح التي تُدار بتحقّق خاص (targets, schedules, constraints...)
#: لها قاعدة `_r_any` هنا ويفحصها فاحص تقاطعي مخصّص في validate_config.
def _r_any(v):
    return None


_SCHEDULE_NAMES = _SCHEDULE_CURVES   # one list of curve names: core/loss_plan.py (the plan validates against it at settings construction)
_LR_SCHEDULE_TYPES = ("constant", "cosine", "cosine_restarts")
_COMMON_TARGET_KEYS = ("true_key", "task_type", "output_keys", "loss_weight", "flood_level")
_TARGET_VALUE_RULES = {
    "true_key": _r_str, "task_type": _r_str, "loss_weight": _r_real(ge=0.0),
    "flood_level": _r_opt(_r_real(ge=0.0)),
    "use_calibration_loss": _r_bool, "lambda_reg_var": _r_str, "lambda_calib_var": _r_str,
    "ic_weight": _r_real(ge=0.0), "var_weight": _r_real(ge=0.0),
    "normalize_reg": _r_bool, "normalize_reg_stop_grad": _r_bool, "alpha_floor_straight_through": _r_bool,
    "beta_nll": _r_real(ge=0.0), "huber_weight": _r_real(ge=0.0), "huber_delta": _r_real(gt=0.0),
    "label_smoothing": _r_real(ge=0.0, lt=1.0), "binary": _r_bool, "loss_fn": _r_enum("mse", "mae", "huber"),
}

_CONFIG_SCHEMA = {
    "run": {
        "run_dir": _r_path, "mirror_dir": _r_opt(_r_path), "mirror_every": _r_int(ge=1), "epochs": _r_int(ge=1),
        "batch_size": _r_int(ge=1), "seed": _r_int(), "verbose": _r_enum(0, 1, 2),
        "train_mode": _r_any,       # فحص مخصّص: غير حسّاس لحالة الأحرف، ويُطبَّع إلى lower()
        "on_existing": _r_enum("archive", "error"),
        "warm_start": {"weights_path": _r_opt(_r_path), "epochs_done": _r_opt(_r_int(ge=0)), "lr_rewarmup_epochs": _r_int(ge=0)},
        "strict_epoch_guard": _r_bool,
    },
    "targets": _r_any,              # فاحص الأهداف المخصّص (مفاتيح كل نوع مهمة تأتي من TASK_REGISTRY)
    "loss": {
        "use_uncertainty_weighting": _r_bool,
        "uncertainty_weighting_exclude_task_types": _r_any,
        "schedules": _r_any, "constraints": _r_any,
    },
    "optimizer": {
        "name": _r_enum("adamw", "adam"), "lr_initial": _r_real(gt=0.0), "lr_min": _r_real(ge=0.0),
        "lr_warmup_epochs": _r_int(ge=0),
        "lr_schedule": {"type": _r_enum(*_LR_SCHEDULE_TYPES), "cycle_length": _r_int(ge=1), "cycle_mult": _r_real(ge=1.0),
                        "total_epochs": _r_int(ge=1)},
        "weight_decay": _r_real(ge=0.0), "weight_decay_exclude": _r_regex_list, "clip_norm": _r_opt(_r_real(gt=0.0)),
        "use_ema": _r_bool, "ema_momentum": _r_real(ge=0.0, lt=1.0), "ema_warmup": _r_bool,
        "ema_window_epochs": _r_opt(_r_real(gt=0.0)), "use_xla": _r_bool,
    },
    "callbacks": {
        "early_stopping": {
            "monitor": _r_str, "mode": _r_enum("min", "max"), "patience": _r_int(ge=1), "min_delta": _r_real(ge=0.0),
            "smoothing": _r_enum("none", "window", "ema"), "smoothing_window": _r_int(ge=1),
            "ema_beta": _r_real(ge=0.0, lt=1.0), "restore_best_weights": _r_bool,
            "weights_snapshot": _r_enum("raw", "ema_weights"),
        },
        "task_weight_update_frequency": _r_int(ge=0), "metrics_log_every": _r_int(ge=1),
        "snapshots": {"epochs": _r_int_list, "dir": _r_str},
        "class_baselines": _r_any,  # يُضاف بعد البناء من val (workflow.make_training_config) ويمرّ بـ build_config ثانيةً (capacity/permutation)
    },
    "checkpoint": {"save_every": _r_int(ge=1), "max_to_keep": _r_int(ge=1), "save_best_weights": _r_bool},
}


def _suggest(name, known):
    close = _difflib.get_close_matches(str(name), [str(k) for k in known], n=1, cutoff=0.6)
    return f" — did you mean {close[0]!r}?" if close else ""


def _check_schema(path, value, schema, errs):
    if not isinstance(value, dict):
        errs.append(f"{path}: must be a dict, got {type(value).__name__}")
        return
    for k, v in value.items():
        if k not in schema:
            errs.append(f"{path}: unknown key {k!r}{_suggest(k, schema)}; known keys: {sorted(schema)}")
            continue
        rule = schema[k]
        if isinstance(rule, dict):
            _check_schema(f"{path}.{k}", v, rule, errs)
        else:
            msg = rule(v)
            if msg:
                errs.append(f"{path}.{k}: {msg}")


def _expected_metric_names(cfg):
    """أسماء المقاييس التي يكتبها المدرّب في logs (قبل بادئة val_): تُشتق من الأهداف ومن TASK_REGISTRY، لا من قائمة ثانية."""
    names = {"loss", "raw_loss"}
    for t, tcfg in cfg["targets"].items():
        if not isinstance(tcfg, dict):
            continue                    # خطؤه مُبلَّغ في _check_targets
        names.add(f"loss_{t}")
        entry = TASK_REGISTRY.get(tcfg.get("task_type"), None)
        if entry:
            names.update(f"{t}_{k}" for k in entry["stat_keys"])
        if tcfg.get("task_type") == "classification":
            names.add("class_loss")
    for c in cfg["loss"]["constraints"]:
        if isinstance(c, dict) and isinstance(c.get("name"), str):
            names.add(f"constraint_{c['name']}")
    return names


def _head_term_keys(task_type):
    """Target keys of the auxiliary loss terms (core.loss_plan.HEAD_LOSS_TERMS) a head of ``task_type`` accepts: the term registry
    is their one declaration (not a second list in register_task_type's config_keys)."""
    return {k for e in _HEAD_LOSS_TERMS.entries() if task_type in e.meta["task_types"] for k in e.meta["options"]}


def _check_targets(cfg, errs):
    targets = cfg["targets"]
    if not isinstance(targets, dict):
        errs.append(f"targets: must be a dict, got {type(targets).__name__}")
        return
    schedules = cfg["loss"]["schedules"] if isinstance(cfg["loss"]["schedules"], dict) else {}
    for name, tcfg in targets.items():
        p = f"targets[{name!r}]"
        if not isinstance(tcfg, dict):
            errs.append(f"{p}: must be a dict, got {type(tcfg).__name__}")
            continue
        ttype = tcfg.get("task_type")
        entry = TASK_REGISTRY.get(ttype, None) if isinstance(ttype, str) else None
        if "task_type" in tcfg and entry is None:
            errs.append(f"{p}.task_type: unregistered task type {ttype!r}; registered: {sorted(TASK_REGISTRY)} "
                        f"(register it with register_task_type first)")
        known = set(_COMMON_TARGET_KEYS) | set(entry["config_keys"] if entry else ()) | _head_term_keys(ttype)
        for k, v in tcfg.items():
            if k not in known:
                if entry is None:
                    continue        # نوع غير مسجَّل: خطؤه الأساسي مذكور أعلاه؛ لا نكرّر
                other = [t for t, e in TASK_REGISTRY.items() if k in e["config_keys"]]
                hint = f" (it is a key of task type {other})" if other else ""
                errs.append(f"{p}: unknown key {k!r} for task type {ttype!r}{hint}{_suggest(k, known)}; known keys: {sorted(known)}")
                continue
            rule = _TARGET_VALUE_RULES.get(k)
            msg = rule(v) if rule else None
            if k == "output_keys" and not v:
                msg = f"must be non-empty, got {v!r}"
            if msg:
                errs.append(f"{p}.{k}: {msg}")
        # المراجع: اسم متغيّر مجدول يجب أن يكون معرَّفاً في loss.schedules (وإلا كان المنظِّم يُطفأ بصمت — tasks.py)
        for key, needed in (("lambda_reg_var", True), ("lambda_calib_var", bool(tcfg.get("use_calibration_loss")))):
            if key in tcfg and isinstance(tcfg[key], str) and needed and tcfg[key] not in schedules:
                errs.append(f"{p}.{key}: names the schedule {tcfg[key]!r}, which is not defined in loss.schedules "
                            f"{sorted(schedules)}{_suggest(tcfg[key], schedules)}")


def _check_loss(cfg, errs):
    loss = cfg["loss"]
    ex = loss["uncertainty_weighting_exclude_task_types"]
    if not isinstance(ex, (list, tuple, set)):
        errs.append(f"loss.uncertainty_weighting_exclude_task_types: must be a list of task type names, got {ex!r}")
    else:
        for t in ex:
            if t not in TASK_REGISTRY:
                errs.append(f"loss.uncertainty_weighting_exclude_task_types: unregistered task type {t!r}; "
                            f"registered: {sorted(TASK_REGISTRY)}{_suggest(t, TASK_REGISTRY)}")
    sched = loss["schedules"]
    if not isinstance(sched, dict):
        errs.append(f"loss.schedules: must be a dict of name -> schedule, got {type(sched).__name__}")
        sched = {}
    sched_schema = {"start": _r_real(), "end": _r_real(), "warmup_epochs": _r_int(ge=0), "schedule": _r_enum(*_SCHEDULE_NAMES)}
    for pname, pcfg in sched.items():
        p = f"loss.schedules[{pname!r}]"
        if not isinstance(pname, str) or not pname:
            errs.append(f"loss.schedules: schedule names must be non-empty str, got {pname!r}")
        _check_schema(p, pcfg, sched_schema, errs)
        if isinstance(pcfg, dict):
            for req in ("start", "end"):
                if req not in pcfg:
                    errs.append(f"{p}: required key {req!r} is missing")
    cons = loss["constraints"]
    if not isinstance(cons, (list, tuple)):
        errs.append(f"loss.constraints: must be a list of dicts, got {type(cons).__name__}")
        return
    seen = set()
    for i, c in enumerate(cons):
        p = f"loss.constraints[{i}]"
        if not isinstance(c, dict):
            errs.append(f"{p}: must be a dict, got {type(c).__name__}")
            continue
        for k in c:
            if k not in ("name", "fn", "weight_var"):
                errs.append(f"{p}: unknown key {k!r}{_suggest(k, ('name', 'fn', 'weight_var'))}; known keys: ['fn', 'name', 'weight_var']")
        if not isinstance(c.get("name"), str) or not c.get("name"):
            errs.append(f"{p}.name: must be a non-empty str, got {c.get('name')!r}")
        elif c["name"] in seen:
            errs.append(f"{p}.name: duplicate constraint name {c['name']!r}")
        else:
            seen.add(c["name"])
        if not callable(c.get("fn")):
            errs.append(f"{p}.fn: must be callable, got {c.get('fn')!r}")
        wv = c.get("weight_var")
        if wv is not None and wv not in sched:
            errs.append(f"{p}.weight_var: names the schedule {wv!r}, which is not defined in loss.schedules "
                        f"{sorted(sched)}{_suggest(wv, sched)}")


def _check_callbacks(cfg, errs):
    cb, es = cfg["callbacks"], cfg["callbacks"]["early_stopping"]
    mon, mode = es["monitor"], es["mode"]
    if isinstance(mon, str) and mon:
        base = mon[4:] if mon.startswith("val_") else mon
        known = _expected_metric_names(cfg)
        if base not in known:
            errs.append(f"callbacks.early_stopping.monitor: {mon!r} is not a metric this trainer logs; "
                        f"use [val_]<one of {sorted(known)}>{_suggest(base, known)}")
        elif mode in ("min", "max"):
            lossy = base in ("loss", "raw_loss", "class_loss") or base.startswith(("loss_", "constraint_")) or base.endswith("_mae")
            if lossy and mode != "min":
                errs.append(f"callbacks.early_stopping.mode: monitor {mon!r} is an error/loss (smaller is better), mode must be 'min'")
            if base.endswith("_accuracy") and mode != "max":
                errs.append(f"callbacks.early_stopping.mode: monitor {mon!r} is an accuracy (larger is better), mode must be 'max'")
    if es["weights_snapshot"] == "ema_weights" and cfg["optimizer"]["use_ema"] is False:
        errs.append("callbacks.early_stopping.weights_snapshot: 'ema_weights' needs optimizer.use_ema=True "
                    "(otherwise the raw weights are saved silently)")
    if cb["task_weight_update_frequency"] and cfg["loss"]["use_uncertainty_weighting"]:
        errs.append("callbacks.task_weight_update_frequency: > 0 has no effect while loss.use_uncertainty_weighting=True "
                    "(the updater is only attached when it is off)")
    cbl = cb.get("class_baselines")
    if cbl is not None:
        if not isinstance(cbl, dict):
            errs.append(f"callbacks.class_baselines: must be a dict target -> baseline, got {type(cbl).__name__}")
        else:
            for t, b in cbl.items():
                if t not in cfg["targets"]:
                    errs.append(f"callbacks.class_baselines: {t!r} is not a target {sorted(cfg['targets'])}{_suggest(t, cfg['targets'])}")
                elif _r_real(ge=0.0)(b):
                    errs.append(f"callbacks.class_baselines[{t!r}]: {_r_real(ge=0.0)(b)}")


def validate_config(cfg: dict) -> dict:
    """المكان الوحيد الذي يُتحقَّق فيه من إعداد المدرّب (يستدعيه build_config على الإعداد بعد الدمج مع الافتراضي، فأي مفتاح
    خارج TRAINER_DEFAULT_CONFIG/المخطط هو مفتاح المستخدم). يجمع كل المشاكل في ValueError واحد. يُرجع cfg نفسه (لا يعدّل
    شيئاً إلا تطبيع run.train_mode إلى أحرف صغيرة كما كان build_config يفعل)."""
    errs: List[str] = []
    _check_schema("config", cfg, _CONFIG_SCHEMA, errs)
    if errs and any(not isinstance(cfg.get(k), dict) for k in ("run", "loss", "optimizer", "callbacks", "checkpoint")):
        raise ValueError("❌ invalid trainer config:\n  - " + "\n  - ".join(errs))
    if not cfg["targets"]:
        raise ValueError("❌ يجب تعريف هدف واحد على الأقل داخل config['targets']")
    if not isinstance(cfg["targets"], dict):
        errs.append(f"config.targets: must be a dict name -> target, got {type(cfg['targets']).__name__}")
        raise ValueError("❌ invalid trainer config:\n  - " + "\n  - ".join(errs))
    for name, tcfg in cfg["targets"].items():
        for required in ("true_key", "task_type", "output_keys"):
            if isinstance(tcfg, dict) and required not in tcfg:
                raise ValueError(f"❌ الهدف '{name}': المفتاح المطلوب '{required}' غير موجود")
    run = cfg["run"]
    if str(run["train_mode"]).lower() not in VALID_TRAIN_MODES:
        errs.append(f"run.train_mode={run['train_mode']!r} غير صالح — اختر واحدًا من {VALID_TRAIN_MODES}")
    else:
        run["train_mode"] = str(run["train_mode"]).lower()
        if run["train_mode"] == "warm_start":
            ws = run["warm_start"]
            if not ws.get("weights_path"):
                errs.append("run.warm_start.weights_path: required when run.train_mode='warm_start'")
            if ws.get("epochs_done") is None:
                errs.append("run.warm_start.epochs_done: required when run.train_mode='warm_start' (not guessed: 0 would reset the penalty schedules)")
    if isinstance(cfg["loss"], dict) and isinstance(cfg["callbacks"], dict) and isinstance(cfg["callbacks"].get("early_stopping"), dict):
        _check_targets(cfg, errs)
        _check_loss(cfg, errs)
        _check_callbacks(cfg, errs)
    if errs:
        raise ValueError(f"❌ invalid trainer config ({len(errs)} problem(s)):\n  - " + "\n  - ".join(errs))
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


def fingerprint_payload(cfg: dict, extra_ignored_run_keys: tuple = ()) -> dict:
    """The canonical (JSON-ready) dict that config_fingerprint hashes: the whole config minus the run keys that do not change what is trained.
    ``extra_ignored_run_keys`` drops more run keys (run identity ignores the run LOCATION keys, see trainer/identity.py)."""
    c = {k: v for k, v in cfg.items() if k != "run"}
    skip = _FINGERPRINT_IGNORED_RUN_KEYS + tuple(extra_ignored_run_keys)
    c["run"] = {k: v for k, v in cfg["run"].items() if k not in skip}
    return _canonical(c)


def config_fingerprint(cfg: dict, extra_ignored_run_keys: tuple = ()) -> str:
    return hashlib.sha256(json.dumps(fingerprint_payload(cfg, extra_ignored_run_keys), sort_keys=True).encode("utf-8")).hexdigest()[:16]


def model_signature(model: tf.keras.Model) -> list:
    """توقيع بنيوي للنموذج: نوع كل طبقة + دالة التنشيط + أشكال الأوزان (لا نستخدم أسماء الطبقات لأن Keras يُرقّمها تلقائيًا)"""
    sig = []
    for layer in model.layers:
        act = getattr(layer, "activation", None)
        shapes = tuple(tuple(int(s) for s in w.shape) for w in layer.weights)
        sig.append((type(layer).__name__, getattr(act, "__name__", None), shapes))
    return sig
