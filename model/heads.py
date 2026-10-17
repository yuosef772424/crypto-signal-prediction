"""
PURPOSE:  HEAD_REGISTRY: pluggable output heads (register_head_type, build_head_outputs, binary/multiclass
          classification heads).
TAGS:     HEAD_REGISTRY, register_head_type, build_head_outputs, binary_classification, multiclass_classification,
          head types
PITFALLS: An unregistered head type raises (no silent default); this is the single place to add a new output type.
          Executed into the one shared model namespace by model/_loader.py (never imported on its own): names from
          other modules resolve at call time.

## 7) سجلّ الرؤوس (`HEAD_REGISTRY`) — نقطة التعديل الوحيدة لإضافة/تسمية مخرجات جديدة

كل نوع رأس هنا دالة واحدة: `(h, target, cfg) -> dict مخرجات`. لإضافة رأس
جديد لاحقاً (مثلاً تصنيف متعدد الفئات لنظام السوق): أضف دالة، سجّلها بـ
`@register_head_type("اسمك")`، ثم أضف اسمها إلى
`MODEL_CONFIG['head_types'][الهدف]` — بلا لمس أي شيء آخر في المعمارية.

استثناء واحد مقصود: `nig_regression` لا يُبنى من هذا السجلّ مباشرة لأنه
يحتاج تنسيقاً عابراً للأهداف (ترتيب high>=close>=low عبر `OrderedMeans`)
قبل أن تُحسَم قيمه النهائية — التنسيق نفسه في `build_nig_timenet_v2`
(القسم ٨)، لكن اسم الرأس ما زال يُقرأ من `head_types` مثل أي رأس آخر،
فحذفه من `head_types` يُعطِّله بلا تعديل كود.
"""
HEAD_REGISTRY = {}


def register_head_type(name):
    """مُزخرِف: يُسجِّل دالة بناء رأس تحت اسم يُستخدَم في MODEL_CONFIG['head_types']."""
    def deco(fn):
        HEAD_REGISTRY[name] = fn
        return fn
    return deco


def build_head_outputs(head_type, h, target, cfg):
    """يستدعي دالة الرأس المسجّلة تحت head_type. يرفع خطأً واضحاً إن كان
    الاسم غير مسجَّل — بدل فشل صامت لاحقاً بمخرج مفقود."""
    if head_type not in HEAD_REGISTRY:
        raise ValueError(
            f"نوع رأس غير مسجَّل: '{head_type}'. الأنواع المتاحة: {sorted(HEAD_REGISTRY)}. "
            f"سجّل دالة جديدة بـ @register_head_type('{head_type}') قبل استخدامه في head_types."
        )
    return HEAD_REGISTRY[head_type](h, target, cfg)


@register_head_type("binary_classification")
def build_binary_classification_head(h, target, cfg):
    """رأس تصنيف ثنائي (مثال: صعود/هبوط) — مطابق لاصطلاح خط الأنابيب
    head_name(target, 'class') = f'{target}_class'، فمفتاح المخرج هنا
    'y_{target}_class_logits' (احتمال بعد sigmoid، لا logit خام)."""
    hidden = cfg.get("class_head_hidden", 64)
    dropout = cfg.get("dropout", 0.1)
    x = layers.Dense(hidden, activation="relu", name=f"class_fc_{target}")(h)
    x = layers.Dropout(dropout, name=f"class_drop_{target}")(x)
    prob = layers.Dense(1, activation="sigmoid", name=f"y_{target}_class_logits", dtype="float32")(x)
    return {f"y_{target}_class_logits": prob}


@register_head_type("multiclass_classification")
def build_multiclass_classification_head(h, target, cfg):
    """رأس تصنيف متعدّد الفئات (مثال: نظام سوق هادئ/متقلّب/اتجاهي).
    عدد الفئات من cfg['n_classes'][target] أو cfg['n_classes'] الرقم المباشر."""
    n_classes_cfg = cfg.get("n_classes", 3)
    n_classes = n_classes_cfg[target] if isinstance(n_classes_cfg, dict) else n_classes_cfg
    hidden = cfg.get("class_head_hidden", 64)
    dropout = cfg.get("dropout", 0.1)
    x = layers.Dense(hidden, activation="relu", name=f"class_fc_{target}")(h)
    x = layers.Dropout(dropout, name=f"class_drop_{target}")(x)
    # softmax لا logits خام: classification_task_loss في trainer_framework يستدعي
    # sparse_categorical_crossentropy بـ from_logits=False (الافتراضي) — يتوقّع
    # توزيع احتمالات جاهزاً، تماماً كرأس التصنيف الثنائي (sigmoid لا logit خام).
    probs = layers.Dense(n_classes, activation="softmax", name=f"y_{target}_class_logits", dtype="float32")(x)
    return {f"y_{target}_class_logits": probs}
