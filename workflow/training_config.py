"""
PURPOSE:  Bridge from model heads to trainer_framework targets: build_target_configs (true_key vs output_keys, derived from core.heads.HeadSet) and the naive majority-class baseline for val_accuracy.
TAGS:     build_target_configs, with_head_loss_terms, head_terms, build_head_set, loss_weights, true_key, output_keys, label_smoothing, class_baselines, _naive_class_baseline, trainer targets
PITFALLS: Pipeline labels are +1/-1 (batches._to_unit_label converts to {0,1} at feed time); the names come from core/heads.py HeadSet (the one place where data keys and model output keys are tied together). Executed into the notebook's shared namespace by workflow/_loader.py (never imported on its own): names from other modules and the %run notebooks resolve at call time. Extracted verbatim from main.ipynb cells 15 and 17 (section 5).
"""
from core.heads import HeadSet as _HeadSet
from core.loss_plan import check_head_loss_terms as _check_head_loss_terms, check_loss_weights as _check_loss_weights


def build_head_set(price_targets, label_smoothing=0.0, class_only=False, loss_weights=None):
    """The HeadSet a run trains: today's heads for ``price_targets`` with the loss plan's static weights applied
    (``loss_weights`` = ``RunSettings.train.loss.weights``, ``{head name: weight}``; an unknown head raises). The one place
    ``train.loss`` weights enter a head (``HeadSet.with_loss_weights``); build_target_configs and the selection-metric check
    both read it."""
    heads = _HeadSet.default(price_targets, class_only=class_only, label_smoothing=label_smoothing)
    if not loss_weights:
        return heads
    return heads.with_loss_weights(_check_loss_weights(loss_weights, heads, "loss weights of the plan (train.loss.weights)"))


def build_target_configs(price_targets, label_smoothing=0.0, class_only=False, loss_weights=None):
    """يبني قسم targets لإعداد trainer_framework — رأسا انحدار وتصنيف معاً
    لكل هدف، مطابقة لِما ينتجه build_model_fn الآن افتراضياً (class_only: رؤوس
    التصنيف وحدها). مشتقّ من ``core.heads.HeadSet.default`` — المصدر الوحيد لأسماء
    الرؤوس ومفاتيح المخرجات والتسميات وأوزان الخسارة (plan_model يشتقّ منه
    head_types أيضاً)؛ مفاتيح القاموس (مثلاً 'high_reg'/'high_class') أسماء الرؤوس.

    الانحدار: منظِّم الأدلة (lambda_reg) ومعايرة رأس الثقة (lambda_calib) — جدولاهما في main_config؛ بلاهما
    lambda_reg=0 ورأس conf_* لا يصله أي تدرّج (تحذير "Gradients do not exist for conf_*") فتبقى
    y_{t}_confidence التي تعرضها chicks أوزاناً عشوائية غير مُدرَّبة (يضيفهما HeadKind nig_regression).

    label_smoothing: لرؤوس التصنيف (مقاومة الحفظ، PR #7) — تسمية اتجاه عائد شبه صفري ضجيج خالص، وبلا تنعيم
    تدفع BCE النموذجَ لحفظها بثقة كاملة.

    loss_weights: أوزان الخسارة الثابتة {اسم الرأس: وزن} من خطة الخسارة (RunSettings.train.loss.weights)؛ رأس مجهول يرفع
    ValueError؛ None/{} = الأوزان الافتراضية (لا يُكتب loss_weight)."""
    return build_head_set(price_targets, label_smoothing, class_only, loss_weights).target_configs()


def with_head_loss_terms(targets, head_terms, heads):
    """``targets`` (HeadSet.target_configs()) plus the per-head auxiliary loss terms of the plan (``train.loss.head_terms``,
    ``{head: {huber_weight|huber_delta|ic_weight: value}}``). Empty (the default) = ``targets`` unchanged, so the config and its fingerprint are the old ones. Returns a new dict; an unknown
    head, an option the head's task type does not accept or a bad value raises (core.loss_plan.check_head_loss_terms / check_report_heads)."""
    terms = _check_head_loss_terms(head_terms, heads, "loss terms of the plan (train.loss.head_terms)")
    out = {h: dict(cfg) for h, cfg in targets.items()}
    for h, opts in terms.items():
        out[h].update(opts)
    return out


import numpy as np


def _naive_class_baseline(split, cfg):
    """نسبة الفئة الأغلب في val لهذا الهدف (ترميز الاتجاه +1.0/-1.0 الخام،
    قبل _to_unit_label) — خطّ أساس ساذج يُقارَن به val_accuracy المُبلَّغ في
    كل ملخّص حقبة (MetricsLogger.class_baselines، trainer_framework_v2)، بنفس
    مبدأ tree_naive_baseline_accuracy في detect_success_failure_patterns:
    دقّة خام قد تبدو جيدة وهي فعلياً لا تتجاوز تخمين الفئة الأغلب، خصوصاً أن
    high_class/low_class تقارنان بأطراف نطاق يومي (high/low) لا بمرجع متماثل
    كـclose، فقد تحملان توازناً مختلفاً تماماً عن close_class."""
    v = np.asarray(split["y"][cfg["true_key"]])
    p = float(np.mean(v > 0))
    return max(p, 1.0 - p)
