"""
PURPOSE:  MODEL_CONFIG, ANTI_MEMORIZATION_CONFIG and build_model_fn, the entry point used by main.ipynb.
TAGS:     MODEL_CONFIG, ANTI_MEMORIZATION_CONFIG, build_model_fn, head_types, model config
PITFALLS: MODEL_CONFIG is a live dict mutated by main.ipynb (MODEL_CONFIG['head_types'] = ...); build_model_fn reads
          it at call time. Executed into the one shared model namespace by model/_loader.py (never imported on its
          own): names from other modules resolve at call time.

## 9) `MODEL_CONFIG` + `build_model_fn` — نقطة الدخول من دفتر `main`

على عكس النسخة السابقة (حيث كانت `MODEL_CONFIG` بمفاتيح معمارية قديمة غير
مستخدَمة فعلياً، و`seq_len`/`n_features` مُثبَّتتين بالكود)، هنا:

* كل مفاتيح `MODEL_CONFIG` تُطابق أسماء معاملات `build_nig_timenet_v2` فعلياً.
* `seq_len`/`n_features` وسيطان **إلزاميان** لـ`build_model_fn` — يُشتقّان من
  بيانات خط الأنابيب الفعلية في دفتر main (`dataset['window_sizes'][model_tf]`
  و`len(dataset['feature_order'])`)، لا يُخمَّنان هنا.
* `ANTI_MEMORIZATION_CONFIG` (PR #7): إعداد أصغر ومُنظَّم يُدمَج فوق `MODEL_CONFIG` —
  `build_model_fn(seq_len, n_features, config=ANTI_MEMORIZATION_CONFIG)`، أو `ANTI_MEMORIZATION = True` في main.
"""
MODEL_CONFIG = dict(
    d_model=128,
    num_layers=4,
    num_heads=4,
    num_kv_heads=2,
    patch_len=4,
    stride=2,
    kernel_sizes=(3, 5, 9, 17),
    max_rel_pos=16,
    dropout=0.1,
    attn_dropout=0.0,
    causal=False,
    window=None,
    use_instance_norm=True,
    use_decomposition=True,
    use_linear_path=True,
    price_targets=("high", "low", "close"),
    # رأسا تصنيف وانحدار معاً لكل هدف — يطابق enabled_heads الافتراضي في
    # خط الأنابيب. عطّل التصنيف بإعادة هذا لـ DEFAULT_HEAD_TYPES (نسخة).
    head_types={
        "high": ["nig_regression", "binary_classification"],
        "low": ["nig_regression", "binary_classification"],
        "close": ["nig_regression", "binary_classification"],
    },
    enforce_order=True,
    head_hidden=128,
    nu_min=0.1,
    alpha_min=2.0,
    beta_min=0.01,
    unc_max=20.0,   # حدّ عدم اليقين y_*_aleatoric/epistemic (بوحدة y_*_reg)؛ None = بلا قصّ. راجع NIGUncertainty
    class_head_hidden=64,
    n_classes=3,
    # مقاومة الحفظ — القيم هنا = السلوك القديم تماماً (نموذج مُدرَّب سابقاً يُستأنف بلا تغيير في المعمارية).
    norm_eps=1e-4,
    input_clip=None,
    stats_mode="full",
    input_noise_std=0.0,
    feature_dropout=0.0,
    linear_path_l2=1e-4,
    level_passthrough=False,
    level_norm=None,
)

# ── إعداد مقاومة الحفظ (PR #7، «RL_gru») — يُدمَج فوق MODEL_CONFIG عبر
# build_model_fn(..., config=ANTI_MEMORIZATION_CONFIG) أو ANTI_MEMORIZATION=True في دفتر main (مع lr_initial=3e-4
# في ANTI_MEMORIZATION_TRAINER هناك). مُختار من 5 اتجاهات معمارية مُقاسة، ومؤكَّد على بيانات Drive الحقيقية (222
# عملة): test AUC 0.645/0.555 (حجم الحركة/الاتجاه النسبي) مقابل 0.609/0.538 للمرجع الخطّي، فجوة train−test
# 0.014/0.016، وAUC التدريب على تسميات مخلوطة 0.508/0.518 (= النموذج الخطّي نفسه). docs/research/anti_memorization_pr7.md
ANTI_MEMORIZATION_CONFIG = dict(
    d_model=64,
    num_layers=2,
    head_hidden=64,
    class_head_hidden=32,
    dropout=0.25,
    input_clip=4.0,
    stats_mode="symlog",
    input_noise_std=0.1,
    feature_dropout=0.1,
    linear_path_l2=1e-3,
    # السبب الجذري للحفظ: InstanceNorm يمحو مستوى الميزة (الإشارة مقطعية بالمستوى) — المستوى يعود للمسار الرئيسي
    # مُقيَّساً عبر العملات، والزمن يُرمَّز بـ GRU (سعة المحوّل غير لازمة للإشارة الزمنية المتبقية ويحفظ أكثر).
    level_passthrough=True,
    level_norm="batch",
    encoder="gru",
)


def build_model_fn(seq_len, n_features, config=None):
    """نقطة الدخول العامة لبناء النموذج. `config` يُدمَج فوق MODEL_CONFIG
    (المفاتيح غير المذكورة تبقى على قيمتها الافتراضية).

    seq_len/n_features: عددان (فريم واحد) أو قاموسان {فريم: قيمة} (فرع مُرمِّز لكل فريم — انظر build_nig_timenet_v2)."""
    cfg = dict(MODEL_CONFIG)
    if config:
        cfg.update(config)
    return build_nig_timenet_v2(seq_len=seq_len, n_features=n_features, **cfg)
