"""
PURPOSE:  Helpers over train/val/test splits: model input (single or multi-timeframe), per-split stamps (reg scale, target mode, entry close definition).
TAGS:     model_x, _tfs_of, _required, model_tf, multi timeframe input, reg_scale_of, target_mode_of, entry_close_reg_of, split stamps
PITFALLS: model_tf (a timeframe name, or the list DatasetInfo.model_tfs) is always an explicit argument: _tfs_of(None) raises (no notebook MODEL_TFS/MODEL_TF fallback); _required(value, name) is the one place that refusal lives. reg_scale_of(split, target) prefers the per-target stamp reg_target_scales (entry_range close with range_pos is unscaled). Executed into the notebook's shared namespace by workflow/_loader.py (never imported on its own): names from other modules and the %run notebooks resolve at call time. Extracted verbatim from main.ipynb cell 8 (section 3).
"""
def _required(value, name):
    """وسيط صريح بلا قيمة افتراضية: None = نسيه المستدعي (لا يُقرأ أي متغيّر من نطاق الدفتر — القيمة من الإعدادات/الخطوات)."""
    if value is None:
        raise ValueError(f"{name} مطلوب صراحةً (لا قيمة افتراضية من نطاق الدفتر) — مثلاً DatasetInfo.model_tfs / ModelPlan.price_targets "
                         "من workflow/run.py")
    return value


def _tfs_of(model_tf=None):
    """فريمات مُدخل النموذج كقائمة: model_tf اسم فريم، أو قائمة (DatasetInfo.model_tfs). إلزامي: لا يُقرأ MODEL_TFS/MODEL_TF من نطاق الدفتر."""
    model_tf = _required(model_tf, "model_tf")
    return [model_tf] if isinstance(model_tf, str) else list(model_tf)


def model_x(split, model_tf=None):
    """مُدخل النموذج من قسم: مصفوفة X_{فريم} لفريم واحد (كما كان)، وقاموس {فريم: X_فريم} لأكثر من فريم — مفاتيحه أسماء
    مُدخلات النموذج متعدّد الفريمات (model_v2). X المخزَّنة float16 (x_storage_dtype) يرفعها Keras إلى float32 عند النداء،
    ودفعات التدريب ترفعها في make_shuffled_dataset."""
    tfs = _tfs_of(model_tf)
    return split[f"X_{tfs[0]}"] if len(tfs) == 1 else {tf: split[f"X_{tf}"] for tf in tfs}


def _split_members(split_or_dict):
    """قسم واحد (train/val) ← [قسم]؛ قاموس عملات (test) ← أقسامه."""
    return [split_or_dict] if "y" in split_or_dict else list(split_or_dict.values())


def reg_scale_of(split_or_dict, target=None):
    """مقياس y_*_reg المختوم على القسم (من dataset['reg_target_scale'] أو retarget_splits)؛ 1.0 إن غاب.
    target: مقياس هدف بعينه — retarget_splits يختم reg_target_scales لكل هدف، لأن وضعاً قد لا يضرب كل أهدافه
    (entry_range مع ENTRY_CLOSE_REG="range_pos": close موقع في المدى [0,1] لا يُضرب). بلا الختم = المقياس المشترك."""
    members = _split_members(split_or_dict)
    if not members:
        return 1.0
    per_target = members[0].get("reg_target_scales") or {}
    if target is not None and target in per_target:
        return float(per_target[target])
    return float(members[0].get("reg_target_scale", 1.0))


def target_mode_of(split_or_dict):
    """وضع الهدف المختوم على القسم (retarget_splits، القسم ٣-ب)؛ None = أهداف خط الأنابيب كما بُنيت."""
    members = _split_members(split_or_dict)
    return members[0].get("target_mode") if members else None


def entry_close_reg_of(split_or_dict):
    """تعريف انحدار close في entry_range المختوم على القسم ("abs_return" | "range_pos")؛ None خارج entry_range."""
    members = _split_members(split_or_dict)
    return members[0].get("entry_close_reg") if members else None
