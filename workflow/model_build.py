"""
PURPOSE:  The zero-argument model_builder required by build_training_system (same architecture every call, needed to resume saved state).
TAGS:     model_builder, build_model_fn, MODEL_OVERRIDES, MODEL_SEQ_LEN, resume
PITFALLS: Reads build_model_fn (model_v2) and MODEL_SEQ_LEN / MODEL_N_FEATURES / MODEL_OVERRIDES from the notebook namespace at call time. Executed into the notebook's shared namespace by workflow/_loader.py (never imported on its own): names from other modules and the %run notebooks resolve at call time. Extracted verbatim from main.ipynb cell 13 (section 4).
"""
def make_model_builder(build_fn, model_seq_len, model_n_features, model_overrides):
    """يُرجع دالة بلا وسائط تبني نفس المعمارية كل مرّة، كما يتطلّب build_training_system (ضروري لصحّة استئناف الحالة المحفوظة).
    build_fn = build_model_fn (دفتر model_v2)؛ الباقي من ModelPlan (workflow/run.py plan_model)."""
    def model_builder():
        return build_fn(model_seq_len, model_n_features, config=model_overrides)
    return model_builder


def model_builder():
    """صفر-وسيط، كما يتطلّب build_training_system — نفس المعمارية كل مرّة
    (ضروري لصحّة استئناف الحالة المحفوظة)."""
    return build_model_fn(MODEL_SEQ_LEN, MODEL_N_FEATURES, config=MODEL_OVERRIDES)
