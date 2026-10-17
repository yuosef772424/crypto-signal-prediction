"""
PURPOSE:  make_model_builder: the zero-argument model_builder required by build_training_system (same architecture every call, needed to resume saved state).
TAGS:     make_model_builder, model_builder, build_model_fn, model_overrides, model_seq_len, resume
PITFALLS: No namespace reads: build_fn (model_v2's build_model_fn) and the plan's model_seq_len / model_n_features / model_overrides are arguments, so the returned zero-argument builder builds the same architecture every call (needed to resume saved state). Executed into the notebook's shared namespace by workflow/_loader.py (never imported on its own): names from other modules and the %run notebooks resolve at call time. Extracted verbatim from main.ipynb cell 13 (section 4).
"""
def make_model_builder(build_fn, model_seq_len, model_n_features, model_overrides):
    """يُرجع دالة بلا وسائط تبني نفس المعمارية كل مرّة، كما يتطلّب build_training_system (ضروري لصحّة استئناف الحالة المحفوظة).
    build_fn = build_model_fn (دفتر model_v2)؛ الباقي من ModelPlan (workflow/run.py plan_model)."""
    def model_builder():
        return build_fn(model_seq_len, model_n_features, config=model_overrides)
    return model_builder

