"""
PURPOSE:  Numeric constants shared by model and evaluation (the NIG alpha-minus-one denominator floor).
TAGS:     NIG_ALPHA_DEN_MIN, nig, alpha, uncertainty floor, constants
PITFALLS: NIG_ALPHA_DEN_MIN is used by model/nig_layers.py (training-time layer) and evaluation/targets.py (post-hoc decoding of the raw NIG outputs): the two MUST agree or the evaluated uncertainty differs from the trained one. Do not change the value without a new experiment card (it changes recorded uncertainty numbers when alpha is close to 1).
"""

#: أرضية مقام (alpha − 1) في عدم اليقين: لا أثر لها إطلاقاً حين alpha ≥ 1.01 (رأس NIG يضمن alpha ≥ alpha_min = 2 افتراضياً)،
#: وتمنع الانفجار β/(α−1) → ∞ إن خُفِّض alpha_min نحو 1 أو جاءت alpha من نموذج آخر.
NIG_ALPHA_DEN_MIN = 1e-2
