"""
PURPOSE:  Converts the pipeline's test split into the shape chicks expects (build_chicks_test_dict) and defines which target modes chicks supports.
TAGS:     build_chicks_test_dict, entry_range_target_spec, EVAL_TARGET_SPECS, CHICKS_TARGET_MODES, chicks, test_dict, relative modes, base_params last_close
PITFALLS: Reads LAST_CLOSE_COL, CHICKS_TARGETS, CONFIG, reg_scale_of, entry_close_reg_of from the notebook namespace. base_params is replaced by [last_close, last_close] only for reg_target_mode='return'. Executed into the notebook's shared namespace by workflow/_loader.py (never imported on its own): names from other modules and the %run notebooks resolve at call time. Extracted verbatim from main.ipynb cell 19 (section 6).
"""
import dataclasses

import numpy as np

from core.schema import LAST_COLUMN_INDEX as _LAST_COLUMN_INDEX
#: فهرس آخر إغلاق في last_candles (سعر الدخول) — من core/schema.py لا من الدفتر.
LAST_CLOSE_COL = _LAST_COLUMN_INDEX["last_close"]
# أوضاع القسم ٣-ب التي تبقى «عائداً نسبة لآخر سعر من نفس النوع» — وهو ما تفكّه EVAL_TARGET_SPECS. في relative يُطرح
# وسيط السوق، فتُمرَّر لـ chicks أسعار مستقبلية مكافئة للهدف (آخر_سعر × (1 + الهدف)) كي يتطابق الفضاءان.
CHICKS_TARGET_MODES = (None, "return", "relative", "return+relative", "entry_range")
_RELATIVE_MODES = ("relative", "return+relative")
_FUTURE_OF = {"close": ("future_close", "last_close"), "high": ("future_high_max", "last_high"),
              "low": ("future_low_min", "last_low")}


def build_chicks_test_dict(pipeline_test, model_tf, reg_target_mode=None, chicks_targets=None):
    """يحوّل `test` (مخرَج split_data، قاموس {أصل: قسم}) إلى الشكل الذي
    تتوقعه دوال chicks (`test_all_assets_v4`/`run_full_analysis`).
    chicks_targets: الأهداف التي يُمرَّر y_{هدف}_reg لها (ChicksInputs.chicks_targets)؛ None = CHICKS_TARGETS من نطاق الدفتر."""
    reg_target_mode = reg_target_mode or CONFIG.get("reg_target_mode", "return")
    if chicks_targets is None:
        chicks_targets = CHICKS_TARGETS
    # أهداف القسم ٣-ب (relative/scaled/magnitude…) ليست عائداً نسبة لسعر الدخول: فكّها في chicks يُنتج «true» خاطئاً،
    # وtearsheet يحسب الربح من «true» — مع relative يصبح الربح انحراف العوائد (متوسطها > وسيطها) لا مهارة.
    modes = {s.get("target_mode") for s in pipeline_test.values()} - set(CHICKS_TARGET_MODES)
    if modes:
        raise ValueError(f"test يحمل أهداف {sorted(modes, key=str)} — chicks يدعم {CHICKS_TARGET_MODES[1:]} فقط")
    out = {}
    for asset, split in pipeline_test.items():
        last_candles = split["last_candles"]
        if split.get("target_mode") in _RELATIVE_MODES:
            last_candles = np.array(last_candles, dtype="float64", copy=True)
            for t, (fut_col, last_col) in _FUTURE_OF.items():
                if f"y_{t}_reg" in split["y"]:
                    last_candles[:, LAST_COLUMNS.index(fut_col)] = (
                        last_candles[:, LAST_COLUMNS.index(last_col)]
                        * (1.0 + np.asarray(split["y"][f"y_{t}_reg"], dtype="float64") / reg_scale_of(split)))
        if reg_target_mode == "return":
            last_close = last_candles[:, LAST_CLOSE_COL]
            base_params_eval = np.stack([last_close, last_close], axis=1).astype("float32")
        else:
            base_params_eval = split["base_params"]
        out[asset] = {
            **{f"X_{tf}": split[f"X_{tf}"] for tf in _tfs_of(model_tf)},
            "base_params": base_params_eval,
            "last_candles": last_candles,
            "y": {t: split["y"][f"y_{t}_reg"] for t in chicks_targets},
        }
    return out


def entry_range_target_spec(s, test):
    """TargetSpec (chicks) of target ``s`` for an entry_range ``test``: high/low من سعر الدخول last_close (العمود 2): high = P·(1 + raw/s)،
    low = P·(1 − raw/s) ⇐ reg_scale سالب؛ close مع range_pos = pred_low + clip(raw, 0, 1)·(pred_high − pred_low) ⇐ range_of — نفس
    entry_range_to_prices في القسم ٣-ب؛ close مع abs_return = P·(1 + s·raw/scale)، s = +1 إن كان p_up_close ≥ 0.5 وإلا −1 ⇐ signed_by_class:
    chicks يقرأ P(صعود) من مخرَج رأس التصنيف. class_key لكل رأس = 'y_{هدف}_class_logits' (احتمال بعد sigmoid رغم الاسم): يُقرأ كخاصية p_up
    لتحليل الأنماط إن وُجد في النموذج، ويُلزَم فقط لاتجاه close."""
    _range_pos = entry_close_reg_of(test) == "range_pos"
    _cls = dict(class_key=f"y_{s.name}_class_logits")
    if s.name == "close" and _range_pos:
        return dataclasses.replace(s, price_index=2, entry_col=2, range_of=("high", "low"), **_cls)
    if s.name == "close":
        return dataclasses.replace(s, relative_to_entry=True, price_index=2, entry_col=2,
                                   reg_scale=reg_scale_of(test, "close"), signed_by_class=True, **_cls)
    return dataclasses.replace(s, relative_to_entry=True, price_index=2, entry_col=2,
                               reg_scale=(-1.0 if s.name == "low" else 1.0) * reg_scale_of(test, s.name), **_cls)
