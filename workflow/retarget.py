"""
PURPOSE:  Target-mode switching on already-built splits (return, return_close, scaled, magnitude, volnorm, entry_range, +relative): retarget_splits recomputes y_*_reg / y_*_class from raw last_candles; entry_range_to_prices inverts entry_range.
TAGS:     target_mode, retarget_splits, target modes, relative, entry_range, entry_range_to_prices, entry_close_reg, reg_target_scale, magnitude, volnorm, scaled
PITFALLS: retarget_splits takes reg_target_scale (the dataset's, DatasetInfo), price_targets and close_reg (TargetSettings.entry_close_reg) as arguments; only the pipeline's CONFIG (default targets) and LAST_COLUMNS come from the namespace. The scale must be the dataset's, not read back from a split's stamp: a 'scaled' retarget stamps 1.0. tools/evaluate_trained_model.py pins TARGET_MODES/ENTRY_CLOSE_REGS (tests/test_entry_range.py). Executed into the notebook's shared namespace by workflow/_loader.py (never imported on its own): names from other modules and the %run notebooks resolve at call time. Extracted verbatim from main.ipynb cell 10 (section 3-b).
"""
import numpy as np
import pandas as pd

# أسماء أوضاع الهدف (+ اختيارياً "+relative") والأوضاع التي لا تقبل "+relative" وتعريفات انحدار close في entry_range: core/schema.py
from core.schema import ENTRY_CLOSE_REGS, NO_RELATIVE_BASES as _NO_RELATIVE_BASES, TARGET_MODES   # noqa: F401
_FUTURE_COL = {"high": "future_high_max", "low": "future_low_min", "close": "future_close"}
_DEFAULT_CLIP = {"scaled": 10.0, "volnorm": 10.0}   # نفس قصّ خط الأنابيب لوضعه القديم المقاس بـ IQR؛ الباقي ±1
# أوضاع بوحدة العائد: تُضرب في REG_TARGET_SCALE (من البيانات) بعد القصّ كخط الأنابيب، فيبقى مقياس أهداف النموذج
# واحداً بين None و"return" و"relative"… scaled/volnorm بوحدة التقلّب أصلاً فلا تُضرب (مقياسها 1).
_RETURN_UNIT_BASES = ("return", "return_close", "magnitude", "entry_range")
# تعريفات انحدار close في entry_range: "abs_return" = |عائد الإغلاق من P| (يُضرب في المقياس كغيره)،
# "range_pos" = موقع الإغلاق في المدى [0,1] (لا يُضرب: وحدته O(1) أصلاً). كل قسم يُختم بمقياس كل هدف
# (reg_target_scales) وبالتعريف (entry_close_reg) فلا يخمّن المستهلك (reg_scale_of(split, target) في القسم ٣).


def _parse_mode(mode):
    """'scaled+relative' ← ('scaled', True)؛ 'relative' ← ('return', True)."""
    base, _, suffix = mode.partition("+")
    if base == "relative" and not suffix:
        return "return", True
    if base not in TARGET_MODES or suffix not in ("", "relative"):
        raise ValueError(f"mode غير معروف: {mode!r} — المتاح: {TARGET_MODES}، ويمكن إلحاق '+relative' بأيّها")
    if suffix and base in _NO_RELATIVE_BASES:
        raise ValueError(f"{mode!r} غير مدعوم: الوسيط المقطعي لا يُطبَّق على موقع الإغلاق (pos)، والناتج لا يُعكس إلى "
                         f"سعر — استخدم {base!r} وحده")
    return base, suffix == "relative"


def _split_parts(split_or_dict):
    """قسم واحد ← [(None, قسم)]؛ قاموس عملات (test) ← [(اسم, قسم), ...]."""
    return [(None, split_or_dict)] if "y" in split_or_dict else list(split_or_dict.items())


def _price_columns(parts):
    lc = np.concatenate([np.asarray(s["last_candles"], dtype="float64") for _, s in parts])
    P = {name: lc[:, i] for i, name in enumerate(LAST_COLUMNS)}
    if all("base_params" in s for _, s in parts):
        bp = np.concatenate([np.asarray(s["base_params"], dtype="float64") for _, s in parts])
        P["bp_center"], P["bp_scale"] = bp[:, 0], bp[:, 1]
    return P


def _raw_target(P, t, base, close_reg="abs_return"):
    """قيمة الهدف المستمرة قبل أي معالجة مقطعية. close_reg: تعريف close في entry_range فقط."""
    fut = P[_FUTURE_COL[t]]
    if base == "return":
        return fut / P[f"last_{t}"] - 1.0
    if base == "return_close":
        return fut / P["last_close"] - 1.0
    if base == "scaled":
        if "bp_scale" not in P:
            raise ValueError("وضع 'scaled' يتطلّب base_params في كل الأقسام")
        scale = np.where(np.abs(P["bp_scale"]) > 1e-12, P["bp_scale"], 1e-8)   # حماية من القسمة على صفر
        return (fut - P[f"last_{t}"]) / scale
    if base == "volnorm":
        # عائد نفس النوع (كـ return) ÷ تقلّب النافذة النسبي (IQR إغلاقات النافذة ÷ آخر إغلاق) — كلاهما معروف وقت
        # الدخول، فلا تسرّب (كـ scaled: لا يُقرأ مركز base_params، وهو وسيط النافذة). قفزة +50% لعملة
        # تقلّبها 20% تصبح 2.5 لا حدثاً يهيمن على الترتيب — يطابق محفظة rank_weighted_vol.
        if "bp_scale" not in P:
            raise ValueError("وضع 'volnorm' يتطلّب base_params في كل الأقسام")
        vol = np.abs(P["bp_scale"]) / np.maximum(np.abs(P["last_close"]), 1e-12)
        vol = np.where(vol > 1e-6, vol, np.nanmedian(vol[vol > 1e-6]) if np.any(vol > 1e-6) else 1.0)
        return (fut / P[f"last_{t}"] - 1.0) / vol
    if base == "magnitude":
        r = fut / P["last_close"] - 1.0
        return {"close": np.abs(r), "high": np.maximum(r, 0.0), "low": np.maximum(-r, 0.0)}[t]
    if base == "entry_range":
        # مقادير غير سالبة من سعر الدخول؛ الاتجاه في رؤوس التصنيف وحدها. max(·، 0) يُطبَّق هنا (قبل القصّ والمقياس)
        # فتبقى الفجوات (قمة الأفق تحت P أو قاعه فوقه) صفراً لا قيمة سالبة.
        entry, hi, lo = P["last_close"], P["future_high_max"], P["future_low_min"]
        if t == "high":
            return np.maximum(hi / entry - 1.0, 0.0)
        if t == "low":
            return np.maximum(1.0 - lo / entry, 0.0)
        if close_reg == "abs_return":
            return np.abs(P["future_close"] / entry - 1.0)
        rng = hi - lo                          # range_pos: موقع الإغلاق داخل المدى المحقَّق
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(rng > 0, (P["future_close"] - lo) / rng, 0.5)
    raise ValueError(f"وضع غير معروف: {base!r}")


def _group_keys(ts, group_freq):
    if not group_freq:
        return ts
    return pd.to_datetime(ts.astype("int64")).floor(group_freq).asi8


def _targets_for(parts, targets, base, cross, clip, center, group_freq, thresholds, scale=1.0,
                 close_reg="abs_return"):
    """يُرجع ({y_key: مصفوفة مدمجة}, معلومات) لكل أجزاء قسم واحد معاً. info["scales"]: مقياس كل هدف فعلاً."""
    P = _price_columns(parts)
    out, info = {}, {"P": P, "scales": {}}
    keys = _group_keys(P["timestamp"], group_freq) if cross else None
    if keys is not None:
        info["group_size"] = pd.Series(keys).groupby(keys).transform("size").to_numpy()
    for t in targets:
        r = _raw_target(P, t, base, close_reg)
        if cross:
            c = pd.Series(r).groupby(keys).transform(center).to_numpy()
            reg, up = r - c, r > c
        elif base == "magnitude":
            reg, up = r, r > thresholds[t]
        elif base == "entry_range":
            # التسمية = اتجاه من نوع الهدف نفسه كخط الأنابيب (لا من الانحدار: المقدار لا يحمل اتجاهاً)
            reg, up = r, P[_FUTURE_COL[t]] > P[f"last_{t}"]
        else:
            reg, up = r, r > 0
        t_scale = 1.0 if (base == "entry_range" and t == "close" and close_reg == "range_pos") else scale
        info["scales"][t] = t_scale
        out[f"y_{t}_reg"] = (np.clip(reg, -clip, clip) * t_scale).astype("float32")   # القصّ بوحدة الهدف ثم الضرب
        # الأوضاع القديمة بترميز ±1 (يحوّله _to_unit_label قبل التدريب)؛ entry_range بترميز 1/0 مباشرة كخط الأنابيب
        out[f"y_{t}_class"] = np.where(up, 1.0, 0.0 if base == "entry_range" else -1.0).astype("float32")
    return out, info


def _rebuild(parts, new_y, keep, mode, scale=1.0, scales=None, stamps=None):
    """يوزّع المصفوفات المدمجة على الأجزاء الأصلية؛ نسخ سطحية — X لا يُنسَخ."""
    rebuilt, start = [], 0
    for name, s in parts:
        n = len(s["last_candles"])
        mask = keep[start:start + n]
        ns = dict(s)
        ns["y"] = dict(s["y"])
        for k, v in new_y.items():
            ns["y"][k] = v[start:start + n]
        if not mask.all():                   # يحدث فقط مع drop_small_groups=True (ينسخ X لهذا الجزء)
            ns = {k: ({kk: vv[mask] for kk, vv in v.items()} if k == "y"
                      else (v[mask] if hasattr(v, "__len__") and not isinstance(v, (str, dict)) and len(v) == n else v))
                  for k, v in ns.items()}
        ns["target_mode"] = mode
        ns["reg_target_scale"] = scale
        ns["reg_target_scales"] = dict(scales or {})   # مقياس كل هدف (قد يختلف عن المشترك: close في entry_range)
        ns.pop("entry_close_reg", None)                # ختم وضع سابق (entry_range) لا يبقى على وضع آخر
        ns.update(stamps or {})
        rebuilt.append((name, ns))
        start += n
    return rebuilt[0][1] if rebuilt[0][0] is None else dict(rebuilt)


def retarget_splits(train_split, val_split, test_split, mode="return", targets=None, clip=None,
                    center="median", group_freq=None, min_group=5, drop_small_groups=False, close_reg="abs_return",
                    verbose=True, reg_target_scale=1.0, price_targets=None):
    """يُرجع (train, val, test) جديدة بأهداف الوضع المطلوب. الأصلية لا تُعدَّل.

    clip            : حدّ قصّ هدف الانحدار (±). الافتراضي 10 لـ scaled و1 لغيره — نفس خط الأنابيب.
    group_freq      : لـ +relative — تقريب الطوابع الزمنية قبل التجميع (مثلاً "1D" أو "4h") حين لا تتطابق
                      طوابع العملات تماماً (stride على فريم الساعة قد يُزيح بدايات العملات).
    min_group       : أقل عدد عملات في الطابع الزمني ليكون «متوسط السوق» ذا معنى (+relative).
    drop_small_groups: True يحذف عيّنات المجموعات الأصغر من min_group (ينسخ X — ذاكرة إضافية).
    close_reg       : تعريف انحدار close في entry_range: "abs_return" | "range_pos" (TargetSettings.entry_close_reg).
    reg_target_scale: مقياس أهداف الانحدار من البيانات (DatasetInfo.reg_target_scale؛ يُضرب فيه بعد القصّ لأوضاع وحدة العائد)؛
                      1.0 = عائد خام (ملف بلا المفتاح).
    price_targets   : الأهداف المفعّلة (ModelPlan.price_targets)؛ None = أهداف CONFIG (كل ما في y)."""
    base, cross = _parse_mode(mode)
    clip = _DEFAULT_CLIP.get(base, 1.0) if clip is None else clip
    scale = float(reg_target_scale) if base in _RETURN_UNIT_BASES else 1.0
    first = _split_parts(train_split)[0][1]
    known = price_targets or list(CONFIG.get("targets", ("high", "low", "close")))
    if base == "entry_range" and not targets:
        # كل أهداف y لا المفعّلة فقط: إعادة تشغيل الخلية بعد القسم ٤ (close معلّق) كانت ستُبقي y_close بوضع سابق
        # تحت ختم entry_range؛ إلغاء التعليق لاحقاً يجده جاهزاً بمعناه الجديد
        known = ("high", "low", "close")
    targets = list(targets or [t for t in known if f"y_{t}_reg" in first["y"] or f"y_{t}_class" in first["y"]])
    close_reg = "abs_return" if close_reg is None else str(close_reg)
    if base == "entry_range" and close_reg not in ENTRY_CLOSE_REGS:
        raise ValueError(f"close_reg غير معروف: {close_reg!r} — المتاح: {ENTRY_CLOSE_REGS}")
    stamps = {"entry_close_reg": close_reg} if base == "entry_range" else {}
    if base == "entry_range" and close_reg == "range_pos":
        stamps_scale_note = "close = موقع في المدى بلا مقياس"
    else:
        stamps_scale_note = "كل الرؤوس مقادير غير سالبة بوحدة العائد × المقياس"

    thresholds = None
    if base == "magnitude" and not cross:   # العتبة من train وحده ثم تُطبَّق كما هي على val وtest
        P_tr = _price_columns(_split_parts(train_split))
        thresholds = {t: float(np.median(_raw_target(P_tr, t, base))) for t in targets}

    results, report = [], []
    for name, sp in (("train", train_split), ("val", val_split), ("test", test_split)):
        parts = _split_parts(sp)
        new_y, info = _targets_for(parts, targets, base, cross, clip, center, group_freq, thresholds, scale,
                                   close_reg)
        n = len(next(iter(new_y.values())))
        keep = np.ones(n, dtype=bool)
        if "group_size" in info:
            small = info["group_size"] < min_group
            report.append((name, "مجموعات أصغر من min_group", f"{small.mean():.2%}",
                           f"وسيط حجم المجموعة {int(np.median(info['group_size']))}"))
            if drop_small_groups:
                keep = ~small
        if base == "scaled":
            P = info["P"]
            report.append((name, "وسيط IQR/السعر", f"{np.median(P['bp_scale'] / P['last_close']):.4f}", ""))
            clipped = np.mean([np.mean(np.abs(new_y[f'y_{t}_reg']) >= clip * scale) for t in targets])
            report.append((name, f"نسبة القصّ عند ±{clip:g}", f"{clipped:.2%}", ""))
        if mode == "return" and parts[0][1].get("target_mode") in (None, "return"):
            # فحص ذاتي: يجب أن يطابق ما بناه خط الأنابيب (فقط إن كانت y الحالية منه لا من وضع آخر)
            for t in targets:
                k = f"y_{t}_class"
                old = np.concatenate([np.asarray(s["y"][k]).ravel() for _, s in parts]) if k in parts[0][1]["y"] else None
                if old is not None:
                    report.append((name, f"تطابق {k} مع خط الأنابيب", f"{np.mean((old > 0) == (new_y[k] > 0)):.2%}", ""))
        for t in targets:
            yc, yr = new_y[f"y_{t}_class"][keep], new_y[f"y_{t}_reg"][keep]
            report.append((name, f"{t}: نسبة الفئة +1", f"{np.mean(yc > 0):.3f}",
                           f"reg: متوسط {np.mean(yr):+.4f} | انحراف {np.std(yr):.4f}"))
        results.append(_rebuild(parts, new_y, keep, mode, scale, info["scales"], stamps))

    if verbose:
        print(f"🎯 وضع الهدف: {mode!r} — الأهداف: {targets} — قصّ الانحدار ±{clip:g} × مقياس {scale:g}"
              + (f" — عتبات magnitude من train: { {t: round(v, 5) for t, v in thresholds.items()} }" if thresholds else "")
              + (f" — close_reg={close_reg!r}: {stamps_scale_note}" if stamps else ""))
        with pd.option_context("display.width", 200, "display.max_colwidth", 60):
            print(pd.DataFrame(report, columns=["split", "فحص", "قيمة", "ملاحظة"]).to_string(index=False))
        if cross and any(r[1].startswith("مجموعات") and float(r[2].rstrip("%")) > 5 for r in report):
            print("⚠️ أكثر من 5% من العيّنات في طوابع زمنية بعملات قليلة — جرّب group_freq (مثلاً '1D') أو drop_small_groups=True")
        if base == "entry_range":
            print("ℹ️ entry_range: القسم ٦ (chicks) يفكّ high/low من آخر إغلاق، وclose باتجاه رأس التصنيف (abs_return) أو موقعاً في المدى (range_pos)، و٧-ب "
                  "يُتخطّى. ⚠️ رؤوس المقدار تتبع التقلّب في معظمها: قارن دائماً بخط أساس التقلّب وحده (hourly_1h.md §١٠).")
        elif mode != "return":
            print("ℹ️ الأقسام ٦ (chicks) و٧-ب (الثقة/الصفقات) تفترض 'return' وستُتخطّى. ٧-ج و٧-د و٧-هـ تعمل على أي هدف.")
    return tuple(results)


def entry_range_to_prices(last_close, high=None, low=None, close=None, close_reg="abs_return", p_close_up=None):
    """عكس entry_range إلى أسعار — المكان الوحيد لهذه الصيغة في الدفتر (collect_signals وreal_price_predictions يستدعيانه).
    الأهداف بوحدة العائد (مخرَج النموذج ÷ reg_scale_of(split, هدف))، وP = last_close:
        high  → P·(1 + high)،  low → P·(1 − low)
        close (abs_return) → P·(1 + s·close) حيث s = +1 إن كان p_close_up ≥ 0.5 وإلا −1، ويُرجَع الاتجاهان أيضاً:
                              close_up = P·(1 + close)، close_down = P·(1 − close). بلا p_close_up لا يُرجَع "close".
        close (range_pos)  → pred_low + clip(close, 0, 1)·(pred_high − pred_low) — يحتاج high وlow معاً.
    يُرجع {اسم: سعر} لما أمكن حسابه فقط."""
    P = np.asarray(last_close, dtype="float64")
    out = {}
    if high is not None:
        out["high"] = P * (1.0 + np.asarray(high, dtype="float64"))
    if low is not None:
        out["low"] = P * (1.0 - np.asarray(low, dtype="float64"))
    if close is not None:
        c = np.asarray(close, dtype="float64")
        if close_reg == "range_pos":
            if "high" in out and "low" in out:
                out["close"] = out["low"] + np.clip(c, 0.0, 1.0) * (out["high"] - out["low"])
        elif close_reg == "abs_return":
            out["close_up"], out["close_down"] = P * (1.0 + c), P * (1.0 - c)
            if p_close_up is not None:
                out["close"] = np.where(np.asarray(p_close_up, dtype="float64") >= 0.5, out["close_up"], out["close_down"])
        else:
            raise ValueError(f"close_reg غير معروف: {close_reg!r} — المتاح: {ENTRY_CLOSE_REGS}")
    return out
