"""
PURPOSE:  Shared imports of the evaluation package plus TargetSpec/resolve_targets (which targets a model outputs: continuous or categorical), NIG uncertainty bounding and the Wilson CI.
TAGS:     TargetSpec, resolve_targets, make_categorical_spec, get_model_target_names, nig_uncertainty_bounded, wilson_ci, imports, shared namespace
PITFALLS: Loaded first: later modules use its imports (re, warnings, np, tf, pd, dataclass, typing names) without importing them. Executed into the one shared evaluation namespace by evaluation/_loader.py (never imported on its own): names from other modules resolve at call time.

## 1️⃣ الإعداد الأساسي: TargetSpec (وصف الأهداف المستمرة/الفئوية)
"""
# ═══════════════════════════════════════════════════════════════════════════════
# 🔧 الإصدار v4.2: TargetSpec الموحّد + resolve_targets (أهداف نصية / جزئية بمرونة)
# ═══════════════════════════════════════════════════════════════════════════════
#
# الجديد في هذا الإصدار:
# 1. ✅ TargetSpec بحقول النسخة الجديدة (reference / magnitude_weight / clip_percentile)
#    مع مزامنة entry_col <-> price_index تلقائياً.
# 2. ✅ resolve_targets: تقبل الأهداف كنصوص ('close' أو 'high,low' أو ['high','low'])
#    أو TargetSpec أو dict أو None/'all' (= كل ما يُخرجه النموذج).
# 3. ✅ إن طلبتَ هدفاً أو هدفين والنموذج يُخرج ثلاثة: يُتجاهل الباقي تماماً
#    (لا يُقرأ ولا يُفكّ ولا يُقيَّم ولا يظهر في أي تقرير).
# 4. ✅ توافق كامل مع الدوال القديمة.
# ═══════════════════════════════════════════════════════════════════════════════

import re
import warnings
import numpy as np
import tensorflow as tf
import pandas as pd
from dataclasses import dataclass, field, replace
from typing import Tuple, Dict, Optional, List, Union, Literal, Iterable


# ─────────────────────────────────────────────────────────────────────────────
# 📋 وصف الأهداف (Target Specification)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TargetSpec:
    """
    وصف موحّد لأي هدف تنبؤ (مستمر أو فئوي)، يجمع بين واجهة النسخة الأصلية
    (kind / class_names / price_index / has_uncertainty) وحقول النسخة الجديدة
    (reference / magnitude_weight / clip_percentile).

    ملاحظة: entry_col و price_index نفس المفهوم، ويتم مزامنتهما تلقائيًا.
    ملاحظة: reference / magnitude_weight / clip_percentile محفوظة هنا للتوافق مع
            كود التدريب؛ دوال التقييم في هذا الدفتر لا تستخدمها.
    """
    name: str
    kind: Literal['continuous', 'categorical'] = 'continuous'
    class_names: Optional[List[str]] = None
    has_uncertainty: bool = True

    # مرجع سعر الدخول (index داخل ['high','low','close'])
    price_index: Optional[int] = None
    entry_col: Optional[int] = None      # اسم بديل لـ price_index

    # القيمة الحقيقية المستقبلية
    future_col: Optional[int] = None

    # إعدادات حساب الاتجاه/المقدار
    reference: str = 'entry'             # 'entry' أو 'median'
    magnitude_weight: float = 0.1
    clip_percentile: float = 99.0

    # True: المخرج الخام عائد نسبة لسعر دخول *هذا الهدف نفسه* (last_candles[:, price_index])،
    # فيُفكّ بـ entry × (1 + raw) — مطابق لـ invert_reg_predictions في خط الأنابيب عند
    # reg_target_mode='return' (high نسبة لآخر high، low لآخر low، close لآخر close).
    # False (الافتراضي، السلوك القديم): (raw × iqr) + median من base_params المشتركة بين الأهداف —
    # base_params واحدة لكل الأهداف لا تستطيع تمثيل مرجع مختلف لكل هدف.
    relative_to_entry: bool = False
    # مع relative_to_entry: المخرج الخام = عائد × reg_scale (reg_target_scale في بيانات خط الأنابيب)، فيُفكّ
    # بـ entry × (1 + raw / reg_scale). 1.0 = عائد خام (السلوك القديم). سالب = مقدار هبوط بإشارة موجبة:
    # entry × (1 − raw / |reg_scale|) (low في وضع entry_range بدفتر main).
    reg_scale: float = 1.0

    # موقع في المدى: المخرج الخام موقع [0,1] بين السعرين المتوقَّعين لهدفين آخرين، range_of = (اسم_القمة، اسم_القاع):
    # pred_real = القاع + clip(raw, 0, 1) × (القمة − القاع). close في entry_range بدفتر main مع ENTRY_CLOSE_REG='range_pos'. يُفكّ بعد الهدفين،
    # فيجب أن يكونا ضمن الأهداف نفسها. None = فكّ مستقل (السلوك القديم).
    range_of: Optional[Tuple[str, str]] = None

    # رأس التصنيف لهذا الهدف: class_key = مفتاح مخرَج النموذج الذي يحمل P(صعود) بعد sigmoid (مثل 'y_close_class_logits'). يُقرأ
    # إن وُجد ويظهر في decode كـ p_up (خاصية تحليل؛ غيابه لا يكسر شيئاً ما لم يكن signed_by_class).
    # signed_by_class=True: المخرج الخام *مقدار* ≥ 0 (close مع abs_return في entry_range بدفتر main) والاتجاه من هذا الرأس وحده:
    # pred_real = entry × (1 + s·raw/reg_scale)، s = +1 إن كان p_up ≥ 0.5 وإلا −1 — نفس entry_range_to_prices في main.
    # يستلزم relative_to_entry وclass_key؛ وبلا اتجاه معروف لا سعر له (NaN) فلا يدخل التقييم.
    class_key: Optional[str] = None
    signed_by_class: bool = False

    def __post_init__(self):
        # 1) فحص الأهداف الفئوية
        if self.kind == 'categorical' and not self.class_names:
            raise ValueError(
                f"الهدف '{self.name}' فئوي لكنه بدون class_names."
            )

        # 2) مزامنة entry_col <-> price_index
        default_price_index = {'close': 2, 'low': 1, 'high': 0}
        if self.price_index is None and self.entry_col is not None:
            self.price_index = self.entry_col
        if self.entry_col is None and self.price_index is not None:
            self.entry_col = self.price_index
        if self.price_index is None:
            self.price_index = default_price_index.get(self.name)
            self.entry_col = self.price_index

        # 3) future_col الافتراضي حسب الاسم
        if self.future_col is None:
            self.future_col = {'close': 4, 'low': 5, 'high': 6}.get(self.name)

        # 4) فحص reference
        assert self.reference in ('entry', 'median'), \
            f"reference غير صالحة: {self.reference}"

        # 5) الاتجاه من رأس التصنيف يحتاج مفتاح الرأس وفكّاً نسبة لسعر الدخول
        if self.signed_by_class and not (self.class_key and self.relative_to_entry):
            raise ValueError(f"الهدف '{self.name}': signed_by_class يحتاج class_key وrelative_to_entry=True.")


# ─── إعداد افتراضي متوافق مع النسخة الأصلية ───
DEFAULT_PRICE_TARGETS: List[TargetSpec] = [
    TargetSpec(name='high',  kind='continuous', has_uncertainty=True, price_index=0, future_col=6),
    TargetSpec(name='low',   kind='continuous', has_uncertainty=True, price_index=1, future_col=5),
    TargetSpec(name='close', kind='continuous', has_uncertainty=True, price_index=2, future_col=4),
]


def make_categorical_spec(name: str, class_names: List[str], has_uncertainty: bool = False) -> TargetSpec:
    """اختصار لإنشاء TargetSpec لهدف فئوي (مثال: اتجاه السوق: صعود/هبوط/تذبذب)."""
    return TargetSpec(name=name, kind='categorical', class_names=class_names, has_uncertainty=has_uncertainty)


# ─────────────────────────────────────────────────────────────────────────────
# 🧰 أدوات صغيرة مشتركة (آمنة مع NaN وأعمدة last_candles الناقصة)
# ─────────────────────────────────────────────────────────────────────────────

def _safe_col(arr, idx):
    """عمود idx من مصفوفة ثنائية، أو None إن لم يوجد (مثال: last_candles في التداول الحي)."""
    if arr is None or idx is None:
        return None
    a = np.asarray(arr)
    if a.ndim < 2 or a.shape[1] <= idx:
        return None
    return a[:, idx]


def _spec_base(spec, median, iqr, last_candles, limit=None):
    """(offset, scale) لفك تشفير هدف مستمر: pred_real = raw × scale + offset.
    relative_to_entry → (entry, entry / reg_scale) أي entry × (1 + raw / reg_scale)؛ وإلا (median, iqr) من base_params."""
    if getattr(spec, 'relative_to_entry', False):
        col = _safe_col(last_candles[:limit] if (last_candles is not None and limit is not None)
                        else last_candles, spec.price_index)
        if col is None:
            raise ValueError(f"الهدف '{spec.name}' (relative_to_entry) يحتاج last_candles بعمود السعر {spec.price_index}.")
        entry = np.asarray(col, dtype=np.float64)
        return entry, entry / float(getattr(spec, 'reg_scale', 1.0))
    return median, iqr


def _nanmean(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    m = np.isfinite(x)
    return float(x[m].mean()) if m.any() else float('nan')


def _nanmax(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    m = np.isfinite(x)
    return float(x[m].max()) if m.any() else float('nan')


# ─────────────────────────────────────────────────────────────────────────────
# 🛡️ عدم يقين NIG محدود + فاصل Wilson
# ─────────────────────────────────────────────────────────────────────────────
# NIG_UNC_MAX: الحدّ الأعلى لـ aleatoric/epistemic الخام (بوحدة y_*_reg، قبل الضرب بـ |s_iqr|). القيم الطبيعية (< ~0.5) لا
# تتغيّر. السبب: β = softplus + beta_min بلا سقف فيكبر √(β/(α−1)) بلا حدّ لعيّنة خارج التوزيع (وα → 1 لو خُفِّضت alpha_min).
# غيّره قبل الاستدعاء (NIG_UNC_MAX = 50.0)؛ float('inf') = بلا قصّ. يوافق unc_max في NIGUncertainty (model_v2).
NIG_UNC_MAX = 20.0
NIG_ALPHA_DEN_MIN = 1e-2     # أرضية (α−1) في المقام؛ لا أثر لها حين α ≥ 1.01


def _bound_unc(x, cap):
    """قصّ عدم اليقين عند cap؛ غير المنتهي (NaN/±∞) يصير cap. يُرجع (قيم، قناع ما قُصّ)."""
    x = np.asarray(x, dtype=np.float64)
    if cap is None or not np.isfinite(cap):
        return x, np.zeros(x.shape, dtype=bool)
    bad = ~np.isfinite(x)
    clipped = bad | (x > cap)
    return np.where(bad, cap, np.minimum(x, cap)), clipped


def nig_uncertainty_bounded(nu, alpha, beta, unc_max=None, alpha_den_min=NIG_ALPHA_DEN_MIN):
    """(epistemic, aleatoric, clipped) من معاملات NIG — نفس صيغة NIGUncertainty في model_v2:
    aleatoric = √(β/max(α−1, ε))، epistemic = √(β/(ν·max(α−1, ε)))، ثم قصّ عند unc_max (None ← NIG_UNC_MAX)."""
    cap = NIG_UNC_MAX if unc_max is None else float(unc_max)
    nu, alpha, beta = (np.asarray(v, dtype=np.float64) for v in (nu, alpha, beta))
    a1 = np.maximum(alpha - 1.0, alpha_den_min)
    with np.errstate(all='ignore'):
        ale = np.sqrt(beta / a1)
        epi = np.sqrt(beta / (nu * a1))
    ale, c1 = _bound_unc(ale, cap)
    epi, c2 = _bound_unc(epi, cap)
    return epi, ale, (c1 | c2)


def wilson_ci(k, n, z: float = 1.96):
    """فاصل Wilson 95% لنسبة k/n → (low, high) بين 0 و1؛ n = 0 ← (NaN, NaN). صالح لـ n صغير ونسب قرب 0/1 (عكس Wald)."""
    n = float(n)
    if n <= 0:
        return float('nan'), float('nan')
    p = float(k) / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / den
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


# ─────────────────────────────────────────────────────────────────────────────
# 🎯 resolve_targets: تحويل أي صيغة أهداف إلى قائمة TargetSpec
# ─────────────────────────────────────────────────────────────────────────────

_UNCERTAINTY_SUFFIXES = ('_epistemic', '_aleatoric', '_confidence')
_ALL_KEYWORDS = {'all', '*', 'الكل', 'كل', 'كلها'}


def _target_name_key(name) -> str:
    """'Close' / ' y_close ' → 'close'."""
    n = str(name).strip().lower()
    return n[2:] if n.startswith('y_') else n


def _default_spec(name: str) -> Optional[TargetSpec]:
    for s in DEFAULT_PRICE_TARGETS:
        if s.name == name:
            return replace(s)          # نسخة مستقلة كي لا نُعدّل الافتراضي بالخطأ
    return None


def _flatten_targets(obj):
    """يُسطّح أي تركيبة (نص، قائمة، مجموعة، مصفوفة، TargetSpec، dict) إلى عناصر مفردة."""
    if obj is None:
        return
    if isinstance(obj, (TargetSpec, dict)):
        yield obj
    elif isinstance(obj, str):
        for part in re.split(r'[,\s;|+/،&]+', obj.strip()):
            if part:
                yield part
    elif isinstance(obj, (set, frozenset)):
        for o in sorted(obj, key=str):
            yield from _flatten_targets(o)
    elif isinstance(obj, (list, tuple)) or hasattr(obj, 'tolist'):
        seq = obj if isinstance(obj, (list, tuple)) else obj.tolist()
        for o in seq:
            yield from _flatten_targets(o)
    else:
        raise TypeError(f"نوع هدف غير مدعوم: {type(obj).__name__} ({obj!r})")


def _is_all_keyword(item) -> bool:
    return isinstance(item, str) and item.strip().lower() in _ALL_KEYWORDS


def resolve_targets(targets=None, available: Optional[Iterable[str]] = None) -> List[TargetSpec]:
    """
    يحوّل أي صيغة أهداف إلى List[TargetSpec] مرتّبة وبلا تكرار.

    الصيغ المقبولة (ويمكن خلطها):
        None أو 'all'         → كل الأهداف (ما يُخرجه النموذج إن مُرِّر available، وإلا high/low/close)
        'close'               → هدف واحد
        'high,low' / 'high low' / ['high', 'low']
        TargetSpec(...) أو dict(name='trend', kind='categorical', class_names=[...])

    Args:
        available: أسماء الأهداف التي يُخرجها النموذج فعلاً (اختياري). إن مُرِّرت:
                   • يُتحقق أن كل هدف مطلوب موجود فيها (وإلا خطأ واضح).
                   • الأهداف غير المطلوبة تُتجاهل تلقائياً.
    """
    avail = [_target_name_key(a) for a in available] if available is not None else None

    items = list(_flatten_targets(targets))
    wants_all = (not items) or any(_is_all_keyword(i) for i in items)
    items = [i for i in items if not _is_all_keyword(i)]
    if wants_all:
        items += list(avail) if avail else [s.name for s in DEFAULT_PRICE_TARGETS]

    specs: List[TargetSpec] = []
    seen = set()
    for it in items:
        if isinstance(it, TargetSpec):
            spec = it
        elif isinstance(it, dict):
            spec = TargetSpec(**it)
        else:
            key = _target_name_key(it)
            spec = _default_spec(key)
            if spec is None:
                if avail is not None and key in avail:
                    warnings.warn(
                        f"الهدف '{key}' خارج (high/low/close): سيُعامَل كهدف مستمر بلا سعر دخول مرجعي. "
                        f"مرّر TargetSpec(name='{key}', price_index=..., future_col=...) لتحديدهما.",
                        stacklevel=3,
                    )
                    spec = TargetSpec(name=key, kind='continuous', has_uncertainty=True)
                else:
                    known = ', '.join(s.name for s in DEFAULT_PRICE_TARGETS)
                    raise ValueError(
                        f"هدف غير معروف: '{it}'. المعروف: {known}. "
                        f"للأهداف الأخرى مرّر TargetSpec(...) أو dict بالحقل name."
                    )
        if spec.name in seen:
            continue
        seen.add(spec.name)
        specs.append(spec)

    if not specs:
        raise ValueError("لم يُحدَّد أي هدف صالح.")

    if avail is not None:
        missing = [s.name for s in specs if s.name not in avail]
        if missing:
            raise ValueError(f"الأهداف {missing} غير موجودة في مخرجات النموذج. المتاح: {avail}")
    return specs


def get_model_target_names(model, X_inputs=None) -> Optional[List[str]]:
    """أسماء الأهداف التي يُخرجها النموذج فعلاً (مفاتيح 'y_<name>' بدون epistemic/aleatoric/confidence)."""
    keys = None
    names = getattr(model, 'output_names', None)
    if names and any(str(n).startswith('y_') for n in names):
        keys = list(names)
    elif X_inputs is not None:
        try:
            probe = tuple(np.asarray(x)[:1] for x in X_inputs)
            out = model(probe, training=False)
            if isinstance(out, dict):
                keys = list(out.keys())
        except Exception:
            keys = None
    if not keys:
        return None
    found = []
    for k in map(str, keys):
        if k.startswith('y_') and not k.endswith(_UNCERTAINTY_SUFFIXES):
            found.append(k[2:])
    return found or None


def _resolve_for_model(target_specs, model=None, X_inputs=None) -> List[TargetSpec]:
    """resolve_targets + استكشاف مخرجات النموذج فقط عند الحاجة (None/'all'/اسم غير معروف)."""
    items = list(_flatten_targets(target_specs))
    needs_probe = (not items) or any(_is_all_keyword(i) for i in items) or any(
        isinstance(i, str) and _default_spec(_target_name_key(i)) is None for i in items
    )
    avail = get_model_target_names(model, X_inputs) if (needs_probe and model is not None) else None
    return resolve_targets(target_specs, available=avail)
