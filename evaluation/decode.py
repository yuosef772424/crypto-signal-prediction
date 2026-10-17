"""
PURPOSE:  Stage 2: flexible decoding of raw model outputs into real units (continuous + categorical), decode_predictions_v4.
TAGS:     decode_predictions_v4, decode_predictions, softmax, scale offset, reg_target_scale, decode
PITFALLS: Executed into the one shared evaluation namespace by evaluation/_loader.py (never imported on its own): names from other modules resolve at call time.

## 4️⃣ المرحلة 2: فك التشفير المرن (مستمر + فئوي)
"""
# ─────────────────────────────────────────────────────────────────────────────
# المرحلة 2: فك التشفير المرن (مستمر + فئوي)
# ─────────────────────────────────────────────────────────────────────────────

def _softmax_stable(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def decode_predictions_v4(
    raw_preds: Dict[str, np.ndarray],
    target_specs: List['TargetSpec'],
    base_params: Optional[np.ndarray] = None,
    last_candles: Optional[np.ndarray] = None,
    limit: Optional[int] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    فك تشفير مرن لكل هدف حسب نوعه:

    • مستمر (continuous):
        pred_scaled  → القيمة الخام كما خرجت من النموذج (فضاء مُطبَّع)
        pred_real    → القيمة بعد فك التطبيع: (scaled * iqr) + median
        p_up         → (إن قُرئ رأس التصنيف) P(صعود)؛ و direction_sign = ±1 لأهداف signed_by_class

    • فئوي (categorical):
        pred_encoded → القيمة الخام: إما logits/probabilities [N, C] أو مؤشر فئة [N]
        pred_index   → مؤشر الفئة المتوقعة (argmax)
        pred_label   → اسم الفئة الأصلي (str) بعد فك الترميز عبر class_names
        pred_proba   → احتمال أعلى فئة (إن كانت المخرجات probabilities)

    Args:
        raw_preds: مخرجات predict_batch_v4
        target_specs: قائمة TargetSpec تصف كل هدف
        base_params: [N, 2] = (median, iqr) — مطلوب فقط للأهداف المستمرة
        last_candles: مصفوفة الشمعة الحالية/المستقبلية (اختياري، لأهداف الأسعار)
        limit: تحديد عدد العينات المُعالجة

    Returns:
        dict[target_name] -> dict بالحقول أعلاه (+ عدم اليقين إن وُجد)
    """
    target_specs = resolve_targets(target_specs)
    any_target = target_specs[0].name
    if limit is None:
        limit = raw_preds[any_target].shape[0]

    if base_params is not None:
        base_params = base_params[:limit]
        median = base_params[:, 0]
        iqr = base_params[:, 1]
    else:
        median = iqr = None

    if last_candles is not None:
        last_candles = last_candles[:limit]

    results: Dict[str, Dict[str, np.ndarray]] = {}

    # أهداف «موقع في المدى» (range_of) تُفكّ بعد الهدفين اللذين تقع بينهما
    ordered = ([s for s in target_specs if getattr(s, 'range_of', None) is None]
               + [s for s in target_specs if getattr(s, 'range_of', None) is not None])
    for spec in ordered:
        raw = raw_preds[spec.name][:limit]
        entry = {}
        range_of = getattr(spec, 'range_of', None) if spec.kind == 'continuous' else None
        if range_of is not None:
            missing = [n for n in range_of if n not in results or 'pred_real' not in results[n]]
            if missing:
                raise ValueError(f"الهدف '{spec.name}' موقع بين {range_of} ويحتاج فكّهما معه — ناقص: {missing}")
            s_median = results[range_of[1]]['pred_real']
            s_iqr = results[range_of[0]]['pred_real'] - s_median
        else:
            s_median, s_iqr = (_spec_base(spec, median, iqr, last_candles)
                               if spec.kind == 'continuous' else (median, iqr))

        # احتمال صعود رأس التصنيف (إن قُرئ)، ومنه اتجاه الأهداف «المقدار بلا اتجاه» (signed_by_class): s = ±1 فيصير
        # المقياس s·entry/reg_scale. p_up = NaN ← اتجاه NaN ← سعر NaN فتُستثنى العيّنة من التقييم بدل أن تُحسب صعوداً.
        p_up = direction = None
        if spec.kind == 'continuous' and f'{spec.name}_p_up' in raw_preds:
            p_up = np.asarray(raw_preds[f'{spec.name}_p_up'][:limit], dtype=np.float64).flatten()
        if spec.kind == 'continuous' and getattr(spec, 'signed_by_class', False):
            if p_up is None:
                raise ValueError(f"الهدف '{spec.name}' اتجاهه من رأس التصنيف ولا '{spec.name}_p_up' في raw_preds "
                                 f"(predict_batch_v4 يملؤه من TargetSpec.class_key).")
            direction = np.where(np.isfinite(p_up), np.where(p_up >= 0.5, 1.0, -1.0), np.nan)
            s_iqr = s_iqr * direction

        # ── عدم اليقين (مشترك بين النوعين إن وُجد) ──────────────────────────
        # |s_iqr|: عرض لا إشارة (reg_scale سالب لهدف هبوط، أو مدى متوقَّع مقلوب)
        if spec.has_uncertainty and f'{spec.name}_epistemic' in raw_preds:
            epi = raw_preds[f'{spec.name}_epistemic'][:limit].flatten()
            ale = raw_preds[f'{spec.name}_aleatoric'][:limit].flatten()
            entry['epistemic'] = epi if s_iqr is None else epi * np.abs(s_iqr)
            entry['aleatoric'] = ale if s_iqr is None else ale * np.abs(s_iqr)
            total_scaled = np.sqrt(epi ** 2 + ale ** 2)
            entry['uncertainty_real'] = total_scaled if s_iqr is None else total_scaled * np.abs(s_iqr)
            entry['uncertainty_scaled'] = total_scaled
            entry['confidence'] = raw_preds[f'{spec.name}_confidence'][:limit].flatten()
            if f'{spec.name}_unc_clipped' in raw_preds:      # ما قُصّ بحدّ NIG_UNC_MAX (عيّنات خارج التوزيع)
                entry['unc_clipped'] = np.asarray(raw_preds[f'{spec.name}_unc_clipped'][:limit]).flatten().astype(bool)

        # ── فك التشفير حسب النوع ────────────────────────────────────────────
        if spec.kind == 'continuous':
            if s_median is None or s_iqr is None:
                raise ValueError(f"الهدف المستمر '{spec.name}' يحتاج base_params (median, iqr) لفك التشفير.")
            raw_flat = raw.flatten()
            pred_real = ((np.clip(raw_flat, 0.0, 1.0) if range_of is not None else raw_flat) * s_iqr) + s_median

            entry_price = np.zeros_like(pred_real)
            target_real = np.full_like(pred_real, np.nan)
            # أعمدة last_candles قد تكون ناقصة في التداول الحي (بلا أعمدة المستقبل) — نتعامل معها بأمان
            _entry_col = _safe_col(last_candles, spec.price_index)
            if _entry_col is not None:
                entry_price = np.asarray(_entry_col, dtype=np.float64)
            _future_col = _safe_col(last_candles, spec.future_col)
            if _future_col is not None:
                target_real = np.asarray(_future_col, dtype=np.float64)

            change_from_entry = ((pred_real - entry_price) / (np.abs(entry_price) + 1e-7)) * 100
            # طابع زمني لكل عيّنة (عمود 3 في تخطيط last_candles لخط الأنابيب) — لتجميع الصفقات
            # المتزامنة في محفظة واحدة لكل فترة بدل سلسلة صفقات متتالية وهمية.
            _ts_col = _safe_col(last_candles, 3) if (last_candles is not None and np.ndim(last_candles) == 2
                                                     and last_candles.shape[1] >= 7) else None

            entry.update({
                'kind': 'continuous',
                'pred_scaled': raw_flat,
                'pred_real': pred_real,
                'entry_price': entry_price,
                'change_from_entry': change_from_entry,
                'target_real': target_real,
                'median': s_median,
                'iqr': s_iqr,
                **({'p_up': p_up} if p_up is not None else {}),
                **({'direction_sign': direction} if direction is not None else {}),
                'timestamp': (np.asarray(_ts_col, dtype=np.float64) if _ts_col is not None
                              else np.full_like(pred_real, np.nan)),
            })

        elif spec.kind == 'categorical':
            n_classes = len(spec.class_names)

            if raw.ndim == 1 or (raw.ndim == 2 and raw.shape[1] == 1):
                # النموذج أخرج مؤشر الفئة مباشرة (وليس احتمالات)
                pred_index = raw.flatten().astype(int)
                pred_proba = np.full_like(pred_index, np.nan, dtype=np.float32)
                raw_encoded = pred_index
            else:
                if raw.shape[1] != n_classes:
                    raise ValueError(
                        f"عدد أعمدة مخرجات '{spec.name}' ({raw.shape[1]}) لا يطابق "
                        f"عدد الفئات المُعرَّفة ({n_classes})."
                    )
                # نتحقق إن كانت logits أو probabilities (مجموع الصف ≈ 1)
                row_sums = raw.sum(axis=1)
                looks_like_proba = np.allclose(row_sums, 1.0, atol=1e-3)
                proba = raw if looks_like_proba else _softmax_stable(raw, axis=1)

                pred_index = np.argmax(proba, axis=1)
                pred_proba = proba[np.arange(len(proba)), pred_index]
                raw_encoded = raw

            pred_label = np.array([spec.class_names[i] for i in pred_index], dtype=object)

            entry.update({
                'kind': 'categorical',
                'pred_encoded': raw_encoded,
                'pred_index': pred_index,
                'pred_label': pred_label,
                'pred_proba': pred_proba,
                'class_names': spec.class_names,
            })
        else:
            raise ValueError(f"نوع هدف غير مدعوم: {spec.kind}")

        results[spec.name] = entry

    return {spec.name: results[spec.name] for spec in target_specs}   # بترتيب الأهداف المطلوب


# دالة قديمة تبقى للتوافق الخلفي
def decode_predictions(raw_preds, base_params, last_candles=None, limit=None):
    """(نسخة سابقة - محفوظة للتوافق) فك تشفير high/low/close المستمرة فقط."""
    decoded = decode_predictions_v4(raw_preds, DEFAULT_PRICE_TARGETS, base_params, last_candles, limit)
    # إعادة تسمية الحقول لتطابق الواجهة القديمة تماماً
    for t, d in decoded.items():
        d['pred_scaled'] = d.pop('pred_scaled')
    return decoded
