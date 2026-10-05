"""
PURPOSE:  Stage 1: batched model prediction for any set of continuous/categorical targets (predict_batch_v4; predict_batch kept for compatibility).
TAGS:     predict_batch_v4, predict_batch, batched predict, model outputs
PITFALLS: Executed into the one shared evaluation namespace by evaluation/_loader.py (never imported on its own): names from other modules resolve at call time.

## 3️⃣ المرحلة 1: التنبؤ على الدفعات (مرن)
"""
# ─────────────────────────────────────────────────────────────────────────────
# المرحلة 1: التنبؤ على الدفعات (مرن لأي عدد/نوع أهداف)
# ─────────────────────────────────────────────────────────────────────────────

def predict_batch_v4(
    model,
    X_inputs: Tuple[np.ndarray, ...],
    target_specs: List['TargetSpec'],
    batch_size: int = 256,
    verbose: bool = False,
    unc_max: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    تنبؤ على الدفعات لأي مجموعة أهداف (مستمرة و/أو فئوية).

    unc_max: حدّ عدم اليقين (None ← NIG_UNC_MAX). إن أخرج النموذج y_{name}_nu/alpha/beta يُعاد حساب epistemic/aleatoric
             منها محدودَين (nig_uncertainty_bounded) — فيسري الحدّ على نموذج مُدرَّب بطبقة NIGUncertainty القديمة بلا
             إعادة تدريب؛ وإلا يُقصّ المخرَج الجاهز. '{name}_unc_clipped' = قناع ما قُصّ.

    يتوقع أن يُخرج النموذج مفاتيح بصيغة:
        'y_{name}'                  → القيمة (مستمرة) أو logits/probabilities (فئوية)
        'y_{name}_epistemic'        → (اختياري) عدم يقين ابستيمي
        'y_{name}_aleatoric'        → (اختياري) عدم يقين أليتوري
        'y_{name}_confidence'       → (اختياري) درجة ثقة
        TargetSpec.class_key        → (اختياري) P(صعود) لرأس تصنيف الهدف، يُخزَّن '<name>_p_up'

    Returns:
        dict خام يحوي لكل هدف: raw (+ epistemic/aleatoric/confidence إن وُجدت)
    """
    target_specs = resolve_targets(target_specs)   # نصوص/جزئية مقبولة؛ غير المطلوب يُتجاهل
    n_samples = X_inputs[0].shape[0]
    ds = tf.data.Dataset.from_tensor_slices((X_inputs,)).batch(batch_size)

    keys = []
    for spec in target_specs:
        keys.append(spec.name)
        if spec.has_uncertainty:
            keys += [f'{spec.name}_epistemic', f'{spec.name}_aleatoric', f'{spec.name}_confidence']

    preds = {k: [] for k in keys}
    prog = tf.keras.utils.Progbar(n_samples // batch_size + 1) if verbose else None

    for i, (b_x,) in enumerate(ds):
        out = model(b_x, training=False)

        for spec in target_specs:
            key = f'y_{spec.name}'
            if key not in out:
                _avail = sorted(str(k)[2:] for k in out
                                if str(k).startswith('y_') and not str(k).endswith(_UNCERTAINTY_SUFFIXES))
                raise KeyError(
                    f"مخرجات النموذج لا تحوي '{key}'. الأهداف المتاحة في النموذج: {_avail}. "
                    f"تأكد من أن اسم الهدف في TargetSpec يطابق مخرجات النموذج."
                )
            preds[spec.name].append(np.asarray(out[key]))

            if spec.class_key:
                # احتمال الصعود من رأس التصنيف (بعد sigmoid). يصل إلى decode عبر raw_preds['<name>_p_up'] — لا عبر test_dict
                if spec.class_key in out:
                    _n_b = np.asarray(out[key]).shape[0]
                    preds.setdefault(f'{spec.name}_p_up', []).append(
                        np.asarray(out[spec.class_key], dtype=np.float64).reshape(_n_b, -1)[:, 0])
                elif spec.signed_by_class:
                    # اتجاه هذا الهدف من الرأس فقط: بدونه لا تخمين (كل السعر يصير صعوداً أو هبوطاً وهمياً)
                    raise KeyError(f"مخرجات النموذج لا تحوي '{spec.class_key}' — اتجاه الهدف '{spec.name}' يُقرأ منه.")

            if spec.has_uncertainty:
                _cap = NIG_UNC_MAX if unc_max is None else float(unc_max)
                _unc, _clipped = {}, None
                if all(f'y_{spec.name}_{s}' in out for s in ('nu', 'alpha', 'beta')):
                    _e, _a, _clipped = nig_uncertainty_bounded(*(np.asarray(out[f'y_{spec.name}_{s}'])
                                                                 for s in ('nu', 'alpha', 'beta')), _cap)
                    _unc = {'epistemic': _e, 'aleatoric': _a}
                else:
                    for s in ('epistemic', 'aleatoric'):
                        if f'y_{spec.name}_{s}' in out:
                            _unc[s], _c = _bound_unc(np.asarray(out[f'y_{spec.name}_{s}']), _cap)
                            _clipped = _c if _clipped is None else (_clipped | _c)
                if _clipped is not None:
                    preds.setdefault(f'{spec.name}_unc_clipped', []).append(_clipped.reshape(len(_clipped), -1)[:, 0])
                for suffix in ['epistemic', 'aleatoric', 'confidence']:
                    k2 = f'y_{spec.name}_{suffix}'
                    if suffix in _unc:
                        preds[f'{spec.name}_{suffix}'].append(_unc[suffix].astype(np.float32))
                    elif k2 in out:
                        preds[f'{spec.name}_{suffix}'].append(np.asarray(out[k2]))
                    else:
                        # نملأ بأصفار حتى لا ينهار concatenate لاحقاً
                        n_b = b_x[0].shape[0] if isinstance(b_x, (tuple, list)) else b_x.shape[0]
                        preds[f'{spec.name}_{suffix}'].append(np.zeros((n_b, 1), dtype=np.float32))

        if prog:
            prog.update(i + 1)

    return {k: np.concatenate(v, axis=0) for k, v in preds.items()}


# دالة قديمة تبقى للتوافق الخلفي (high/low/close مستمرة فقط)
def predict_batch(model, X_inputs, batch_size: int = 256, verbose: bool = False) -> Dict[str, np.ndarray]:
    """(نسخة سابقة - محفوظة للتوافق) تنبؤ للأهداف الثلاثة high/low/close فقط."""
    return predict_batch_v4(model, X_inputs, DEFAULT_PRICE_TARGETS, batch_size, verbose)
