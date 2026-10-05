"""
PURPOSE:  Stage 2.5: verify_decoding - checks decoding in the normalized and original space with a round-trip test.
TAGS:     verify_decoding, round-trip, decoding check
PITFALLS: Executed into the one shared evaluation namespace by evaluation/_loader.py (never imported on its own): names from other modules resolve at call time.

## 5️⃣ المرحلة 2.5: التحقق من صحة فك التشفير

تقارن `verify_decoding` النتائج في مسارين مستقلين: **الفضاء المطبّع** (قبل أي تحويل) و**الفضاء الأصلي** (بعد فك التطبيع)، مع اختبار Round-trip يكتشف أي خطأ حسابي في معادلة فك التشفير نفسها.
"""
# ─────────────────────────────────────────────────────────────────────────────
# المرحلة 2.5: التحقق من صحة فك التشفير (مطبّع مقابل أصلي)
# ─────────────────────────────────────────────────────────────────────────────

def verify_decoding(
    raw_preds: Dict[str, np.ndarray],
    decoded: Dict[str, Dict[str, np.ndarray]],
    target_specs: List['TargetSpec'],
    base_params: Optional[np.ndarray] = None,
    y_true: Optional[Dict[str, np.ndarray]] = None,
    last_candles: Optional[np.ndarray] = None,
    limit: Optional[int] = None,
    atol: float = 1e-4,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    ✅ يتحقق من أن فك التشفير صحيح عبر مقارنتين مستقلتين لكل هدف:

    1) فضاء القيم المطبّعة (Normalized / Scaled space):
       - للمستمر: مقارنة raw_pred (scaled) مع y_true scaled مباشرة (قبل فك التطبيع)
       - للفئوي:  مقارنة pred_index المُستخرج مع argmax(y_true) في نفس الفضاء المُرمّز

    2) فضاء القيم الأصلية (Real / Denormalized space):
       - للمستمر: مقارنة pred_real المُعاد حسابه يدوياً ((scaled*iqr)+median) مع
         القيمة المخزّنة فعلاً في decoded[target]['pred_real'] (اختبار Round-trip)،
         ثم مقارنته بالقيمة الحقيقية الفعلية (target_real من last_candles أو
         y_true بعد فك تطبيعه) لحساب اتساق الخطأ بين الفضاءين:
             نسبة الاتساق = |خطأ الفضاء الحقيقي| / (|خطأ الفضاء المطبّع| * iqr)
         يجب أن تكون قريبة من 1.0 إن كان فك التشفير صحيحاً حسابياً.
       - للفئوي: مقارنة pred_label مع اسم الفئة الحقيقي (إن أمكن اشتقاقه من y_true)

    Returns:
        dict[target_name] -> تقرير تحقق (مطابقة round-trip، اتساق الفضاءين، أخطاء)
    """
    target_specs = resolve_targets(target_specs)
    report: Dict[str, Dict] = {}

    if limit is None:
        any_name = target_specs[0].name
        limit = raw_preds[any_name].shape[0]

    if base_params is not None:
        bp = base_params[:limit]
        median, iqr = bp[:, 0], bp[:, 1]
    else:
        median = iqr = None

    if verbose:
        print("=" * 100)
        print("🔍 التحقق من صحة فك التشفير (Decoding Verification)")
        print("=" * 100)

    for spec in target_specs:
        d = decoded[spec.name]
        entry_report = {'target': spec.name, 'kind': spec.kind}

        if spec.kind == 'continuous':
            range_of = getattr(spec, 'range_of', None)
            signed = bool(getattr(spec, 'signed_by_class', False))   # مقدار بلا اتجاه، واتجاهه من رأس التصنيف
            s_median, s_iqr = ((d['median'], d['iqr']) if range_of is not None
                               else _spec_base(spec, median, iqr, last_candles, limit))
            if signed and s_iqr is not None:
                s_iqr = s_iqr * d['direction_sign']    # نفس مقياس decode: round-trip مطابق لـ pred_real
            if s_median is None:
                entry_report['status'] = 'skipped'
                entry_report['reason'] = 'لا يوجد base_params للتحقق'
            else:
                median_s, iqr_s = s_median, s_iqr
                # 1) round-trip: إعادة حساب pred_real من pred_scaled يدوياً
                recomputed_real = ((np.clip(d['pred_scaled'], 0.0, 1.0) if range_of is not None else d['pred_scaled'])
                                   * iqr_s) + median_s
                roundtrip_diff = np.abs(recomputed_real - d['pred_real'])
                roundtrip_ok = bool(np.all(roundtrip_diff <= atol))

                entry_report['roundtrip_max_diff'] = float(roundtrip_diff.max())
                entry_report['roundtrip_ok'] = roundtrip_ok

                # 2) مقارنة الفضاء المطبّع مقابل الفضاء الأصلي (إن وُجدت الحقيقة)
                true_scaled = None
                if y_true is not None and f'y_{spec.name}' in y_true:
                    true_scaled = y_true[f'y_{spec.name}'][:limit].flatten()

                true_real = d.get('target_real')
                has_real_truth = true_real is not None and not np.all(np.isnan(true_real))

                if true_scaled is not None:
                    err_scaled = np.abs(d['pred_scaled'] - true_scaled)
                    entry_report['mae_scaled'] = _nanmean(err_scaled)

                    # فك تطبيع الحقيقة يدوياً للمقارنة مع target_real المخزّن (إن وُجد)
                    if signed:
                        # y_true مقدار بلا اتجاه: اتجاه الحقيقة من السعر الفعلي (بلا سعر فعلي لا مقارنة في الفضاء الأصلي)
                        true_real_from_scaled = ((true_scaled * np.abs(iqr_s) * np.sign(true_real - d['entry_price'])) + median_s
                                                 if has_real_truth else None)
                    else:
                        true_real_from_scaled = (true_scaled * iqr_s) + median_s
                    if range_of is not None:
                        # الموقع الحقيقي يُفكّ داخل المدى *الحقيقي* (لا المتوقَّع) لهذين الهدفين
                        t_hi, t_lo = (np.asarray(decoded[n].get('target_real', np.nan), dtype=np.float64)
                                      for n in range_of)
                        true_real_from_scaled = t_lo + true_scaled * (t_hi - t_lo)
                    if has_real_truth:
                        consistency_diff = np.abs(true_real_from_scaled - true_real)
                        entry_report['true_value_consistency_max_diff'] = _nanmax(consistency_diff)
                        # ✅ يدخل الحكم الآن: فكّ الهدف الحقيقي بنفس صيغة فكّ التنبؤ يجب أن يعيد السعر
                        # المستقبلي الفعلي. اختلاف كبير = مرجع فكّ التشفير خاطئ لهذا الهدف (مثلاً high
                        # يُفكّ حول آخر close بينما الهدف عائد نسبة لآخر high) — كان يُطبع رقماً فقط.
                        rel = consistency_diff / (np.abs(true_real) + 1e-12)
                        # بالنسبة لا بأسوأ عيّنة: هدف الانحدار مقصوص (reg_target_clip، ±100% افتراضياً) فعملة
                        # تضاعفت في يوم لا يُعاد بناؤها حرفياً — استثناء نادر لا خطأ فكّ. مرجع خاطئ يُفسد كل العيّنات تقريباً.
                        finite = np.isfinite(rel)
                        bad_frac = float(np.mean(rel[finite] > 1e-3)) if finite.any() else 0.0
                        entry_report['true_value_inconsistent_frac'] = bad_frac
                        entry_report['true_value_consistent'] = bool(bad_frac <= 0.01)
                    elif true_real_from_scaled is not None:
                        true_real = true_real_from_scaled
                        has_real_truth = True

                if has_real_truth:
                    err_real = np.abs(d['pred_real'] - true_real)
                    entry_report['mae_real'] = _nanmean(err_real)

                    if true_scaled is not None and iqr_s is not None and range_of is None and not signed:
                        # اتساق الخطأ بين الفضاءين: يجب أن تكون النسبة ≈ 1 (لا معنى له لموقع في مدى متوقَّع ≠ الحقيقي،
                        # ولا لمقدار بلا اتجاه: خطؤه في الفضاء الأصلي يشمل خطأ الاتجاه)
                        denom = err_scaled * np.abs(iqr_s)
                        denom_safe = np.where(denom < 1e-9, np.nan, denom)
                        ratio = err_real / denom_safe
                        entry_report['space_consistency_ratio_mean'] = _nanmean(ratio)

                entry_report['status'] = ('ok' if roundtrip_ok and entry_report.get('true_value_consistent', True)
                                          else 'FAILED')

        elif spec.kind == 'categorical':
            entry_report['n_classes'] = len(spec.class_names)

            # round-trip: التأكد أن كل pred_index يقع ضمن مجال الفئات الصحيح
            valid_range = np.all((d['pred_index'] >= 0) & (d['pred_index'] < len(spec.class_names)))
            entry_report['roundtrip_ok'] = bool(valid_range)

            # التأكد من تطابق pred_label مع class_names[pred_index] (اتساق داخلي)
            expected_labels = np.array([spec.class_names[i] for i in d['pred_index']], dtype=object)
            label_match = bool(np.all(expected_labels == d['pred_label']))
            entry_report['label_mapping_ok'] = label_match

            # مقارنة الفضاء المُرمّز (index/one-hot) بالفضاء الأصلي (label) مقابل y_true
            if y_true is not None and f'y_{spec.name}' in y_true:
                true_raw = y_true[f'y_{spec.name}'][:limit]
                if true_raw.ndim > 1 and true_raw.shape[1] > 1:
                    true_index = np.argmax(true_raw, axis=1)
                else:
                    true_index = true_raw.flatten().astype(int)

                acc_encoded_space = float(np.mean(d['pred_index'] == true_index))
                true_label = np.array([spec.class_names[i] for i in true_index], dtype=object)
                acc_label_space = float(np.mean(d['pred_label'] == true_label))

                entry_report['accuracy_encoded_space'] = acc_encoded_space
                entry_report['accuracy_label_space'] = acc_label_space
                # يجب أن تتطابق الدقتان تماماً إن كان فك التشفير صحيحاً
                entry_report['spaces_agree'] = bool(np.isclose(acc_encoded_space, acc_label_space))

            entry_report['status'] = 'ok' if (valid_range and label_match) else 'FAILED'

        report[spec.name] = entry_report

        if verbose:
            icon = "✅" if entry_report['status'] == 'ok' else ("⚠️" if entry_report['status'] == 'skipped' else "❌")
            print(f"\n{icon} الهدف: {spec.name} ({spec.kind})")
            for k, v in entry_report.items():
                if k in ('target', 'kind', 'status'):
                    continue
                if isinstance(v, float):
                    print(f"     • {k}: {v:.6f}")
                else:
                    print(f"     • {k}: {v}")

    if verbose:
        n_failed = sum(1 for r in report.values() if r['status'] == 'FAILED')
        print("\n" + "-" * 100)
        if n_failed == 0:
            print("✅ جميع الأهداف اجتازت التحقق من فك التشفير بنجاح")
        else:
            print(f"❌ {n_failed} هدف/أهداف فشلت في التحقق — راجع التفاصيل أعلاه")

    return report
