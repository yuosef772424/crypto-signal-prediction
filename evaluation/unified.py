"""
PURPOSE:  Stage 4: predict_with_evaluation_v4 - prediction + decoding + verification + evaluation in one call (v3 kept for compatibility).
TAGS:     predict_with_evaluation_v4, predict_with_evaluation_v3, unified pipeline, n_display
PITFALLS: Executed into the one shared evaluation namespace by evaluation/_loader.py (never imported on its own): names from other modules resolve at call time.

## 7️⃣ المرحلة 4: الدالة الموحّدة predict_with_evaluation_v4
"""
# ─────────────────────────────────────────────────────────────────────────────
# المرحلة 4: الدالة الرئيسية المُوحّدة (v4) — تنبؤ + فك تشفير + تحقق + تقييم
# ─────────────────────────────────────────────────────────────────────────────

def predict_with_evaluation_v4(
    model,
    X_inputs: Tuple[np.ndarray, ...],
    target_specs,
    base_params: Optional[np.ndarray] = None,
    last_candles: Optional[np.ndarray] = None,
    y_true: Optional[Dict[str, np.ndarray]] = None,
    verbose: bool = False,
    n_display: int = 5,
    batch_size: int = 256,
    direction_tolerance: float = 0.2,
    price_tolerance: Optional[float] = None,
    min_samples: int = 1,
    run_verification: bool = True,
    timestamps=None,
    timestamp_col: Optional[int] = None,
    show_stages: Optional[bool] = None,
    show_verification: Optional[bool] = None,
    asset: Optional[str] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    التنبؤ + فك التشفير المرن + التحقق + التقييم — الإصدار v4.2

    المراحل: predict_batch_v4 → decode_predictions_v4 → verify_decoding
              → evaluate_predictions_v4

    target_specs: List[TargetSpec] أو نصوص ('close' / 'high,low' / ['high','low']) أو None/'all'.
                  يُقرأ ويُقيَّم المطلوب فقط، وباقي مخرجات النموذج تُتجاهل.
    n_display:    عدد **آخر** العينات المعروضة (الأحدث في الأسفل).
    timestamps / timestamp_col: تاريخ كل عينة (مصفوفة أوقات، أو رقم عمود داخل last_candles).
    show_stages / show_verification: افتراضياً = verbose (تُغلق من test_all_assets_v4
                  ليظهر ملخص التحقق مرة واحدة فقط).
    """
    n_samples = X_inputs[0].shape[0]
    if n_samples < min_samples:
        raise ValueError(f"عدد العينات ({n_samples}) أقل من الحد الأدنى ({min_samples})")

    target_specs = _resolve_for_model(target_specs, model, X_inputs)
    show_stages = verbose if show_stages is None else show_stages
    show_verification = verbose if show_verification is None else show_verification

    if show_stages:
        print(f"\n🔮 Predicting & Evaluating v4 (مرن: مستمر + فئوي) — الأهداف: {[s.name for s in target_specs]}")
        print("   Stage 1: Batch Predictions...")
    raw_preds = predict_batch_v4(model, X_inputs, target_specs, batch_size, show_stages)

    if show_stages:
        print("   Stage 2: Decoding (Flexible)...")
    decoded = decode_predictions_v4(raw_preds, target_specs, base_params, last_candles)

    verification_report = None
    if run_verification:
        if show_stages:
            print("   Stage 2.5: Verifying decoding (normalized vs. original)...")
        verification_report = verify_decoding(
            raw_preds, decoded, target_specs, base_params, y_true, last_candles,
            verbose=show_verification,
        )

    if show_stages:
        print("   Stage 3: Evaluation...")
    results = evaluate_predictions_v4(
        decoded, target_specs, y_true, base_params, direction_tolerance, price_tolerance
    )

    if verification_report is not None:
        results['_verification'] = verification_report

    if verbose:
        ts = _align_timestamps(_extract_timestamps(timestamps, last_candles, timestamp_col), n_samples, n_samples)
        _print_v4_report(results, target_specs, n_samples, n_display, y_true is not None,
                         timestamps=ts, asset=asset)

    return results


def _print_v4_report(results, target_specs, n_samples, n_display, has_truth,
                     timestamps=None, asset=None, legend=True):
    """
    تقرير مضغوط: جدول واحد بآخر `n_display` عينة لكل الأهداف (الأحدث في الأسفل)،
    ثم سطر أداء واحد لكل هدف على كامل العينات (إن وُجدت أهداف حقيقية).
    """
    specs = [s for s in resolve_targets(target_specs) if isinstance(results.get(s.name), dict)]
    k = int(min(n_display, n_samples)) if n_display else 0

    if k > 0:
        table = build_latest_table(results, specs, k, timestamps, asset)
        print(f"\n📊 آخر {k} عينة لكل هدف (الأحدث في الأسفل):")
        print_latest_table(table, legend=legend)

    if has_truth:
        lines = []
        for spec in specs:
            r = results[spec.name]
            if spec.kind == 'continuous' and 'error_abs' in r:
                n_valid = int(r.get('n_valid', len(r['error_abs'])))
                lines.append(
                    f"   ▸ {spec.name:<6}│ MAE {_fmt_price(_nanmean(r['error_abs']))}"
                    f" │ MAPE {_nanmean(r['error_pct']):.4f}%"
                    f" │ Win {r.get('win_rate', float('nan')):.2f}%"
                    f" │ corr {r.get('correlation', 0.0):+.3f}"
                    f" │ n={n_valid:,}"
                )
            elif spec.kind == 'categorical' and 'accuracy' in r:
                lines.append(f"   ▸ {spec.name:<6}│ Accuracy {r['accuracy']:.2f}% │ n={len(r['correct']):,}")
        if lines:
            print("\n📈 الأداء على كامل العينات:")
            print("\n".join(lines))


# دالة قديمة تبقى للتوافق الخلفي
def predict_with_evaluation_v3(model, X_inputs, base_params, last_candles=None, y_true=None,
                                verbose=False, n_display=5, batch_size=256,
                                direction_tolerance=0.2, price_tolerance=None, min_samples=1):
    """(نسخة سابقة - محفوظة للتوافق) تستخدم داخلياً محرك v4 بأهداف الأسعار الافتراضية."""
    return predict_with_evaluation_v4(
        model, X_inputs, DEFAULT_PRICE_TARGETS, base_params, last_candles, y_true,
        verbose, n_display, batch_size, direction_tolerance, price_tolerance, min_samples,
        run_verification=False,
    )
