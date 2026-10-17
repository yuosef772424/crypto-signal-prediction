"""
PURPOSE:  Stage 3: flexible evaluation metrics (continuous MAE/direction/tolerance, categorical accuracy), evaluate_predictions_v4.
TAGS:     evaluate_predictions_v4, evaluate_predictions, MAE, direction accuracy, tolerance, metrics
PITFALLS: Executed into the one shared evaluation namespace by evaluation/_loader.py (never imported on its own): names from other modules resolve at call time.

## 6️⃣ المرحلة 3: التقييم المرن (Metrics مستمرة + فئوية)
"""
# ─────────────────────────────────────────────────────────────────────────────
# المرحلة 3: التقييم المرن (مستمر + فئوي) — آمن مع العينات التي بلا هدف
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_predictions_v4(
    decoded_preds: Dict[str, Dict[str, np.ndarray]],
    target_specs: List['TargetSpec'],
    y_true: Optional[Dict[str, np.ndarray]] = None,
    base_params: Optional[np.ndarray] = None,
    direction_tolerance: float = 0.2,
    price_tolerance: Optional[float] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    تقييم مرن: للأهداف المستمرة يحافظ على منطق v3 (MAE/اتجاه/تسامح)،
    وللأهداف الفئوية يضيف accuracy/precision/recall/F1 لكل فئة.

    ✅ v4.2: العينات التي بلا هدف حقيقي (NaN — مثل آخر شمعة في التداول الحي) تُستثنى
       من كل المقاييس بدل أن تُحسب "خطأً"، ويُسجَّل قناع `has_truth` لكل هدف.
    """
    target_specs = resolve_targets(target_specs)
    if y_true is None:
        return decoded_preds

    limit = None
    for spec in target_specs:
        limit = decoded_preds[spec.name].get('pred_real', decoded_preds[spec.name].get('pred_index'))
        limit = len(limit)
        break

    if base_params is not None:
        bp = base_params[:limit]
        median, iqr = bp[:, 0], bp[:, 1]
    else:
        median = iqr = None

    # المدى الفعلي (لحساب within_tolerance) — يُحسب فقط إن وُجد high/low مستمرين
    actual_range = None
    if 'high' in decoded_preds and 'low' in decoded_preds:
        h, l = decoded_preds['high'].get('target_real'), decoded_preds['low'].get('target_real')
        if h is not None and l is not None and not (np.all(np.isnan(h)) or np.all(np.isnan(l))):
            actual_range = np.maximum(h - l, 1e-7)

    for spec in target_specs:
        r = decoded_preds[spec.name]
        key = f'y_{spec.name}'
        if key not in y_true:
            continue

        if spec.kind == 'continuous':
            true_val = r.get('target_real')
            if true_val is None or np.all(np.isnan(true_val)):
                true_scaled = y_true[key][:limit].flatten()
                if getattr(spec, 'range_of', None) is not None:
                    t_hi, t_lo = (np.asarray(decoded_preds[n].get('target_real', np.nan), dtype=np.float64)
                                  for n in spec.range_of)
                    true_val = t_lo + true_scaled * (t_hi - t_lo)
                elif getattr(spec, 'signed_by_class', False):
                    # y_true مقدار بلا اتجاه، وبلا last_candles المستقبل لا سعر فعلي: NaN فتُستثنى العيّنات (لا تُخمَّن)
                    true_val = np.full(len(true_scaled), np.nan)
                elif getattr(spec, 'relative_to_entry', False):
                    true_val = np.asarray(r['entry_price'], dtype=np.float64) * (
                        1.0 + true_scaled / float(getattr(spec, 'reg_scale', 1.0)))
                else:
                    true_val = (true_scaled * iqr) + median
            true_val = np.asarray(true_val, dtype=np.float64)

            entry = np.asarray(r['entry_price'], dtype=np.float64)
            pred_real = np.asarray(r['pred_real'], dtype=np.float64)
            change_from_entry = np.asarray(r['change_from_entry'], dtype=np.float64)

            # عينات بلا هدف (NaN) تُستثنى من كل المقاييس
            valid = np.isfinite(true_val) & np.isfinite(pred_real) & np.isfinite(entry)

            abs_error = np.abs(true_val - pred_real)                      # NaN حيث لا هدف
            pct_error = (abs_error / (np.abs(true_val) + 1e-7)) * 100
            true_change_pct = ((true_val - entry) / (np.abs(entry) + 1e-7)) * 100

            pred_direction = np.sign(change_from_entry)
            true_direction = np.sign(true_change_pct)
            direction_correct = ((pred_direction == true_direction) & valid).astype(int)

            tol = price_tolerance if price_tolerance is not None else (
                direction_tolerance * actual_range if actual_range is not None else direction_tolerance * np.abs(true_val)
            )
            within_tolerance = ((abs_error <= tol) & valid).astype(int)

            if valid.sum() > 2 and np.std(change_from_entry[valid]) > 1e-7 and np.std(true_change_pct[valid]) > 1e-7:
                corr = np.corrcoef(change_from_entry[valid], true_change_pct[valid])[0, 1]
                corr = 0.0 if np.isnan(corr) else corr
            else:
                corr = 0.0

            win_rate = float(np.mean(direction_correct[valid]) * 100) if valid.any() else float('nan')
            uncertainty_pct = None
            within_1_std = within_2_std = None
            if 'uncertainty_real' in r:
                unc = np.asarray(r['uncertainty_real'], dtype=np.float64)
                uncertainty_pct = (unc / (np.abs(pred_real) + 1e-7)) * 100
                if valid.any():
                    within_1_std = float(np.mean(abs_error[valid] <= unc[valid]) * 100)
                    within_2_std = float(np.mean(abs_error[valid] <= 2 * unc[valid]) * 100)
                else:
                    within_1_std = within_2_std = float('nan')

            r.update({
                'true_real': true_val,
                'has_truth': valid,
                'n_valid': int(valid.sum()),
                'true_change_from_entry_pct': true_change_pct,
                'error_abs': abs_error,
                'error_pct': pct_error,
                'same_direction': direction_correct,
                'within_tolerance': within_tolerance,
                'direction_correct': direction_correct,
                'correlation': corr,
                'win_rate': win_rate,
                'uncertainty_pct': uncertainty_pct,
                'within_1_std': within_1_std,
                'within_2_std': within_2_std,
            })

        elif spec.kind == 'categorical':
            true_raw = y_true[key][:limit]
            if true_raw.ndim > 1 and true_raw.shape[1] > 1:
                true_index = np.argmax(true_raw, axis=1)
            else:
                true_index = true_raw.flatten().astype(int)
            true_label = np.array([spec.class_names[i] for i in true_index], dtype=object)

            correct = (r['pred_index'] == true_index).astype(int)
            accuracy = float(np.mean(correct)) * 100

            try:
                from sklearn.metrics import classification_report, confusion_matrix
                report_dict = classification_report(
                    true_index, r['pred_index'],
                    labels=list(range(len(spec.class_names))),
                    target_names=spec.class_names,
                    output_dict=True, zero_division=0,
                )
                cm = confusion_matrix(true_index, r['pred_index'], labels=list(range(len(spec.class_names))))
            except Exception:
                report_dict, cm = None, None

            r.update({
                'true_index': true_index,
                'true_label': true_label,
                'correct': correct,
                'accuracy': accuracy,
                'classification_report': report_dict,
                'confusion_matrix': cm,
            })

    return decoded_preds


# دالة قديمة تبقى للتوافق الخلفي
def evaluate_predictions(decoded_preds, y_true=None, base_params=None, last_candles=None,
                          direction_tolerance=0.2, price_tolerance=None):
    """(نسخة سابقة - محفوظة للتوافق) تقييم high/low/close المستمرة فقط."""
    if y_true is None or base_params is None:
        return decoded_preds
    return evaluate_predictions_v4(
        decoded_preds, DEFAULT_PRICE_TARGETS, y_true, base_params,
        direction_tolerance, price_tolerance,
    )
