"""
PURPOSE:  Live trading report of the last N samples: predict_latest_v4 / predict_latest_all_assets.
TAGS:     predict_latest_v4, predict_latest_all_assets, live trading, latest_only, last N samples
PITFALLS: Executed into the one shared evaluation namespace by evaluation/_loader.py (never imported on its own): names from other modules resolve at call time.

## 🔴 التداول الحي: تقرير آخر N عينات — `predict_latest_v4` / `predict_latest_all_assets`
"""
# ═══════════════════════════════════════════════════════════════════════════
# 🔴 التداول الحي: تقرير آخر N عينات (التاريخ · التوقع · الثقة · عدم اليقين · التحرك · السعر)
# ═══════════════════════════════════════════════════════════════════════════
#
# • تشغّل النموذج على آخر N عينة فقط (سريعة — مناسبة للاستدعاء المتكرر لايف).
# • لا تحتاج أي هدف حقيقي: آخر عينة بلا هدف تظهر بـ ⏳ ولا تُحسب في أي مقياس.
# • تعمل على بيانات الاختبار أيضاً (فإن وُجدت أهداف فعلية تظهر true وعلامة ✅/❌).
# • تعتمد ترتيب العينات زمنياً (الأقدم أولاً، والأحدث آخراً).
# ═══════════════════════════════════════════════════════════════════════════

def predict_latest_v4(
    model,
    X_inputs: Tuple[np.ndarray, ...],
    base_params: np.ndarray,
    last_candles: Optional[np.ndarray] = None,
    target_specs=None,
    n_display: int = 5,
    timestamps=None,
    timestamp_col: Optional[int] = None,
    y_true: Optional[Dict[str, np.ndarray]] = None,
    batch_size: int = 256,
    asset: Optional[str] = None,
    verbose: bool = True,
    legend: bool = True,
) -> pd.DataFrame:
    """
    يتنبأ بآخر `n_display` عينة ويُرجع جدولاً (وتطبعه إن verbose=True).

    Args:
        X_inputs:      نفس مدخلات النموذج (tuple لكل إطار زمني)، مرتبة زمنياً.
        base_params:   [N, 2] = (median, iqr) لفك التطبيع.
        last_candles:  الشمعة المرجعية (سعر الدخول) لكل عينة؛ الأعمدة price_index ضرورية لحساب
                       التحرك المتوقع. أعمدة الهدف المستقبلي (future_col) اختيارية.
        target_specs:  نصوص ('close' / 'high,low') أو TargetSpec أو None/'all'.
        timestamps:    تاريخ كل عينة (datetime / نص / epoch بالثواني أو مللي)، بطول N أو n_display.
        timestamp_col: بديلاً عن timestamps: رقم عمود التاريخ داخل last_candles.
        y_true:        اختياري: {'y_close': ...} لإظهار الهدف الفعلي عند توفره.

    Returns:
        DataFrame: idx, date, [asset], target, signal, confidence_%, uncertainty_%, move_%,
                   pred, pred_lo, pred_hi, entry, true, ok ...
    """
    X_inputs = tuple(np.asarray(x) for x in X_inputs)
    n_total = X_inputs[0].shape[0]
    if n_total == 0:
        raise ValueError("لا توجد عينات.")
    k = int(min(max(int(n_display), 1), n_total))

    specs = _resolve_for_model(target_specs, model, X_inputs)
    sl = slice(n_total - k, n_total)

    X_last = tuple(x[sl] for x in X_inputs)
    bp = None if base_params is None else np.asarray(base_params)[sl]
    lc = None if last_candles is None else np.asarray(last_candles)[sl]
    ts_last = _align_timestamps(_extract_timestamps(timestamps, last_candles, timestamp_col), n_total, k)
    yt = None if not y_true else {kk: np.asarray(v)[sl] for kk, v in y_true.items()}

    raw = predict_batch_v4(model, X_last, specs, batch_size)
    decoded = decode_predictions_v4(raw, specs, bp, lc)
    if yt:
        decoded = evaluate_predictions_v4(decoded, specs, yt, bp)

    table = build_latest_table(decoded, specs, n_display=k, timestamps=ts_last, asset=asset)

    if verbose:
        title = f"\n🔴 آخر {k} عينة" + (f" — {asset}" if asset else "") + " (الأحدث في الأسفل):"
        print_latest_table(table, title=title, legend=legend)
    return table


def predict_latest_all_assets(
    model,
    test_dict: Dict,
    timeframes: List[str],
    target_specs=None,
    n_display: int = 5,
    latest_only: bool = False,
    timestamp_col: Optional[int] = None,
    timestamp_key: Optional[str] = None,
    batch_size: int = 256,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    تقرير آخر N عينات لكل الأصول في test_dict (أو آخر عينة فقط لكل أصل إن latest_only=True،
    فيُطبع جدول واحد مضغوط بكل الأصول — مناسب للقرار اللحظي).
    """
    k = 1 if latest_only else int(n_display)
    frames = []
    specs = None

    for asset_name, test_data in test_dict.items():
        try:
            X_inputs = tuple(test_data[f'X_{tf_name}'] for tf_name in timeframes)
            if specs is None:
                specs = _resolve_for_model(target_specs, model, X_inputs)
            ts = _find_timestamps(test_data, timestamp_key, timestamp_col)
            y_true = {f'y_{s.name}': test_data['y'][s.name]
                      for s in specs if 'y' in test_data and s.name in test_data['y']} or None
            frames.append(predict_latest_v4(
                model, X_inputs, test_data['base_params'], test_data.get('last_candles'),
                specs, n_display=k, timestamps=ts, y_true=y_true, batch_size=batch_size,
                asset=asset_name, verbose=(verbose and not latest_only), legend=False,
            ))
        except Exception as e:
            print(f"   ❌ {asset_name}: {e}")

    if not frames:
        return pd.DataFrame()
    table = pd.concat(frames, ignore_index=True)

    if verbose:
        if latest_only:
            print(f"\n🔴 آخر عينة لكل أصل ({table['asset'].nunique()} أصل):")
            print_latest_table(table, legend=True)
        else:
            print("\n  " + _latest_legend(table))
    return table


# ══════════════════════════════════════════════════════════════════════════
# 🚀 أمثلة استخدام
# ══════════════════════════════════════════════════════════════════════════
#
# # أصل واحد — آخر 5 عينات، هدف close فقط (والنموذج يُخرج الثلاثة؛ الباقي يُتجاهل):
# table = predict_latest_v4(model, X_inputs, base_params, last_candles,
#                           target_specs='close', n_display=5, timestamps=times)
#
# # هدفان بنصوص، والتاريخ من عمود داخل last_candles (مثلاً العمود 3):
# table = predict_latest_v4(model, X_inputs, base_params, last_candles,
#                           target_specs='high,low', n_display=5, timestamp_col=3)
#
# # كل الأصول — آخر عينة لكل أصل (لقطة لايف):
# snap = predict_latest_all_assets(model, test_dict, ['1h', '4h', '1D'],
#                                  target_specs=['high', 'low', 'close'], latest_only=True)
#
# # في اختبار كل الأصول: n_display يعرض آخر العينات، وأهداف نصية مقبولة:
# res = test_all_assets_v4(model, test_dict, ['1h', '4h', '1D'], 'close', n_display=5)
