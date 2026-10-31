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
    table = latest_table_from_raw(raw, specs, bp, lc, yt, k, ts_last, asset)

    if verbose:
        _print_latest_asset_table(table, k, asset, legend)
    return table


def _print_latest_asset_table(table, k, asset, legend):
    title = f"\n🔴 آخر {k} عينة" + (f" — {asset}" if asset else "") + " (الأحدث في الأسفل):"
    print_latest_table(table, title=title, legend=legend)


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

    # أصول بلا عينات تُبلَّغ كما كانت (خطأ مطبوع) وتُستثنى؛ باقي الأصول: استدعاء نموذج واحد على آخر k_i صف من كل منها
    # (evaluation/pooled.py)، ثم الفكّ/التقييم/الجدول لكل أصل من شريحته (numpy فقط).
    names = []
    for asset_name, test_data in test_dict.items():
        if len(test_data[f'X_{timeframes[0]}']) == 0:
            print(f"   ❌ {asset_name}: لا توجد عينات.")
        else:
            names.append(asset_name)

    raw_by_asset, ks = {}, {}
    if names:
        first = test_dict[names[0]]
        specs = _resolve_for_model(target_specs, model, tuple(np.asarray(first[f'X_{tf}']) for tf in timeframes))
        ks = {a: int(min(max(k, 1), len(test_dict[a][f'X_{timeframes[0]}']))) for a in names}
        pooled_x, bounds = pool_x(test_dict, timeframes, names=names, n_per_asset=ks)   # آخر ks[a] صف من كل أصل
        raw_pooled, specs = predict_pooled_raw(model, pooled_x, timeframes, specs, batch_size)
        raw_by_asset = {b['name']: r for b, r in slice_by_asset(raw_pooled, bounds)}

    for asset_name in names:
        test_data = test_dict[asset_name]
        try:
            n_total = len(test_data[f'X_{timeframes[0]}'])
            ka = ks[asset_name]
            sl = slice(n_total - ka, n_total)
            ts = _find_timestamps(test_data, timestamp_key, timestamp_col)
            y_all = {f'y_{s.name}': test_data['y'][s.name]
                     for s in specs if 'y' in test_data and s.name in test_data['y']} or None
            ts_last = _align_timestamps(_extract_timestamps(ts, test_data.get('last_candles'), None), n_total, ka)
            lc, bp = test_data.get('last_candles'), test_data.get('base_params')
            frame = latest_table_from_raw(
                raw_by_asset[asset_name], specs, None if bp is None else np.asarray(bp)[sl],
                None if lc is None else np.asarray(lc)[sl],
                None if not y_all else {kk: np.asarray(v)[sl] for kk, v in y_all.items()},
                ka, ts_last, asset_name)
            if verbose and not latest_only:
                _print_latest_asset_table(frame, ka, asset_name, False)
            frames.append(frame)
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
