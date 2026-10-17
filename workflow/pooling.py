"""
PURPOSE:  Pool per-asset arrays into one batch (_pool_by_asset), pool a test dict, and predict a pooled batch back per asset.
TAGS:     pool_test_dict, predict_pooled_batch_by_asset, _pool_by_asset, asset_bounds
PITFALLS: Same asset_bounds format as the pipeline's rolling_splits / build_dataset_live. Executed into the notebook's shared namespace by workflow/_loader.py (never imported on its own): names from other modules and the %run notebooks resolve at call time. Extracted verbatim from main.ipynb cell 24 (section 7).
"""
def _pool_by_asset(per_asset_arrays: dict, order=None):
    """يدمج {أصل: مصفوفة} في مصفوفة واحدة + قائمة asset_bounds (نفس تنسيق
    `rolling_splits`/`build_dataset_live` في خط الأنابيب) — أساس بسيط لدمج
    عدّة أصول في دفعة واحدة."""
    order = order or list(per_asset_arrays.keys())
    parts, bounds, pos = [], [], 0
    for name in order:
        arr = np.asarray(per_asset_arrays[name])
        n = len(arr)
        bounds.append({"name": name, "start": pos, "end": pos + n})
        parts.append(arr)
        pos += n
    return np.concatenate(parts, axis=0), bounds


def pool_test_dict(test_dict, model_tf, n_per_asset=None):
    """يحوّل `test_dict` ({أصل: قسم}، شكل chicks) إلى **دفعة واحدة مدمجة**
    (X/base_params/last_candles + `asset_bounds`) — بدل تشغيل النموذج على
    كل أصل على حدة. `n_per_asset`: خذ آخر N عيّنة فقط لكل أصل (مثلاً 5 أو
    10) قبل الدمج — مفيد لاختبار سريع بدفعة واحدة بدل حلقة يدوية لكل عملة.

    يُرجع ``(pooled, y_true_pooled)``: ``pooled`` جاهز لـ
    `predict_pooled_batch_by_asset` مباشرة، و``y_true_pooled`` بمفاتيح
    ``y_{هدف}`` (إن وُجدت أهداف فعلية) لتمريره اختيارياً لنفس الدالة."""
    order = list(test_dict.keys())
    tfs = _tfs_of(model_tf)
    x_by_asset, bp_by_asset, lc_by_asset, y_by_target = {}, {}, {}, {}
    for name, split in test_dict.items():
        n = len(split["base_params"])
        k = n if n_per_asset is None else min(n_per_asset, n)
        sl = slice(n - k, n)
        x_by_asset[name] = {tf: split[f"X_{tf}"][sl] for tf in tfs}
        bp_by_asset[name] = split["base_params"][sl]
        if split.get("last_candles") is not None:
            lc_by_asset[name] = split["last_candles"][sl]
        for t, arr in (split.get("y") or {}).items():
            y_by_target.setdefault(t, {})[name] = np.asarray(arr)[sl]

    X_pooled = {}
    for tf in tfs:
        X_pooled[tf], asset_bounds = _pool_by_asset({a: x[tf] for a, x in x_by_asset.items()}, order)
    base_params_pooled, _ = _pool_by_asset(bp_by_asset, order)
    last_candles_pooled = _pool_by_asset(lc_by_asset, order)[0] if lc_by_asset else None
    y_true_pooled = {f"y_{t}": _pool_by_asset(d, order)[0] for t, d in y_by_target.items()}
    pooled = {**{f"X_{tf}": X_pooled[tf] for tf in tfs}, "base_params": base_params_pooled,
              "last_candles": last_candles_pooled, "asset_bounds": asset_bounds}
    return pooled, y_true_pooled


def predict_pooled_batch_by_asset(model, pooled, model_tf, target_specs=None, n_display=5,
                                  y_true_pooled=None, timestamp_col=None, verbose=True):
    """يُشغِّل النموذج **مرّة واحدة فقط** على دفعة مدمجة من عدّة أصول معاً
    (``pooled`` بتنسيق `asset_bounds` — من `pool_test_dict` أعلاه، أو مباشرة
    من `extract_last_batch(build_dataset_live(...), n)` في خط الأنابيب
    للتداول الحيّ)، ثم يقسّم النتائج حسب كل أصل عبر `asset_bounds` ليبني
    تقريراً واحداً — بدل استدعاء النموذج مرّة لكل أصل كما في
    `predict_latest_all_assets`/`test_all_assets_v4`. فكّ التشفير والتجميع
    (`decode_predictions_v4`/`build_latest_table`) رخيصان (numpy فقط، بلا
    نموذج) فيُعادان لكل قسم أصل بأمان بلا كلفة إضافية تُذكَر.

    مثال تداول حيّ مباشر (بلا `test_dict` إطلاقاً):
        live_ds = build_dataset_live(["BTCUSDT", "ETHUSDT", "SOLUSDT"])
        live_batch = extract_last_batch(live_ds, n=1)   # آخر شمعة لكل عملة
        predict_pooled_batch_by_asset(model, live_batch, info.model_tf,
                                      target_specs=chicks.eval_target_specs, n_display=1)
    """
    X_inputs = tuple(pooled[f"X_{tf}"] for tf in _tfs_of(model_tf))   # فريم واحد = tuple بعنصر (كما كان)
    specs = _resolve_for_model(target_specs, model, X_inputs)
    raw = predict_batch_v4(model, X_inputs, specs)  # ← استدعاء نموذج واحد فقط لكل الأصول معاً

    tables = []
    for b in pooled["asset_bounds"]:
        s, e = b["start"], b["end"]
        raw_slice = {k: v[s:e] for k, v in raw.items()}
        bp_slice = pooled["base_params"][s:e] if pooled.get("base_params") is not None else None
        lc_slice = pooled["last_candles"][s:e] if pooled.get("last_candles") is not None else None
        decoded = decode_predictions_v4(raw_slice, specs, bp_slice, lc_slice)
        y_true = ({k: v[s:e] for k, v in y_true_pooled.items()} if y_true_pooled else None)
        if y_true:
            decoded = evaluate_predictions_v4(decoded, specs, y_true, bp_slice)
        k_disp = min(n_display, e - s) if n_display else (e - s)
        table = build_latest_table(decoded, specs, n_display=k_disp,
                                   timestamps=_extract_timestamps(None, lc_slice, timestamp_col),
                                   asset=b["name"])
        tables.append(table)

    result = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    if verbose:
        print(f"\n🚀 دُفعة واحدة مدمجة ({len(pooled['asset_bounds'])} أصل، استدعاء نموذج واحد فقط):")
        print_latest_table(result, legend=True)
    return result
