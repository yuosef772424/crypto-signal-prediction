"""
PURPOSE:  The ONE batched evaluation predict path (lifted from workflow/pooling.py): pool the windows of all assets (pool_x / pool_test_dict), call the model once in fixed-size batches (predict_pooled_raw), slice the raw outputs back per asset by asset_bounds (slice_by_asset), build a per-asset latest table from a raw slice (latest_table_from_raw), and predict_pooled_batch_by_asset (the pooled live/test table).
TAGS:     pooling, predict_pooled_raw, pool_x, pool_test_dict, predict_pooled_batch_by_asset, _pool_by_asset, slice_by_asset, asset_bounds, latest_table_from_raw, one model call, batched evaluation, performance
PITFALLS: Every per-asset evaluation site (test_all_assets_v4, predict_latest_all_assets, predict_pooled_batch_by_asset, workflow's classification_accuracy_report) must go through this module instead of calling the model once per asset: 500 coins x one tiny predict call each was ~50x slower than one pooled call. The model is independent of batch composition (no BatchNorm in inference), so pooled raw outputs equal the per-asset ones up to float reduction order (rtol 1e-5); decode / evaluate / table code is numpy-only and unchanged. asset_bounds = [{'name', 'start', 'end'}] (same format as the pipeline's rolling_splits / build_dataset_live). Executed into the one shared evaluation namespace by evaluation/_loader.py (never imported on its own): names from other modules resolve at call time.
"""
from core.schema import TS_COL as _TS_COL
from stats.prediction_table import build_table as _build_prediction_table, write_table as _write_prediction_table


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


def _timeframes_of(model_tf):
    """model_tf: اسم فريم أو قائمة فريمات ← قائمة (كما `_tfs_of` في workflow، بلا اعتماد عليه)."""
    return [model_tf] if isinstance(model_tf, str) else list(model_tf)


def _tail_slices(test_dict, names, timeframes, n_per_asset=None):
    """{أصل: slice(آخر n_per_asset صف)} (كل الصفوف إن None) للأصول في names. n_per_asset رقم واحد للكل أو {أصل: رقم}."""
    out = {}
    for name in names:
        n = len(test_dict[name][f"X_{timeframes[0]}"])
        want = n_per_asset.get(name) if isinstance(n_per_asset, dict) else n_per_asset
        k = n if want is None else min(want, n)
        out[name] = slice(n - k, n)
    return out


def pool_x(test_dict, timeframes, names=None, n_per_asset=None):
    """مُدخلات النموذج فقط من test_dict ({أصل: قسم}) في دفعة مدمجة: ``({'X_<tf>': مصفوفة مدمجة}, asset_bounds)``.
    names: الأصول المشمولة وترتيبها (الافتراضي: كل مفاتيح test_dict بترتيبها). n_per_asset: آخر N صف لكل أصل فقط (رقم أو {أصل: رقم})."""
    order = list(test_dict.keys()) if names is None else list(names)
    timeframes = _timeframes_of(timeframes)
    sls = _tail_slices(test_dict, order, timeframes, n_per_asset)
    X, bounds = {}, None
    for tf in timeframes:
        X[f"X_{tf}"], bounds = _pool_by_asset({a: test_dict[a][f"X_{tf}"][sls[a]] for a in order}, order)
    return X, bounds


def pool_test_dict(test_dict, model_tf, n_per_asset=None, names=None, with_y=True):
    """يحوّل `test_dict` ({أصل: قسم}، شكل chicks) إلى **دفعة واحدة مدمجة**
    (X/base_params/last_candles + `asset_bounds`) — بدل تشغيل النموذج على كل أصل على حدة.
    `n_per_asset`: خذ آخر N عيّنة فقط لكل أصل قبل الدمج. model_tf: اسم فريم أو قائمة.

    يُرجع ``(pooled, y_true_pooled)``: ``pooled`` جاهز لـ `predict_pooled_raw`/`predict_pooled_batch_by_asset`، و``y_true_pooled`` بمفاتيح
    ``y_{هدف}`` (فارغ إن with_y=False). تحذير: y المدمجة تفترض أن كل أصل يحمل كل الأهداف (وإلا تنزاح الحدود)."""
    order = list(test_dict.keys()) if names is None else list(names)
    timeframes = _timeframes_of(model_tf)
    sls = _tail_slices(test_dict, order, timeframes, n_per_asset)
    X, asset_bounds = pool_x(test_dict, timeframes, order, n_per_asset)
    bp_by_asset, lc_by_asset, y_by_target = {}, {}, {}
    for name in order:
        split, sl = test_dict[name], sls[name]
        bp_by_asset[name] = split["base_params"][sl]
        if split.get("last_candles") is not None:
            lc_by_asset[name] = split["last_candles"][sl]
        if with_y:
            for t, arr in (split.get("y") or {}).items():
                y_by_target.setdefault(t, {})[name] = np.asarray(arr)[sl]
    base_params_pooled, _ = _pool_by_asset(bp_by_asset, order)
    last_candles_pooled = _pool_by_asset(lc_by_asset, order)[0] if lc_by_asset else None
    y_true_pooled = {f"y_{t}": _pool_by_asset(d, order)[0] for t, d in y_by_target.items()}
    pooled = {**X, "base_params": base_params_pooled, "last_candles": last_candles_pooled, "asset_bounds": asset_bounds}
    return pooled, y_true_pooled


def predict_pooled_raw(model, pooled, timeframes, target_specs=None, batch_size: int = 256, unc_max=None):
    """**استدعاء النموذج الوحيد** لكل الأصول معاً: دفعات ثابتة الحجم على المدخلات المدمجة (``pooled['X_<tf>']``، من
    pool_x/pool_test_dict أو من extract_last_batch). يُرجع ``(raw, specs)``: raw = مخرج predict_batch_v4 الخام لكل الصفوف
    المدمجة (يُقسَّم لكل أصل بـ slice_by_asset)، وspecs = TargetSpec المحلولة من النموذج."""
    X_inputs = tuple(pooled[f"X_{tf}"] for tf in _timeframes_of(timeframes))   # فريم واحد = tuple بعنصر (كما كان)
    specs = _resolve_for_model(target_specs, model, X_inputs)
    return predict_batch_v4(model, X_inputs, specs, batch_size, unc_max=unc_max), specs


def slice_by_asset(raw, asset_bounds):
    """{مفتاح: مصفوفة مدمجة} -> [(الحدّ، {مفتاح: شريحة الأصل})] بترتيب asset_bounds (شرائح عروض لا نسخ)."""
    return [(b, {k: v[b["start"]:b["end"]] for k, v in raw.items()}) for b in asset_bounds]


def latest_table_from_raw(raw, specs, base_params, last_candles, y_true, n_display, timestamps=None, asset=None):
    """جدول آخر `n_display` عيّنة لأصل واحد من مخرجه الخام: فكّ ← تقييم (إن وُجد y_true) ← جدول. numpy فقط (بلا نموذج)."""
    decoded = decode_predictions_v4(raw, specs, base_params, last_candles)
    if y_true:
        decoded = evaluate_predictions_v4(decoded, specs, y_true, base_params)
    return build_latest_table(decoded, specs, n_display=n_display, timestamps=timestamps, asset=asset)


def predict_pooled_batch_by_asset(model, pooled, model_tf, target_specs=None, n_display=5,
                                  y_true_pooled=None, timestamp_col=None, verbose=True):
    """يُشغِّل النموذج **مرّة واحدة فقط** على دفعة مدمجة من عدّة أصول معاً
    (``pooled`` بتنسيق `asset_bounds` — من `pool_test_dict`، أو مباشرة
    من `extract_last_batch(build_dataset_live(...), n)` في خط الأنابيب
    للتداول الحيّ)، ثم يقسّم النتائج حسب كل أصل عبر `asset_bounds` ليبني
    تقريراً واحداً. فكّ التشفير والتجميع (`decode_predictions_v4`/`build_latest_table`) رخيصان
    (numpy فقط، بلا نموذج) فيُعادان لكل قسم أصل. (نفس المسار الذي صارت تمرّ منه
    `predict_latest_all_assets`/`test_all_assets_v4`.)

    مثال تداول حيّ مباشر (بلا `test_dict` إطلاقاً):
        live_ds = build_dataset_live(["BTCUSDT", "ETHUSDT", "SOLUSDT"])
        live_batch = extract_last_batch(live_ds, n=1)   # آخر شمعة لكل عملة
        predict_pooled_batch_by_asset(model, live_batch, info.model_tf,
                                      target_specs=chicks.eval_target_specs, n_display=1)
    """
    raw, specs = predict_pooled_raw(model, pooled, model_tf, target_specs)   # ← استدعاء نموذج واحد فقط لكل الأصول معاً

    tables = []
    for b, raw_slice in slice_by_asset(raw, pooled["asset_bounds"]):
        s, e = b["start"], b["end"]
        bp_slice = pooled["base_params"][s:e] if pooled.get("base_params") is not None else None
        lc_slice = pooled["last_candles"][s:e] if pooled.get("last_candles") is not None else None
        y_true = ({k: v[s:e] for k, v in y_true_pooled.items()} if y_true_pooled else None)
        k_disp = min(n_display, e - s) if n_display else (e - s)
        tables.append(latest_table_from_raw(raw_slice, specs, bp_slice, lc_slice, y_true, k_disp,
                                            _extract_timestamps(None, lc_slice, timestamp_col), b["name"]))

    result = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    if verbose:
        print(f"\n🚀 دُفعة واحدة مدمجة ({len(pooled['asset_bounds'])} أصل، استدعاء نموذج واحد فقط):")
        print_latest_table(result, legend=True)
    return result


def write_prediction_table(spec, test_dict, raw_by_asset, names):
    """يكتب مخرجات النموذج الخام (``raw_by_asset`` من slice_by_asset) كـ PredictionTable واحد بترتيب ``names`` (stats/prediction_table.py،
    ``spec`` = PredictionTableSpec): صف لكل (أصل، طابع) بأعمدة prediction_made_at / label_available_at / train_end / fold / role /
    identity_hash ومخرجات الرؤوس ``out_<مفتاح>``. اختياري: بلا spec لا يُستدعى. يُرجع مسار الملف؛ مخالفة العقد ترفع PredictionTableError."""
    names = list(names)
    no_ts = [a for a in names if test_dict[a].get('last_candles') is None]
    if no_ts:
        raise ValueError(f"prediction_table needs last_candles (its timestamp column) on every asset; missing on {no_ts[:5]}")
    ts = np.concatenate([np.asarray(test_dict[a]['last_candles'])[:, _TS_COL] for a in names])
    assets = np.concatenate([np.full(len(test_dict[a]['last_candles']), a, dtype=object) for a in names])
    keys = list(raw_by_asset[names[0]])
    outputs = {k: np.concatenate([np.asarray(raw_by_asset[a][k]).reshape(len(raw_by_asset[a][k]), -1) for a in names]) for k in keys}
    table = _build_prediction_table(asset=assets, timestamp=ts, outputs=outputs, split_role=spec.split_role, train_end=spec.train_end,
                                    fold=spec.fold, identity_hash=spec.identity_hash, horizon_bars=spec.horizon_bars,
                                    timeframe=spec.timeframe)
    return _write_prediction_table(spec.path, table, spec.meta)
