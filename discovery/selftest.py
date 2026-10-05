"""
PURPOSE:  Synthetic-data self-test of the whole lab (run_discovery_lab_selftest): no Drive, checks the predictors, scanner and batch runner.
TAGS:     run_discovery_lab_selftest, self test, synthetic, lab
PITFALLS: Reads FEATURE_ORDER from the notebook namespace when present. The notebook cell calls it right after loading. Executed into the notebook's shared namespace by discovery/_loader.py (never imported on its own): names from the axis/pipeline notebooks resolve at call time. Extracted verbatim from signal_discovery_lab.ipynb cell 92 (section 29).
"""
import numpy as np
import pandas as pd


def run_discovery_lab_selftest():
    rng = np.random.default_rng(0)
    n_assets, n_per_asset, T, F = 3, 200, 8, len(FEATURE_ORDER) if 'FEATURE_ORDER' in globals() else 37
    feature_order = FEATURE_ORDER if 'FEATURE_ORDER' in globals() else [f"f{i}" for i in range(F)]
    n = n_assets * n_per_asset
    ts0 = pd.Timestamp("2022-01-01", tz="UTC")
    ts = pd.concat([pd.Series(pd.date_range(ts0, periods=n_per_asset, freq="1D"))
                    for _ in range(n_assets)], ignore_index=True)

    X = rng.normal(size=(n, T, F)).astype("float32")
    last_close = 100.0 * np.exp(rng.normal(scale=0.05, size=n).cumsum() / n_per_asset)
    body_idx = feature_order.index("BODY_ratio") if "BODY_ratio" in feature_order else 0
    # ✅ نزرع أثر مرجع "نفس النوع" عمداً: last_high يعتمد على BODY_ratio لا على
    #    حركة سعرية حقيقية — clean_reg_target يجب أن يُزيله، والهدف الخام (لو
    #    استُخدم بالخطأ) يجب أن يُظهره بوضوح.
    body_last = X[:, -1, body_idx]
    last_high = last_close * (1.0 + np.clip(-body_last, 0, None) * 0.05 + 1e-3)
    last_low = last_close * (1.0 - np.clip(body_last, 0, None) * 0.05 - 1e-3)
    future_high_max = last_close * (1.0 + rng.normal(scale=0.01, size=n))  # لا علاقة حقيقية بـbody_last
    future_low_min = last_close * (1.0 - np.abs(rng.normal(scale=0.01, size=n)))
    future_close = last_close * (1.0 + rng.normal(scale=0.01, size=n))

    y_high_reg_dirty = (future_high_max - last_high) / last_high  # مرجع "نفس النوع" (ملوَّث)
    y_low_reg_dirty = (future_low_min - last_low) / last_low
    y_close_reg = (future_close - last_close) / last_close

    last_candles = np.stack([last_high, last_low, last_close, ts.values.astype("int64"),
                             future_close, future_low_min, future_high_max], axis=1)
    flat = {"base_params": np.zeros((n, 2), "float32"), "last_candles": last_candles,
            "X_1D": X, "y": {"y_high_reg": y_high_reg_dirty, "y_low_reg": y_low_reg_dirty,
                             "y_close_reg": y_close_reg}}

    # ١) الحارس يُزيل الأثر المزروع فعلاً
    clean_high = clean_reg_target(flat, "high")
    from scipy.stats import spearmanr
    rho_dirty, _ = spearmanr(body_last, y_high_reg_dirty)
    rho_clean, _ = spearmanr(body_last, clean_high)
    assert abs(rho_dirty) > 0.3, f"الأثر المزروع ضعيف جداً للاختبار ({rho_dirty:.3f}) — أصلح البيانات التركيبية."
    assert abs(rho_clean) < abs(rho_dirty) / 3, (
        f"❌ clean_reg_target لم يُزل الأثر المزروع: dirty={rho_dirty:.3f} clean={rho_clean:.3f}")
    print(f"  ✅ clean_reg_target يُزيل أثر مرجع نفس النوع (dirty={rho_dirty:+.3f} → clean={rho_clean:+.3f})")

    # ٢) extract_feature_last_value / make_feature_predict_fn يعملان
    v = extract_feature_last_value(flat, feature_order[0], tf="1D", feature_order=feature_order)
    assert v.shape == (n,), "extract_feature_last_value: شكل خاطئ."
    predict_fn = make_feature_predict_fn(feature_order[0], transform=lambda x: -x,
                                         tf="1D", feature_order=feature_order)
    preds = predict_fn(flat, flat, flat)
    assert np.allclose(preds, -v), "make_feature_predict_fn: التحويل لم يُطبَّق بشكل صحيح."
    print("  ✅ extract_feature_last_value / make_feature_predict_fn تعملان بشكل صحيح")

    # ٢-ب) make_candidate_predict_fn: كل الأنواع الأربعة (feature/interaction/custom/trained)
    feat_b = feature_order[1] if len(feature_order) > 1 else feature_order[0]
    inter_cand = {"kind": "interaction", "feat_a": feature_order[0], "feat_b": feat_b, "op": "mul"}
    inter_fn = make_candidate_predict_fn(inter_cand, tf="1D", feature_order=feature_order)
    a = extract_feature_last_value(flat, feature_order[0], tf="1D", feature_order=feature_order)
    b = extract_feature_last_value(flat, feat_b, tf="1D", feature_order=feature_order)
    assert np.allclose(inter_fn(flat, flat, flat), a * b), "make_candidate_predict_fn: مسار التفاعل خاطئ."

    custom_cand = {"kind": "custom", "fn": lambda X_last, fo: X_last[:, 0] * 2.0}
    custom_fn = make_candidate_predict_fn(custom_cand, tf="1D", feature_order=feature_order)
    X_last_expected = extract_feature_matrix(flat, tf="1D", feature_order=feature_order)
    assert np.allclose(custom_fn(flat, flat, flat), X_last_expected[:, 0] * 2.0), (
        "make_candidate_predict_fn: مسار kind='custom' خاطئ.")

    trained_cand = {"kind": "trained", "builder": lambda feature_order: make_isolation_forest_predict_fn(
        [feature_order[0], feat_b], feature_order=feature_order, contamination=0.1)}
    trained_fn = make_candidate_predict_fn(trained_cand, feature_order=feature_order)
    trained_preds = trained_fn(flat, flat, flat)
    assert trained_preds.shape == (n,), "make_candidate_predict_fn: مسار kind='trained' أرجع شكلاً خاطئاً."

    series_cand = {"kind": "series", "feature": feature_order[0], "fn": lambda s: s[:, -1] * 3.0}
    series_fn = make_candidate_predict_fn(series_cand, tf="1D", feature_order=feature_order)
    series_full = extract_feature_series(flat, feature_order[0], tf="1D", feature_order=feature_order)
    assert series_full.shape == (n, T), "extract_feature_series: شكل خاطئ."
    assert np.allclose(series_fn(flat, flat, flat), series_full[:, -1] * 3.0), (
        "make_candidate_predict_fn: مسار kind='series' خاطئ.")
    print("  ✅ make_candidate_predict_fn يدعم الأنواع الخمسة (feature/interaction/custom/trained/series)")

    # ٣) scan_candidates يُرجع لوحة قيادة بالأعمدة المتوقَّعة، بلا انهيار
    windows_synth = [(flat, flat, flat)]
    tiny_candidates = [{"name": "f0", "track": "data_driven", "feature": feature_order[0], "transform": None}]
    board = scan_candidates(tiny_candidates, windows_synth, targets=("close", "high", "low"),
                            feature_order=feature_order, n_shuffles=20, min_samples=5)
    expected_cols = {"name", "track", "target", "status"}
    assert expected_cols.issubset(board.columns), f"أعمدة ناقصة في اللوحة: {board.columns.tolist()}"
    assert (board["status"] == "ok").all(), f"فشل تقييم بعض المرشّحين:\n{board}"
    print("  ✅ scan_candidates يُرجع لوحة قيادة سليمة بلا أخطاء")

    # ٤) classify_result: منطق حتمي بمعزل عن أي تدريب/عشوائية
    assert classify_result({"n_ok": 2, "consistent_sign": True, "frac_significant": 1.0}) == "قيد الاختبار", (
        "classify_result: n_ok قليل يجب أن يُرجع 'قيد الاختبار' بصرف النظر عن باقي الحقول")
    assert classify_result({"n_ok": 10, "consistent_sign": True, "frac_significant": 0.5}) == "مقبولة", (
        "classify_result: consistent_sign=True + frac_significant كافية يجب أن يُرجع 'مقبولة'")
    assert classify_result({"n_ok": 10, "consistent_sign": False, "frac_significant": 0.9}) == "مرفوضة", (
        "classify_result: consistent_sign=False يجب أن يُرجع 'مرفوضة' مهما كانت frac_significant")
    assert classify_result({"n_ok": 10, "consistent_sign": True, "frac_significant": 0.1}) == "مرفوضة", (
        "classify_result: frac_significant دون الحدّ الأدنى يجب أن يُرجع 'مرفوضة'")
    print("  ✅ classify_result يُطبِّق معيار القبول الموحّد بشكل صحيح (٤ حالات)")

    # ٥) run_batch_and_register: تسجيل فعلي إلى ملف مؤقّت (لا experiment_registry الحقيقي إطلاقاً)
    import tempfile, json as _json
    from pathlib import Path
    tmp_registry = Path(tempfile.mkdtemp()) / "registry_selftest.json"
    batch_board, registered_ids = run_batch_and_register(
        tiny_candidates, windows_synth, targets=("close", "high", "low"), feature_order=feature_order,
        id_prefix="SELFTEST", max_workers=2, registry_path=tmp_registry, n_shuffles=20, min_samples=5)
    assert len(registered_ids) == 3, f"يُتوقَّع تسجيل 3 (مرشّح واحد × 3 أهداف)، وُجد {len(registered_ids)}"
    assert tmp_registry.exists(), "run_batch_and_register: لم يُكتَب ملف السجلّ المؤقّت إطلاقاً"
    entries = _json.loads(tmp_registry.read_text(encoding="utf-8"))
    assert {e["id"] for e in entries} == set(registered_ids), "معرّفات السجلّ المكتوبة لا تطابق registered_ids"
    assert all(e["status"] in REGISTRY_STATUSES for e in entries), "حالة غير صالحة في سجلّ مكتوب فعلياً"
    assert all(e["id"].startswith("SELFTEST_") for e in entries), "id_prefix لم يُطبَّق على المعرّفات"
    print(f"  ✅ run_batch_and_register يقيّم بالتوازي (imap_ordered/default_workers) ويسجّل فعلياً "
          f"في ملف JSON مستقلّ ({len(registered_ids)} مدخلات، registry_path مُخصَّص لا الحقيقي)")

    print("✅ نجحت كل اختبارات مختبر بحث الإشارات الذاتية.")
