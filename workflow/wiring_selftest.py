"""
PURPOSE:  Section 8: self-test of the wiring between the notebooks on synthetic data (no Drive, no real training).
TAGS:     run_wiring_selftest, wiring, self test, synthetic data, +1/-1 labels, chicks
PITFALLS: Skipped by tools/evaluate_trained_model.py (the cell calling run_wiring_selftest). Executed into the notebook's shared namespace by workflow/_loader.py (never imported on its own): names from other modules and the %run notebooks resolve at call time. Extracted verbatim from main.ipynb cell 44 (section 8).
"""
import dataclasses


def run_wiring_selftest(kit, verbose=True):
    """kit: workflow.run.Toolkit (يوفّر مواصفات الفكّ الافتراضية في chicks)؛ لا شيء يُقرأ من نطاق الدفتر."""
    # أهداف بيانات الاختبار الوهمية (high/low/close بمقياس 1) ومواصفات فكّها تُبنى هنا
    price_targets = ("high", "low", "close")
    eval_specs = [dataclasses.replace(s, relative_to_entry=True, reg_scale=1.0) for s in kit.default_price_targets]
    # يثبّت التحويل نفسه أولاً — تأكيد حتمي بمعزل عن عشوائية التدريب
    assert _to_unit_label(np.array([-1.0, 1.0])).tolist() == [0.0, 1.0], "_to_unit_label خاطئة"
    assert _to_unit_label(np.array([0.0, 1.0])).tolist() == [0.0, 1.0], "_to_unit_label خاطئة (ترميز 1/0)"

    rng = np.random.default_rng(0)
    seq_len, n_features = 16, 10
    tf_name = "1D"

    def make_split(n):
        X = rng.normal(size=(n, seq_len, n_features)).astype("float32")
        last_close = 100 + rng.normal(size=(n,)) * 5
        last_high = last_close + np.abs(rng.normal(size=(n,)))
        last_low = last_close - np.abs(rng.normal(size=(n,)))
        ret = {t: rng.normal(scale=0.01, size=(n,)).astype("float32") for t in ("high", "low", "close")}
        # ترميز خط الأنابيب الحقيقي هو +1.0/-1.0 (اتجاه)، لا {0,1} — القسم ٠.٤
        cls = {t: (rng.integers(0, 2, size=(n,)).astype("float32") * 2 - 1) for t in ("high", "low", "close")}
        last_candles = np.stack([
            last_high, last_low, last_close, np.arange(n).astype("float64"),
            last_close * (1 + ret["close"]), last_low * (1 + ret["low"]), last_high * (1 + ret["high"]),
        ], axis=1)
        y = {f"y_{t}_reg": v for t, v in ret.items()}
        y.update({f"y_{t}_class": v for t, v in cls.items()})
        return {
            f"X_{tf_name}": X,
            "y": y,
            "base_params": np.zeros((n, 2), dtype="float32"),
            "last_candles": last_candles,
        }

    fake_train, fake_val = make_split(256), make_split(64)
    fake_test = {"A": make_split(50), "B": make_split(40)}

    fake_model_builder = lambda: build_model_fn(seq_len, n_features)
    fake_config = build_config({
        "run": {"run_dir": "/tmp/_wiring_selftest_run", "epochs": 1, "batch_size": 32,
                "verbose": 0, "train_mode": "new"},
        "targets": build_target_configs(("high", "low", "close")),
    })
    fake_train_ds = tf.data.Dataset.from_tensor_slices(
        (fake_train[f"X_{tf_name}"], _y_for_targets(fake_train, fake_config))
    ).batch(32, drop_remainder=True)
    fake_val_ds = tf.data.Dataset.from_tensor_slices(
        (fake_val[f"X_{tf_name}"], _y_for_targets(fake_val, fake_config))
    ).batch(32, drop_remainder=True)
    sample = next(iter(fake_train_ds))

    fake_trainer, fake_callbacks, fake_ie = build_training_system(fake_model_builder, fake_config, sample)
    fake_history = fake_trainer.fit(
        fake_train_ds, validation_data=fake_val_ds, initial_epoch=fake_ie,
        epochs=fake_config["run"]["epochs"], callbacks=fake_callbacks, verbose=0)
    fake_model = fake_trainer.model

    # التحقّق أن أهداف التصنيف فعلياً دخلت التدريب (مقياس accuracy مُسجَّل لكل منها)
    for t in ("high", "low", "close"):
        assert any(k.startswith(f"{t}_class") and "accuracy" in k for k in fake_history.history), (
            f"لا مقياس accuracy لهدف {t}_class — رأس التصنيف لم يُدرَّب فعلياً")

    fake_test_dict = {}
    for asset, split in fake_test.items():
        last_close = split["last_candles"][:, LAST_CLOSE_COL]
        fake_test_dict[asset] = {
            f"X_{tf_name}": split[f"X_{tf_name}"],
            "base_params": np.stack([last_close, last_close], axis=1).astype("float32"),
            "last_candles": split["last_candles"],
            "y": {t: split["y"][f"y_{t}_reg"] for t in ("high", "low", "close")},
        }

    results = run_full_analysis(
        model=fake_model, test_dict=fake_test_dict, timeframes=[tf_name],
        target_specs=eval_specs, make_plots=False, verbose=False,
        out_dir="/tmp/_wiring_selftest_analysis",
    )
    assert "per_asset_results" in results and len(results["per_asset_results"]) == 2

    # التقييم الانتقائي (القسم ٧-ب) على مخرجات النموذج الفعلية وبنية val (قسم واحد) وtest (قاموس أصول)
    sel = selective_evaluation(fake_model, fake_val, fake_test, tf_name, min_n=10, verbose=False, price_targets=price_targets)
    assert len(sel["val_df"]) == 64 and len(sel["test_df"]) == 90
    assert {"class_margin", "nig_edge", "conf_head", "agree_margin"} <= set(sel["direction"]["score"])

    # خطوط أساس شكل الشمعة (القسم ٧-ج) على نفس البيانات الوهمية
    cb = candle_baseline_report(fake_model, fake_train, fake_val, fake_test, tf_name, n_boot=20, verbose=False)
    assert set(cb["verdicts"]) == {"high", "low"} and len(cb["table"]) == 2 * 2 * 5

    # التحقق المتكامل (القسم ٧-د) — يعمل من طرفه لطرفه على مخرجات النموذج الفعلية
    ver = run_full_verification(fake_model, fake_train, fake_val, fake_test, tf_name, n_boot=50, verbose=False,
                                price_targets=price_targets)
    assert {"أ) البيانات", "ب) خطوط الأساس", "ج) الثقة", "ج) التداول", "د) شكل الشمعة"} <= set(ver["summary"]["section"])

    # المحفظة المحايدة (القسم ٧-و) — تعمل من طرفها لطرفها على مخرجات النموذج الفعلية وخط أساس GBM
    mn = market_neutral_report(fake_model, fake_train, fake_val, fake_test, tf_name, quantiles=(0.5,), min_assets=2,
                               min_per_leg=1, n_boot=20, verbose=False, price_targets=price_targets)
    assert {"model", "gbm"} <= set(mn) and mn["gbm"] is not None and len(mn["model"]["grid"]) == 2 * len(_MN_SIDES)

    # مقاومة الحفظ (القسم ٧-ز): بيانات عشوائية مُطبَّعة نظيفة، وميزة مستوى مزروعة يجب أن تُكشف
    aud = normalization_audit(fake_train, fake_val, fake_test, model_tf=tf_name, verbose=False)
    assert len(aud) == n_features and (aud["flags"] == "✅").all(), aud[aud["flags"] != "✅"]
    lvl_train = dict(fake_train)
    lvl_X = fake_train[f"X_{tf_name}"].copy()
    lvl_X[..., 0] = rng.uniform(0.01, 6.5e4, size=(len(lvl_X), 1)) * (1 + 0.01 * lvl_X[..., 0])
    lvl_train[f"X_{tf_name}"] = lvl_X
    aud_lvl = normalization_audit(lvl_train, model_tf=tf_name, verbose=False)
    assert "مستوى" in aud_lvl.loc[0, "flags"] and (aud_lvl.loc[1:, "flags"] == "✅").all()
    gap = generalization_gap_report(fake_model, fake_train, fake_val, fake_test, model_tf=tf_name, verbose=False,
                                    price_targets=price_targets)
    assert set(gap["verdicts"]) == {"high", "low", "close"}
    assert {"train", "val", "test", "gap_train_val"} <= set(gap["table"].columns)

    # تغيير الهدف (القسم ٣-ب): الأوضاع الأخرى تمرّ عبر ٧-ب و٧-د بلا استثناء، وتُتخطّى أجزاء 'return' بوضوح
    for mode in ("magnitude", "relative", "scaled+relative", "volnorm+relative", "entry_range"):
        r_tr, r_va, r_te = retarget_splits(fake_train, fake_val, fake_test, mode=mode, group_freq="1D",
                                           verbose=False, reg_target_scale=1.0, price_targets=price_targets)
        assert all(s["target_mode"] == mode for s in [r_tr, r_va, *r_te.values()])
        assert selective_evaluation(fake_model, r_va, r_te, tf_name, verbose=False, price_targets=price_targets) is None
        ver_m = run_full_verification(fake_model, r_tr, r_va, r_te, tf_name, n_boot=20, verbose=False,
                                      price_targets=price_targets)
        assert (ver_m["summary"].query("section == 'ج) الثقة والتداول'")["verdict"] == "ℹ️").all()
    r_tr, _, _ = retarget_splits(fake_train, fake_val, fake_test, mode="return", verbose=False,
                                 reg_target_scale=1.0, price_targets=price_targets)
    for k, v in fake_train["y"].items():
        if k.endswith("_reg"):   # الأهداف الوهمية هنا: الانحدار متّسق مع الأسعار، التصنيف عشوائي عمداً
            assert np.allclose(r_tr["y"][k], v, atol=1e-6), k

    # الدفعة المدمجة (predict_pooled_batch_by_asset): يجب أن تُعطي نتائج مطابقة عملياً
    # لاستدعاء منفصل لكل أصل (predict_latest_all_assets) رغم أنها تستدعي predict_batch_v4
    # مرّة واحدة فقط لكل الأصول معاً بدل استدعاء منفصل لكل أصل — التحقّق هنا رقمي
    # (تطابق القيم)؛ استدعاء النموذج مرّة واحدة فقط خاصية بنيوية واضحة من كود الدالة نفسها.
    # تسامح (rtol/atol) بدل التطابق الحتمي: تركيب الدفعة يُغيّر ترتيب عمليات الجمع
    # العائم (matmul/reduction) فيُنتج فروقاً دقيقة (~1e-5 على قيم بمئات) لا علاقة
    # لها بصحّة الحساب — النموذج لا يحوي BatchNorm فهو مستقلّ عن تركيب الدفعة رياضياً.
    pooled, y_true_pooled = pool_test_dict(fake_test_dict, tf_name)
    assert [b["name"] for b in pooled["asset_bounds"]] == list(fake_test_dict.keys())
    pooled_table = predict_pooled_batch_by_asset(
        fake_model, pooled, tf_name, target_specs=eval_specs,
        n_display=1000, y_true_pooled=y_true_pooled, verbose=False)
    separate_table = predict_latest_all_assets(
        fake_model, fake_test_dict, timeframes=[tf_name], target_specs=eval_specs,
        n_display=1000, verbose=False)
    for asset in fake_test_dict:
        for t in ("high", "low", "close"):
            a = pooled_table.query("asset == @asset and target == @t")["pred"].to_numpy()
            b = separate_table.query("asset == @asset and target == @t")["pred"].to_numpy()
            assert a.shape == b.shape and np.allclose(a, b, rtol=1e-3, atol=1e-2, equal_nan=True), (
                f"الدفعة المدمجة تختلف عن الحلقة اليدوية لـ{asset}/{t}")
    if verbose:
        print("  ✅ predict_pooled_batch_by_asset: نتائج مطابقة (بحدود دقّة float32) لاستدعاء منفصل لكل أصل، "
              "باستدعاء نموذج واحد فقط للأصلين معاً بدل استدعاء لكل أصل")

    # مسار التصنيف المنفصل (classification_accuracy_report) بنفس المنطق، على البيانات الوهمية —
    # بترميز +1.0/-1.0 المُحوَّل عبر _to_unit_label، تماماً كما في classification_accuracy_report الفعلية
    from sklearn.metrics import accuracy_score
    for asset, split in fake_test_dict.items():
        out = fake_model(split[f"X_{tf_name}"], training=False)
        for t in ("high", "low", "close"):
            y_true = _to_unit_label(np.asarray(fake_test[asset]["y"][f"y_{t}_class"]))
            assert set(np.unique(y_true).tolist()) <= {0.0, 1.0}, "التحويل لم ينتج {0,1}"
            y_prob = out[f"y_{t}_class_logits"].numpy().ravel()
            accuracy_score(y_true, (y_prob >= 0.5).astype("float32"))  # لا يرفع استثناءً يكفي هنا

    # متعدّد الفريمات: نفس المسار بفريمين (tf_name أساسي + سياق 4h): تغذية بقاموس، تدريب حقبة، chicks، دفعة مدمجة، إشارات
    tf2, seq2 = "4h", 8
    for s in (fake_train, fake_val, *fake_test.values()):
        s[f"X_{tf2}"] = rng.normal(size=(len(s["last_candles"]), seq2, n_features)).astype("float32")
    for a in fake_test_dict:
        fake_test_dict[a][f"X_{tf2}"] = fake_test[a][f"X_{tf2}"]
    two = [tf_name, tf2]
    two_builder = lambda: build_model_fn({tf_name: seq_len, tf2: seq2}, n_features)
    two_config = build_config({
        "run": {"run_dir": "/tmp/_wiring_selftest_run_2tf", "epochs": 1, "batch_size": 32,
                "verbose": 0, "train_mode": "new"},
        "targets": build_target_configs(("high", "low", "close")),
    })
    two_ds = make_shuffled_dataset(model_x(fake_train, two), _y_for_targets(fake_train, two_config), 32)
    two_val = (tf.data.Dataset.from_tensor_slices((model_x(fake_val, two), _y_for_targets(fake_val, two_config)))
               .batch(32, drop_remainder=True).map(_to_float32_inputs))
    two_trainer, two_cbs, two_ie = build_training_system(two_builder, two_config, next(iter(two_ds)))
    two_trainer.fit(two_ds, validation_data=two_val, initial_epoch=two_ie, epochs=1, callbacks=two_cbs, verbose=0)
    two_model = two_trainer.model
    assert [i.name for i in two_model.inputs] == two, [i.name for i in two_model.inputs]
    assert (len(collect_signals(two_model, fake_val, two, price_targets=price_targets)) == 64
            and len(collect_signals(two_model, fake_test, two, price_targets=price_targets)) == 90)
    two_res = run_full_analysis(model=two_model, test_dict=fake_test_dict, timeframes=two,
                                target_specs=eval_specs, make_plots=False, verbose=False,
                                out_dir="/tmp/_wiring_selftest_analysis_2tf")
    assert "per_asset_results" in two_res and len(two_res["per_asset_results"]) == 2
    two_pooled, two_y = pool_test_dict(fake_test_dict, two)
    two_tab = predict_pooled_batch_by_asset(two_model, two_pooled, two, target_specs=eval_specs,
                                            n_display=1000, y_true_pooled=two_y, verbose=False)
    two_sep = predict_latest_all_assets(two_model, fake_test_dict, timeframes=two, target_specs=eval_specs,
                                        n_display=1000, verbose=False)
    for asset in fake_test_dict:
        for t in ("high", "low", "close"):
            a = two_tab.query("asset == @asset and target == @t")["pred"].to_numpy()
            b = two_sep.query("asset == @asset and target == @t")["pred"].to_numpy()
            assert a.shape == b.shape and np.allclose(a, b, rtol=1e-3, atol=1e-2, equal_nan=True), (
                f"متعدّد الفريمات: الدفعة المدمجة تختلف عن الحلقة اليدوية لـ{asset}/{t}")
    if verbose:
        print("  ✅ متعدّد الفريمات (فريمان): تدريب حقبة بتغذية القاموس، chicks، الدفعة المدمجة، والإشارات")

    # نموذج اللوحة عبر العملات (القسم ٧-ح): تجميع الأيام، القناع، الخسائر، تدريب قصير، استئناف وتصدير —
    # بالمُرمِّز الحقيقي (build_model_fn بإعداد مقاومة الحفظ). يُتخطّى إن لم تكن الحزمة بجانب الدفاتر.
    try:
        from cross_asset.selftest import run_panel_selftest
    except ImportError:
        print("  ⏭️ اختبار نموذج اللوحة متخطّى (مجلد cross_asset غير موجود بجانب الدفاتر)")
    else:
        run_panel_selftest(lambda: build_model_fn(seq_len, n_features, config=ANTI_MEMORIZATION_CONFIG),
                           seq_len, n_features, verbose=verbose, strict=False)

    if verbose:
        print("✅ نجح اختبار التوصيل: بيانات ← نموذج (انحدار+تصنيف، بترميز +1/-1 الحقيقي) ← "
              f"تدريب ← chicks + تقرير تصنيف مستقلّ، بلا أي استثناء، عبر {len(results)} تقريراً من chicks")
    return True


def _y_for_targets(split, config):
    y = {}
    for cfg in config["targets"].values():
        v = split["y"][cfg["true_key"]]
        y[cfg["true_key"]] = _to_unit_label(v) if cfg["task_type"] == "classification" else v
    return y
