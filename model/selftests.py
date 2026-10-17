"""
PURPOSE:  run_model_selftests: data-free self-test of registry, head expansion and full save/load; runs at load.
TAGS:     run_model_selftests, self-test, selftest, smoke, head registry, save load
PITFALLS: RUNS AT LOAD (last statement). Tests and tools that only need the model exclude this module: load_into(ns,
          exclude=('selftests',)). Executed into the one shared model namespace by model/_loader.py (never imported on
          its own): names from other modules resolve at call time.

## 10) اختبار ذاتي (بلا بيانات حقيقية) — يثبت أن السجلّ والتوسعة والحفظ الكامل تعمل

يبني نموذجاً بالإعداد الافتراضي (تصنيف+انحدار معاً، تحقّق من مفاتيح
المخرجات واحتمالات صالحة)، يعطّل التصنيف بالكامل عبر `config` ليثبت أن
الاتجاه المعاكس يعمل أيضاً، يوسّع بإضافة رأس متعدّد الفئات لهدف واحد،
ثم — الأهم — **يحفظ نموذجاً مبنياً كاملاً إلى القرص ويُعيد تحميله بدون أي
استدعاء لاحق لـ`build_model_fn`**، للتأكّد أن `model.save(...)` يكفي وحده
(لا حاجة لإعادة البناء ثم `load_weights` فقط).
"""
def run_model_selftests(verbose: bool = True) -> bool:
    seq_len, n_features = 32, 38
    batch = 4
    x = np.random.randn(batch, seq_len, n_features).astype("float32")

    # 1) الإعداد الافتراضي: تصنيف + انحدار معاً لكل هدف (يطابق enabled_heads
    #    الافتراضي في خط الأنابيب) — 21 مخرج NIG + 3 احتمالات تصنيف ثنائي
    model_default = build_model_fn(seq_len, n_features)
    out = model_default(x, training=False)
    expected_default = set()
    for t in ("high", "low", "close"):
        expected_default |= {f"y_{t}", f"y_{t}_nu", f"y_{t}_alpha", f"y_{t}_beta",
                              f"y_{t}_epistemic", f"y_{t}_aleatoric", f"y_{t}_confidence",
                              f"y_{t}_class_logits"}
    assert set(out.keys()) == expected_default, (set(out.keys()), expected_default)
    for k, v in out.items():
        assert v.shape == (batch, 1), f"{k}: {v.shape}"
    for t in ("high", "low", "close"):
        p = out[f"y_{t}_class_logits"].numpy()
        assert np.all((p >= 0.0) & (p <= 1.0)), f"y_{t}_class_logits ليست احتمالات صالحة: {p}"

    # 2) high>=close>=low يجب أن يصمد فعلياً (enforce_order) — بمعزل عن التصنيف
    h, c, l = out["y_high"].numpy(), out["y_close"].numpy(), out["y_low"].numpy()
    assert np.all(h >= c - 1e-5) and np.all(c >= l - 1e-5), "OrderedMeans لم يحفظ الترتيب"

    # 3) تعطيل التصنيف بالكامل عبر config — يعيد سلوك النسخة السابقة تماماً
    reg_only = {t: ["nig_regression"] for t in ("high", "low", "close")}
    model_reg_only = build_model_fn(seq_len, n_features, config={"head_types": reg_only})
    out_reg_only = model_reg_only(x, training=False)
    assert not any(k.endswith("_class_logits") for k in out_reg_only), "تعطيل التصنيف لم يُطبَّق"
    assert len(out_reg_only) == 21

    # 4) توسعة إضافية بلا لمس كود المعمارية: رأس متعدّد فئات فوق ما هو مُفعَّل أصلاً
    extended_heads = {"high": ["nig_regression", "binary_classification"],
                       "low": ["nig_regression", "binary_classification"],
                       "close": ["nig_regression", "binary_classification", "multiclass_classification"]}
    model_ext = build_model_fn(seq_len, n_features,
                                config={"head_types": extended_heads, "n_classes": 3})
    out_ext = model_ext(x, training=False)
    assert out_ext["y_close_class_logits"].shape == (batch, 3), "الرأس متعدّد الفئات لم يظهر بالشكل الصحيح"
    probs = out_ext["y_close_class_logits"].numpy()
    assert np.allclose(probs.sum(axis=-1), 1.0, atol=1e-4), "مخرج multiclass ليس توزيع احتمالات (softmax)"

    # 5) نوع رأس غير مسجَّل يرفع خطأً واضحاً لا فشلاً صامتاً
    try:
        build_model_fn(seq_len, n_features, config={"head_types": {"close": ["does_not_exist"]}})
        raise AssertionError("كان يجب رفع ValueError لنوع رأس غير مسجَّل")
    except ValueError:
        pass

    # 6) حفظ كامل + تحميل بلا أي إعادة بناء (لا build_model_fn ولا custom_objects) —
    #    كل الطبقات المخصَّصة مُسجَّلة (register_keras_serializable) وتُبنى صراحةً
    #    في build() (لا انتظار أول call)، فالحفظ الأصلي (native .keras) يكفي وحده.
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.keras")
        model_default.save(path)
        reloaded = tf.keras.models.load_model(path)
        out_reloaded = reloaded(x, training=False)
        assert set(out_reloaded.keys()) == expected_default
        for k in expected_default:
            assert np.allclose(out[k].numpy(), out_reloaded[k].numpy(), atol=1e-5), (
                f"تباين بعد التحميل في {k} — النموذج المُعاد لا يطابق الأصلي")

    # 7) إعداد مقاومة الحفظ: يُبنى ويُحفظ ويُحمَّل كاملاً، عشوائي في التدريب فقط وحتمي في الاستدلال
    model_am = build_model_fn(seq_len, n_features, config={**ANTI_MEMORIZATION_CONFIG, "level_passthrough": True,
                                                           "level_norm": "batch"})
    out_a = model_am(x, training=False)
    out_b = model_am(x, training=False)
    for k in out_a:
        assert np.allclose(out_a[k].numpy(), out_b[k].numpy()), f"{k}: الاستدلال ليس حتمياً"
    tr_a = model_am(x, training=True)["y_close"].numpy()
    tr_b = model_am(x, training=True)["y_close"].numpy()
    assert not np.allclose(tr_a, tr_b), "الضجيج/الإسقاط لا يعملان أثناء التدريب"
    out_a = model_am(x, training=False)   # مرجع بعد استدعاءات التدريب (تُحدِّث متوسطات BatchNorm المتحرّكة)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model_am.keras")
        model_am.save(path)
        out_re = tf.keras.models.load_model(path)(x, training=False)
        for k in out_a:
            assert np.allclose(out_a[k].numpy(), out_re[k].numpy(), atol=1e-5), f"{k}: تباين بعد التحميل"

    # 8) stats_mode="symlog" يحصر مستوى ميزة غير مُطبَّعة (سعر خام ~10^5) في بضع وحدات بدل تمريره كما هو
    x_lvl = x.copy()
    x_lvl[..., 0] = 6.5e4 + 100.0 * x_lvl[..., 0]
    _, st_full = InstanceNorm(stats_mode="full")(tf.constant(x_lvl))
    _, st_sym = InstanceNorm(stats_mode="symlog")(tf.constant(x_lvl))
    assert float(tf.reduce_max(tf.abs(st_full))) > 6e4 and float(tf.reduce_max(tf.abs(st_sym))) < 12.0

    # 9) البدائل المعمارية (encoder/FiLM/برجان/تضمين العملة): تُبنى وتُحفظ وتُحمَّل بمخرجات مطابقة
    tiny = dict(d_model=16, num_layers=1, num_heads=2, num_kv_heads=1, head_hidden=16, class_head_hidden=8)
    variants = {"tcn_film": dict(encoder="tcn", level_film=True), "gru": dict(encoder="gru"),
                "two_tower_gate_coin": dict(architecture="two_tower", fusion="gate", n_coins=5),
                "film_coin": dict(level_film=True, n_coins=5)}
    for vname, v in variants.items():
        m = build_model_fn(seq_len, n_features, config={**ANTI_MEMORIZATION_CONFIG, **tiny, **v,
                                                          "level_passthrough": True, "level_norm": "batch"})
        inp = {"input_sequence": x, "coin_id": np.arange(batch) % 5} if v.get("n_coins") else x
        o1 = m(inp, training=False)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, f"{vname}.keras")
            m.save(path)
            o2 = tf.keras.models.load_model(path)(inp, training=False)
        for k in o1:
            assert np.allclose(o1[k].numpy(), o2[k].numpy(), atol=1e-5), f"{vname}/{k}: تباين بعد التحميل"

    # 10) متعدّد الفريمات (1h+4h): فرع لكل فريم بأوزان منفصلة، دمج بـ branch_concat، تغذية بقاموس أو قائمة بنفس النتيجة،
    #     وتدرّج يصل الفرعين؛ وفريم واحد (عدد أو قاموس بفريم) يبقى البناء القديم نفسه (أسماء الطبقات وعدد المعاملات).
    ms = {**ANTI_MEMORIZATION_CONFIG, **tiny, "level_passthrough": True, "level_norm": "batch"}
    m_one = build_model_fn(seq_len, n_features, config=ms)
    m_one_d = build_model_fn({"1h": seq_len}, {"1h": n_features}, config=ms)
    assert [l.name for l in m_one.layers] == [l.name for l in m_one_d.layers] and \
        m_one.count_params() == m_one_d.count_params(), "قاموس بفريم واحد يجب أن يبني النموذج القديم نفسه"
    m_two = build_model_fn({"1h": seq_len, "4h": 24}, n_features, config=ms)
    x4 = np.random.randn(batch, 24, n_features).astype("float32")
    o_d = m_two({"1h": x, "4h": x4}, training=False)
    o_l = m_two([x, x4], training=False)
    for k in o_d:
        assert np.allclose(o_d[k].numpy(), o_l[k].numpy(), atol=1e-6), f"{k}: قاموس ≠ قائمة"
    names = [l.name for l in m_two.layers]
    assert len(names) == len(set(names)) and "branch_concat" in names
    assert {"instance_norm_1h", "instance_norm_4h"} <= set(names) and "instance_norm" not in names
    assert m_two.count_params() > 1.6 * m_one.count_params(), "الفرعان لا يملكان أوزاناً منفصلة"
    with tf.GradientTape() as tape:
        loss = tf.add_n([tf.reduce_mean(tf.square(v)) for v in m_two({"1h": x, "4h": x4}, training=True).values()])
    grads = dict(zip([w.path for w in m_two.trainable_weights], tape.gradient(loss, m_two.trainable_weights)))
    for sfx in ("_1h", "_4h"):
        assert any(g is not None and float(tf.reduce_sum(tf.abs(g))) > 0 for p, g in grads.items() if sfx in p), \
            f"لا تدرّج يصل فرع {sfx}"

    if verbose:
        print(f"✅ نجحت كل اختبارات النموذج الذاتية ({len(expected_default)} مخرجاً افتراضياً "
              f"شاملة التصنيف، تعطيل/توسعة عبر config فقط، وحفظ+تحميل كامل بلا إعادة بناء، وإعداد مقاومة الحفظ والبدائل المعمارية)")
    return True

run_model_selftests()
