"""أدوات تشخيص النموذج (اختيارية، معطَّلة افتراضياً) على بيانات تركيبية صغيرة بلا Drive:  python -m pytest tests/test_model_diagnostics.py -q

تُنفَّذ خلايا الدفاتر نفسها (لا نسخاً):
  model (ex model_v2) — diagnose_model / model_health_verdicts / layer_probe_report / layer_compare_report / random_init_copy
  trainer       — TrainingDiagnostics (+ with_sample_index) فوق GenericTrainer الحقيقي
  main          — make_training_diagnostics / model_health_report / model_layer_report (مع تعريفات main الصغيرة المستخرَجة بـ ast)

ما يُثبَت: (١) لا تغيير: أوزان التدريب مطابقة بمسجّل وبدونه، والنموذج لا يتغيّر بعد التشخيص (أسماء، أوزان، مخارج، عشوائية بايثون)؛
(٢) المخرجات سليمة الأشكال؛ (٣) دفعة مُعكَّسة التسميات تُصنَّف الأضرّ (تأثير الدفعات) وعيّناتها الأصعب (خريطة البيانات)؛
(٤) **أدلة لا تخمين**: نموذج عشوائي التهيئة لا يُعلَّم في أي مقارنة صائب/خاطئ، ومسبار الطبقات لا يتجاوز النسخة العشوائية فيه.
"""
import ast
import contextlib
import io
import json
import os
import random
import sys
import tempfile
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))
import _nbload  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

T_LEN, N_FEAT, BS = 8, 4, 32
FLIP = 3                                   # فهرس الدفعة المُعكَّسة تسمياتها (بلا خلط ⇒ العيّنات 96..127)


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _cells(name):
    return json.load(open(os.path.join(ROOT, name), encoding="utf-8"))["cells"]


def _run_cells(name, ns, skip=()):
    for c in _cells(name):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if any(s in src for s in skip):
            continue
        lines = [ln for ln in src.splitlines() if not ln.lstrip().startswith(("!", "%")) and ln.strip() != "run_model_selftests()"]
        _quiet(exec, compile("\n".join(lines), name, "exec"), ns)
    return ns


_NS = {}


def _ns():
    """نطاق واحد: model_v2 + المدرّب (دون خلايا المثال) + build_target_configs من main."""
    if not _NS:
        import tensorflow as tf  # noqa: F401
        ns = {"__name__": "audit_nb"}
        _quiet(_nbload.load_model, ns)
        _quiet(_nbload.load_trainer, ns)                # بلا Smoke Test ولا K-Fold كما كان
        _module_into("training_config", ns)                                                  # build_target_configs
        _NS.update(ns)
    return _NS


def _data(n, seed, flip=None):
    """إشارة: آخر خطوة للميزة 0 مقابل متوسّط النافذة. «منطقة ضجيج»: العيّنات التي مستوى ميزتها 3 (إزاحة العيّنة) > 0.67 تسمياتها عشوائية
    ⇒ يخطئ النموذج فيها بنسبة ~50% (وفي غيرها نادراً) وعلامتها (مستوى الميزة 3 عبر مسار stats) مرئية له، فيمكن في المبدأ فصل الصائب
    عن الخاطئ من التنشيطات. الدفعة FLIP (إن طُلب flip) كلها في المنطقة النظيفة."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, T_LEN, N_FEAT)).astype("float32")
    off = rng.normal(size=n).astype("float32")
    if flip is not None:
        off[flip * BS:(flip + 1) * BS] = -1.0
    x[:, :, 3] = x[:, :, 3] * 0.3 + off[:, None]
    sig = x[:, -1, 0] - x[:, :-1, 0].mean(1)
    noisy = off > 0.67
    s = np.where(noisy, rng.normal(size=n) * 2, sig + rng.normal(size=n) * 0.1)
    y = {"y_high_reg": (0.01 * s).astype("float32"), "y_low_reg": (-0.01 * s).astype("float32"),
         "y_high_class": (s > 0).astype("float32"), "y_low_class": (s < 0).astype("float32")}
    if flip is not None:
        sl = slice(flip * BS, (flip + 1) * BS)
        for k in ("y_high_class", "y_low_class"):
            y[k][sl] = 1 - y[k][sl]
        for k in ("y_high_reg", "y_low_reg"):
            y[k][sl] = -y[k][sl]
    return x, y


def _model_cfg(ns, encoder="gru", **over):
    heads = {t: ["nig_regression", "binary_classification"] for t in ("high", "low")}
    return dict(price_targets=("high", "low"), head_types=heads, d_model=16, num_layers=1, head_hidden=16, class_head_hidden=8,
                encoder=encoder, num_heads=2, num_kv_heads=1, dropout=0.1, **over)


def _train(diag_factory=None, n=1024, epochs=30, lr=3e-3, flip=None, seed=7, shuffle=False):
    """يدرّب GenericTrainer حقيقياً؛ يُرجع (trainer, diag, x, y). بلا خلط افتراضياً (دفعة i = العيّنات i·BS..)."""
    import tensorflow as tf
    ns = _ns()
    x, y = _data(n, 0, flip)
    cfg = ns["build_config"]({"run": {"run_dir": tempfile.mkdtemp(), "epochs": epochs, "batch_size": BS, "verbose": 0,
                                      "train_mode": "new", "seed": seed},
                              "optimizer": {"lr_initial": lr}, "targets": ns["build_target_configs"](("high", "low"))})
    mcfg = _model_cfg(ns)
    builder = lambda: ns["build_model_fn"](T_LEN, N_FEAT, config=mcfg)  # noqa: E731
    yi = ns["with_sample_index"](y)
    raw = tf.data.Dataset.from_tensor_slices((x, yi))
    if shuffle:
        raw = raw.shuffle(n, seed=1, reshuffle_each_iteration=True)
    ds = raw.batch(BS, drop_remainder=True)
    diag = diag_factory(x, y, yi) if diag_factory else None
    if diag is not None:
        ds = diag.tap(ds)
    trainer, callbacks, _ = _quiet(ns["build_training_system"], builder, cfg, next(iter(raw.batch(BS, drop_remainder=True))))
    _quiet(trainer.fit, ds, epochs=epochs, callbacks=callbacks + ([diag] if diag else []), verbose=0)
    return trainer, diag, x, y


_RUN = {}


def _main_run():
    """تدريب واحد مشترك (30 حقبة): دفعة معكوسة التسميات + مسجّل يقيس كل 4 دفعات (فيقع قياس عند الموضع FLIP في كل حقبة)."""
    if not _RUN:
        ns = _ns()
        report = lambda m: ns["layer_probe_report"](m, *_data(512, 0, FLIP), *_data(400, 3), max_train=512, max_val=400, n_rand=1)  # noqa: E731

        def factory(x, y, yi):
            xv, yv = _data(400, 3)
            return ns["TrainingDiagnostics"](probe=(xv, yv), probe_size=256, grad_every=4, influence_every=4, cartography=(x, yi),
                                             report_fn=report, verbose=0)
        tr, diag, x, y = _train(factory, flip=FLIP, lr=1e-2)
        _RUN.update(trainer=tr, diag=diag, x=x, y=y)
    return _RUN


# ══════════════════════════════════════════════════════════════════════════════════════════════════
class DiagnoseModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tensorflow as tf
        cls.ns = _ns()
        cls.x, cls.y = _data(160, 5)
        tf.keras.utils.set_random_seed(0)
        cls.models = {enc: cls.ns["build_model_fn"](T_LEN, N_FEAT, config=_model_cfg(cls.ns, enc)) for enc in ("transformer", "gru")}

    def test_all_sections_shapes(self):
        rep = self.ns["diagnose_model"](self.models["transformer"], self.x, self.y)
        self.assertEqual(set(self.ns["DIAG_SECTIONS"]) - set(rep), set())
        self.assertEqual(set(rep["classification"]["head"]), {"high_class", "low_class"})
        self.assertEqual(set(rep["nig"]["target"]), {"high", "low"})
        att = rep["attention"]
        self.assertEqual(len(att), 2)                                    # كتلة واحدة × رأسا انتباه (num_heads=2) × ... = 2 صفوف
        self.assertTrue((att["entropy_ratio"].between(0, 1)).all() and (att["sync_err"] < 1e-4).all(), att)
        self.assertEqual(set(rep["sensitivity"]["feature"]), {f"f{i}" for i in range(N_FEAT)})
        np.testing.assert_allclose(rep["sensitivity"].groupby("head")["share"].sum(), 1.0, atol=1e-6)
        self.assertTrue({"instance_norm", "block_1", "readout_fc", "nig_high.fc"} <= set(rep["activations"]["layer"]))
        self.assertGreater(len(rep["samples"]), 0)
        self.assertTrue(rep["samples"]["row"].between(0, len(self.x) - 1).all())

    def test_gru_has_no_attention_and_permutation_sensitivity(self):
        rep = self.ns["diagnose_model"](self.models["gru"], self.x, self.y, sections=("attention", "sensitivity"), sensitivity="permutation")
        self.assertEqual(len(rep["attention"]), 0)
        self.assertEqual(set(rep["sensitivity"]["head"]), {"high_class", "low_class", "high_reg", "low_reg"})

    def test_unknown_names_raise(self):
        with self.assertRaises(ValueError):
            self.ns["diagnose_model"](self.models["gru"], self.x, self.y, sections=("nope",))
        with self.assertRaises(ValueError):
            self.ns["diagnose_model"](self.models["gru"], self.x, None, sections=("sensitivity",), sensitivity="permutation")
        with self.assertRaises(ValueError):
            self.ns["model_health_verdicts"]({}, thresholds={"not_a_threshold": 1})
        with self.assertRaises(TypeError):
            self.ns["TrainingDiagnostics"](not_an_option=1)
        with self.assertRaises(ValueError):
            self.ns["TrainingDiagnostics"](influence_scope="everything")

    def test_model_untouched_by_diagnostics(self):
        """التشخيص لا يغيّر النموذج: الطبقات/الأوزان/المخارج وحالة random العامة في بايثون."""
        m = self.models["transformer"]
        names, weights = [l.name for l in m.layers], [w.numpy().copy() for w in m.weights]
        before = m(self.x[:8], training=False)
        st = random.getstate()
        self.ns["diagnose_model"](m, self.x, self.y)
        self.ns["layer_probe_report"](m, self.x, self.y, self.x, self.y, max_train=100, max_val=100, n_rand=1)
        self.assertEqual(st, random.getstate())
        self.assertEqual(names, [l.name for l in m.layers])
        self.assertTrue(all(np.array_equal(a, w.numpy()) for a, w in zip(weights, m.weights)))
        after = m(self.x[:8], training=False)
        for k in before:
            np.testing.assert_array_equal(before[k].numpy(), after[k].numpy())

    def test_verdicts_flag_a_biased_class_head(self):
        """رأس تصنيف مُجبَر على «إيجابي دائماً» ⇒ 🚨 توازن الفئات + ⚠️ تشبّع؛ والنموذج السليم لا 🚨 فيه."""
        import tensorflow as tf
        ns = self.ns
        m = ns["build_model_fn"](T_LEN, N_FEAT, config=_model_cfg(ns, "gru"))
        out = m.get_layer("y_high_class_logits")
        out.bias.assign(np.full(out.bias.shape, 25.0, "float32"))
        rep = ns["diagnose_model"](m, self.x, self.y, sections=("heads", "classification"))
        v = ns["model_health_verdicts"](rep).set_index("check")
        self.assertEqual(v.loc["توازن الفئات المتوقَّعة", "status"], "🚨")
        self.assertEqual(v.loc["تشبّع احتمالات التصنيف", "status"], "⚠️")
        self.assertIsInstance(tf, object)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ns["print_verdicts"](v.reset_index())
        self.assertIn("🚨", buf.getvalue())

    def test_random_init_copy_is_deterministic_and_different(self):
        ns, m = self.ns, self.models["gru"]
        a, b, c = ns["random_init_copy"](m, 1), ns["random_init_copy"](m, 1), ns["random_init_copy"](m, 2)
        self.assertEqual([l.name for l in a.layers], [l.name for l in m.layers])
        wa, wb, wc, wm = ([w.numpy() for w in x.weights] for x in (a, b, c, m))
        self.assertTrue(all(np.array_equal(p, q) for p, q in zip(wa, wb)))
        self.assertFalse(all(np.array_equal(p, q) for p, q in zip(wa, wc)))
        self.assertFalse(all(np.array_equal(p, q) for p, q in zip(wa, wm)))


# ══════════════════════════════════════════════════════════════════════════════════════════════════
class TrainingDiagnosticsTests(unittest.TestCase):
    def test_training_unchanged_by_recorder(self):
        """نفس البذرة: أوزان بعد fit بالمسجّل (يقيس كل دفعة، تدرّجات + تأثير + خريطة) = أوزان بلا مسجّل (حرفياً إن كان التشغيلان
        المرجعيان متطابقين، وإلا بتسامح ضيّق)."""
        ns = _ns()

        def factory(x, y, yi):
            xv, yv = _data(100, 3)
            return ns["TrainingDiagnostics"](probe=(xv, yv), grad_every=1, influence_every=1, cartography=(x, yi), verbose=0)
        kw = dict(n=128, epochs=3, lr=3e-3, shuffle=True)
        a, _, _, _ = _train(None, **kw)
        b, diag, _, _ = _train(factory, **kw)
        c, _, _, _ = _train(None, **kw)
        wa, wb, wc = ([w.numpy() for w in t.model.weights] for t in (a, b, c))
        self.assertGreater(diag.stats()["n_influence_measures"], 0)
        if all(np.array_equal(p, q) for p, q in zip(wa, wc)):
            for p, q in zip(wa, wb):
                np.testing.assert_array_equal(p, q)
        else:
            for p, q in zip(wa, wb):
                np.testing.assert_allclose(p, q, rtol=1e-5, atol=1e-6)

    def test_batch_losses_match_history_and_shapes(self):
        r = _main_run()
        diag, trainer = r["diag"], r["trainer"]
        s = diag.summary()
        n_batches = len(r["x"]) // BS
        self.assertEqual(len(s["batches"]), 30 * n_batches)
        h = trainer.history.history["loss"]
        per_epoch = s["batches"].groupby("epoch")["loss"].mean().to_numpy()
        np.testing.assert_allclose(per_epoch, h, rtol=1e-3)               # متوسط الخسائر المُستخلصة لكل حقبة = متوسط Keras
        self.assertEqual(len(s["grad"]), 30 * (n_batches // 4))
        gcols = [c for c in s["grad"].columns if c.startswith("gn_trunk:")]
        self.assertEqual(set(gcols), {f"gn_trunk:{t}" for t in ("high_reg", "high_class", "low_reg", "low_class")})
        self.assertTrue((s["grad"][gcols] > 0).all().all())
        self.assertGreaterEqual(s["grad"]["dominance"].min(), 1.0)
        self.assertIn("cos:high_reg|high_class", s["grad"].columns)
        self.assertTrue({"gru_1", "readout_fc", "nig_high", "class_fc_high"} <= set(s["groups"]["group"]))
        self.assertTrue((s["groups"]["update_ratio"].dropna() >= 0).all() and s["groups"]["update_ratio"].notna().any())
        tr = s["group_trend"]
        self.assertTrue(np.isfinite(tr["grad_trend"]).all() or tr["grad_trend"].notna().any())
        self.assertEqual(len(s["influence"]), 30 * (n_batches // 4))
        self.assertTrue({"influence", "cos", "infl:high_class"} <= set(s["influence"].columns))
        st = s["stats"]
        self.assertEqual(st["missed_measures"], 0)
        self.assertEqual(set(st["head_trunk_norm_median"]), {"high_reg", "high_class", "low_reg", "low_class"})
        self.assertEqual(len(s["cartography_summary"]), 4)
        self.assertEqual(len(s["cartography"]), len(r["x"]))

    def test_tap_aligns_batches_with_steps(self):
        """الدفعة المُودَعة لخطوة k هي نفسها التي دخلت التدريب فيها: فهارس الدفعة الموضع FLIP = العيّنات 96..127 بلا خلط."""
        diag = _main_run()["diag"]
        t = diag.influence_table()
        top = diag.top_batches("harmful", k=len(t))
        pos3 = top[top["batch"] == FLIP]
        self.assertEqual(len(pos3), 30)
        for sm in pos3["samples"]:
            self.assertEqual(sm, list(range(FLIP * BS, (FLIP + 1) * BS)))

    def test_flipped_batch_ranked_harmful(self):
        """دفعة تسمياتها معكوسة (تصنيف وانحدار) هي الأضرّ بتوافق نفس المهمة، وعيّناتها الأضرّ تأثيراً."""
        diag = _main_run()["diag"]
        t = diag.influence_table()
        late = t[t["epoch"] >= 10]                                         # بعد أن يتعلّم النموذج القاعدة
        for tgt in ("high_class", "low_class"):
            by_pos = late.groupby("batch")[f"infl:{tgt}"].mean()
            self.assertEqual(int(by_pos.idxmin()), FLIP, by_pos.sort_values().head(3))
            si = diag.sample_influence(target=tgt)
            worst = set(si["idx"].head(BS))
            self.assertGreaterEqual(len(worst & set(range(FLIP * BS, (FLIP + 1) * BS))), BS // 2, tgt)
        top = diag.top_batches("harmful", k=30, target="high_class")
        self.assertGreaterEqual(int((top["batch"] == FLIP).sum()), 8, top["batch"].value_counts().head())     # حاضرة في أضرّ 30 قياساً مراراً
        helpful = diag.top_batches("helpful", k=3, target="high_class")
        self.assertNotIn(FLIP, set(helpful["batch"]))
        self.assertLessEqual(top.iloc[0]["infl:high_class"], helpful.iloc[0]["infl:high_class"])

    def test_cartography_marks_flipped_samples_hard(self):
        r = _main_run()
        diag = r["diag"]
        flipped = sorted(range(FLIP * BS, (FLIP + 1) * BS))
        for tgt in ("high_class", "low_class"):
            h = diag.hardest_samples(tgt, k=BS)
            self.assertGreaterEqual(len(set(h["idx"]) & set(flipped)), BS // 4, tgt)       # + عيّنات منطقة الضجيج الحقيقية تنافسها
            tab = diag.cartography_table().set_index("idx")
            conf = tab[f"conf_mean:{tgt}"]
            self.assertLess(conf.loc[flipped].mean(), conf.mean() - 0.2)
            self.assertLess((conf < conf.loc[flipped].mean()).mean(), 0.10)                 # أدنى من 90% من كل العيّنات
        summ = diag.cartography_summary().set_index("target")
        self.assertTrue(((summ[["easy", "ambiguous", "hard", "middle"]].sum(axis=1) - 1).abs() < 1e-9).all())
        with self.assertRaises(KeyError):
            diag.hardest_samples("nope")

    def test_report_hook_collects_layer_reports(self):
        diag = _main_run()["diag"]
        self.assertEqual(len(diag.layer_reports), 1)                       # report_every=0: عند نهاية التدريب فقط
        epoch, probe = diag.layer_reports[0]
        self.assertEqual(epoch, 29)
        self.assertTrue({"layer", "head", "skill", "null_max", "rand_skill"} <= set(probe.columns))


# ══════════════════════════════════════════════════════════════════════════════════════════════════
class LayerEvidenceTests(unittest.TestCase):
    """أدلة لا تخمين: كل فرق يأتي مع أثر وخط أساس، ونموذج عشوائي التهيئة لا يُعلَّم."""

    @classmethod
    def setUpClass(cls):
        cls.ns = _ns()
        cls.model = _main_run()["trainer"].model
        cls.rand = cls.ns["random_init_copy"](cls.model, 99)
        cls.x, cls.y = _main_run()["x"], _main_run()["y"]
        cls.xv, cls.yv = _data(600, 3)

    def test_probe_trained_beats_random_baseline(self):
        ns = self.ns
        pr = ns["layer_probe_report"](self.model, self.x, self.y, self.xv, self.yv, max_train=1024, max_val=600, n_rand=1)
        self.assertTrue({"layer", "stage", "head", "metric", "skill", "null_mean", "null_max", "se", "z", "rand_skill",
                         "significant", "beyond_random"} <= set(pr.columns))
        g = pr[(pr["head"] == "high_class") & (pr["stage"].isin(["readout", "trunk", "head"]))]
        self.assertTrue(g["significant"].all(), g)                                         # المعلومة موجودة بعد الإحماء فوق التسميات المخلوطة
        self.assertTrue(g["beyond_random"].any(), g)                                       # وتجاوزت النسخة العشوائية في عمق ما
        self.assertTrue((g["skill"] > g["null_max"]).all())
        self.assertTrue(set(pr[pr["stage"] == "head"]["layer"]) <= {"class_fc_high", "class_fc_low", "nig_high.fc", "nig_low.fc"})
        lines = ns["layer_probe_verdict"](pr)
        self.assertEqual(len(lines), 4)
        # نموذج عشوائي: معلومة المدخل ظاهرة (significant) لكن لا شيء «فوق النسخة العشوائية»
        pr_r = ns["layer_probe_report"](self.rand, self.x, self.y, self.xv, self.yv, max_train=1024, max_val=600, n_rand=2)
        self.assertEqual(int(pr_r["beyond_random"].sum()), 0, pr_r[pr_r["beyond_random"]])

    def test_compare_correct_vs_wrong_flags_planted_structure(self):
        """نزرع بنية: تسميات y = تنبؤ النموذج حيث ثقته عالية، وعكسه حيث ثقته منخفضة ⇒ الصواب = «ثقة عالية»، وهذا مُرمَّز فعلاً في تنشيطات
        الرأس/الجذع (دالة خطّية فيها) لا في ميزات عشوائية ⇒ تُرفع علامة في طبقة واحدة على الأقل فوق الخلط والنسخة العشوائية."""
        ns, m = self.ns, self.model
        p = ns["_predict"](m, self.xv)["y_high_class_logits"].reshape(-1)
        conf = np.abs(p - 0.5)
        pred = (p >= 0.5).astype("float32")
        y2 = dict(self.yv)
        y2["y_high_class"] = np.where(conf > np.median(conf), pred, 1 - pred).astype("float32")
        cmp_ = ns["layer_compare_report"](m, self.xv, y2, heads=["high_class"], n_perm=40, n_rand=3)
        t = cmp_["table"]
        self.assertTrue({"effect_d", "sep_auc", "null_q95_d", "null_q95_sep", "rand_d", "rand_sep", "flag", "excess_auc"} <= set(t.columns))
        fl = t[(t["kind"] == "activation") & t["flag"]]
        self.assertGreater(len(fl), 0, t)
        self.assertTrue((fl["sep_auc"] > fl["null_q95_sep"]).all() and (fl["sep_auc"] > fl["rand_sep"]).all())
        self.assertIn("class_fc_high", set(fl["layer"]) | set(fl["layer"]))                  # الرأس الذي يرمّز الثقة مباشرةً
        self.assertIn("أكثر طبقة تفرّق", cmp_["verdict"][0])
        # وعلى تسميات حقيقية (بلا بنية مزروعة): الجداول سليمة الأشكال
        c2 = ns["layer_compare_report"](m, self.xv, self.yv, heads=["high_class", "low_class"], n_perm=20, n_rand=2)
        self.assertEqual(set(c2["table"]["head"]), {"high_class", "low_class"})
        self.assertEqual(set(c2["features"]["feature"]), {f"f{i}" for i in range(N_FEAT)})

    def test_random_init_model_is_never_flagged(self):
        """النموذج عشوائي التهيئة لا يُعلَّم في أي طبقة ولا ميزة (الأدلة: خلط التسميات + نسخ عشوائية أخرى)."""
        for seed in (99, 100):
            rm = self.ns["random_init_copy"](self.model, seed)
            cmp_ = self.ns["layer_compare_report"](rm, self.xv, self.yv, n_perm=40, n_rand=3, seed=seed)
            self.assertEqual(int(cmp_["table"]["flag"].sum()), 0, cmp_["table"][cmp_["table"]["flag"]])
            self.assertEqual(int(cmp_["features"]["flag"].sum()), 0, cmp_["features"][cmp_["features"]["flag"]])
            self.assertTrue(all("لا طبقة تفرّق" in v or v.startswith(("high_", "low_")) for v in cmp_["verdict"]))

    def test_compare_input_validation(self):
        with self.assertRaises(ValueError):
            self.ns["layer_compare_report"](self.model, self.xv, self.yv, heads=["nope"])


# ══════════════════════════════════════════════════════════════════════════════════════════════════
def _module_src(module):
    """نص وحدة من حزمة workflow/ (كود main.ipynb سابقاً)."""
    with open(os.path.join(ROOT, "workflow", f"{module}.py"), encoding="utf-8") as f:
        return f.read()


def _module_into(module, ns):
    """يشغّل وحدة workflow/ كاملة في ns (ما تفعله workflow.load_into لوحدة واحدة)."""
    _quiet(exec, compile(_module_src(module), f"workflow/{module}.py", "exec", dont_inherit=True), ns)
    return ns


def _extract_defs(module, names, ns):
    """يعرّف دوال main الصغيرة (من وحدة workflow/) في ns بالاستخراج عبر ast — الكود الحقيقي لا نسخة."""
    tree = ast.parse(_module_src(module))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            _quiet(exec, compile(ast.Module([node], []), f"workflow/{module}.py", "exec"), ns)
    return ns


class MainWrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tensorflow as tf  # noqa: F401
        ns = dict(_ns())
        ns.update(MODEL_TF="1h", LAST_COLUMNS=["timestamp"], main_config={"run": {"batch_size": BS}, "targets": {}})
        _extract_defs("splits", {"_tfs_of", "model_x"}, ns)
        _extract_defs("retarget", {"_split_parts"}, ns)
        _extract_defs("batches", {"make_shuffled_dataset", "_to_unit_label", "make_eval_dataset", "_to_float32_inputs"}, ns)
        ns["main_config"] = ns["build_config"]({"run": {"run_dir": tempfile.mkdtemp(), "epochs": 1, "batch_size": BS, "verbose": 0,
                                                        "train_mode": "new"},
                                                "optimizer": {"lr_initial": 1e-2}, "targets": ns["build_target_configs"](("high", "low"))})
        ns["_y_for"] = lambda sp: dict(sp["y"])
        _module_into("diagnostics", ns)
        _module_into("capacity", ns)
        cls.ns = ns
        x, y = _data(256, 0)
        xv, yv = _data(200, 3)
        mk = lambda x, y: {"X_1h": x, "y": y, "last_candles": np.arange(len(x), dtype="float64")[:, None]}  # noqa: E731
        cls.train, cls.val, cls.test = mk(x, y), mk(xv, yv), {"A": mk(*_data(100, 8)), "B": mk(*_data(100, 9))}
        cls.model = _main_run()["trainer"].model

    def test_diag_xy_samples_before_concat_and_handles_asset_dict(self):
        X, y = self.ns["_diag_xy"](self.test, None, max_n=60)
        self.assertEqual((len(X), len(y["y_high_class"])), (60, 60))
        X2, _ = self.ns["_diag_xy"](self.train, None, max_n=10_000)
        self.assertEqual(len(X2), 256)

    def test_model_health_report(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = self.ns["model_health_report"](self.model, self.train, self.val, recorder=_main_run()["diag"], max_n=200)
        v = out["verdicts"]
        self.assertTrue({"check", "status", "detail", "action"} <= set(v.columns))
        self.assertTrue(set(v["status"]) <= {"✅", "⚠️", "🚨"})
        self.assertIn("هيمنة رأس على الجذع", set(v["check"]))                 # بنود المسجّل حاضرة
        self.assertIn("خريطة البيانات", set(v["check"]))
        self.assertIn("سعة مقابل عيّنات فعّالة", set(v["check"]))             # سطر السعة (رخيص: عيّنات فعّالة + مرجع خطّي)
        self.assertGreater(len(out["capacity"]), 0)
        self.assertIn("تقرير صحّة النموذج", buf.getvalue())

    def test_model_layer_report(self):
        out = self.ns["model_layer_report"](self.model, self.train, self.val, self.test, recorder=_main_run()["diag"],
                                            max_train=256, max_val=200, max_eval=150, n_rand=2, n_perm=20, verbose=False)
        tab = out["table"]
        self.assertTrue({"layer", "stage", "dead_unit_frac", "probe:high_class", "sep:high_class", "⚑", "grad_trend"} <= set(tab.columns), tab.columns)
        self.assertIn("trunk_norm", set(tab["layer"]))
        self.assertGreater(len(out["verdict"]), 0)
        self.assertTrue(any(v.startswith("📐") for v in out["verdict"]), out["verdict"])
        self.assertIsInstance(out["compare"]["features"], pd.DataFrame)

    def test_make_training_diagnostics_does_not_change_dataset(self):
        diag, ds = self.ns["make_training_diagnostics"](self.train, self.val, probe_size=64, cartography_size=100, grad_every=2,
                                                        influence_every=0, verbose=0)
        x, y = next(iter(ds))
        self.assertEqual(x.shape, (BS, T_LEN, N_FEAT))
        self.assertIn("__sample_idx__", y)
        with self.assertRaises(TypeError):
            self.ns["make_training_diagnostics"](self.train, self.val, bogus=1)


# ══════════════════════════════════════════════════════════════════════════════════════════════════
def _panel_split(n_assets, steps, rho, seed, n_feat=6, informative=(0,), noise_labels=False):
    """قسم بشكل split_data: عملة عملة، كل عملة زمنياً. هدف الانحدار = عامل مشترك (ارتباط rho بين الأصول) + ضجيج مستقلّ؛ تصنيف من الميزة 0
    (آخر خطوة − متوسط النافذة) إلا إن noise_labels. الميزة الأخيرة تحمل رتبة الزمن (للتحقّق من زمنية منحنى التعلّم)."""
    rng = np.random.default_rng(seed)
    factor = rng.normal(size=steps)
    xs, ys_reg, ys_cls, ts = [], [], [], []
    for a in range(n_assets):
        x = rng.normal(size=(steps, T_LEN, n_feat)).astype("float32")
        sig = x[:, -1, informative[0]] - x[:, :-1, informative[0]].mean(1)
        cls = (rng.normal(size=steps) > 0) if noise_labels else (sig + rng.normal(size=steps) * 0.2 > 0)
        reg = np.sqrt(rho) * factor + np.sqrt(1 - rho) * rng.normal(size=steps)
        x[:, :, -1] = (np.arange(steps) / steps)[:, None]
        xs.append(x), ys_reg.append(reg), ys_cls.append(cls), ts.append(np.arange(steps, dtype="float64"))
    cat = lambda L: np.concatenate(L)  # noqa: E731
    y = {"y_high_reg": cat(ys_reg).astype("float32"), "y_high_class": cat(ys_cls).astype("float32")}
    return {"X_1h": cat(xs), "y": y, "last_candles": cat(ts)[:, None]}


def _flat_curve(slope=0.0):
    df = pd.DataFrame({"fraction": [0.5, 1.0], "n": [100, 200], "val_skill": [0.5, 0.5 + slope], "train_skill": [0.6, 0.6]})
    df.attrs["slope_per_doubling"] = slope
    return df


class CapacityToolsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        MainWrapperTests.setUpClass()
        cls.ns = MainWrapperTests.ns

    def _build_fn(self):
        ns = self.ns
        cfg = dict(price_targets=("high",), head_types={"high": ["nig_regression", "binary_classification"]}, d_model=16,
                   num_layers=1, head_hidden=16, class_head_hidden=8, encoder="gru", dropout=0.1)
        return lambda k: ns["build_model_fn"](T_LEN, k, config=cfg)

    def _cfg(self):
        ns = self.ns
        return ns["build_config"]({"run": {"run_dir": tempfile.mkdtemp(), "epochs": 1, "batch_size": BS, "verbose": 0, "train_mode": "new"},
                                   "optimizer": {"lr_initial": 1e-2}, "targets": ns["build_target_configs"](("high",))})

    def test_effective_sample_size_discounts_correlated_assets(self):
        ess = self.ns["effective_sample_size"]
        indep = ess(_panel_split(8, 120, 0.0, 0), window=T_LEN, n_params=5000)
        corr = ess(_panel_split(8, 120, 0.9, 0), window=T_LEN, n_params=5000)
        self.assertEqual(indep["n_rows"], corr["n_rows"])
        self.assertEqual(indep["n_assets"], 8)
        self.assertLess(indep["rho_bar"], 0.2)
        self.assertGreater(corr["rho_bar"], 0.6)
        self.assertGreater(indep["n_eff_assets"], 5.0)
        self.assertLess(corr["n_eff_assets"], 2.0)                                  # ثمانية أصول متّسقة الحركة ≈ أصل واحد ونصف
        self.assertLess(corr["n_effective"], indep["n_effective"] / 2)
        self.assertAlmostEqual(indep["n_time_indep"], 120 / T_LEN, places=6)         # خطوات غير متداخلة = المدى ÷ النافذة
        self.assertAlmostEqual(corr["eff_per_param"], corr["n_effective"] / 5000)
        self.assertAlmostEqual(corr["eff_per_feature"], corr["n_effective"] / 6)
        self.assertTrue(corr["verdict"].startswith("🚨"))                              # 5000 معامل ≫ العيّنات الفعّالة
        small = ess(_panel_split(8, 120, 0.0, 0), window=T_LEN, n_params=10)
        self.assertTrue(small["verdict"].startswith(("✅", "⚠️")))
        self.assertEqual(ess(_panel_split(3, 60, 0.0, 1), window=T_LEN, horizon=20)["n_time_indep"], 3.0)   # أفق الهدف يزيد التداخل

    def test_simple_baseline_signal_vs_noise(self):
        ns = self.ns
        sig = ns["simple_baseline"](_panel_split(4, 150, 0.0, 0), _panel_split(4, 100, 0.0, 1))
        noi = ns["simple_baseline"](_panel_split(4, 150, 0.0, 0, noise_labels=True), _panel_split(4, 100, 0.0, 1, noise_labels=True))
        a, b = sig.set_index("head").loc["high_class"], noi.set_index("head").loc["high_class"]
        self.assertGreater(a["skill"], 0.7)
        self.assertGreater(a["z"], 3.0)
        self.assertLess(abs(b["z"]), 3.0)
        self.assertLess(abs(b["null_mean"] - 0.5), 0.1)

    def test_feature_count_sweep_ranks_without_test_and_has_null(self):
        import inspect
        ns = self.ns
        self.assertNotIn("test_split", inspect.signature(ns["feature_count_sweep"]).parameters)    # test لا يدخل الترتيب أصلاً
        tr, va = _panel_split(4, 128, 0.0, 0, n_feat=6), _panel_split(4, 100, 0.0, 1, n_feat=6)
        names = [f"f{i}" for i in range(6)]
        sw = _quiet(ns["feature_count_sweep"], self._build_fn(), tr, va, names, ks=(1, 6, 99), epochs=5, seed=0, config=self._cfg(), verbose=False)
        self.assertEqual(list(sw["k"]), [1, 6])                                       # 99 قُصّت إلى 6 (ودُمجت مع 6)
        self.assertEqual(sw.iloc[0]["features"], ["f0"])                              # الميزة الوحيدة المعلوماتية رُتّبت أولاً (train فقط)
        self.assertTrue({"train_skill", "val_skill", "gap", "baseline_val", "null_train", "null_val", "z", "se"} <= set(sw.columns))
        self.assertGreater(sw.iloc[0]["val_skill"], 0.6)
        self.assertGreater(sw.iloc[0]["z"], 3.0)
        self.assertLess(abs(sw["null_val"] - 0.5).max(), 3 * sw["se"].max())               # تسميات مخلوطة ⇒ val ≈ 0.5 (لا تسرّب)
        self.assertGreater(sw.iloc[0]["baseline_val"], 0.7)
        with self.assertRaises(ValueError):
            ns["feature_count_sweep"](self._build_fn(), tr, va, names, rank_by="test")

    def test_learning_curve_is_chronological(self):
        ns = self.ns
        seen = []
        orig = ns["_capacity_fit"]
        ns["_capacity_fit"] = lambda builder, X, y, Xv, yv, *a, **k: (seen.append(np.asarray(X)[:, 0, -1].copy()), orig(builder, X, y, Xv, yv, *a, **k))[1]
        try:
            tr, va = _panel_split(3, 128, 0.0, 0), _panel_split(3, 100, 0.0, 1)
            lc = _quiet(ns["learning_curve"], self._build_fn(), tr, va, fractions=(0.25, 0.5, 1.0), epochs=4, config=self._cfg(), verbose=False)
        finally:
            ns["_capacity_fit"] = orig
        self.assertEqual(list(lc["fraction"]), [0.25, 0.5, 1.0])
        self.assertEqual(list(lc["n"]), [96, 192, 384])
        self.assertGreater(seen[0].min(), 0.7)                                          # 25% = أحدث الصفوف (رتبة الزمن عالية)، لا عيّنة عشوائية
        self.assertGreater(seen[1].min(), 0.45)
        self.assertLess(seen[2].min(), 0.01)
        self.assertTrue(np.isfinite(lc.attrs["slope_per_doubling"]))
        with self.assertRaises(ValueError):
            ns["learning_curve"](self._build_fn(), tr, va, fractions=(0.0, 1.0))
        with self.assertRaises(ValueError):
            ns["learning_curve"](self._build_fn(), tr, va, anchor="random")

    def test_capacity_verdict_separates_setup_failure_from_no_signal(self):
        v = self.ns["capacity_verdict"]
        base_sig = pd.DataFrame([{"head": "h_class", "metric": "auc", "skill": 0.80, "null_mean": 0.5, "null_max": 0.52, "z": 12.0}])
        base_nul = pd.DataFrame([{"head": "h_class", "metric": "auc", "skill": 0.51, "null_mean": 0.5, "null_max": 0.52, "z": 0.6}])
        ess_bad = {"n_params": 50000, "n_effective": 300.0, "n_rows": 90000, "rho_bar": 0.7, "n_features": 40, "eff_per_param": 0.006, "eff_per_feature": 7.5}
        ess_ok = dict(ess_bad, n_params=1000, eff_per_param=300.0, n_effective=3e5, eff_per_feature=7500.0)
        sw_flat = pd.DataFrame({"k": [1, 5], "val_skill": [0.51, 0.50], "se": [0.02, 0.02], "z": [0.4, 0.0], "gap": [0.05, 0.4],
                                "null_val": [0.5, 0.51], "null_train": [0.55, 0.9]})
        # لا إشارة: مرجع صفري، مسح مسطّح، بلا أعلام سعة
        self.assertTrue(v(baseline=base_nul, sweep=sw_flat.assign(null_train=0.52), curve=_flat_curve(), model_skill=0.5)[0].startswith("➖ لا إشارة"))
        # إخفاق سعة: النموذج تحت المرجع الخطّي القويّ
        low = v(ess=ess_ok, baseline=base_sig, model_skill=0.6)
        self.assertTrue(low[0].startswith("⚠️ إخفاق إعداد (سعة)") and "تحت المرجع الخطّي" in low[0], low)
        # إخفاق سعة: معاملات ≫ عيّنات فعّالة + حفظ تسميات مخلوطة
        mem = v(ess=ess_bad, sweep=sw_flat, baseline=base_nul)
        self.assertTrue(mem[0].startswith("⚠️ إخفاق إعداد (سعة)") and "معامل" in mem[0] and "يحفظ" in mem[0], mem)
        # خفض الميزات يحسّن val
        gain = pd.DataFrame({"k": [1, 40], "val_skill": [0.70, 0.52], "se": [0.02, 0.02], "z": [10.0, 1.0], "gap": [0.05, 0.3],
                             "null_val": [0.5, 0.5], "null_train": [0.5, 0.5]})
        self.assertIn("خفض الميزات يحسّن", v(sweep=gain)[0])
        # منحنى لا يزال صاعداً
        self.assertIn("منحنى التعلّم ما زال صاعداً", v(curve=_flat_curve(slope=0.05))[0])
        # إشارة سليمة
        ok = v(ess=ess_ok, baseline=base_sig, model_skill=0.8)
        self.assertTrue(ok[0].startswith("✅ إشارة"), ok)
        with self.assertRaises(ValueError):
            v(thresholds={"nope": 1})

    def test_end_to_end_verdicts_on_real_runs(self):
        """بيانات بلا معلومة ⇒ «لا إشارة»؛ ومعلومة قوية ⇒ «إشارة» — بالمسح والمنحنى الحقيقيين."""
        ns = self.ns
        for noise, expect in ((True, ("➖ لا إشارة",)), (False, ("✅ إشارة", "ℹ️ توجد إشارة"))):
            tr = _panel_split(4, 128, 0.0, 0, noise_labels=noise)
            va = _panel_split(4, 100, 0.0, 1, noise_labels=noise)
            out = _quiet(ns["capacity_report"], self._build_fn(), tr, va, ks=(1, 6), fractions=(0.5, 1.0), epochs=5,
                         config=self._cfg(), verbose=False)
            self.assertTrue(any(line.startswith(expect) for line in out["verdict"]), (noise, out["verdict"]))
            self.assertEqual(any(line.startswith("➖") for line in out["verdict"]), noise)
            self.assertEqual(set(out), {"ess", "baseline", "sweep", "curve", "verdict"})


if __name__ == "__main__":
    unittest.main()
