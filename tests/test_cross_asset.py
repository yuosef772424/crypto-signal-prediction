"""اختبارات نموذج اللوحة عبر العملات (سريعة، بيانات تركيبية، بلا Drive):  python -m unittest tests.test_cross_asset -v"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cross_asset import selftest as st  # noqa: E402


class PanelDataTests(unittest.TestCase):
    def test_grouping(self):
        self.assertTrue(st.test_grouping())

    def test_batching(self):
        self.assertTrue(st.test_batching())

    def test_subset_keeps_same_coins_across_splits(self):
        a = st.synthetic_split(seed=0).subset(last_days=10, coins=5, seed=7)
        b = st.synthetic_split(seed=0, start_day=18100).subset(last_days=10, coins=5, seed=7)
        self.assertLessEqual(len(set(a.assets)), 5)
        self.assertEqual(set(a.assets), set(b.assets))
        self.assertTrue(a.check()["ok"])
        self.assertEqual(a.n_days, 10)

    def test_class_label_encoding_agnostic(self):
        """خط الأنابيب يُصدر _class بترميز 1/0، وretarget_splits (main) وبيانات أقدم بـ ±1 — كلاهما يعطي نفس ycls."""
        import numpy as np
        from cross_asset.data import PanelSplit
        ps = st.synthetic_split(seed=3)                      # يبني تسميات ±1
        for enc in ("pm1", "unit"):
            y = {}
            for i, t in enumerate(ps.targets):
                up = ps.ycls[:, i]
                y[f"y_{t}_class"] = up * 2.0 - 1.0 if enc == "pm1" else up.copy()
                y[f"y_{t}_reg"] = ps.yreg[:, i]
            other = PanelSplit(ps.X, y, ps.lc, ps.assets, enc, ps.targets)
            np.testing.assert_array_equal(other.ycls, ps.ycls)
            self.assertEqual(set(np.unique(other.ycls).tolist()), {0.0, 1.0})
        sub = ps.take(np.arange(0, ps.n, 3))                 # take يمرّ بترميز 1/0 ولا يفقد التسميات
        np.testing.assert_array_equal(sub.ycls, ps.ycls[np.arange(0, ps.n, 3)])

    def test_group_width_for_unaligned_timestamps(self):
        """فريم الساعة بـ stride=32: طوابع كل عملة بطور مختلف ⇒ day_ns=32 ساعة يجمع كل عملة مرّة في كل مجموعة، ويبقى
        عبر take/subset، ويطابق أرضية report.prepare(group_ns) وgroup_freq في retarget_splits (pandas floor)."""
        import numpy as np
        import pandas as pd
        from cross_asset.data import DAY_NS, PanelSplit
        from cross_asset.report import prepare
        H, G = 3600 * 10**9, 32 * 3600 * 10**9
        n_coins, n_steps = 6, 20
        ts = np.concatenate([(1_000 * G + c * 5 * H) + np.arange(n_steps) * G for c in range(n_coins)])  # طور 5c ساعة
        assets = np.repeat([f"C{c}" for c in range(n_coins)], n_steps)
        n = len(ts)
        lc = np.c_[np.ones((n, 3)), ts.astype("float64"), np.ones((n, 3))]
        y = {f"y_{t}_{k}": np.zeros(n, "float32") for t in ("high", "low", "close") for k in ("class", "reg")}
        X = np.zeros((n, 4, 2), "float32")
        exact = PanelSplit(X, y, lc, assets, "exact", day_ns=H)
        self.assertEqual(exact.sizes.max(), 1)                               # الطابع الدقيق: مجموعات من عملة واحدة
        ps = PanelSplit(X, y, lc, assets, "g32", day_ns=G)
        self.assertEqual(set(ps.sizes.tolist()), {n_coins})
        self.assertTrue(ps.check()["ok"])
        self.assertEqual(ps.take(np.arange(0, n, 2)).day_ns, G)
        self.assertEqual(ps.subset(last_days=5).n_days, 5)
        np.testing.assert_array_equal(ps.day_key, pd.to_datetime(ts).floor("32h").asi8)
        df = pd.DataFrame({"asset": assets, "timestamp": ts.astype("float64"), "entry": 1.0, "fut_close": 1.0,
                           "last_high": 1.0, "last_low": 1.0, "fut_high": 1.0, "fut_low": 1.0})
        np.testing.assert_array_equal(np.sort(prepare(df, group_ns=G)["timestamp"].unique()), ps.days)
        self.assertEqual(PanelSplit(X, y, lc, assets, "d").day_ns, DAY_NS)   # الافتراضي لم يتغيّر


def _grid_split(n_coins=12, n_groups=30, step_h=8, seed=0, seq_len=8, n_features=5, name="grid"):
    """طوابع على شبكة step_h ساعة مشتركة بين العملات (كمخرَج خط الأنابيب بـ align_windows_to_grid وstride=8)."""
    import numpy as np
    from cross_asset.data import PanelSplit
    rng = np.random.default_rng(seed)
    H = 3600 * 10**9
    g0 = 20_000 * 24 // step_h
    ts = np.concatenate([(g0 + np.arange(n_groups)) * step_h * H for _ in range(n_coins)])
    assets = np.repeat([f"C{c:02d}" for c in range(n_coins)], n_groups)
    n = len(ts)
    close = 100 * np.exp(rng.normal(size=n))
    r, eh, el = rng.normal(scale=0.01, size=(3, n))
    lc = np.c_[close * 1.01, close * 0.99, close, ts.astype("float64"), close * (1 + r), close * (0.99 + el),
               close * (1.01 + eh)]
    y = {}
    for t, fut, last in (("high", 6, 0), ("low", 5, 1), ("close", 4, 2)):
        ret = lc[:, fut] / lc[:, last] - 1
        y[f"y_{t}_reg"], y[f"y_{t}_class"] = ret.astype("float32"), (ret > 0).astype("float32")
    X = rng.normal(size=(n, seq_len, n_features)).astype("float32")
    return PanelSplit(X, y, lc, assets, name, targets=("high", "low"), day_ns=step_h * H)


class DynamicCoinTests(unittest.TestCase):
    """المجموعات بالطابع الدقيق (8h)، عدد العملات العشوائي في التدريب (min_coins)، وتقييم k عملة."""

    def test_exact_8h_grouping_pure_and_full(self):
        import numpy as np
        from cross_asset.data import group_ns_for
        ps = _grid_split(n_coins=12, n_groups=30)
        self.assertEqual(ps.day_ns, group_ns_for("1h", 8))
        c = ps.check()
        self.assertTrue(c["ok"] and c["day_groups_pure"])
        self.assertEqual(c["misaligned_timestamps"], 0)
        self.assertEqual(c["coins_per_day_median"], 12)
        self.assertEqual(c["assets"], 12)
        self.assertEqual(ps.n_days, 30)
        self.assertTrue(all(len(np.unique(ps.ts[ps.day_indices(d)])) == 1 for d in range(ps.n_days)))

    def test_min_coins_sampling(self):
        import numpy as np
        ps = st.synthetic_split(n_assets=30, n_days=60, seed=4)
        kw = dict(batch_samples=200, max_days=6, shuffle=True, seed=3, epoch=2)
        sizes, full = [], []
        for b in ps.iter_batches(min_coins=5, **kw):
            for g in np.unique(b["day"]):
                m = b["day"] == g
                n_day = int(ps.sizes[ps.days == ps.day_key[b["idx"][m][0]]][0])
                k = int(m.sum())
                self.assertLessEqual(k, n_day)
                self.assertGreaterEqual(k, min(5, n_day))
                self.assertEqual(sorted(b["pos"][m].tolist()), list(range(k)))
                self.assertEqual(len(np.unique(ps.day_key[b["idx"][m]])), 1)
                sizes.append(k)
                full.append(n_day)
        self.assertGreater(len(set(sizes)), 10)                 # أحجام متنوّعة فعلاً
        self.assertLess(np.mean(sizes), np.mean(full))
        # حتمي من (seed, epoch)، وmax_coins سقف أعلى
        a = [b["idx"] for b in ps.iter_batches(min_coins=5, **kw)]
        b2 = [b["idx"] for b in ps.iter_batches(min_coins=5, **kw)]
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(a, b2)))
        for b in ps.iter_batches(min_coins=5, max_coins=8, **kw):
            self.assertLessEqual(np.bincount(b["day"]).max(), 8)
        # الافتراضي (None) وval/test (بلا shuffle) = كل عيّنة مرّة واحدة كما كان
        for it in (ps.iter_batches(**kw), ps.iter_batches(200, 6, shuffle=False, min_coins=5)):
            seen = np.concatenate([b["idx"] for b in it])
            np.testing.assert_array_equal(np.sort(seen), np.arange(ps.n))

    def test_chunk_groups_cover_each_sample_once(self):
        """كل عيّنة تُحسب مرّة واحدة بالضبط، وكل جزء فيه k عملة بالضبط من طابع واحد (الباقي يُكمَّل بعملات سياق)."""
        import numpy as np
        ps = _grid_split(n_coins=23, n_groups=10)
        for k in (5, 10, 20, 30, None):
            members, scored = ps.chunk_groups(k, seed=1)
            np.testing.assert_array_equal(np.sort(np.concatenate(scored)), np.arange(ps.n))
            for g, sc in zip(members, scored):
                self.assertEqual(len(np.unique(ps.day_key[g])), 1)
                self.assertEqual(len(g), min(k or 23, 23))
                self.assertTrue(np.isin(sc, g).all())
            seen = np.concatenate([b["idx"][b["score"]] for b in ps.iter_group_batches(members, 50, 4, scored)])
            np.testing.assert_array_equal(np.sort(seen), np.arange(ps.n))
        a, b = ps.chunk_groups(5, 1)[0], ps.chunk_groups(5, 2)[0]
        self.assertFalse(all(np.array_equal(x, y) for x, y in zip(a, b)))

    def test_model_built_once_applies_to_any_group_size(self):
        """نموذج واحد (بُني مرّة) على مجموعات بأحجام 1..30: مخرج المجموعة وحدها = مخرجها داخل دفعة مع مجموعات أخرى
        بأحجام مختلفة (القناع يعزل الحشو)، ومع الانتباه يتغيّر مخرج العملة بتغيّر سياقها، وبدونه لا."""
        import numpy as np
        import tensorflow as tf
        from cross_asset.model import build_panel_model, tiny_encoder
        ps = _grid_split(n_coins=30, n_groups=6)
        rng = np.random.default_rng(0)
        for att in (True, False):
            tf.keras.utils.set_random_seed(2)
            m = build_panel_model(tiny_encoder(8, 5), 8, 5, d_model=16, num_heads=4, cross_attention=att, dropout=0.0,
                                  targets=("high", "low"))
            groups = [np.sort(rng.choice(ps.day_indices(d), k, replace=False)) for d, k in enumerate((1, 2, 5, 17, 30))]
            big = ps.assemble(groups)
            o_big = m({k: big[k] for k in ("x", "day", "pos")}, training=False)["logit"].numpy()
            where = {i: r for r, i in enumerate(big["idx"])}
            for g in groups:
                alone = ps.assemble([g])
                o = m({k: alone[k] for k in ("x", "day", "pos")}, training=False)["logit"].numpy()
                np.testing.assert_allclose(o_big[[where[i] for i in alone["idx"]]], o, atol=1e-5)
            # نفس العملة في سياقين مختلفين من طابعها
            d0 = ps.day_indices(0)
            full = ps.assemble([d0])
            sub = ps.assemble([d0[:5]])
            o_full = m({k: full[k] for k in ("x", "day", "pos")}, training=False)["logit"].numpy()
            o_sub = m({k: sub[k] for k in ("x", "day", "pos")}, training=False)["logit"].numpy()
            same = np.allclose(o_full[:5], o_sub, atol=1e-5)
            self.assertEqual(same, not att)

    def test_evaluate_k_coins(self):
        """جدول k: سياق ≤ k، نفس الصفوف لكل k، k=None = predict العادي؛ وبلا انتباه (B) المقاييس لا تتغيّر بـ k."""
        import tempfile
        import numpy as np
        import tensorflow as tf
        from cross_asset.experiment import evaluate_k_coins
        from cross_asset.model import build_panel_model, tiny_encoder
        from cross_asset.train import PanelTrainer, robust_scales
        ps = _grid_split(n_coins=24, n_groups=20, seed=5)
        tabs = {}
        for att in (True, False):
            tf.keras.utils.set_random_seed(0)
            m = build_panel_model(tiny_encoder(8, 5), 8, 5, d_model=16, num_heads=4, cross_attention=att, dropout=0.0,
                                  targets=ps.targets)
            tr = PanelTrainer(m, dict(batch_samples=100, max_days=4), tempfile.mkdtemp(), robust_scales(ps.yreg), 8, 5,
                              verbose=False)
            base = tr.predict(ps)[0]
            np.testing.assert_allclose(tr.predict(ps, groups=ps.chunk_groups(None))[0], base, atol=1e-5)
            out = tempfile.mkdtemp()
            tabs[att] = evaluate_k_coins(tr, ps, (5, 10, None), draws=2, seed=0, export_dir=out)
            import os
            self.assertTrue(os.path.exists(os.path.join(out, "signals_test_k5.csv.gz")))
        t = tabs[True]
        self.assertEqual(list(t.index), [5, 10, "all"])
        self.assertEqual(t.loc[5, "context_mean"], 5)
        self.assertEqual(t.loc[10, "context_mean"], 10)
        self.assertEqual(t.loc["all", "context_mean"], 24)
        self.assertEqual(t.loc["all", "draws"], 1)
        self.assertIn("ic_asym", t.columns)
        self.assertTrue(np.isfinite(t[["auc_high", "auc_low", "ic_high", "ic_low", "ic_asym"]].to_numpy()).all())
        self.assertGreater(abs(t.loc[5, "auc_high"] - t.loc["all", "auc_high"]), 1e-7)    # الانتباه: السياق يغيّر
        b = tabs[False]
        for col in ("auc_high", "auc_low", "ic_high", "ic_asym"):
            self.assertAlmostEqual(b.loc[5, col], b.loc["all", col], places=5)


class PanelModelTests(unittest.TestCase):
    def test_best_epoch_restore_uses_val_metric(self):
        """الإيقاف المبكر والأفضل على مقياس val (أصغر أفضل لـ val_loss، أكبر أفضل لغيره)، وload_best يعيد أوزان
        تلك الحقبة بالضبط (مقياس val نفسه)، والبصمة لا تتغيّر بإضافة min_coins=None (استئناف التشغيلات القديمة)."""
        import hashlib
        import json
        import tempfile
        import numpy as np
        import tensorflow as tf
        from cross_asset.model import build_panel_model, tiny_encoder
        from cross_asset.train import PanelTrainer, robust_scales
        tr_ps = st.synthetic_split(20, 40, seed=1, signal=2.0)
        va_ps = st.synthetic_split(20, 15, start_day=18200, seed=2, signal=2.0)
        for monitor, sign in (("val_loss", -1), ("val_ic_close", 1)):
            tf.keras.utils.set_random_seed(0)
            m = build_panel_model(tiny_encoder(8, 5), 8, 5, d_model=16, num_heads=4)
            cfg = dict(epochs=5, patience=2, batch_samples=150, max_days=8, lr_warmup_epochs=0, lr_initial=3e-3,
                       lr_schedule={"type": "constant"}, monitor=monitor)
            t = PanelTrainer(m, cfg, tempfile.mkdtemp(), robust_scales(tr_ps.yreg), 8, 5, verbose=False)
            state = t.fit(tr_ps, va_ps)
            h = [r[monitor] for r in state["history"]]
            self.assertEqual(state["best_epoch"], int(np.argmax(sign * np.asarray(h))) + 1, h)
            waits = 0
            for i in range(1, len(h)):
                waits = 0 if sign * h[i] > sign * max(h[:i], key=lambda v: sign * v) else waits + 1
            self.assertTrue(len(h) == cfg["epochs"] or waits >= cfg["patience"], (h, waits))
            t.load_best()
            self.assertAlmostEqual(t.evaluate(va_ps, use_ema=False)[monitor], state["best"], places=5)
        # البصمة القديمة (قبل min_coins) = الجديدة حين min_coins=None
        old = {k: v for k, v in t.cfg.items() if k not in ("epochs", "patience", "min_coins")}
        blob = json.dumps({"cfg": old, **t.fingerprint}, sort_keys=True, default=str)
        self.assertEqual(t._fp(), hashlib.sha1(blob.encode()).hexdigest()[:12])


    def test_suspended_close_high_low_only(self):
        """close معلّق (SUSPENDED_TARGETS في main): رأسان فقط، حدّ IC على low، تصدير وتقرير بلا أعمدة close."""
        import shutil
        import tempfile
        import numpy as np
        from cross_asset.experiment import run_panel_experiment
        tr = st.synthetic_split(20, 30, seed=1, name="train")
        va = st.synthetic_split(20, 12, start_day=18200, seed=2, name="val")
        te = st.synthetic_split(20, 12, start_day=18300, seed=3, name="test")

        def as_split(ps):
            y = {f"y_{t}_{k}": (ps.ycls[:, i] * 2 - 1 if k == "class" else ps.yreg[:, i])
                 for i, t in enumerate(ps.targets) for k in ("class", "reg")}
            return {"X_1D": ps.X, "y": y, "last_candles": ps.lc}
        root = tempfile.mkdtemp(prefix="panel_hl_")
        try:
            res = run_panel_experiment(
                as_split(tr), as_split(va),
                {a: as_split(te.take(np.flatnonzero(te.assets == a))) for a in np.unique(te.assets)}, "1D", lambda: st._TinyBase(8, 5), 8, 5, root,
                variants=("A_ic",), seeds=(0,), model_cfg=dict(d_model=16, num_heads=4),
                train_cfg=dict(epochs=1, batch_samples=150, max_days=8, lr_warmup_epochs=0), verbose=False,
                train_assets=tr.assets, val_assets=va.assets, targets=("high", "low"))
            v, t = res["runs"]["A_ic_s0"]
            self.assertIn("p_up_low", t)
            self.assertNotIn("p_up_close", t)
            self.assertEqual(len(t), te.n)
            self.assertTrue(np.isfinite(t[["p_up_high", "p_up_low", "mu_high", "mu_low"]].to_numpy()).all())
            import json
            with open(f"{root}/panel_A_ic_s0/state.json") as f:
                self.assertIn("val_ic_low", json.load(f)["history"][0])
            self.assertTrue(np.isfinite(res["table"].loc["AUC low", "A_ic_s0"]))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_masking(self):
        self.assertTrue(st.test_masking())

    def test_ic_loss(self):
        self.assertTrue(st.test_ic_loss())

    def test_training_export_resume(self):
        st.test_training()

    def test_attention_uses_same_day_coins(self):
        """ضابط إيجابي: المعلومة في انحراف العملة عن يومها (إزاحة يومية مشتركة كبيرة على الميزة). A يلتقطها في
        AUC المُجمَّع عبر الأيام، وB (كل عملة وحدها) لا يستطيع معرفة الإزاحة."""
        import tensorflow as tf
        from cross_asset.model import build_panel_model, tiny_encoder
        from cross_asset.train import PanelTrainer, robust_scales
        from sklearn.metrics import roc_auc_score
        import tempfile
        kw = dict(n_assets=40, seq_len=8, n_features=5, signal=3.0, day_shift=3.0)
        tr = st.synthetic_split(n_days=150, seed=1, **kw)
        va = st.synthetic_split(n_days=40, start_day=18300, seed=2, **kw)
        auc = {}
        for att in (True, False):
            tf.keras.utils.set_random_seed(0)
            m = build_panel_model(tiny_encoder(8, 5), 8, 5, d_model=16, num_heads=4, cross_attention=att, dropout=0.0)
            cfg = dict(epochs=8, patience=20, batch_samples=200, max_days=8, lr_warmup_epochs=0,
                       lr_schedule={"type": "constant"}, lr_initial=3e-3, ema_window_epochs=0.5, label_smoothing=0.0)
            t = PanelTrainer(m, cfg, tempfile.mkdtemp(), robust_scales(tr.yreg), 8, 5, verbose=False)
            t.fit(tr, va)
            t.load_best()
            logit, _ = t.predict(va)
            auc[att] = roc_auc_score(va.ycls[:, 2], logit[:, 2])
        print(f"\n   AUC close (val): A {auc[True]:.3f} | B {auc[False]:.3f}")
        self.assertGreater(auc[True], auc[False] + 0.05, auc)


if __name__ == "__main__":
    unittest.main()
