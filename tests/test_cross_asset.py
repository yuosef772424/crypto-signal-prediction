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


class PanelModelTests(unittest.TestCase):
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
